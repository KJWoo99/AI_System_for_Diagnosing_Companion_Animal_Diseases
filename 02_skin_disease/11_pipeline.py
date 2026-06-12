"""
투트랙 추론 파이프라인 — 바이너리 YOLO + EfficientNet 분류

Stage 1: YOLO11l-seg  → 병변 위치 탐지 (바이너리: 병변/정상)
Stage 2: EfficientNet-B7 → masked crop 기반 질병 분류 (A4/A6)

역할 분리:
  YOLO:         "어디에 병변이 있는가" (위치 + 유무)
  EfficientNet: "어떤 질병인가" (A4 농포/여드름 vs A6 결절/종괴)

사용법:
  단일 이미지: python 11_pipeline.py <이미지경로.jpg>
  전체 평가:   python 11_pipeline.py --eval
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from ultralytics import YOLO


YOLO_PT   = Path("runs/pet_skin_seg/weights/best.pt")
EFFNET_PT = Path("runs/stage2_efficientnet/best.pt")

# EfficientNet 분류 클래스 (A4, A6)
CLASSES  = ["농포_여드름", "결절_종괴"]   # A4, A6
COLORS   = [(0, 200, 255), (0, 255, 100)]  # 하늘, 초록
IMG_W, IMG_H = 1920, 1080
EFFNET_SIZE  = 600
PAD_RATIO    = 0.15
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# EfficientNet 최종 확률 max가 이 값 미만이면 무증상으로 처리
# (YOLO false positive + EfficientNet 불확실 케이스 필터링)
CONF_THRESHOLD = 0.45


def load_effnet() -> torch.nn.Module:
    model = timm.create_model("efficientnet_b7", pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(str(EFFNET_PT), map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def preprocess_crop(crop: np.ndarray) -> torch.Tensor:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(img, (EFFNET_SIZE, EFFNET_SIZE)).astype(np.float32) / 255.0
    img  = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def masked_crop(img: np.ndarray, mask_data, box_xyxy: list) -> np.ndarray | None:
    x1, y1, x2, y2 = map(int, box_xyxy)
    pw = int((x2 - x1) * PAD_RATIO)
    ph = int((y2 - y1) * PAD_RATIO)
    cx1 = max(0, x1 - pw);  cy1 = max(0, y1 - ph)
    cx2 = min(img.shape[1], x2 + pw); cy2 = min(img.shape[0], y2 + ph)
    if cx2 - cx1 < 5 or cy2 - cy1 < 5:
        return None

    if mask_data is not None:
        mask = cv2.resize(mask_data.cpu().numpy(), (img.shape[1], img.shape[0]))
        result = img.copy()
        result[mask < 0.5] = 128   # 배경 → 회색
    else:
        result = img.copy()
    return result[cy1:cy2, cx1:cx2]


class TwoStagePipeline:
    def __init__(self):
        print("모델 로딩 중...")
        self.yolo   = YOLO(str(YOLO_PT))
        self.effnet = load_effnet()
        print("완료!")

    @torch.no_grad()
    def predict(self, img_path: str, conf: float = 0.2) -> dict:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)

        # ── Stage 1: YOLO (병변 위치 탐지) ──────────────────────
        results = self.yolo.predict(
            source=img_path, imgsz=1280, conf=conf, iou=0.5,
            augment=True, retina_masks=True, verbose=False,
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return {"detections": [], "is_symptomatic": False}

        detections = []

        for i, box in enumerate(result.boxes):
            yolo_conf = float(box.conf.item())
            bbox      = box.xyxy[0].tolist()

            # ── Stage 2: EfficientNet (질병 분류) ──────────────
            mask_data = result.masks.data[i] if result.masks else None
            crop = masked_crop(img, mask_data, bbox)

            if crop is not None and crop.size > 0:
                if np.mean(np.all(crop == 128, axis=2)) > 0.95:
                    effnet_probs = [1 / len(CLASSES)] * len(CLASSES)
                else:
                    tensor = preprocess_crop(crop)
                    logits = self.effnet(tensor)
                    effnet_probs = F.softmax(logits, dim=1)[0].cpu().tolist()
            else:
                effnet_probs = [1 / len(CLASSES)] * len(CLASSES)

            final_cls  = int(np.argmax(effnet_probs))
            final_conf = max(effnet_probs)

            # EfficientNet 확률 threshold 미만이면 스킵 (불확실 분류 필터링)
            if final_conf < CONF_THRESHOLD:
                continue

            detections.append({
                "final_class":  CLASSES[final_cls],
                "yolo_conf":    round(yolo_conf, 4),
                "effnet_probs": [round(p, 4) for p in effnet_probs],
                "bbox":         [round(v, 1) for v in bbox],
            })

        return {"detections": detections, "is_symptomatic": len(detections) > 0}

    def visualize(self, img_path: str, output: dict) -> np.ndarray:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        for det in output["detections"]:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls_id = CLASSES.index(det["final_class"])
            color  = COLORS[cls_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{det['final_class'].split('_')[0]} {max(det['effnet_probs']):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        status = "유증상" if output["is_symptomatic"] else "무증상"
        color  = (0, 0, 255) if output["is_symptomatic"] else (0, 200, 0)
        cv2.putText(img, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        return img

    def evaluate(self):
        """YOLO 병변 탐지 mAP 평가 (Stage 1 성능 기준점)"""
        metrics = self.yolo.val(
            data="dataset.yaml",
            imgsz=1280, batch=4, conf=0.001, iou=0.6, augment=True, verbose=True,
        )
        print("\n=== YOLO Stage 1 평가 결과 (병변 탐지) ===")
        print(f"Box  mAP@50:    {metrics.box.map50:.4f}")
        print(f"Box  mAP@50-95: {metrics.box.map:.4f}")
        print(f"Seg  mAP@50:    {metrics.seg.map50:.4f}")
        print(f"Seg  mAP@50-95: {metrics.seg.map:.4f}")


if __name__ == "__main__":
    if "--eval" in sys.argv:
        TwoStagePipeline().evaluate()
    elif len(sys.argv) >= 2:
        import json
        pipe   = TwoStagePipeline()
        output = pipe.predict(sys.argv[1])
        print(json.dumps(output, ensure_ascii=False, indent=2))
        vis = pipe.visualize(sys.argv[1], output)
        out = str(Path(sys.argv[1]).with_suffix("")) + "_pred.jpg"
        _, buf = cv2.imencode('.jpg', vis)
        buf.tofile(out)
        print(f"\n저장: {out}")
    else:
        print(__doc__)
