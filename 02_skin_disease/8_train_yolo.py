"""
반려묘 피부질환 Instance Segmentation 학습 (Stage 1)
모델: YOLO11l-seg (VRAM 16GB 기준 최적)
클래스: 1클래스 바이너리 (병변 위치 탐지 전용, 분류는 EfficientNet 담당)

설치:
  pip install ultralytics
"""

from pathlib import Path
from ultralytics import YOLO

# ── 경로 설정 ──────────────────────────────────────────────
DATASET_YAML = Path("dataset.yaml")
RUNS_DIR     = Path("runs")

# ── 하이퍼파라미터 (정확도 최우선) ────────────────────────
CFG = dict(
    # YOLO11l-seg: VRAM 16GB 기준 최적 (s보다 정확도 높음, x보다 VRAM 절약)
    model   = "yolo11l-seg.pt",

    data    = str(DATASET_YAML),
    project = str(RUNS_DIR),
    name    = "pet_skin_seg",

    # 해상도: 원본 1920x1080 → 1280으로 리사이즈 (1024보다 세부 병변 탐지 유리)
    imgsz   = 1280,

    epochs  = 200,
    patience= 30,          # Early stopping (30 epoch 개선 없으면 종료)

    batch   = 4,           # VRAM 16GB 기준 안전값
    nbs     = 64,          # Gradient accumulation: batch=4이지만 64 분량 누적 → 안정적 학습
    workers = 4,

    # 옵티마이저
    optimizer = "AdamW",
    lr0     = 1e-3,
    lrf     = 1e-2,        # 최종 lr = lr0 * lrf
    momentum= 0.937,
    weight_decay = 5e-4,

    # 스케줄러: cosine annealing (warmup 포함)
    warmup_epochs  = 5,
    warmup_momentum= 0.8,
    warmup_bias_lr = 0.1,
    cos_lr         = True,

    # 손실 가중치 (seg mask 정확도 높이기)
    box  = 7.5,
    cls  = 0.5,
    dfl  = 1.5,

    # ── Augmentation (정확도 최우선 세팅) ─────────────────
    # 피부 병변 특성상: 색상·밝기 변환 강하게, 기하 변환 적당히
    hsv_h  = 0.015,   # Hue 변화
    hsv_s  = 0.7,     # Saturation 변화 (병변 색 다양성)
    hsv_v  = 0.4,     # Brightness 변화 (촬영 환경 차이)

    degrees  = 15,    # 회전 ±15도
    translate= 0.1,   # 이동 10%
    scale    = 0.5,   # 스케일 0.5~1.5배 (병변 크기 다양성)
    shear    = 5.0,   # 전단 변환
    perspective= 0.0001,

    flipud   = 0.3,   # 상하 반전 (털 방향 무관한 병변)
    fliplr   = 0.5,   # 좌우 반전

    mosaic   = 1.0,   # Mosaic (4장 합성, 소형 병변 학습 강화)
    mixup    = 0.15,  # Mixup (일반화 성능 향상)
    copy_paste= 0.3,  # Copy-Paste (seg 특화 augmentation)
    erasing  = 0.4,   # Random Erasing (오탐 억제)

    # 학습 안정화
    amp      = True,   # Mixed Precision (fp16)
    seed     = 42,
    deterministic= False,  # True 하면 느려짐

    # 검증/저장
    val      = True,
    save     = True,
    save_period= 10,   # 10 epoch마다 체크포인트 저장
    plots    = True,
    verbose  = True,

    # 클래스 불균형 대응
    # 무증상(negative sample)이 많으므로 cls_pw로 양성 가중
    cls_pw   = 1.5,
    obj_pw   = 1.0,
)


def main():
    # 사전학습 가중치 로드 (COCO pretrained → transfer learning)
    model = YOLO(CFG.pop("model"))

    print("=" * 60)
    print("YOLO11l-seg 학습 시작 (1클래스 바이너리, VRAM 16GB 최적)")
    print(f"데이터셋: {DATASET_YAML}")
    print(f"출력:     {RUNS_DIR}/pet_skin_seg")
    print("=" * 60)

    results = model.train(**CFG)

    print("\n학습 완료!")
    print(f"최고 모델: {RUNS_DIR}/pet_skin_seg/weights/best.pt")
    print(f"Box mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"Seg mAP50-95: {results.results_dict.get('metrics/mAP50-95(M)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
