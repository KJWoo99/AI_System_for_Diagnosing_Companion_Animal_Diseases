"""
Stage 2용 Crop 데이터셋 생성기 (멀티프로세싱 + tqdm + 안전한 파일명)

JSON 어노테이션 → bbox crop (+ polygon masked crop) 이미지 저장
출력 구조:
  crop_dataset/
    train/
      A4_농포_여드름/
      A6_결절_종괴/
    val/
      ...

★ YOLO 없이 JSON bbox 직접 활용 → Stage 2 학습에 사용
   (실제 추론 시에는 YOLO bbox로 crop)
"""

import json
import cv2
import numpy as np
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

BASE      = Path(".")
OUT       = Path("crop_dataset")
PAD_RATIO = 0.15   # bbox 주변 15% 패딩 (병변 경계 맥락 포함)
USE_MASK  = True   # True: polygon으로 배경 제거 (권장), False: bbox crop만
MAX_WORKERS = 4    # I/O 병목 고려 → CPU 수 전체보다 4~6 권장

CLASSES = {
    "A4_농포_여드름": "A4_농포_여드름",
    "A6_결절_종괴":   "A6_결절_종괴",
}

IMG_W, IMG_H = 1920, 1080


def polygon_dict_to_numpy(location: dict) -> np.ndarray:
    """{"x1":..,"y1":..,...} → numpy array shape (N, 2)"""
    pts = []
    i = 1
    while f"x{i}" in location:
        pts.append([location[f"x{i}"], location[f"y{i}"]])
        i += 1
    return np.array(pts, dtype=np.int32)


def masked_crop(img: np.ndarray, poly_pts: np.ndarray, box: dict, pad: float) -> np.ndarray | None:
    """
    polygon 마스크로 배경 제거 후 bbox 기준 crop
    배경은 회색(128)으로 채움 → EfficientNet 학습 시 배경 혼동 방지
    """
    x, y, w, h = int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])

    pad_x = int(w * pad)
    pad_y = int(h * pad)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(IMG_W, x + w + pad_x)
    y2 = min(IMG_H, y + h + pad_y)

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    if USE_MASK and len(poly_pts) >= 3:
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_pts], 255)
        result = img.copy()
        bg = np.full_like(img, 128)
        result[mask == 0] = bg[mask == 0]
    else:
        result = img.copy()

    return result[y1:y2, x1:x2]


def box_loc_to_hash(box_loc: dict) -> str:
    """box 좌표를 짧은 해시로 변환 → 파일명 충돌 방지"""
    key = f"{box_loc.get('x')}_{box_loc.get('y')}_{box_loc.get('width')}_{box_loc.get('height')}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def process_single_json(args: tuple) -> tuple[dict, int, list]:
    """
    단일 JSON 파일 처리
    Returns:
        counters: 클래스별 저장 수
        skipped: 스킵 수
        errors: 에러 메시지 리스트
    """
    jf_path, split_name, dst_split = args

    label_root = BASE / split_name / "2_라벨링데이터_240422_add"
    img_root   = BASE / split_name / "1_원천데이터_240422_add"

    counters = {k: 0 for k in CLASSES}
    skipped = 0
    errors = []

    try:
        with open(jf_path, encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metaData", {})
        if meta.get("Path") != "유증상":
            return counters, skipped, errors

        stem = jf_path.stem
        img_candidates = list(img_root.rglob(f"{stem}.jpg"))
        if not img_candidates:
            skipped += 1
            return counters, skipped, errors

        img_array = np.fromfile(str(img_candidates[0]), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            errors.append(f"[WARN] 이미지 디코드 실패: {img_candidates[0]}")
            return counters, skipped, errors

        labeling = data.get("labelingInfo", [])

        poly_list = []
        box_list  = []
        for ann in labeling:
            if "polygon" in ann:
                lbl = ann["polygon"].get("label", "")
                loc = ann["polygon"]["location"][0]
                poly_list.append((lbl, polygon_dict_to_numpy(loc)))
            if "box" in ann:
                lbl = ann["box"].get("label", "")
                box_list.append((lbl, ann["box"]["location"][0]))

        for box_lbl, box_loc in box_list:
            cls_name = None
            for key in CLASSES:
                if key in box_lbl:
                    cls_name = CLASSES[key]
                    break
            if cls_name is None:
                continue

            poly_pts = np.array([], dtype=np.int32)
            for poly_lbl, pts in poly_list:
                if poly_lbl == box_lbl:
                    poly_pts = pts
                    break

            crop = masked_crop(img, poly_pts, box_loc, PAD_RATIO)
            if crop is None:
                skipped += 1
                continue

            # polygon-box 좌표 불일치로 완전 회색인 crop 필터링
            if USE_MASK and np.all(crop == 128, axis=2).mean() > 0.95:
                skipped += 1
                continue

            # ★ 파일명 충돌 방지: stem + box 좌표 해시 조합
            loc_hash = box_loc_to_hash(box_loc)
            fname = f"{stem}_{loc_hash}.jpg"

            out_dir = OUT / dst_split / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)
            _, buf = cv2.imencode('.jpg', crop)
            buf.tofile(str(out_dir / fname))
            counters[cls_name] += 1

    except Exception as e:
        skipped += 1
        errors.append(f"[ERROR] {jf_path.name}: {e}")

    return counters, skipped, errors


def process_split_parallel(split_name: str, dst_split: str):
    label_root = BASE / split_name / "2_라벨링데이터_240422_add"
    json_files = list(label_root.rglob("*.json"))
    print(f"[{split_name}] JSON 수: {len(json_files)}")

    total_counters = {k: 0 for k in CLASSES}
    total_skipped = 0
    all_errors = []

    # args를 튜플로 묶어서 전달 (ProcessPoolExecutor는 단일 인자만 지원)
    args_list = [(jf, split_name, dst_split) for jf in json_files]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_json, args): args[0] for args in args_list}

        with tqdm(total=len(futures), desc=f"Processing {split_name}", unit="file") as pbar:
            for future in as_completed(futures):
                jf_path = futures[future]
                try:
                    counters, skipped, errors = future.result()
                    for k in counters:
                        total_counters[k] += counters[k]
                    total_skipped += skipped
                    all_errors.extend(errors)
                except Exception as e:
                    total_skipped += 1
                    all_errors.append(f"[FATAL] {jf_path.name}: {e}")
                finally:
                    pbar.update(1)

    print(f"\n저장 완료: {total_counters}")
    print(f"스킵: {total_skipped}")

    if all_errors:
        print(f"\n⚠️  경고/에러 ({len(all_errors)}건):")
        for msg in all_errors[:20]:   # 최대 20건만 출력
            print(f"  {msg}")
        if len(all_errors) > 20:
            print(f"  ... 외 {len(all_errors) - 20}건")


if __name__ == "__main__":
    print("=== Training Crop 생성 ===")
    process_split_parallel("1.Training", "train")

    print("\n=== Validation Crop 생성 ===")
    process_split_parallel("2.Validation", "val")

    print("\nCrop 데이터셋 완료!")
    print(f"출력: {OUT}")