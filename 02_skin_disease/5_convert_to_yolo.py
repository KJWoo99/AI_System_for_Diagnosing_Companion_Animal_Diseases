"""
JSON 어노테이션 → YOLO Segmentation 포맷 변환기
출력 구조:
  yolo_dataset/
    images/train/, images/val/, images/test/
    labels/train/, labels/val/, labels/test/

클래스: A4(농포/여드름), A6(결절/종괴) 2클래스

분할 전략: 1.Training + 2.Validation 전체 합산 후 8:1:1 랜덤 분할
  (원본 2분할 그대로 쓰면 val이 너무 작고 test가 없음)
"""

import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm


BASE = Path(".")
OUT  = Path("yolo_dataset")

# 바이너리 YOLO: A4/A6 모두 class 0 (병변) — 분류는 EfficientNet이 담당
CLASSES = {"A4_농포_여드름", "A6_결절_종괴"}

IMG_W, IMG_H  = 1920, 1080
SEED          = 42
MIN_POLY_SIDE = 10   # 폴리곤 bbox 최소 픽셀 길이 (너무 작은 노이즈 어노테이션 제거)
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.1
# TEST_RATIO  = 0.1 (나머지)

# 무증상(negative) 이미지 최대 비율
# YOLO 학습 시 무증상이 너무 많으면 Recall 저하 → 전체의 20%로 제한
# 0.0 으로 설정하면 무증상 전부 제외, 1.0 이면 제한 없음
MAX_NEG_RATIO = 0.15


def polygon_dict_to_coords(location: dict) -> list[tuple[float, float]]:
    """
    {"x1":..,"y1":..,"x2":..,"y2":..,...} → [(x1,y1),(x2,y2),...]
    """
    pts = []
    i = 1
    while f"x{i}" in location:
        pts.append((location[f"x{i}"], location[f"y{i}"]))
        i += 1
    return pts


def normalize(pts: list[tuple[float, float]]) -> list[float]:
    """픽셀 좌표 → 0~1 정규화 + 클램핑, YOLO seg 형식으로 펼치기"""
    flat = []
    for x, y in pts:
        flat.append(round(max(0.0, min(1.0, x / IMG_W)), 6))
        flat.append(round(max(0.0, min(1.0, y / IMG_H)), 6))
    return flat


def collect_all_samples() -> list[tuple[Path, Path]]:
    """
    1.Training + 2.Validation 전체 JSON 수집
    반환: [(json_path, img_root), ...]
    """
    samples = []
    for split_name in ["1.Training", "2.Validation"]:
        label_root = BASE / split_name / "2_라벨링데이터_240422_add"
        img_root   = BASE / split_name / "1_원천데이터_240422_add"
        for jf in label_root.rglob("*.json"):
            samples.append((jf, img_root))
    return samples


def apply_neg_ratio(samples: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    """
    유증상/무증상 분리 후 MAX_NEG_RATIO 기준으로 무증상 수 제한.
    유증상은 전부 유지, 무증상만 샘플링.
    A2 등 CLASSES에 없는 클래스만 있는 유증상 샘플은 사전 제외.
    samples는 이미 shuffle된 상태여야 함 (앞쪽 무증상이 유지됨).
    """
    symp     = []
    asym     = []
    skipped_a2 = 0
    for jf, img_root in tqdm(samples, desc="유증상/무증상 분류"):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
        path_label = data.get("metaData", {}).get("Path", "")
        if path_label == "유증상":
            # CLASSES에 해당하는 어노테이션이 하나라도 있어야 유효
            labeling = data.get("labelingInfo", [])
            has_valid = any(
                any(cls_key in ann["polygon"].get("label", "") for cls_key in CLASSES)
                for ann in labeling if "polygon" in ann
            )
            if has_valid:
                symp.append((jf, img_root))
            else:
                skipped_a2 += 1   # A2 등 미사용 클래스만 있는 샘플
        else:
            asym.append((jf, img_root))

    print(f"  유증상 {len(symp) + skipped_a2:,}개 중 미사용 클래스(A2 등) {skipped_a2:,}개 제외 → {len(symp):,}개")

    max_neg = int(len(symp) * MAX_NEG_RATIO / (1 - MAX_NEG_RATIO))
    if len(asym) > max_neg:
        print(f"  무증상 {len(asym):,}개 → {max_neg:,}개로 제한 (MAX_NEG_RATIO={MAX_NEG_RATIO})")
        asym = asym[:max_neg]
    else:
        print(f"  무증상 {len(asym):,}개 — 비율 양호, 전부 사용")

    combined = symp + asym
    random.shuffle(combined)   # 유증상/무증상 섞기
    return combined


def convert_samples(samples: list[tuple[Path, Path]], dst_folder: str):
    out_img = OUT / "images" / dst_folder
    out_lbl = OUT / "labels" / dst_folder
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    skipped_no_ann = 0   # 유증상인데 유효 어노테이션 없음 (A2 등 미사용 클래스)
    skipped_no_img = 0   # 이미지 파일 없음
    converted = 0

    for jf, img_root in tqdm(samples, desc=f"{dst_folder} 변환"):
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)

        stem = jf.stem
        meta = data.get("metaData", {})
        is_symptomatic = (meta.get("Path", "") == "유증상")

        lines = []

        if is_symptomatic:
            labeling = data.get("labelingInfo", [])
            for ann in labeling:
                if "polygon" not in ann:
                    continue
                poly = ann["polygon"]
                label_name = poly.get("label", "")

                class_id = None
                for cls_key in CLASSES:
                    if cls_key in label_name:
                        class_id = 0   # 바이너리: 모든 병변 → class 0
                        break
                if class_id is None:
                    continue

                loc = poly["location"][0]
                pts = polygon_dict_to_coords(loc)
                if len(pts) < 3:
                    continue

                # 너무 작은 폴리곤 필터링 (노이즈 어노테이션)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                if (max(xs) - min(xs) < MIN_POLY_SIDE or
                        max(ys) - min(ys) < MIN_POLY_SIDE):
                    continue

                coords = normalize(pts)
                line = f"{class_id} " + " ".join(map(str, coords))
                lines.append(line)

        # 유증상인데 유효 어노테이션이 하나도 없으면 스킵
        # (빈 txt 저장 시 YOLO가 유증상 이미지를 배경으로 학습하는 오류 방지)
        if is_symptomatic and not lines:
            skipped_no_ann += 1
            continue

        img_candidates = list(img_root.rglob(f"{stem}.jpg"))
        if not img_candidates:
            skipped_no_img += 1
            continue

        src_img = img_candidates[0]
        dst_img = out_img / f"{stem}.jpg"
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        lbl_path = out_lbl / f"{stem}.txt"
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))

        converted += 1

    print(f"  변환 완료: {converted}, 스킵: {skipped_no_ann + skipped_no_img}"
          f" (어노테이션 없음: {skipped_no_ann}, 이미지 없음: {skipped_no_img})")


if __name__ == "__main__":
    # 전체 샘플 수집 + 셔플
    all_samples = collect_all_samples()
    random.seed(SEED)
    random.shuffle(all_samples)

    # 무증상 비율 제한 (MAX_NEG_RATIO)
    print(f"\n무증상 비율 관리 (목표: ≤ {MAX_NEG_RATIO * 100:.0f}%)")
    all_samples = apply_neg_ratio(all_samples)

    n = len(all_samples)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:n_train + n_val]
    test_samples  = all_samples[n_train + n_val:]

    print(f"전체: {n}개 → train {len(train_samples)} / val {len(val_samples)} / test {len(test_samples)}")

    print("\n=== Train 변환 ===")
    convert_samples(train_samples, "train")

    print("\n=== Val 변환 ===")
    convert_samples(val_samples, "val")

    print("\n=== Test 변환 ===")
    convert_samples(test_samples, "test")

    print("\n변환 완료!")
    print(f"출력 경로: {OUT}")
