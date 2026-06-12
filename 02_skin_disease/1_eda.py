"""
0_0_eda.py — 학습 전 데이터 현황 파악 (EDA)

확인 항목:
  1. 유증상/무증상 비율 + 권장 무증상 수 제안
  2. 클래스별 병변 인스턴스 수 (A2/A4/A6 불균형 확인)
  3. metaData 필드 분석 — 동물/환자 ID 키 존재 여부 탐지
     ★ ID 발견 시: 1_convert_to_yolo.py의 랜덤 split 전략을 ID 기준으로 변경 필요

실행:
  python 0_0_eda.py

스레드: ThreadPoolExecutor (JSON 읽기 = I/O 바운드 → 멀티스레드 적합)
"""

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


BASE   = Path(".")
SPLITS = ["1.Training", "2.Validation"]

CLASSES = {
    "A2_비듬_각질_상피성잔고리": "A2",
    "A4_농포_여드름":           "A4",
    "A6_결절_종괴":             "A6",
}

IMG_W, IMG_H  = 1920, 1080
MAX_NEG_RATIO = 0.20     # 무증상 권장 최대 비율
NUM_WORKERS   = 8        # 스레드 수 (I/O 바운드 → 8~16 효과적)

# 동물/환자 ID로 의심할 키워드 (소문자 비교)
ID_KEYWORDS = ["id", "patient", "animal", "pet", "no", "번호", "개체", "동물", "uid", "seq"]


# ── 스레드 워커 ────────────────────────────────────────────────
def _process_one(args: tuple) -> dict | None:
    """단일 JSON 파일 처리 → 결과 dict 반환"""
    jf, split_name = args
    try:
        with open(jf, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    meta    = data.get("metaData", {})
    is_symp = (meta.get("Path", "") == "유증상")

    class_counts = Counter()
    bbox_areas   = defaultdict(list)

    if is_symp:
        for ann in data.get("labelingInfo", []):
            if "box" not in ann:
                continue
            box  = ann["box"]
            lbl  = box.get("label", "")
            locs = box.get("location", [])
            if not locs:
                continue
            for cls_key, cls_code in CLASSES.items():
                if cls_key in lbl:
                    class_counts[cls_code] += 1
                    loc = locs[0]
                    w, h = loc.get("width", 0), loc.get("height", 0)
                    if w > 0 and h > 0:
                        bbox_areas[cls_code].append((w * h) / (IMG_W * IMG_H))
                    break

    return {
        "split":       split_name,
        "is_symp":     is_symp,
        "meta_keys":   list(meta.keys()),
        "meta_data":   meta,          # 샘플 출력용
        "class_counts": class_counts,
        "bbox_areas":   dict(bbox_areas),
    }


# ── 통계 수집 ──────────────────────────────────────────────────
def collect_stats() -> dict:
    # 전체 JSON 목록 수집
    all_args = []
    for split_name in SPLITS:
        label_root = BASE / split_name / "2_라벨링데이터_240422_add"
        for jf in label_root.rglob("*.json"):
            all_args.append((jf, split_name))

    print(f"JSON 총 {len(all_args):,}개 → 스레드 {NUM_WORKERS}개로 병렬 분석")

    stats = {
        "total":        0,
        "symptomatic":  0,
        "asymptomatic": 0,
        "per_split":    {s: {"symp": 0, "asym": 0} for s in SPLITS},
        "class_counts": Counter(),
        "bbox_areas":   defaultdict(list),
        "meta_keys":    Counter(),
        "meta_sample":  None,   # 유증상 첫 번째 JSON metaData 예시
    }

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(_process_one, arg): arg for arg in all_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="분석"):
            result = future.result()
            if result is None:
                continue

            stats["total"] += 1
            split = result["split"]

            for k in result["meta_keys"]:
                stats["meta_keys"][k] += 1

            if result["is_symp"]:
                stats["symptomatic"]         += 1
                stats["per_split"][split]["symp"] += 1
                for code, cnt in result["class_counts"].items():
                    stats["class_counts"][code] += cnt
                for code, areas in result["bbox_areas"].items():
                    stats["bbox_areas"][code].extend(areas)
                if stats["meta_sample"] is None:
                    stats["meta_sample"] = result["meta_data"]
            else:
                stats["asymptomatic"]        += 1
                stats["per_split"][split]["asym"] += 1

    return stats


# ── 보고서 출력 ────────────────────────────────────────────────
def print_report(stats: dict):
    total = stats["total"]
    symp  = stats["symptomatic"]
    asym  = stats["asymptomatic"]
    asym_ratio = asym / total if total > 0 else 0

    W = 58
    print("\n" + "=" * W)
    print("  데이터 현황 보고서")
    print("=" * W)

    # ── 1. 유증상/무증상 비율 ──────────────────────────────
    print(f"\n[1] 유증상/무증상 비율")
    print(f"    전체:     {total:,} 개")
    print(f"    유증상:   {symp:,} 개  ({symp / total * 100:.1f}%)")
    print(f"    무증상:   {asym:,} 개  ({asym_ratio * 100:.1f}%)")

    for split, cnt in stats["per_split"].items():
        t = cnt["symp"] + cnt["asym"]
        if t == 0:
            continue
        print(f"    [{split}]  유증상 {cnt['symp']:,} / 무증상 {cnt['asym']:,}"
              f"  ({cnt['asym'] / t * 100:.1f}%)")

    rec_asym = int(symp * MAX_NEG_RATIO / (1 - MAX_NEG_RATIO))
    if asym_ratio > MAX_NEG_RATIO:
        print(f"\n  ⚠  무증상 {asym_ratio * 100:.1f}% — 권장({MAX_NEG_RATIO * 100:.0f}%) 초과")
        print(f"     권장 무증상: {rec_asym:,}개  (현재 {asym:,}개 → {asym - rec_asym:,}개 감축)")
        print(f"     → 1_convert_to_yolo.py 의 MAX_NEG_RATIO={MAX_NEG_RATIO} 로 자동 제한됨")
    else:
        print(f"\n  ✓  무증상 비율 양호 ({asym_ratio * 100:.1f}% ≤ {MAX_NEG_RATIO * 100:.0f}%)")

    # ── 2. 클래스별 병변 수 ────────────────────────────────
    print(f"\n[2] 클래스별 병변 인스턴스 수 (유증상 bbox 기준)")
    counts = {}
    for code in ["A2", "A4", "A6"]:
        n = stats["class_counts"].get(code, 0)
        counts[code] = n
        print(f"    {code}: {n:,} 개")

    vals = list(counts.values())
    if min(vals) > 0:
        imbalance = max(vals) / min(vals)
        min_cls = min(counts, key=counts.get)
        if imbalance > 3:
            print(f"\n  ⚠  클래스 불균형 {imbalance:.1f}배 — {min_cls} 부족")
            print(f"     → WeightedRandomSampler(4_train_efficientnet.py)가 자동 보정")
            print(f"     → 5배 이상이면 부족한 클래스의 3_build_crops.py 증강 강화 고려")
        else:
            print(f"\n  ✓  클래스 분포 양호 ({imbalance:.1f}배)")

    # ── 3. metaData 키 분석 ────────────────────────────────
    # ★ 핵심: 동물 ID 필드 발견 여부에 따라 split 전략이 달라짐
    print(f"\n[3] metaData 필드 분석")
    all_keys = sorted(stats["meta_keys"].keys())
    print(f"    확인된 키: {all_keys}")

    found_id = [k for k in all_keys if any(kw in k.lower() for kw in ID_KEYWORDS)]

    if found_id:
        print(f"\n  ⚠  동물/환자 ID 관련 키 발견: {found_id}")
        sample = stats["meta_sample"] or {}
        for k in found_id:
            print(f"     {k}: {sample.get(k, '(없음)')}")
        print(f"""
  → 동일 동물 이미지가 train/test에 양쪽 들어가면 데이터 누수!
     모델이 특정 동물 피부 특성을 외워버릴 수 있음.
  → 1_convert_to_yolo.py 를 ID 기준 split으로 변경 필요.
     (요청 시 코드 수정 가능)""")
    else:
        print(f"\n  ✓  명시적 동물 ID 키 없음")
        print(f"     → 각 이미지 독립 케이스로 간주 가능 → 랜덤 split 유효")

    if stats["meta_sample"]:
        print(f"\n    metaData 전체 내용 (첫 번째 유증상 JSON):")
        for k, v in stats["meta_sample"].items():
            print(f"      {k}: {v}")

    print("\n" + "=" * W)
    print("  분석 완료")
    print("=" * W)


if __name__ == "__main__":
    stats = collect_stats()
    print_report(stats)
