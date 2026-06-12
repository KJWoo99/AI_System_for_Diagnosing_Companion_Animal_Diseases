"""
데이터셋 무결성 검사 (변환 전 필수 실행)

검사 항목:
  1. 원천데이터 이미지 ↔ 라벨링데이터 JSON 1:1 매칭 여부
  2. 이미지는 있는데 JSON 없는 파일
  3. JSON은 있는데 이미지 없는 파일
  4. 분할(Train/Val)별, 클래스별 이미지 수 통계

출력:
  verify_issues.txt  ← 0_1_fix_dataset.py 가 읽는 파일 (JSON 형식)
"""

import json
from pathlib import Path
from collections import defaultdict


BASE   = Path(".")
OUT_TXT = Path("verify_issues.txt")
SPLITS = {
    "Train": "1.Training",
    "Val":   "2.Validation",
}


def collect(split_dir: str):
    img_root   = BASE / split_dir / "1_원천데이터_240422_add"
    label_root = BASE / split_dir / "2_라벨링데이터_240422_add"
    img_stems  = {p.stem: p for p in img_root.rglob("*.jpg")}
    json_stems = {p.stem: p for p in label_root.rglob("*.json")}
    return img_stems, json_stems


def verify_split(split_label: str, split_dir: str, all_img: dict, all_json: dict):
    img_stems, json_stems = all_img[split_label], all_json[split_label]

    print(f"\n{'='*60}")
    print(f"[{split_label}] 검사 중...")
    print(f"{'='*60}")
    print(f"원천데이터  이미지 수: {len(img_stems):,}")
    print(f"라벨링데이터 JSON 수:  {len(json_stems):,}")

    no_json = sorted(img_stems.keys() - json_stems.keys())
    no_img  = sorted(json_stems.keys() - img_stems.keys())

    print(f"\n[검사 1] 이미지 있음 / JSON 없음: {len(no_json)}건")
    for stem in no_json[:20]:
        print(f"  ✗ {stem}  ({img_stems[stem].parent.name})")
    if len(no_json) > 20:
        print(f"  ... 외 {len(no_json)-20}건")

    print(f"\n[검사 2] JSON 있음 / 이미지 없음: {len(no_img)}건")
    for stem in no_img[:20]:
        print(f"  ✗ {stem}  ({json_stems[stem].parent.name})")
    if len(no_img) > 20:
        print(f"  ... 외 {len(no_img)-20}건")

    matched = img_stems.keys() & json_stems.keys()
    print(f"\n[검사 3] 정상 매칭: {len(matched):,}건")

    # 클래스별 통계
    print(f"\n[검사 4] 클래스별 이미지 수")
    class_counts = defaultdict(lambda: defaultdict(int))
    img_root = BASE / split_dir / "1_원천데이터_240422_add"
    for p in img_root.rglob("*.jpg"):
        symptom = cls = None
        for part in p.parts:
            if part in ("유증상", "무증상"):
                symptom = part
            if part.startswith(("A2_", "A4_", "A6_")):
                cls = part.replace("_잔여", "")
        if symptom and cls:
            class_counts[cls][symptom] += 1

    total = 0
    for cls in sorted(class_counts):
        row = class_counts[cls]
        subtotal = sum(row.values())
        total += subtotal
        유 = row.get("유증상", 0)
        무 = row.get("무증상", 0)
        print(f"  {cls:<30} 유증상: {유:>6,}  무증상: {무:>6,}  합계: {subtotal:>6,}")
    print(f"  {'합계':<30}                              전체: {total:>6,}")

    print(f"\n{'─'*60}")
    if not no_json and not no_img:
        print(f"✅ [{split_label}] 이상 없음")
    else:
        print(f"⚠️  [{split_label}] 누락 파일 발견")
    print(f"{'─'*60}")

    return no_json, no_img


if __name__ == "__main__":
    # 전체 수집 (교차 split 탐색을 위해 양쪽 모두 미리 수집)
    all_img  = {}
    all_json = {}
    for label, split_dir in SPLITS.items():
        img_stems, json_stems = collect(split_dir)
        all_img[label]  = img_stems
        all_json[label] = json_stems

    issues = []

    for label, split_dir in SPLITS.items():
        no_json, no_img = verify_split(label, split_dir, all_img, all_json)

        # 이미지 있음 / JSON 없음
        for stem in no_json:
            img_path = all_img[label][stem]
            # 반대 split에 JSON이 있는지 확인
            other = [s for s in SPLITS if s != label][0]
            json_path = all_json[other].get(stem)
            issues.append({
                "stem":       stem,
                "type":       "img_no_json",
                "img_split":  label,
                "img_path":   str(img_path),
                "json_split": other if json_path else None,
                "json_path":  str(json_path) if json_path else None,
            })

        # JSON 있음 / 이미지 없음
        for stem in no_img:
            json_path = all_json[label][stem]
            # 반대 split에 이미지가 있는지 확인
            other = [s for s in SPLITS if s != label][0]
            img_path = all_img[other].get(stem)
            issues.append({
                "stem":       stem,
                "type":       "json_no_img",
                "json_split": label,
                "json_path":  str(json_path),
                "img_split":  other if img_path else None,
                "img_path":   str(img_path) if img_path else None,
            })

    # 중복 제거 (같은 stem이 양방향으로 잡힐 수 있음)
    seen = set()
    unique_issues = []
    for issue in issues:
        key = (issue["stem"], issue["type"], issue.get("img_split"), issue.get("json_split"))
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)

    # 분류 요약
    cross  = [i for i in unique_issues if i["img_path"] and i["json_path"]]
    orphan = [i for i in unique_issues if not (i["img_path"] and i["json_path"])]

    print(f"\n{'='*60}")
    print(f"이슈 분류:")
    print(f"  교차 split 오배치 (이미지↔JSON이 다른 split): {len(cross)}건  → 자동 수정 가능")
    print(f"  완전 누락 (이미지 또는 JSON 없음):            {len(orphan)}건  → 수정 불가 (스킵)")
    print(f"{'='*60}")

    # TXT 저장 (JSON 형식)
    OUT_TXT.write_text(
        json.dumps(unique_issues, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n✅ 이슈 목록 저장 완료: {OUT_TXT}")
    print(f"   → 0_1_fix_dataset.py 를 실행하여 자동 보정하세요")
