"""
데이터셋 자동 보정 (0_verify_dataset.py 실행 후 사용)

verify_issues.txt 를 읽어 두 가지 케이스 처리:

  [케이스 A] 교차 split 오배치 — 이미지와 JSON이 다른 split에 있음
             → 이미지를 JSON이 있는 split의 원천데이터로 이동
             예) Train JSON + Val 이미지 → 이미지를 Train 원천으로 이동

  [케이스 B] 완전 누락 — 이미지나 JSON 한쪽만 존재
             → 수정 불가. 변환 시 자동 스킵되므로 그대로 둠

보정 후 0_verify_dataset.py 재실행하여 결과 확인
"""

import json
import shutil
from pathlib import Path


BASE     = Path(".")
ISSUES_F = Path("verify_issues.txt")
SPLITS   = {
    "Train": "1.Training",
    "Val":   "2.Validation",
}


def find_img_folder_in_split(stem: str, split_label: str) -> Path | None:
    """
    JSON의 상위 폴더 구조를 기반으로 원천데이터의 대응 폴더 경로 반환
    예) 라벨링데이터/.../A2_비듬_각질_상피성잔고리/IMG.json
      → 원천데이터/.../A2_비듬_각질_상피성잔고리/
    """
    split_dir  = SPLITS[split_label]
    label_root = BASE / split_dir / "2_라벨링데이터_240422_add"
    img_root   = BASE / split_dir / "1_원천데이터_240422_add"

    # JSON 경로에서 클래스 폴더 이하 상대경로 추출
    json_matches = list(label_root.rglob(f"{stem}.json"))
    if not json_matches:
        return None

    json_path = json_matches[0]
    # label_root 이후 상대경로 (클래스 폴더명 포함)
    rel = json_path.parent.relative_to(label_root)
    target_dir = img_root / rel
    return target_dir


if __name__ == "__main__":
    if not ISSUES_F.exists():
        print("❌ verify_issues.txt 없음 — 먼저 0_verify_dataset.py 를 실행하세요")
        exit(1)

    issues = json.loads(ISSUES_F.read_text(encoding="utf-8"))

    cross  = [i for i in issues if i["img_path"] and i["json_path"]]
    orphan = [i for i in issues if not (i["img_path"] and i["json_path"])]

    print(f"이슈 총 {len(issues)}건: 교차오배치 {len(cross)}건 / 완전누락 {len(orphan)}건\n")

    # ── 케이스 A: 교차 split 오배치 → 이미지 이동 ───────────────
    fixed = 0
    failed = 0

    print(f"[케이스 A] 교차 오배치 자동 수정 ({len(cross)}건)")
    for issue in cross:
        stem       = issue["stem"]
        img_path   = Path(issue["img_path"])
        json_split = issue["json_split"]   # JSON이 있는 split → 이미지가 가야 할 곳

        # 이미지가 이동해야 할 원천데이터 폴더
        target_dir = find_img_folder_in_split(stem, json_split)
        if target_dir is None:
            print(f"  ✗ [{stem}] 대상 폴더 탐색 실패")
            failed += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / img_path.name

        if target_path.exists():
            print(f"  ↷ [{stem}] 이미 존재, 스킵")
            continue

        try:
            shutil.move(str(img_path), str(target_path))
            print(f"  ✅ [{stem}]  {img_path.parent.parent.parent.name}/{img_path.parent.name}")
            print(f"       → {target_path.parent.parent.parent.name}/{target_path.parent.name}")
            fixed += 1
        except Exception as e:
            print(f"  ✗ [{stem}] 이동 실패: {e}")
            failed += 1

    print(f"\n  수정 완료: {fixed}건 / 실패: {failed}건")

    # ── 케이스 B: 완전 누락 → 처리 불가 ───────────────────────
    print(f"\n[케이스 B] 완전 누락 — 수정 불가 ({len(orphan)}건)")
    for issue in orphan:
        stem = issue["stem"]
        if not issue["img_path"]:
            print(f"  ⚠️  [{stem}] 이미지 없음 (JSON만 존재) → 변환 시 자동 스킵")
        else:
            print(f"  ⚠️  [{stem}] JSON 없음 (이미지만 존재) → 변환 시 자동 스킵")

    # ── 결과 요약 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"보정 완료. 0_verify_dataset.py 재실행으로 결과 확인하세요.")
    print(f"  수정: {fixed}건  /  스킵(불가): {len(orphan)}건  /  실패: {failed}건")
    print(f"{'='*60}")
