"""
1_학습미사용정리.py
===================
train_cat / train_dog 기준으로 사용하지 않는 파일 삭제

[사용 데이터]
  */원천데이터/고양이/**/*.jpg  → train_cat
  */원천데이터/개/**/*.jpg      → train_dog

[삭제 대상]
  ① */라벨링데이터/             (고양이+개 모두 — crop 이미지 + JSON, 폴더째)
  ② */원천데이터/**/*.json      (고양이+개 모두 — 메타 JSON, 이미지만 사용하므로)

[실행]
  python 1_학습미사용정리.py            # dry-run
  python 1_학습미사용정리.py --execute  # 실제 삭제
"""

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from tqdm import tqdm

BASE = Path(".")

# ── ① 폴더째 삭제 ─────────────────────────────────────────────────
DELETE_FOLDERS = [
    BASE / "1.Training"   / "라벨링데이터",
    BASE / "2.Validation" / "라벨링데이터",
]

# ── ② JSON 파일 삭제 루트 ─────────────────────────────────────────
JSON_ROOTS = [
    BASE / "1.Training"   / "원천데이터",
    BASE / "2.Validation" / "원천데이터",
]


def count_files(folder: Path) -> int:
    return sum(1 for _ in folder.rglob("*") if _.is_file())


def delete_folder(folder: Path) -> tuple[Path, int, str]:
    try:
        n = count_files(folder)
        shutil.rmtree(folder)
        return folder, n, ""
    except Exception as e:
        return folder, 0, str(e)


def delete_file(path: Path) -> tuple[Path, str]:
    try:
        path.unlink()
        return path, ""
    except Exception as e:
        return path, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    folder_targets = [f for f in DELETE_FOLDERS if f.exists()]

    print("\n[집계 중...]")

    # 폴더 파일 수 (병렬)
    folder_counts: dict[Path, int] = {}
    if folder_targets:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(count_files, t): t for t in folder_targets}
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc="  폴더 집계", unit="폴더"):
                folder_counts[futs[fut]] = fut.result()

    # JSON 파일 목록
    print("  JSON 파일 목록 수집 중...")
    json_files: list[Path] = []
    for root in JSON_ROOTS:
        if root.exists():
            json_files.extend(root.rglob("*.json"))

    # ── 출력 ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  ① 라벨링데이터 전체 (고양이+개, 폴더 삭제)")
    print(f"{'─'*60}")
    for t in folder_targets:
        print(f"    {t}  →  {folder_counts[t]:,}개")
    for m in [f for f in DELETE_FOLDERS if not f.exists()]:
        print(f"    [이미 없음] {m}")

    print(f"\n{'─'*60}")
    print(f"  ② 원천데이터/**/*.json (고양이+개, 파일 삭제)")
    print(f"{'─'*60}")
    for root in JSON_ROOTS:
        n = sum(1 for f in json_files if f.is_relative_to(root))
        print(f"    {root}  →  {n:,}개")

    total = sum(folder_counts.values()) + len(json_files)
    print(f"\n{'─'*60}")
    print(f"  삭제 예정: {total:,}개")
    print(f"  유지:      */원천데이터/고양이+개/**/*.jpg  (학습 이미지)")
    print(f"{'─'*60}")

    if not args.execute:
        print(f"\n  [DRY-RUN] 실제 삭제: --execute")
        return

    # ── 실제 삭제 ─────────────────────────────────────────────────
    deleted = 0
    errors  = []

    # ① 폴더 삭제
    if folder_targets:
        print(f"\n  [①] 라벨링데이터 폴더 삭제...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(delete_folder, t): t for t in folder_targets}
            with tqdm(as_completed(futs), total=len(futs),
                      desc="  폴더", unit="폴더") as pbar:
                for fut in pbar:
                    path, n, err = fut.result()
                    if err:
                        errors.append((path, err))
                        pbar.write(f"    [오류] {path}: {err}")
                    else:
                        deleted += n
                        pbar.write(f"    [완료] {path}  ({n:,}개)")

    # ② JSON 파일 삭제
    if json_files:
        print(f"\n  [②] JSON 파일 삭제 ({len(json_files):,}개)...")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(delete_file, f): f for f in json_files}
            with tqdm(as_completed(futs), total=len(futs),
                      desc="  JSON", unit="개") as pbar:
                for fut in pbar:
                    path, err = fut.result()
                    if err:
                        errors.append((path, err))
                    else:
                        deleted += 1

    print(f"\n{'─'*60}")
    print(f"  삭제 완료: {deleted:,}개")
    if errors:
        print(f"  오류: {len(errors)}개")
        for p, e in errors[:5]:
            print(f"    {p}: {e}")
    print(f"{'─'*60}")

    print(f"\n[남은 구조]")
    for split in ["1.Training", "2.Validation"]:
        for animal in ["고양이", "개"]:
            d = BASE / split / "원천데이터" / animal
            if d.exists():
                n = count_files(d)
                print(f"  {split}/원천데이터/{animal}/  →  {n:,}개 이미지")


if __name__ == "__main__":
    main()
