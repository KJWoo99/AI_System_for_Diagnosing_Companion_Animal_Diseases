"""
불필요 파일 정리

삭제 대상:
  원천데이터 CSV        → 코드에서 미사용 (이미지+JSON만 필요)
  라벨링데이터 이미지   → 코드에서 미사용 (JSON만 필요, 이미지는 원천에서 읽음)
"""

from pathlib import Path
from tqdm import tqdm   # tqdm 추가

BASE   = Path(".")
SPLITS = ["1.Training", "2.Validation"]


def collect_targets() -> tuple[list[Path], list[Path]]:
    csvs = []
    label_imgs = []

    for split in SPLITS:
        # 원천데이터 CSV
        src_root = BASE / split / "1_원천데이터_240422_add"
        csvs.extend(src_root.rglob("*.csv"))

        # 라벨링데이터 JPG (JSON은 유지)
        lbl_root = BASE / split / "2_라벨링데이터_240422_add"
        label_imgs.extend(lbl_root.rglob("*.jpg"))

    return csvs, label_imgs


def main():
    csvs, label_imgs = collect_targets()

    print("=== 삭제 예정 파일 ===")
    print(f"  원천데이터 CSV:       {len(csvs):,} 개")
    print(f"  라벨링데이터 이미지:  {len(label_imgs):,} 개")
    print(f"  합계:                 {len(csvs) + len(label_imgs):,} 개\n")

    ans = input("삭제하시겠습니까? (yes 입력 시 진행): ").strip().lower()
    if ans != "yes":
        print("취소됨.")
        return

    deleted = 0

    # tqdm progress bar 적용
    targets = csvs + label_imgs
    for f in tqdm(targets, desc="Deleting", unit="file"):
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            print(f"  실패: {f.name} — {e}")

    print(f"\n완료: {deleted:,} 개 삭제")


if __name__ == "__main__":
    main()