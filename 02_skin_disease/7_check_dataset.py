"""
yolo_dataset 검증 스크립트
train / val / test 각 split별로:
  - 전체 이미지 수
  - 무증상 수 (빈 txt)
  - 유증상 수 (비어있지 않은 txt)
  - 클래스별 인스턴스 수 (0=병변(A4+A6))
"""

from pathlib import Path
from collections import defaultdict

LABELS_DIR = Path("yolo_dataset/labels")
CLASSES = {0: "병변(A4+A6)"}


def check_split(split: str):
    split_dir = LABELS_DIR / split
    if not split_dir.exists():
        print(f"  [{split}] 폴더 없음: {split_dir}")
        return

    txt_files = list(split_dir.glob("*.txt"))
    total = len(txt_files)
    asym = 0   # 무증상 (빈 파일)
    symp = 0   # 유증상
    class_counts = defaultdict(int)

    for txt in txt_files:
        content = txt.read_text(encoding="utf-8").strip()
        if not content:
            asym += 1
        else:
            symp += 1
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1

    print(f"\n[{split}]")
    print(f"  전체 이미지 : {total:,}")
    print(f"  무증상      : {asym:,} ({asym/total*100:.1f}%)" if total else "  무증상      : 0")
    print(f"  유증상      : {symp:,} ({symp/total*100:.1f}%)" if total else "  유증상      : 0")
    print(f"  클래스별 인스턴스:")
    for cls_id, cls_name in CLASSES.items():
        cnt = class_counts.get(cls_id, 0)
        print(f"    {cls_id} {cls_name}: {cnt:,}")


if __name__ == "__main__":
    print("=== yolo_dataset 클래스 분포 검증 ===")
    for split in ["train", "val", "test"]:
        check_split(split)
    print()
