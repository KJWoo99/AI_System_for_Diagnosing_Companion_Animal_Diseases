"""
YOLO 데이터셋 시각화 뷰어

train / val / test 각 split에서 N_SAMPLES개 랜덤 샘플링 후
polygon 어노테이션을 이미지에 오버레이하여 저장.

출력: yolo_viewer/{split}/
사용: python 1_1_view_yolo.py
"""

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

YOLO_DIR    = Path("yolo_dataset")
OUT_DIR     = Path("yolo_viewer")
N_SAMPLES   = 20   # 각 split당 샘플 수
SEED        = 42
MAX_WORKERS = 8

CLASSES = ["A2_비듬/각질", "A4_농포/여드름", "A6_결절/종괴"]
COLORS  = [
    (255,  80,   0),   # A2 주황
    (  0, 200,  60),   # A4 초록
    ( 30, 100, 255),   # A6 파랑
]
ALPHA = 0.35   # polygon fill 투명도


def load_font(size: int = 24) -> ImageFont.FreeTypeFont:
    for path in [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        "C:/Windows/Fonts/gulim.ttc",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_image(path: Path) -> np.ndarray | None:
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def parse_label(txt_path: Path, img_w: int, img_h: int) -> list[dict]:
    anns = []
    if not txt_path.exists():
        return anns
    with open(txt_path) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 7:
                continue
            cls_id = int(vals[0])
            coords = list(map(float, vals[1:]))
            pts = [
                (int(coords[i] * img_w), int(coords[i + 1] * img_h))
                for i in range(0, len(coords), 2)
            ]
            anns.append({"cls": cls_id, "pts": pts})
    return anns


def draw_annotations(img_bgr: np.ndarray, anns: list[dict], font) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ann in anns:
        cls_id = ann["cls"]
        pts    = ann["pts"]
        if not pts:
            continue
        color_rgb = COLORS[cls_id] if cls_id < len(COLORS) else (200, 200, 200)
        draw.polygon(pts, fill=(*color_rgb, int(255 * ALPHA)), outline=(*color_rgb, 230))

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        tx = max(0, min(xs) + 4)
        ty = max(0, min(ys) - 28)
        label = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
        draw.rectangle((tx - 2, ty, tx + len(label) * 14, ty + 26), fill=(*color_rgb, 200))
        draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

    composited = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


# ── 병렬 워커 함수 ─────────────────────────────────────────────

def _count_label_file(lbl_file: Path) -> tuple[int, list[int]]:
    """txt 1개 → (유증상여부, [cls0, cls1, cls2] 카운트)"""
    counts = [0, 0, 0]
    labeled = 0
    with open(lbl_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    if lines:
        labeled = 1
    for line in lines:
        vals = line.split()
        if vals:
            c = int(vals[0])
            if 0 <= c < 3:
                counts[c] += 1
    return labeled, counts


def _render_one(args: tuple) -> int:
    """이미지 1개 시각화 후 저장 → 저장 성공 시 1 반환"""
    img_path, lbl_dir, out_split, font = args
    img = read_image(img_path)
    if img is None:
        return 0

    h, w = img.shape[:2]
    anns = parse_label(lbl_dir / (img_path.stem + ".txt"), w, h)
    vis  = draw_annotations(img, anns, font)

    ann_count = len(anns)
    status   = f"{'유증상' if ann_count > 0 else '무증상'}_{ann_count}개"
    out_path = out_split / f"{img_path.stem}_{status}.jpg"
    _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    buf.tofile(str(out_path))
    return 1


# ── split 처리 ─────────────────────────────────────────────────

def process_split(split: str):
    img_dir = YOLO_DIR / "images" / split
    lbl_dir = YOLO_DIR / "labels" / split

    if not img_dir.exists():
        print(f"  [{split}] 폴더 없음, 스킵")
        return

    img_files = list(img_dir.glob("*.jpg"))
    if not img_files:
        print(f"  [{split}] 이미지 없음")
        return

    random.seed(SEED)
    samples = random.sample(img_files, min(N_SAMPLES, len(img_files)))

    out_split = OUT_DIR / split
    out_split.mkdir(parents=True, exist_ok=True)

    # ── 통계 집계 (병렬) ──
    lbl_files    = list(lbl_dir.glob("*.txt"))
    cls_counts   = [0, 0, 0]
    labeled_imgs = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_count_label_file, lf): lf for lf in lbl_files}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"{split} 통계 집계", leave=False):
            labeled, counts = fut.result()
            labeled_imgs += labeled
            for i in range(3):
                cls_counts[i] += counts[i]

    print(f"\n[{split}]  이미지: {len(img_files):,}  |  "
          f"유증상: {labeled_imgs:,}  |  무증상: {len(img_files) - labeled_imgs:,}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {cls_counts[i]:,} 인스턴스")

    # ── 시각화 (병렬) ──
    font = load_font(24)   # 폰트는 1회만 로드
    args_list = [(img_path, lbl_dir, out_split, font) for img_path in samples]

    saved = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_render_one, args): args[0] for args in args_list}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"{split} 시각화"):
            saved += fut.result()

    print(f"  → {saved}개 저장: {out_split}")


if __name__ == "__main__":
    print("=== YOLO 데이터셋 뷰어 ===")
    for split in ["train", "val", "test"]:
        process_split(split)
    print(f"\n완료! 출력: {OUT_DIR}")
