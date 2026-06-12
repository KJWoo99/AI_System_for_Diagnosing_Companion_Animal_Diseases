"""
6_evaluate_cat.py - 반려묘 안구질환 평가 (Validation 데이터)
==============================================================
runs_cat/{질환}_vit/best.pt 로드 → Validation 배치 평가
출력: 질환별 Accuracy / Precision / Recall / F1 / AUROC
"""

from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, roc_auc_score,
        classification_report,
    )
except ImportError:
    print("[오류] scikit-learn 필요:  pip install scikit-learn")
    sys.exit(1)

import numpy as np

# ─────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
RUNS_DIR   = SCRIPT_DIR / "runs_cat"

# Validation 원천데이터의 실제 구조:
#   ...\2.Validation\원천데이터\VS\고양이\안구\일반\{질환}\{무|유}\
VAL_ROOT = Path(
    r"D:\self_study\153.반려동물 안구질환 데이터"
    r"\01.데이터\2.Validation\원천데이터\VS\고양이\안구\일반"
)

DISEASES = ["각막궤양", "각막부골편", "결막염", "비궤양성각막염", "안검염"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]
BATCH    = 32


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class ValDataset(Dataset):
    def __init__(self, items, transform):
        self.items     = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def collect_val(disease: str):
    """VAL_ROOT/{질환}/{무|유}/ 에서 이미지 수집."""
    items = []
    for label, grade in enumerate(["무", "유"]):
        d = VAL_ROOT / disease / grade
        if not d.exists():
            print(f"  [경고] 없음: {d}")
            continue
        for f in d.iterdir():
            if f.suffix in IMG_EXTS:
                items.append((f, label))
    return items


# ─────────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────────
def load_model(disease: str, device: torch.device) -> nn.Module:
    pt = RUNS_DIR / f"{disease}_vit" / "best.pt"
    if not pt.exists():
        raise FileNotFoundError(f"best.pt 없음: {pt}")

    model = vit_b_16(weights=None)
    in_f  = model.heads.head.in_features  # 768
    model.heads = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_f, 1),
    )
    state = torch.load(pt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
# 평가
# ─────────────────────────────────────────────────────────────
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def evaluate(disease: str, device: torch.device) -> dict:
    items = collect_val(disease)
    if not items:
        print(f"  [건너뜀] {disease}: 데이터 없음")
        return None

    n0 = sum(1 for _, l in items if l == 0)
    n1 = sum(1 for _, l in items if l == 1)
    print(f"\n  {disease}  무={n0}, 유={n1}, 합={len(items)}")

    loader = DataLoader(
        ValDataset(items, eval_tf),
        batch_size=BATCH, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    model  = load_model(disease, device)
    all_logits, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)   # [B]
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs  = 1 / (1 + np.exp(-logits))       # sigmoid
    preds  = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")

    return dict(disease=disease, n=len(items), n0=n0, n1=n1,
                acc=acc, precision=p, recall=r, f1=f1, auroc=auroc,
                labels=labels, preds=preds)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"VAL_ROOT: {VAL_ROOT}")
    if not VAL_ROOT.exists():
        print(f"[오류] VAL_ROOT 없음: {VAL_ROOT}")
        sys.exit(1)

    results = []
    for disease in DISEASES:
        try:
            r = evaluate(disease, device)
            if r:
                results.append(r)
        except FileNotFoundError as e:
            print(f"  [건너뜀] {e}")

    if not results:
        print("\n평가 결과 없음")
        return

    # ── 결과 출력 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'질환':<14}  {'N':>5}  {'Acc':>6}  {'Prec':>6}  "
          f"{'Recall':>7}  {'F1':>6}  {'AUROC':>6}")
    print("=" * 70)
    for r in results:
        print(
            f"  {r['disease']:<14}  {r['n']:>5}  "
            f"{r['acc']:>6.4f}  {r['precision']:>6.4f}  "
            f"{r['recall']:>7.4f}  {r['f1']:>6.4f}  {r['auroc']:>6.4f}"
        )
    print("=" * 70)

    # 전체 평균
    avg_acc   = np.mean([r["acc"]   for r in results])
    avg_auroc = np.mean([r["auroc"] for r in results if not np.isnan(r["auroc"])])
    print(f"  {'평균':<14}  {'':>5}  {avg_acc:>6.4f}  {'':>6}  "
          f"{'':>7}  {'':>6}  {avg_auroc:>6.4f}")
    print("=" * 70)

    # ── classification report ──────────────────────────────
    print()
    for r in results:
        print(f"--- {r['disease']} ---")
        print(classification_report(
            r["labels"], r["preds"],
            target_names=["무(정상)", "유(질환)"], digits=4
        ))

    # ── 이미지 저장 ────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from sklearn.metrics import ConfusionMatrixDisplay
    import csv
    from datetime import datetime

    # Windows 한글 폰트 설정
    _kr_font = next(
        (f.name for f in fm.fontManager.ttflist if "Malgun" in f.name),
        None
    )
    if _kr_font:
        plt.rcParams["font.family"] = _kr_font

    save_dir = SCRIPT_DIR / "eval_results"
    save_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV 저장
    csv_path = save_dir / f"cat_eye_eval_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["disease","n","n0","n1","acc","precision","recall","f1","auroc"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ["disease","n","n0","n1","acc","precision","recall","f1","auroc"]})
        writer.writerow({"disease":"평균","n":"","n0":"","n1":"",
                         "acc":avg_acc,"precision":"","recall":"","f1":"","auroc":avg_auroc})
    print(f"\n[저장] {csv_path}")

    # TXT 전체 리포트 저장
    txt_path = save_dir / f"cat_eye_eval_{ts}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"반려묘 안구질환 평가 결과  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"Validation 데이터: {VAL_ROOT}\n\n")
        f.write("=" * 70 + "\n")
        f.write(f"  {'질환':<14}  {'N':>5}  {'Acc':>6}  {'Prec':>6}  {'Recall':>7}  {'F1':>6}  {'AUROC':>6}\n")
        f.write("=" * 70 + "\n")
        for r in results:
            f.write(f"  {r['disease']:<14}  {r['n']:>5}  "
                    f"{r['acc']:>6.4f}  {r['precision']:>6.4f}  "
                    f"{r['recall']:>7.4f}  {r['f1']:>6.4f}  {r['auroc']:>6.4f}\n")
        f.write(f"  {'평균':<14}  {'':>5}  {avg_acc:>6.4f}  {'':>6}  {'':>7}  {'':>6}  {avg_auroc:>6.4f}\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"--- {r['disease']} ---\n")
            f.write(classification_report(r["labels"], r["preds"],
                                          target_names=["무(정상)", "유(질환)"], digits=4))
            f.write("\n")
    print(f"[저장] {txt_path}")

    # 질환별 Confusion Matrix (5개 subplot)
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        ConfusionMatrixDisplay.from_predictions(
            r["labels"], r["preds"],
            display_labels=["무(정상)", "유(질환)"],
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"{r['disease']}\nAcc={r['acc']:.3f} AUROC={r['auroc']:.3f}", fontsize=9)
    plt.suptitle("반려묘 안구질환 Confusion Matrix", fontsize=12, y=1.02)
    plt.tight_layout()
    cm_path = save_dir / f"cat_eye_confusion_matrix_{ts}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {cm_path}")

    # 전체 성능 요약 막대그래프
    names  = [r["disease"] for r in results]
    accs   = [r["acc"]   for r in results]
    aurocs = [r["auroc"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - 0.2, accs,   0.35, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + 0.2, aurocs, 0.35, label="AUROC",    color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("반려묘 안구질환 분류 성능")
    ax.legend()
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    bar_path = save_dir / f"cat_eye_performance_{ts}.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[저장] {bar_path}")


if __name__ == "__main__":
    main()
