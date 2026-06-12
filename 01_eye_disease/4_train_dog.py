"""
train_dog.py - 반려견 안구질환 이진 분류
==========================================

[데이터셋 - 강아지 10개 질환, 모두 binary (무=0, 유=1)]
  이진 질환 (폴더 그대로):
    결막염, 색소침착성각막염, 안검내반증, 안검염, 안검종양, 유루증, 핵경화
  이진화 질환 (무 vs 유(상+하 또는 초기+비성숙+성숙 합침)):
    궤양성각막질환:   무 / 유(상+하)
    비궤양성각막질환: 무 / 유(상+하)
    백내장:           무 / 유(초기+비성숙+성숙)

[논문1 - EfficientNet, 2024.9]
  EfficientNet-B0 Acc=0.92, F1=0.92
  SGD > Adam (복잡한 패턴에서 로컬미니멈 탈출 우수)
  BatchNorm + Dropout 조합

[논문2 - ViT 서강대, 2024.6]
  질환별 개별 이진 모델 → BCE Loss
  ViT-Base-16 전이학습
  Adam + StepLR(7 step×0.1), Dropout=0.5, EarlyStopping

[모델 선택]
  --model vit    : ViT-Base-16      (논문2 권고, 기본값)
  --model effnet : EfficientNet-B4  (논문1 권고, SGD 사용)

[실행]
  cd <데이터셋 루트 디렉토리>

  python 4_train_dog.py                    # ViT (기본)
  python 4_train_dog.py --model effnet     # EfficientNet-B4
"""

import argparse
import contextlib
import copy
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    vit_b_16,        ViT_B_16_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
)

try:
    from sklearn.metrics import classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────
BASE     = Path(".")
SAVE_DIR = Path("./runs_dog")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp",
            ".JPG", ".JPEG", ".PNG", ".BMP"}
CLASSES  = ["무", "유"]   # 0: 정상, 1: 질환

# ── 질환 정의 ──────────────────────────────────────────────────
# str  → 단일 양성 폴더 (결막염/유 등)
# list → 여러 폴더 합쳐서 label=1 (이진화)
DISEASES = {
    "결막염":           "유",
    "색소침착성각막염":  "유",
    "안검내반증":        "유",
    "안검염":           "유",
    "안검종양":         "유",
    "유루증":           "유",
    "핵경화":           "유",
    # 이진화: 상+하 → 유
    "궤양성각막질환":    ["상", "하"],
    "비궤양성각막질환":  ["상", "하"],
    # 이진화: 초기+비성숙+성숙 → 유
    "백내장":           ["초기", "비성숙", "성숙"],
}

DISEASE_NAMES = list(DISEASES.keys())


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class DogEyeDataset(Dataset):
    def __init__(self, items: list, transform=None):
        self.items     = items    # [(Path, 0|1), ...]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), 0)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def collect(split_root: Path, disease: str) -> list:
    """
    split_root/개/{disease}/ 에서 이미지 수집.
    무/ → label=0
    유/ 또는 [상,하] 또는 [초기,비성숙,성숙] → label=1
    """
    pos_folders = DISEASES[disease]
    if isinstance(pos_folders, str):
        pos_folders = [pos_folders]

    items = []
    base  = split_root / "개" / disease

    # 음성 (무)
    neg_dir = base / "무"
    if neg_dir.exists():
        for f in neg_dir.iterdir():
            if f.suffix in IMG_EXTS:
                items.append((f, 0))
    else:
        print(f"  [경고] 없음: {neg_dir}")

    # 양성 (이진화 포함)
    for folder in pos_folders:
        pos_dir = base / folder
        if pos_dir.exists():
            for f in pos_dir.iterdir():
                if f.suffix in IMG_EXTS:
                    items.append((f, 1))
        else:
            print(f"  [경고] 없음: {pos_dir}")

    return items


def split_items(items: list, val_ratio: float = 0.1,
                seed: int = 42) -> tuple:
    rng     = random.Random(seed)
    buckets = defaultdict(list)
    for it in items:
        buckets[it[1]].append(it)
    tr, va = [], []
    for lbl, lst in sorted(buckets.items()):
        rng.shuffle(lst)
        n   = max(1, int(len(lst) * val_ratio))
        va += lst[:n]
        tr += lst[n:]
    return tr, va


# ─────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_tf(mode: str) -> transforms.Compose:
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ─────────────────────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────────────────────
def build_model(name: str, device: torch.device) -> nn.Module:
    if name == "vit":
        m    = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_f = m.heads.head.in_features   # 768
        m.heads = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_f, 1),
        )
    else:  # effnet
        m    = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_f, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
        )
    return m.to(device)


def freeze_backbone(model: nn.Module, name: str):
    key = "heads" if name == "vit" else "classifier"
    for n, p in model.named_parameters():
        if key not in n:
            p.requires_grad = False


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def get_param_groups(model: nn.Module, name: str,
                     backbone_lr: float, head_lr: float):
    key = "heads" if name == "vit" else "classifier"
    return [
        {"params": [p for n, p in model.named_parameters() if key not in n],
         "lr": backbone_lr},
        {"params": [p for n, p in model.named_parameters() if key in n],
         "lr": head_lr},
    ]


# ─────────────────────────────────────────────────────────────
# 학습 / 평가
# ─────────────────────────────────────────────────────────────
CRIT = nn.BCEWithLogitsLoss()


def train_epoch(model, loader, opt, device, scaler, accum: int):
    model.train()
    loss_s = correct = total = 0
    is_cu  = device.type == "cuda"
    opt.zero_grad()

    for step, (imgs, labels) in enumerate(
            tqdm(loader, desc="  train", leave=False)):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        ctx = torch.amp.autocast("cuda") if is_cu else contextlib.nullcontext()
        with ctx:
            out  = model(imgs).squeeze(1)
            loss = CRIT(out, labels) / accum

        (scaler.scale(loss) if is_cu else loss).backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            if is_cu:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad()

        preds   = (out.detach() > 0).long()
        loss_s += loss.item() * accum * imgs.size(0)
        correct += preds.eq(labels.long()).sum().item()
        total   += imgs.size(0)

    return loss_s / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    is_cu  = device.type == "cuda"
    loss_s = correct = total = 0
    all_p, all_l = [], []
    ctx = torch.amp.autocast("cuda") if is_cu else contextlib.nullcontext()

    for imgs, labels in tqdm(loader, desc="  eval ", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), \
                       labels.to(device, non_blocking=True)
        with ctx:
            out  = model(imgs).squeeze(1)
            loss = CRIT(out, labels)
        preds   = (out > 0).long()
        loss_s += loss.item() * imgs.size(0)
        correct += preds.eq(labels.long()).sum().item()
        total   += imgs.size(0)
        all_p.extend(preds.cpu().tolist())
        all_l.extend(labels.long().cpu().tolist())

    return loss_s / total, correct / total, all_p, all_l


# ─────────────────────────────────────────────────────────────
# 평가 리포트
# ─────────────────────────────────────────────────────────────
def report(preds, labels):
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    print(f"\n  [혼동행렬]")
    print(f"             예측:무  예측:유")
    print(f"  실제:무  {tn:8d}  {fp:8d}")
    print(f"  실제:유  {fn:8d}  {tp:8d}")
    print(f"\n  Accuracy  = {acc:.4f}")
    print(f"  Precision = {prec:.4f}")
    print(f"  Recall    = {rec:.4f}")
    print(f"  F1-Score  = {f1:.4f}")

    if HAS_SKLEARN:
        print(f"\n  [sklearn 상세 리포트]")
        print(classification_report(labels, preds,
                                    target_names=CLASSES, digits=4))
    return acc, prec, rec, f1


# ─────────────────────────────────────────────────────────────
# 단일 질환 학습
# ─────────────────────────────────────────────────────────────
def run(disease: str, model_name: str, cfg: dict) -> float:
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cu  = device.type == "cuda"
    accum  = 2

    # 이진화 정보 표시
    pos = DISEASES[disease]
    pos_label = f"유({'+'.join(pos)})" if isinstance(pos, list) else "유"

    print(f"\n{'='*60}")
    print(f"  강아지 / {disease}  (binary: 무=0, {pos_label}=1)")
    print(f"  모델: {model_name.upper()}  |  device: {device}")
    print(f"  epochs={cfg['epochs']}, batch={cfg['batch']}×{accum}={cfg['batch']*accum}")
    print(f"  patience={cfg['patience']}, seed={cfg['seed']}")
    print(f"{'='*60}")

    # ── 데이터 ────────────────────────────────────────────
    tr_root   = BASE / "1.Training"   / "원천데이터"
    test_root = BASE / "2.Validation" / "원천데이터"

    all_items  = collect(tr_root,   disease)
    test_items = collect(test_root, disease)
    tr_items, va_items = split_items(all_items, 0.1, cfg["seed"])

    if not all_items:
        print(f"  [오류] 데이터 없음")
        return 0.0

    n0_tr = sum(1 for _, l in tr_items   if l == 0)
    n1_tr = sum(1 for _, l in tr_items   if l == 1)
    n0_va = sum(1 for _, l in va_items   if l == 0)
    n1_va = sum(1 for _, l in va_items   if l == 1)
    n0_te = sum(1 for _, l in test_items if l == 0)
    n1_te = sum(1 for _, l in test_items if l == 1)

    print(f"  {'':>4}  {'학습':>8}  {'검증':>8}  {'테스트':>8}")
    print(f"  {'무':>4}  {n0_tr:>8,}  {n0_va:>8,}  {n0_te:>8,}")
    print(f"  {pos_label:>4}  {n1_tr:>8,}  {n1_va:>8,}  {n1_te:>8,}")
    print(f"  {'합':>4}  {n0_tr+n1_tr:>8,}  {n0_va+n1_va:>8,}  {n0_te+n1_te:>8,}")

    kw = dict(batch_size=cfg["batch"], num_workers=cfg["workers"],
              pin_memory=is_cu, persistent_workers=cfg["workers"] > 0)
    tr_dl = DataLoader(DogEyeDataset(tr_items,   get_tf("train")), shuffle=True,  **kw)
    va_dl = DataLoader(DogEyeDataset(va_items,   get_tf("eval")),  shuffle=False, **kw)

    # ── 모델 & 저장 경로 ──────────────────────────────────
    model    = build_model(model_name, device)
    save_dir = SAVE_DIR / f"{disease}_{model_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    scaler   = torch.amp.GradScaler("cuda") if is_cu else None
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    pat_cnt  = 0

    # ═══════════════════════════════════════════════════
    # Phase 1: backbone 고정, head만 학습 (5 epochs)
    # ═══════════════════════════════════════════════════
    freeze_backbone(model, model_name)
    head_params = [p for p in model.parameters() if p.requires_grad]

    if model_name == "vit":
        opt1 = torch.optim.Adam(head_params, lr=1e-3, weight_decay=1e-4)
    else:
        opt1 = torch.optim.SGD(head_params, lr=1e-2,
                               momentum=0.9, weight_decay=1e-4)
    sch1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=7, gamma=0.1)

    FREEZE = 5
    print(f"\n  [Phase 1] head만 학습 ({FREEZE} epochs)")
    for ep in range(1, FREEZE + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, tr_dl, opt1, device, scaler, accum)
        vl, va, _, _ = eval_epoch(model, va_dl, device)
        sch1.step()
        if va > best_acc:
            best_acc, best_wts = va, copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_dir / "best.pt")
        print(f"  ep{ep:02d}  tr {tl:.4f}/{ta:.4f}  "
              f"va {vl:.4f}/{va:.4f}  best={best_acc:.4f}  {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════
    # Phase 2: 전체 fine-tuning (differential LR)
    # ═══════════════════════════════════════════════════
    unfreeze_all(model)
    pat_cnt   = 0
    remaining = cfg["epochs"] - FREEZE

    if model_name == "vit":
        opt2 = torch.optim.Adam(
            get_param_groups(model, model_name, 1e-5, 1e-4),
            weight_decay=1e-4)
    else:
        opt2 = torch.optim.SGD(
            get_param_groups(model, model_name, 1e-3 * 0.1, 1e-3),
            momentum=0.9, weight_decay=1e-4)
    sch2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=7, gamma=0.1)

    print(f"\n  [Phase 2] 전체 fine-tuning (최대 {remaining}ep, patience={cfg['patience']})")
    for ep in range(FREEZE + 1, cfg["epochs"] + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, tr_dl, opt2, device, scaler, accum)
        vl, va, _, _ = eval_epoch(model, va_dl, device)
        sch2.step()
        if va > best_acc:
            best_acc, best_wts = va, copy.deepcopy(model.state_dict())
            pat_cnt = 0
            torch.save(model.state_dict(), save_dir / "best.pt")
        else:
            pat_cnt += 1
            if pat_cnt >= cfg["patience"]:
                print(f"  ★ Early stop (ep={ep})")
                break
        print(f"  ep{ep:02d}  tr {tl:.4f}/{ta:.4f}  "
              f"va {vl:.4f}/{va:.4f}  best={best_acc:.4f}  "
              f"pat={pat_cnt}/{cfg['patience']}  {time.time()-t0:.1f}s")

    torch.save(model.state_dict(), save_dir / "last.pt")
    torch.save({"disease": disease, "model_name": model_name,
                "classes": CLASSES}, save_dir / "meta.pt")

    # ═══════════════════════════════════════════════════
    # 테스트 평가
    # ═══════════════════════════════════════════════════
    model.load_state_dict(best_wts)
    te_dl = DataLoader(DogEyeDataset(test_items, get_tf("eval")),
                       shuffle=False, **kw)
    _, te_acc, preds, labs = eval_epoch(model, te_dl, device)

    print(f"\n  ★★ TEST  {disease}  Accuracy = {te_acc:.4f} ★★")
    acc, prec, rec, f1 = report(preds, labs)
    print(f"\n  저장: {save_dir}")
    return te_acc


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="반려견 안구질환 이진 분류 (논문1+2 기반) — 10개 질환 전체 학습"
    )
    ap.add_argument("--model",    default="vit",
                    choices=["vit", "effnet"])
    ap.add_argument("--epochs",   type=int, default=50)
    ap.add_argument("--batch",    type=int, default=16)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers",  type=int, default=4)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    cfg = dict(epochs=args.epochs, batch=args.batch,
               patience=args.patience, workers=args.workers,
               seed=args.seed)

    print(f"\n  반려견 안구질환 이진 분류 — 10개 질환 전체")
    print(f"  모델: {args.model.upper()}  (BCE Loss)")

    results = {}
    for d in DISEASE_NAMES:
        acc = run(d, args.model, cfg)
        results[d] = acc

    print(f"\n{'='*60}")
    print(f"  최종 결과 [{args.model.upper()}]")
    for d, acc in results.items():
        bar = "█" * int(acc * 20)
        print(f"  {d:>15}: {acc:.4f}  {bar}")
    print(f"  평균:            {sum(results.values())/len(results):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
