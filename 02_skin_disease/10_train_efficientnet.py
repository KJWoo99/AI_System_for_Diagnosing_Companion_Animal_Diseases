"""
Stage 2: EfficientNet-B7 학습 (병변 crop 이미지 기반 분류)

사전 조건: python build_crop_dataset.py 실행 완료

설치:
  pip install timm torch torchvision albumentations tqdm
"""

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ── 설정 ──────────────────────────────────────────────────
CROP_DIR   = Path("crop_dataset")
SAVE_DIR   = Path("runs/stage2_efficientnet")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CLASSES    = ["A4_농포_여드름", "A6_결절_종괴"]
NUM_CLASSES = len(CLASSES)
CLS2IDX    = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE   = 600   # EfficientNet-B7 권장 입력 크기
BATCH_SIZE = 16
EPOCHS     = 80
EARLY_STOP = 15    # val_acc 개선 없으면 15 epoch 후 종료
LR         = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── 데이터셋 ──────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.samples = []
        self.transform = transform
        for cls_name in CLASSES:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((img_path, CLS2IDX[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


def get_transforms(is_train: bool):
    if is_train:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            # 피부 병변 특화 augmentation
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.6),

            # 색상/질감 변환 (피부 촬영 환경 다양성)
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.1, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=30, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),

            # 노이즈/블러 (실제 촬영 품질 차이 모방)
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            # Elastic / Grid distortion (병변 형태 다양성)
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                A.GridDistortion(p=1.0),
            ], p=0.3),

            # Cutout (오탐 억제)
            A.CoarseDropout(max_holes=4, max_height=40, max_width=40, p=0.4),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


# ── 모델 ──────────────────────────────────────────────────
def build_model() -> nn.Module:
    """EfficientNet-B7, ImageNet pretrained, fine-tune 전략 적용"""
    model = timm.create_model(
        "efficientnet_b7",
        pretrained=True,
        num_classes=NUM_CLASSES,
    )
    # 초기에는 classifier만 학습 (feature extractor 고정)
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    return model


def unfreeze_all(model: nn.Module):
    """전체 레이어 fine-tuning (warm-up 후 호출)"""
    for param in model.parameters():
        param.requires_grad = True


# ── 학습 루프 ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                                    # clip 전에 unscale 필수
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient explosion 방지
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # 클래스별 정확도
    per_class = {}
    for i, cls in enumerate(CLASSES):
        mask = [l == i for l in all_labels]
        if sum(mask) == 0:
            per_class[cls] = 0.0
            continue
        per_class[cls] = sum(
            p == l for p, l, m in zip(all_preds, all_labels, mask) if m
        ) / sum(mask)

    return total_loss / total, correct / total, per_class


def main():
    print(f"Device: {DEVICE}")

    # 데이터로더
    train_ds = CropDataset(CROP_DIR / "train", get_transforms(True))
    val_ds   = CropDataset(CROP_DIR / "val",   get_transforms(False))
    print(f"Train: {len(train_ds)}장, Val: {len(val_ds)}장")

    # 클래스 분포 확인
    from collections import Counter
    train_dist = Counter(lbl for _, lbl in train_ds.samples)
    print("클래스 분포:", {CLASSES[k]: v for k, v in sorted(train_dist.items())})

    # 클래스 불균형 대응: weighted sampler
    weights = [1.0 / train_dist[lbl] for _, lbl in train_ds.samples]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = build_model().to(DEVICE)

    # Label Smoothing: 과적합 방지, 정확도 향상
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AMP GradScaler (fp16 gradient underflow 방지)
    scaler = torch.amp.GradScaler()

    # Phase 1: classifier만 학습 (10 epoch warm-up)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_acc = 0.0
    history  = []
    unfreeze_done = False
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # Phase 2: 10 epoch 후 전체 fine-tuning
        if epoch == 11 and not unfreeze_done:
            print("\n[전체 레이어 Unfreeze → Fine-tuning 시작]")
            unfreeze_all(model)
            optimizer = AdamW(model.parameters(), lr=LR * 0.1, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - 10, eta_min=1e-7)
            unfreeze_done = True

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_acc, per_class = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
            "per_class":  {k: round(v, 4) for k, v in per_class.items()},
        }
        history.append(record)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")
        print(f"         클래스별: " + ", ".join(f"{k.split('_')[0]}={v:.3f}"
                                                  for k, v in per_class.items()))

        # 최고 모델 저장 + Early Stopping 카운터
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), SAVE_DIR / "best.pt")
            print(f"         ★ Best 저장 (val_acc={best_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"\n[Early Stopping] {EARLY_STOP} epoch 개선 없음 → 종료")
                break

        # 히스토리 저장
        with open(SAVE_DIR / "history.json", "w") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n학습 완료! 최고 Val Accuracy: {best_acc:.4f}")
    print(f"저장 경로: {SAVE_DIR}/best.pt")


if __name__ == "__main__":
    main()
