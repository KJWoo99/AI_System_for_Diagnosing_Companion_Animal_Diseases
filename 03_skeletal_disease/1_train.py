"""Skeletal Disease Classification - Training  (EfficientNet-B5, 2-phase)"""
import os, time
import torch, torch.nn as nn, torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B5_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import CustomDataset

CLASS_NAMES = ['Mu03 (갈비뼈골절)', 'Mu05 (슬개골탈구)', 'Mu06 (전십자인대파열)', 'Mu07 (추간판질환)']
DATA_ROOT   = './data/skeletal_disease'
BATCH_SIZE  = 8
IMG_SIZE    = 456
SAVE_PATH   = './best.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = CustomDataset(os.path.join(DATA_ROOT, 'train'), transform=train_tf)
val_ds   = CustomDataset(os.path.join(DATA_ROOT, 'val'),   transform=val_tf)
cls_counts = [0] * len(CLASS_NAMES)
for _, lbl in train_ds.samples: cls_counts[lbl] += 1
weights  = [1.0 / cls_counts[lbl] for _, lbl in train_ds.samples]
sampler  = WeightedRandomSampler(weights, len(weights))
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
print(f'Train: {len(train_ds)}  Val: {len(val_ds)}')

efficientnet = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
for param in efficientnet.parameters():
    param.requires_grad = False
efficientnet.classifier[1] = nn.Linear(2048, len(CLASS_NAMES))
for param in efficientnet.classifier.parameters():
    param.requires_grad = True
efficientnet = efficientnet.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))


def train_phase(net, loader_tr, loader_va, num_epochs, lr, label, best_acc, save_path,
                accumulation_steps=2, patience=0):
    optimizer = optim.RAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                      patience=5, min_lr=1e-7)
    tl, vl, va = [], [], []
    pat_cnt = 0
    for epoch in range(num_epochs):
        net.train()
        running = 0.0
        optimizer.zero_grad()
        t0 = time.time()
        for i, (x, y) in enumerate(tqdm(loader_tr, desc=f'[{label}] {epoch+1}/{num_epochs}')):
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                loss = criterion(net(x), y) / accumulation_steps
            scaler.scale(loss).backward()
            running += loss.item() * accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader_tr):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        net.eval()
        v_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader_va:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    out = net(x)
                v_loss  += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total   += y.size(0)
        t_l = running / len(loader_tr)
        v_l = v_loss  / len(loader_va)
        v_a = 100 * correct / total
        tl.append(t_l); vl.append(v_l); va.append(v_a)
        scheduler.step(v_a)
        if v_a > best_acc:
            best_acc = v_a
            pat_cnt  = 0
            torch.save(net.state_dict(), save_path)
        else:
            pat_cnt += 1
            if patience > 0 and pat_cnt >= patience:
                print(f'  [{label}] Early stop (ep={epoch+1})')
                break
        print(f'[{label}] Epoch {epoch+1}/{num_epochs} Loss:{t_l:.4f} ValAcc:{v_a:.2f}% Best:{best_acc:.2f}% ({time.time()-t0:.1f}s)')
    return tl, vl, va, best_acc


if __name__ == '__main__':
    best = 0.0
    print('=== Phase 1: Head (20 epochs) ===')
    tl1, vl1, va1, best = train_phase(efficientnet, train_ld, val_ld, 20, 3e-4, 'P1', best, SAVE_PATH)
    print('=== Phase 2: Full fine-tuning (50 epochs) ===')
    for p in efficientnet.parameters(): p.requires_grad = True
    tl2, vl2, va2, best = train_phase(efficientnet, train_ld, val_ld, 50, 3e-5, 'P2', best, SAVE_PATH, patience=10)
    all_tl = tl1 + tl2
    all_vl = vl1 + vl2
    all_va = va1 + va2
    p1     = len(tl1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(all_tl, label='Train'); axes[0].plot(all_vl, label='Val')
    axes[0].axvline(p1, color='gray', linestyle='--', label='Phase 2')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Train / Val Loss'); axes[0].legend()
    axes[1].plot(all_va); axes[1].axvline(p1, color='gray', linestyle='--', label='Phase 2')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', title='Validation Accuracy'); axes[1].legend()
    plt.tight_layout(); plt.savefig('training_curves.png', dpi=120); plt.close()
    print(f'Best val acc: {max(all_va):.2f}%')

