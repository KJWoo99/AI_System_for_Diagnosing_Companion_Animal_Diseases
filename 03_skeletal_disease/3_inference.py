"""Inference visualization - load best.pt and show predictions on test images."""
import torch, torch.nn as nn, numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import EfficientNet_B5_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import CustomDataset
import os

CLASS_NAMES = ['Mu03 (갈비뼈골절)', 'Mu05 (슬개골탈구)', 'Mu06 (전십자인대파열)', 'Mu07 (추간판질환)']
DATA_ROOT   = './data/skeletal_disease'
MODEL_PATH  = './best.pt'
IMG_SIZE    = 456

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = CustomDataset(os.path.join(DATA_ROOT, 'test'), transform=val_tf)

loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(2048, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

plt.figure(figsize=(20, 8))
for idx, (x, y) in enumerate(loader):
    if idx >= 10:
        break
    with torch.no_grad():
        pred = model(x.to(device)).argmax(1).item()
    img = x.squeeze().numpy().transpose(1, 2, 0)
    img = np.clip(img * std + mean, 0, 1)
    ax  = plt.subplot(2, 5, idx + 1)
    ax.imshow(img)
    color = 'green' if pred == y.item() else 'red'
    ax.set_title(f'GT: {CLASS_NAMES[y.item()]}\nPred: {CLASS_NAMES[pred]}',
                 color=color, fontsize=9)
    ax.axis('off')
plt.suptitle('Skeletal Disease - Inference')
os.makedirs('result', exist_ok=True)
plt.tight_layout(); plt.savefig('result/inference_result.png', dpi=120); plt.close()
