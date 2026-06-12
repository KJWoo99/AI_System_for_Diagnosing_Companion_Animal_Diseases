"""CustomDataset for skeletal disease — JSON annotation-based."""
import os, json, glob
from PIL import Image
from torch.utils.data import Dataset

DISEASE_TO_LABEL = {'Mu03': 0, 'Mu05': 1, 'Mu06': 2, 'Mu07': 3}

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform        = transform
        self.disease_to_label = DISEASE_TO_LABEL
        self.samples          = []

        img_dir = os.path.join(root_dir, 'images')
        ann_dir = os.path.join(root_dir, 'annotations')
        for fname in os.listdir(img_dir):
            ann_path = os.path.join(ann_dir, os.path.splitext(fname)[0] + '.json')
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
            disease = ann.get('metadata', {}).get('Disease-Name', None)
            label   = self.disease_to_label.get(disease, -1)
            if label != -1:
                self.samples.append((os.path.join(img_dir, fname), label))

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label
