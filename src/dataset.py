import csv
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet stats: used because we're fine-tuning ResNet which was trained on ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transform(train=True):
    if train:
    
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ImageDataset(Dataset):
    def __init__(self, manifest_path, split, transform=None, limit=None):
        self.samples = []
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.samples.append((row["path"], int(row["label"])))

        if limit:
            self.samples = self.samples[:limit]

        self.transform = transform or get_transform(train=(split == "train"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        # some images are corrupt: just skip with a blank if it fails
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"bad image: {path} ({e})")
            img = Image.new("RGB", (224, 224))
        return self.transform(img), torch.tensor(label)
