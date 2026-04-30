"""
Kaggle gives you a P100 GPU and everything trains in 30-45 min.
Outputs (model weights, history json) land in /kaggle/working.
"""
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


AI_DIR = Path("/kaggle/input/datasets/rambodorrani/ai-images/ai")
UNSPLASH_DIR = Path("/kaggle/input/datasets/jettchentt/unsplash-dataset-images-downloaded-250x250/downloads")
FFHQ_DIR = Path("/kaggle/input/datasets/tommykamaz/faces-dataset-small")
OUT_DIR = Path("/kaggle/working")
UNSPLASH_FRACTION = 0.75  # rest comes from FFHQ to balance face coverage

SEED = 42
BATCH = 64
EPOCHS = 15
LR = 1e-3
PATIENCE = 3


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


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


def list_images(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)


def make_splits():
    ai = list_images(AI_DIR)
    unsplash = list_images(UNSPLASH_DIR)
    ffhq = list_images(FFHQ_DIR)
    print(f"found {len(ai)} AI, {len(unsplash)} unsplash, {len(ffhq)} ffhq")

    target = len(ai)
    n_unsplash = int(target * UNSPLASH_FRACTION)
    n_ffhq = target - n_unsplash
    print(f"real mix: {n_unsplash} unsplash + {n_ffhq} ffhq = {target}")

    rng = random.Random(SEED)
    ai = rng.sample(ai, target)
    real = rng.sample(unsplash, n_unsplash) + rng.sample(ffhq, n_ffhq)
    rng.shuffle(real)

    def split(items):
        shuffled = list(items)
        random.Random(SEED).shuffle(shuffled)
        n = len(shuffled)
        return {
            "train": shuffled[:int(n * 0.70)],
            "val": shuffled[int(n * 0.70):int(n * 0.85)],
            "test": shuffled[int(n * 0.85):],
        }

    return split(ai), split(real)


class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"bad image: {path} ({e})")
            img = Image.new("RGB", (224, 224))
        return self.transform(img), torch.tensor(label)


def make_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def one_epoch(model, loader, loss_fn, opt, device, train):
    model.train() if train else model.eval()
    total_loss = 0
    correct = 0
    seen = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        if train:
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                out = model(imgs)
                loss = loss_fn(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        seen += imgs.size(0)
    return total_loss / seen, correct / seen


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ai_splits, real_splits = make_splits()

    train_samples = [(p, 1) for p in ai_splits["train"]] + [(p, 0) for p in real_splits["train"]]
    val_samples = [(p, 1) for p in ai_splits["val"]] + [(p, 0) for p in real_splits["val"]]
    test_samples = [(p, 1) for p in ai_splits["test"]] + [(p, 0) for p in real_splits["test"]]
    print(f"train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}")

    train_loader = DataLoader(ImageDataset(train_samples, get_transform(True)),
                              batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ImageDataset(val_samples, get_transform(False)),
                            batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    model = make_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = 0
    no_improve = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, ta = one_epoch(model, train_loader, loss_fn, opt, device, True)
        vl, va = one_epoch(model, val_loader, loss_fn, opt, device, False)
        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl); history["val_acc"].append(va)
        print(f"epoch {epoch}/{EPOCHS}  train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}  ({time.time()-t0:.0f}s)")

        if va > best:
            best = va
            no_improve = 0
            torch.save({"state_dict": model.state_dict(), "val_acc": va, "epoch": epoch},
                       OUT_DIR / "best.pt")
            print(f"  -> saved best.pt (val acc {va:.3f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  -> early stop (no improvement for {PATIENCE} epochs)")
                break

    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # also save the test split paths so we can evaluate later
    with open(OUT_DIR / "test_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        for p, l in test_samples:
            w.writerow([str(p), l])

    print(f"done. best val acc: {best:.3f}")


if __name__ == "__main__":
    main()
