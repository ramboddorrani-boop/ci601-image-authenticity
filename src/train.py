# Trains ResNet50 on real vs AI images.
# python src/train.py          - full run
# python src/train.py --smoke  - 1 epoch on tiny subset for sanity checks
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import ImageDataset, get_transform

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "code" / "models"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model():
    ''' Load ResNet50 pretrained on ImageNet, then replace the final layer
to output 2 classes (real / AI) instead of 1000 ImageNet classes.
Freezing the backbone so we only train the new classifier head -
faster, less chance of overfitting. '''
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def one_epoch(model, loader, loss_fn, optimiser, device, train):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    seen = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        if train:
            optimiser.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimiser.step()
        else:
            with torch.no_grad():
                out = model(imgs)
                loss = loss_fn(out, labels)

        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        seen += imgs.size(0)

    return total_loss / seen, correct / seen


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", type=int, default=15)
    args.add_argument("--batch", type=int, default=32)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--workers", type=int, default=4)
    args.add_argument("--patience", type=int, default=3,
                      help="stop if val acc doesn't improve for this many epochs")
    args.add_argument("--smoke", action="store_true")
    args = args.parse_args()

    if args.smoke:
        args.epochs = 1
        train_limit, val_limit = 200, 40
    else:
        train_limit = val_limit = None

    device = pick_device()
    print(f"using device: {device}")

    train_set = ImageDataset(MANIFEST, "train", get_transform(True), limit=train_limit)
    val_set = ImageDataset(MANIFEST, "val", get_transform(False), limit=val_limit)
    print(f"train: {len(train_set)}, val: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = make_model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0
    epochs_no_improve = 0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = one_epoch(model, train_loader, loss_fn, optimiser, device, train=True)
        val_loss, val_acc = one_epoch(model, val_loader, loss_fn, optimiser, device, train=False)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"epoch {epoch}/{args.epochs}  "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f}  "
              f"val loss {val_loss:.3f} acc {val_acc:.3f}  "
              f"({elapsed:.0f}s)")

        # save whenever val accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            fname = "smoke.pt" if args.smoke else "best.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
            }, MODELS_DIR / fname)
            print(f"  -> saved {fname} (val acc {val_acc:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience and not args.smoke:
                print(f"  -> no improvement for {args.patience} epochs, stopping early")
                break

    hist_name = "training_history_smoke.json" if args.smoke else "training_history.json"
    with open(RESULTS_DIR / hist_name, "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved history to {hist_name}")


if __name__ == "__main__":
    main()
