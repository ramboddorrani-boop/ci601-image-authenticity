
from pathlib import Path
import csv
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "figures"

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight"})
LABELS = {0: "Real", 1: "AI"}
COLOURS = {0: "#4C72B0", 1: "#DD8452"}


def dataset_split_figure() -> None:
    if not MANIFEST.exists():
        print("manifest.csv not found, skipping split figure")
        return
    counts: dict[tuple[str, int], int] = {}
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            k = (row["split"], int(row["label"]))
            counts[k] = counts.get(k, 0) + 1

    splits = ["train", "val", "test"]
    real = [counts.get((s, 0), 0) for s in splits]
    ai = [counts.get((s, 1), 0) for s in splits]
    x = np.arange(len(splits))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, real, w, label="Real", color=COLOURS[0])
    ax.bar(x + w / 2, ai, w, label="AI-generated", color=COLOURS[1])
    for i, (r, a) in enumerate(zip(real, ai)):
        ax.text(i - w / 2, r, f"{r:,}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, a, f"{a:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("Image count")
    ax.set_title("Dataset split (balanced, 70/15/15, seed=42)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out = OUT / "fig_dataset_split.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def training_curves_figure() -> None:
    f = RESULTS / "training_history.json"
    if not f.exists():
        return
    h = json.loads(f.read_text())
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(epochs, h["train_loss"], label="train")
    a1.plot(epochs, h["val_loss"], label="val")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.set_title("Loss"); a1.legend()
    a2.plot(epochs, h["train_acc"], label="train")
    a2.plot(epochs, h["val_acc"], label="val")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy"); a2.set_title("Accuracy"); a2.legend()
    for ax in (a1, a2):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    out = OUT / "fig_training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def confusion_matrix_figure() -> None:
    f = RESULTS / "confusion_matrix.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    cm = np.array(d["matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "AI"]); ax.set_yticklabels(["Real", "AI"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (test, n={cm.sum():,})")
    for i in range(2):
        for j in range(2):
            colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=colour)
    fig.colorbar(im, ax=ax, fraction=0.046)
    out = OUT / "fig_confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_split_figure()
    training_curves_figure()
    confusion_matrix_figure()


if __name__ == "__main__":
    sys.exit(main())
