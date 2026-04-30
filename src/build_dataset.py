# Builds the 70/15/15 train/val/test split and writes a manifest CSV
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AI_DIR = ROOT / "data" / "raw" / "ai"
UNSPLASH_DIR = ROOT / "data" / "raw" / "real" / "downloads"
FFHQ_DIR = ROOT / "data" / "raw" / "ffhq"

# mix the real side: 75% unsplash + 25% ffhq faces
UNSPLASH_FRACTION = 0.75
OUT = ROOT / "data" / "processed" / "manifest.csv"

SEED = 42
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)


def split_70_15_15(items):
    # same seed every time = reproducible splits
    shuffled = list(items)
    random.Random(SEED).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def main():
    ai = list_images(AI_DIR)
    unsplash = list_images(UNSPLASH_DIR)
    ffhq = list_images(FFHQ_DIR)
    print(f"found {len(ai)} AI, {len(unsplash)} unsplash, {len(ffhq)} ffhq")

    # target class size = however many AI images we have
    target = len(ai)
    n_unsplash = int(target * UNSPLASH_FRACTION)
    n_ffhq = target - n_unsplash
    print(f"real mix: {n_unsplash} unsplash + {n_ffhq} ffhq = {target}")

    if len(unsplash) < n_unsplash or len(ffhq) < n_ffhq:
        raise SystemExit(f"not enough images: need {n_unsplash} unsplash / {n_ffhq} ffhq")

    rng = random.Random(SEED)
    ai = rng.sample(ai, target)
    real = rng.sample(unsplash, n_unsplash) + rng.sample(ffhq, n_ffhq)
    rng.shuffle(real)  # mix the sources so splits aren't segregated
    print(f"balanced to {target} per class")

    ai_splits = split_70_15_15(ai)
    real_splits = split_70_15_15(real)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for split in ["train", "val", "test"]:
            for p in ai_splits[split]:
                w.writerow([str(p), 1, split])
            for p in real_splits[split]:
                w.writerow([str(p), 0, split])

    for split in ["train", "val", "test"]:
        a, r = len(ai_splits[split]), len(real_splits[split])
        print(f"  {split}: {a} AI + {r} real = {a + r}")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
