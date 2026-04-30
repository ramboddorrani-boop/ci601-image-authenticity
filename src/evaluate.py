''' Runs best.pt on the held-out test set and saves the confusion matrix
and per-class precision/recall/f1 as JSON. The test split CSV came from
Kaggle so the paths inside it are /kaggle/input/... - have to remap to
local paths by filename. '''
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "code" / "models" / "best.pt"
TEST_CSV = ROOT / "results" / "test_split.csv"
RESULTS = ROOT / "results"

# where the images actually live locally
LOCAL_AI = ROOT / "data" / "raw" / "ai"
LOCAL_UNSPLASH = ROOT / "data" / "raw" / "real" / "downloads"
LOCAL_FFHQ = ROOT / "data" / "raw" / "ffhq"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def remap_to_local(kaggle_path):
    
    name = Path(kaggle_path).name
    stem = Path(kaggle_path).stem  

   
    for base in [LOCAL_AI, LOCAL_UNSPLASH, LOCAL_FFHQ]:
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = list(base.rglob(stem + ext))
            if candidate:
                return candidate[0]
    return None


class TestSet(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return transform(img), torch.tensor(label)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # load model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    print(f"loaded best.pt (val acc {ckpt['val_acc']:.3f}, epoch {ckpt['epoch']})")

    # load test split, remap paths
    samples = []
    missing = 0
    with open(TEST_CSV) as f:
        for row in csv.DictReader(f):
            local = remap_to_local(row["path"])
            if local is None:
                missing += 1
                continue
            samples.append((local, int(row["label"])))
    print(f"test set: {len(samples)} images ({missing} couldn't be mapped locally)")

    loader = DataLoader(TestSet(samples), batch_size=64, shuffle=False, num_workers=4)

    # inference
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            probs = torch.softmax(out, dim=1)[:, 1]  # P(AI)
            preds = out.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    n = len(all_preds)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = correct / n

   
    cm = [[0, 0], [0, 0]]
    for p, l in zip(all_preds, all_labels):
        cm[l][p] += 1

    # per-class precision / recall / f1
    def prf(cls):
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in range(2) if other != cls)
        fn = sum(cm[cls][other] for other in range(2) if other != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return precision, recall, f1

    real_p, real_r, real_f = prf(0)
    ai_p, ai_r, ai_f = prf(1)

    print("\n--- test set results ---")
    print(f"accuracy: {acc:.4f}  ({correct}/{n})")
    print(f"confusion matrix (rows=actual, cols=predicted):")
    print(f"              pred_real  pred_ai")
    print(f"  actual_real   {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"  actual_ai     {cm[1][0]:6d}    {cm[1][1]:6d}")
    print(f"\nper-class:")
    print(f"  real:  precision {real_p:.3f}  recall {real_r:.3f}  f1 {real_f:.3f}")
    print(f"  ai:    precision {ai_p:.3f}  recall {ai_r:.3f}  f1 {ai_f:.3f}")

    out = {
        "accuracy": acc,
        "n_test": n,
        "matrix": cm,
        "per_class": {
            "real": {"precision": real_p, "recall": real_r, "f1": real_f},
            "ai": {"precision": ai_p, "recall": ai_r, "f1": ai_f},
        },
    }
    with open(RESULTS / "confusion_matrix.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved to {RESULTS / 'confusion_matrix.json'}")

    # save predictions for analysis
    with open(RESULTS / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "pred", "prob_ai"])
        for s, p, pr in zip(samples, all_preds, all_probs):
            w.writerow([str(s[0]), s[1], p, pr])
    print(f"saved predictions to {RESULTS / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
