import sys
import uuid
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, render_template, request
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from forensics import analyse as analyse_forensics

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"
UPLOAD_DIR = ROOT / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# load the model once at startup
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model = model.to(device).eval()
print(f"model loaded on {device} (val acc {ckpt['val_acc']:.3f})")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def predict(img):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
    return probs[1]  # P(AI)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    f = request.files.get("image")
    if not f:
        return render_template("index.html", error="No file received.")

    # save with a random name to avoid collisions
    ext = Path(f.filename).suffix.lower() or ".jpg"
    name = uuid.uuid4().hex + ext
    path = UPLOAD_DIR / name
    f.save(path)

    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return render_template("index.html", error=f"Could not read image: {e}")

    p_ai = predict(img)
    verdict = "AI-generated" if p_ai > 0.5 else "Real photograph"
    confidence = max(p_ai, 1 - p_ai)

    forensics = analyse_forensics(path)

    return render_template(
        "result.html",
        image_url=f"/static/uploads/{name}",
        filename=f.filename,
        width=img.width,
        height=img.height,
        verdict=verdict,
        confidence=f"{confidence * 100:.1f}",
        p_ai=f"{p_ai:.3f}",
        p_real=f"{1 - p_ai:.3f}",
        ela=forensics["ela"]["explanation"],
        exif=forensics["exif"]["explanation"],
        noise=forensics["noise"]["explanation"],
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
