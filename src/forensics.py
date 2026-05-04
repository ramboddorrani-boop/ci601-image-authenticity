
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


# ---------- 1. Error Level Analysis ----------


def ela(image_path, quality=90):
    original = Image.open(image_path).convert("RGB")


    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)


    diff = ImageChops.difference(original, resaved)
    arr = np.array(diff).astype(np.float32)


    mean_err = arr.mean()
    std_err = arr.std()


    if std_err < 0.6:
        score = 0.85
        explanation = f"Very uniform error level (std={std_err:.2f}), characteristic of generated images."
    elif std_err < 1.0:
        score = 0.6
        explanation = f"Low error variation (std={std_err:.2f}), possibly generated or heavily processed."
    elif std_err < 1.8:
        score = 0.3
        explanation = f"Error variation in typical real-photo range (std={std_err:.2f})."
    else:
        score = 0.1
        explanation = f"High error variation (std={std_err:.2f}), consistent with a real photo."

    return {
        "score": score,
        "mean_error": float(mean_err),
        "std_error": float(std_err),
        "explanation": explanation,
    }


# ---------- 2. EXIF metadata check ----------




def exif(image_path):
    img = Image.open(image_path)
    data = img.getexif() if hasattr(img, "getexif") else None

    if not data:
        return {
            "score": 0.7,
            "has_exif": False,
            "camera_tags_found": 0,
            "explanation": "No EXIF metadata. Common in AI-generated or stripped images.",
        }

    found = [t for t in CAMERA_TAGS if t in data]
    n = len(found)

    if n >= 4:
        score = 0.1
        explanation = f"Rich camera EXIF present ({n} tags), consistent with a real photo."
    elif n >= 1:
        score = 0.4
        explanation = f"Partial EXIF ({n} camera tags), could be real but stripped or edited."
    else:
        score = 0.65
        explanation = "EXIF present but no camera tags - likely edited or re-exported."

    return {
        "score": score,
        "has_exif": True,
        "camera_tags_found": n,
        "explanation": explanation,
    }


# ---------- 3. Noise pattern check ----------


def noise(image_path):
    img = Image.open(image_path).convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32)


    padded = np.pad(arr, 1, mode="edge")
    lap = (
        4 * padded[1:-1, 1:-1]
        - padded[:-2, 1:-1]
        - padded[2:, 1:-1]
        - padded[1:-1, :-2]
        - padded[1:-1, 2:]
    )

    noise_level = lap.std()


    if noise_level < 15:
        score = 0.7
        explanation = f"Low noise variance ({noise_level:.1f}), image is unusually smooth."
    elif noise_level < 25:
        score = 0.5
        explanation = f"Moderate noise level ({noise_level:.1f}), ambiguous signal."
    elif noise_level < 40:
        score = 0.3
        explanation = f"Noise level typical of real photos ({noise_level:.1f})."
    else:
        score = 0.1
        explanation = f"High noise variance ({noise_level:.1f}), strongly consistent with a real photo."

    return {
        "score": score,
        "noise_level": float(noise_level),
        "explanation": explanation,
    }


def analyse(image_path):
    """Run all three forensic checks. Returns a dict with each signal's output."""
    return {
        "ela": ela(image_path),
        "exif": exif(image_path),
        "noise": noise(image_path),
    }


if __name__ == "__main__":

    import sys
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("usage: python forensics.py <image>")
        sys.exit(1)
    print(json.dumps(analyse(path), indent=2))
