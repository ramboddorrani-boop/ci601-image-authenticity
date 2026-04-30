# Runs all three forensic checks on 100 images per class from the val split
import csv
import random
import os
from pathlib import Path

import numpy as np

from forensics import ela, exif, noise

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
RESULTS = ROOT / "results"
N_PER_CLASS = 100


def main():
    samples = {0: [], 1: []}  # 0 = real, 1 = ai
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                samples[int(row["label"])].append(row["path"])

    random.Random(42).shuffle(samples[0])
    random.Random(42).shuffle(samples[1])
    real_paths = samples[0][:N_PER_CLASS]
    ai_paths = samples[1][:N_PER_CLASS]

    def collect(paths, label):
        ela_stds, noise_levels, exif_counts = [], [], []
        for p in paths:
            try:
                ela_stds.append(ela(p)["std_error"])
                noise_levels.append(noise(p)["noise_level"])
                exif_counts.append(exif(p)["camera_tags_found"])
            except Exception as e:
                print(f"skip {p}: {e}")
        return ela_stds, noise_levels, exif_counts

    print(f"processing {N_PER_CLASS} real...")
    real_ela, real_noise, real_exif = collect(real_paths, 0)
    print(f"processing {N_PER_CLASS} AI...")
    ai_ela, ai_noise, ai_exif = collect(ai_paths, 1)

    def stats(name, real_vals, ai_vals):
        r = np.array(real_vals); a = np.array(ai_vals)
        print(f"\n{name}")
        print(f"  real: mean={r.mean():.2f} std={r.std():.2f} min={r.min():.2f} max={r.max():.2f}")
        print(f"  ai:   mean={a.mean():.2f} std={a.std():.2f} min={a.min():.2f} max={a.max():.2f}")
       
        print(f"  midpoint between class means: {(r.mean() + a.mean()) / 2:.2f}")

    stats("ELA std_error", real_ela, ai_ela)
    stats("Noise level (Laplacian std)", real_noise, ai_noise)
    stats("EXIF camera tags", real_exif, ai_exif)

    # save raw numbers so we can make a distribution plot for the report
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "forensic_calibration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "ela_std", "noise_level", "exif_tags"])
        for v in zip(real_ela, real_noise, real_exif):
            w.writerow(["real", *v])
        for v in zip(ai_ela, ai_noise, ai_exif):
            w.writerow(["ai", *v])
    print(f"\nsaved raw data to {RESULTS / 'forensic_calibration.csv'}")


if __name__ == "__main__":
    main()
