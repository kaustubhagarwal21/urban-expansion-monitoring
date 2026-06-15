"""One-time: curate a small, self-contained set of demo patches.

Picks a few patches per (city, class) from data/indian_cities_locked, copies the
raw .npy into webapp/backend/sample_patches/, renders an RGB preview PNG, and
writes samples.json with ground-truth labels. This makes the live-inference demo
work offline without the full ~hundreds-of-MB patch dataset.

Run once:  python bundle_samples.py
"""
import json
import os
import shutil
import sys

import numpy as np

import app_paths as P

sys.path.insert(0, str(P.BACKEND_DIR))
import inference  # for rgb_array / preview rendering  # noqa: E402

CLASS_NAMES = ["Urban", "Non-Urban", "Transition"]
CITIES = ["Mumbai", "Delhi_NCR", "Bangalore"]
PER_CLASS_PER_CITY = 2


def _majority_label(lbl_path):
    lbl = np.load(lbl_path).flatten().astype(int)
    return int(np.bincount(lbl).argmax())


def collect():
    samples = []
    for city in CITIES:
        city_dir = P.DATA / city
        if not city_dir.is_dir():
            print(f"  [skip] no data for {city}")
            continue
        # class -> list of (img_path, lbl_path)
        buckets = {0: [], 1: [], 2: []}
        for root, _dirs, files in os.walk(city_dir):
            for f in sorted(files):
                if not f.endswith("_img.npy"):
                    continue
                img_path = os.path.join(root, f)
                lbl_path = img_path.replace("_img.npy", "_lbl.npy")
                if not os.path.exists(lbl_path):
                    continue
                lab = _majority_label(lbl_path)
                if len(buckets[lab]) < PER_CLASS_PER_CITY * 4:  # collect a few extra to choose from
                    buckets[lab].append((img_path, lbl_path))
        for lab, items in buckets.items():
            for img_path, lbl_path in items[:PER_CLASS_PER_CITY]:
                samples.append((city, lab, img_path))
    return samples


def main():
    out_dir = P.SAMPLES
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = collect()
    manifest = []
    for i, (city, lab, img_path) in enumerate(samples):
        patch = np.load(img_path).astype(np.float32)
        sid = f"sample_{i:03d}"
        np.save(out_dir / f"{sid}.npy", patch)
        with open(out_dir / f"{sid}.png", "wb") as fh:
            fh.write(inference.preview_png(patch))
        manifest.append({
            "id": sid,
            "city": city,
            "true_class": lab,
            "true_label": CLASS_NAMES[lab],
            "preview": f"{sid}.png",
        })
        print(f"  + {sid}  {city:10s}  {CLASS_NAMES[lab]}")

    with open(out_dir / "samples.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nBundled {len(manifest)} sample patches -> {out_dir}")


if __name__ == "__main__":
    main()
