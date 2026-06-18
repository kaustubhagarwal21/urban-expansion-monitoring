"""
Phase 7 GradCAM runner on real Indian-city patches.

Generates a compact grid of real-patch GradCAM visualizations for the
active deep-learning models across the locked 3-city benchmark.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import FIGURE_DIR, MODEL_DIR
from src.explainability import GradCAM, _to_rgb
from src.models import UrbanClassifier
from src.real_data_loaders import IndianCityDataset


LOCKED_CITIES = ["Mumbai", "Delhi_NCR", "Bangalore"]
ACTIVE_MODELS = ["resnet50", "efficientnet_b0", "swin_tiny", "mobilenet_v3_small"]
CLASS_NAMES = ["Urban", "Non-Urban", "Transition"]


def _ensure_locked_env():
    os.environ.setdefault("INDIAN_PATCH_ROOT", "data/indian_cities_locked")
    os.environ.setdefault("INDIAN_CITIES_FILTER", ",".join(LOCKED_CITIES))


def _pick_sample_paths():
    _ensure_locked_env()
    ds = IndianCityDataset(cities=LOCKED_CITIES)
    selected = {}
    for img_path, label in ds.samples:
        norm = os.path.normpath(img_path)
        for city in LOCKED_CITIES:
            if city not in selected and f"{os.sep}{city}{os.sep}" in norm:
                selected[city] = {"path": img_path, "label": int(label)}
        if len(selected) == len(LOCKED_CITIES):
            break
    if len(selected) != len(LOCKED_CITIES):
        raise RuntimeError("Could not find at least one patch for each locked city.")
    return selected


def _load_model(backbone, device):
    ckpt_path = Path(MODEL_DIR) / f"{backbone}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    model = UrbanClassifier(backbone_name=backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _target_layer(model):
    return model.fpn.smooth_convs[-1]


def generate_real_gradcam_grid(device="cuda"):
    samples = _pick_sample_paths()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    rows = len(ACTIVE_MODELS)
    cols = len(LOCKED_CITIES)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_2d(axes)

    metadata = {"cities": {}, "models": ACTIVE_MODELS}

    for row_idx, backbone in enumerate(ACTIVE_MODELS):
        model = _load_model(backbone, device)
        cam = GradCAM(model, _target_layer(model))
        try:
            for col_idx, city in enumerate(LOCKED_CITIES):
                item = samples[city]
                patch = np.load(item["path"]).astype(np.float32)
                tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                heatmap, pred_class = cam.generate(tensor)
                rgb = _to_rgb(patch)

                ax = axes[row_idx, col_idx]
                ax.imshow(rgb)
                ax.imshow(heatmap, cmap="jet", alpha=0.35)
                ax.set_xticks([])
                ax.set_yticks([])
                title = f"{backbone}\n{city} | gt={CLASS_NAMES[item['label']]} pred={CLASS_NAMES[pred_class]}"
                ax.set_title(title, fontsize=9)

                metadata["cities"].setdefault(city, {})[backbone] = {
                    "path": item["path"],
                    "ground_truth": CLASS_NAMES[item["label"]],
                    "prediction": CLASS_NAMES[pred_class],
                }
        finally:
            cam.remove_hooks()

    fig.suptitle("Phase 7: Real Indian GradCAM", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, "phase7_real_gradcam.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    meta_path = os.path.join(FIGURE_DIR, "phase7_real_gradcam_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved GradCAM figure to {fig_path}")
    print(f"Saved GradCAM metadata to {meta_path}")
    return fig_path, meta_path


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_real_gradcam_grid(device=device)
