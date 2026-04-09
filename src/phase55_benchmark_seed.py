"""
Phase 5.5 benchmark runner for the locked 3-city Indian setup.

Runs either:
  - one deep backbone with a specified random seed, or
  - the classical baselines (SVM + Random Forest) with a specified seed.

Outputs:
  outputs/research_results/phase2_{backbone}_seed{seed}.json
  outputs/research_results/phase2_baselines_seed{seed}.json
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import BATCH_SIZE, OUTPUT_DIR, SEED, STAGES, MODEL_DIR
from main import get_device, set_seed
from src.baselines import run_baselines
from src.real_data_loaders import RealDataManager
from src.train import progressive_train


LOCKED_CITIES = ["Mumbai", "Delhi_NCR", "Bangalore"]
RESEARCH_RESULTS_DIR = Path(OUTPUT_DIR) / "research_results"
SUPPORTED_BACKBONES = [
    "efficientnet_b0",
    "resnet50",
    "swin_tiny",
    "mobilenet_v3_small",
]


def _ensure_locked_env():
    os.environ.setdefault("INDIAN_PATCH_ROOT", "data/indian_cities_locked")
    os.environ.setdefault("INDIAN_CITIES_FILTER", ",".join(LOCKED_CITIES))


def _build_loaders(seed):
    manager = RealDataManager(batch_size=BATCH_SIZE, num_workers=0, seed=seed)
    return manager.get_indian_city_loaders(cities=LOCKED_CITIES)


def _legacy_metrics_key(backbone):
    return {
        "mobilenet_v3_small": "mobilenet_v3_small",
        "efficientnet_b0": "efficientnet_b0",
        "resnet50": "resnet50",
        "swin_tiny": "swin_tiny",
    }[backbone]


def run_backbone_seed(backbone, seed, device, epochs_per_stage=5):
    _ensure_locked_env()
    set_seed(seed)
    original_stages = copy.deepcopy(STAGES)
    for stage in STAGES:
        stage["epochs"] = epochs_per_stage

    try:
        loaders = _build_loaders(seed)
        _, _, test_metrics, total_time = progressive_train(
            backbone_name=backbone,
            device=device,
            loaders=loaders,
        )
    finally:
        STAGES[:] = original_stages

    payload = {
        "backbone": backbone,
        "seed": int(seed),
        "oa": float(test_metrics["oa"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "miou": float(test_metrics["miou"]),
        "training_time_min": round(total_time / 60.0, 2),
        "cities": LOCKED_CITIES,
        "dataset": "indian_cities_locked",
    }
    source_ckpt = Path(MODEL_DIR) / f"{backbone}_best.pth"
    if source_ckpt.exists():
        seed_ckpt = Path(MODEL_DIR) / f"{backbone}_seed{seed}.pth"
        shutil.copy2(source_ckpt, seed_ckpt)
        payload["checkpoint"] = str(seed_ckpt)
    out_path = RESEARCH_RESULTS_DIR / f"phase2_{backbone}_seed{seed}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path, payload


def run_baselines_seed(seed):
    _ensure_locked_env()
    set_seed(seed)
    train_loader, _, test_loader = _build_loaders(seed)
    raw_results = run_baselines(
        train_dataset=train_loader.dataset,
        test_dataset=test_loader.dataset,
    )
    results = {}
    for name, metrics in raw_results.items():
        results[name] = {
            "oa": float(metrics["oa"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "miou": float(metrics["miou"]),
        }
    payload = {
        "seed": int(seed),
        "dataset": "indian_cities_locked",
        "cities": LOCKED_CITIES,
        "results": results,
    }
    out_path = RESEARCH_RESULTS_DIR / f"phase2_baselines_seed{seed}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path, payload


def main():
    parser = argparse.ArgumentParser(description="Phase 5.5 benchmark runner")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--epochs-per-stage", type=int, default=5)
    args = parser.parse_args()

    if not args.baselines and not args.backbone:
        parser.error("Specify either --baselines or --backbone.")
    if args.baselines and args.backbone:
        parser.error("Use either --baselines or --backbone, not both.")

    RESEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.baselines:
        out_path, payload = run_baselines_seed(args.seed)
        print(f"Saved baselines to {out_path}")
        print(
            " | ".join(
                f"{name}: OA={metrics['oa']:.4f} F1={metrics['f1']:.4f} mIoU={metrics['miou']:.4f}"
                for name, metrics in payload["results"].items()
            )
        )
        return

    device = get_device()
    out_path, payload = run_backbone_seed(
        backbone=args.backbone,
        seed=args.seed,
        device=device,
        epochs_per_stage=args.epochs_per_stage,
    )
    print(f"Saved {args.backbone} seed {args.seed} results to {out_path}")
    print(
        f"OA={payload['oa']:.4f} F1={payload['f1']:.4f} "
        f"mIoU={payload['miou']:.4f} time={payload['training_time_min']:.2f} min"
    )


if __name__ == "__main__":
    main()
