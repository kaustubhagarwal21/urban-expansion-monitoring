"""
Seed-averaged experiment runner for paper-grade results.

Runs a given experiment function across multiple random seeds,
collects per-seed metrics, and computes mean +/- std for each metric.
This produces the statistical evidence needed for paper tables.

Usage:
    python -m src.seed_runner                        # default 3 seeds, synthetic
    python -m src.seed_runner --seeds 42 123 456 789 2024
    python -m src.seed_runner --data-source real --real-dataset eurosat
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import OUTPUT_DIR, SEED, BATCH_SIZE
from src.dataset import get_dataloaders
from src.train import progressive_train
from src.pillar1_sar_fusion import train_fusion
from src.pillar2_self_supervised import run_self_supervised_pipeline


RESULTS_DIR = os.path.join(OUTPUT_DIR, "paper_experiment")

# Metrics we care about for paper tables
PAPER_METRICS = ["oa", "precision", "recall", "f1", "miou"]


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)


def _extract_metrics(raw_metrics: dict) -> dict:
    """Extract only the numeric metrics we need for averaging."""
    out = {}
    for key in PAPER_METRICS:
        val = raw_metrics.get(key)
        if val is not None:
            out[key] = float(val)
    return out


def aggregate_seed_results(
    all_results: List[Dict[str, float]],
) -> Dict[str, dict]:
    """
    Compute mean, std, min, max across seeds for each metric.

    Returns:
        {metric_name: {"mean": ..., "std": ..., "min": ..., "max": ..., "values": [...]}}
    """
    if not all_results:
        return {}

    keys = all_results[0].keys()
    agg = {}
    for key in keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            agg[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }
    return agg


# =========================================================================
# Experiment wrappers
# =========================================================================

def run_supervised_with_seed(
    seed: int,
    device: str,
    backbone: str = "efficientnet_b0",
    data_source: str = "synthetic",
    real_dataset: str = "eurosat",
) -> dict:
    """Run supervised optical baseline for one seed."""
    set_all_seeds(seed)
    loaders = get_dataloaders(
        data_source=data_source,
        real_dataset=real_dataset,
    )
    _, _, metrics, train_time = progressive_train(
        backbone_name=backbone,
        device=device,
        loaders=loaders,
    )
    result = _extract_metrics(metrics)
    result["time_sec"] = float(train_time)
    result["seed"] = seed
    return result


def run_ssl_with_seed(
    seed: int,
    device: str,
    pretrain_epochs: int = 5,
    finetune_epochs: int = 5,
    data_source: str = "synthetic",
    real_dataset: str = "eurosat",
) -> dict:
    """Run self-supervised pipeline for one seed."""
    set_all_seeds(seed)
    loaders = get_dataloaders(
        data_source=data_source,
        real_dataset=real_dataset,
    )
    unlabeled_dataset = loaders[0].dataset
    t0 = time.time()
    _, _, metrics = run_self_supervised_pipeline(
        device=device,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        loaders=loaders,
        unlabeled_dataset=unlabeled_dataset,
    )
    result = _extract_metrics(metrics)
    result["time_sec"] = float(time.time() - t0)
    result["seed"] = seed
    return result


def run_fusion_with_seed(
    seed: int,
    device: str,
    epochs: int = 5,
    data_source: str = "synthetic",
) -> dict:
    """Run multimodal fusion for one seed."""
    set_all_seeds(seed)
    t0 = time.time()
    _, _, metrics = train_fusion(
        device=device,
        epochs=epochs,
        data_source=data_source,
    )
    result = _extract_metrics(metrics)
    result["time_sec"] = float(time.time() - t0)
    result["seed"] = seed
    return result


# =========================================================================
# Main runner
# =========================================================================

def run_seed_averaged_experiments(
    seeds: List[int] = None,
    device: str = "cuda",
    backbone: str = "efficientnet_b0",
    data_source: str = "synthetic",
    real_dataset: str = "eurosat",
    multimodal_data_source: str = "synthetic",
    pretrain_epochs: int = 5,
    finetune_epochs: int = 5,
    fusion_epochs: int = 5,
) -> dict:
    """
    Run all three paper experiments across multiple seeds.

    Returns:
        {
            "config": {...},
            "supervised_optical": {"per_seed": [...], "aggregated": {...}},
            "self_supervised_optical": {...},
            "multimodal_fusion": {...},
        }
    """
    if seeds is None:
        seeds = [42, 123, 456]

    print("=" * 70)
    print("  SEED-AVERAGED PAPER EXPERIMENTS")
    print("=" * 70)
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print(f"  Backbone: {backbone}")
    print(f"  Optical data: {data_source} ({real_dataset})")
    print(f"  Multimodal data: {multimodal_data_source}")
    print()

    full_results = {
        "config": {
            "seeds": seeds,
            "backbone": backbone,
            "data_source": data_source,
            "real_dataset": real_dataset,
            "multimodal_data_source": multimodal_data_source,
            "pretrain_epochs": pretrain_epochs,
            "finetune_epochs": finetune_epochs,
            "fusion_epochs": fusion_epochs,
        },
    }

    # --- Supervised optical ---
    print("\n[1/3] Supervised optical baseline")
    print("-" * 40)
    sup_results = []
    for i, seed in enumerate(seeds):
        print(f"\n  Seed {seed} ({i+1}/{len(seeds)})")
        r = run_supervised_with_seed(
            seed, device, backbone, data_source, real_dataset
        )
        sup_results.append(r)
        print(f"    OA={r['oa']:.4f}  F1={r['f1']:.4f}  mIoU={r['miou']:.4f}")

    sup_agg = aggregate_seed_results(sup_results)
    full_results["supervised_optical"] = {
        "per_seed": sup_results,
        "aggregated": sup_agg,
    }
    print(f"\n  Supervised avg: OA={sup_agg['oa']['mean']:.4f}+-{sup_agg['oa']['std']:.4f}  "
          f"F1={sup_agg['f1']['mean']:.4f}+-{sup_agg['f1']['std']:.4f}")

    # --- Self-supervised optical ---
    print("\n[2/3] Self-supervised optical")
    print("-" * 40)
    ssl_results = []
    for i, seed in enumerate(seeds):
        print(f"\n  Seed {seed} ({i+1}/{len(seeds)})")
        r = run_ssl_with_seed(
            seed, device, pretrain_epochs, finetune_epochs,
            data_source, real_dataset,
        )
        ssl_results.append(r)
        print(f"    OA={r['oa']:.4f}  F1={r['f1']:.4f}  mIoU={r['miou']:.4f}")

    ssl_agg = aggregate_seed_results(ssl_results)
    full_results["self_supervised_optical"] = {
        "per_seed": ssl_results,
        "aggregated": ssl_agg,
    }
    print(f"\n  SSL avg: OA={ssl_agg['oa']['mean']:.4f}+-{ssl_agg['oa']['std']:.4f}  "
          f"F1={ssl_agg['f1']['mean']:.4f}+-{ssl_agg['f1']['std']:.4f}")

    # --- Multimodal fusion ---
    print("\n[3/3] Multimodal fusion")
    print("-" * 40)
    fus_results = []
    for i, seed in enumerate(seeds):
        print(f"\n  Seed {seed} ({i+1}/{len(seeds)})")
        r = run_fusion_with_seed(
            seed, device, fusion_epochs, multimodal_data_source,
        )
        fus_results.append(r)
        print(f"    OA={r['oa']:.4f}  F1={r['f1']:.4f}  mIoU={r['miou']:.4f}")

    fus_agg = aggregate_seed_results(fus_results)
    full_results["multimodal_fusion"] = {
        "per_seed": fus_results,
        "aggregated": fus_agg,
    }
    print(f"\n  Fusion avg: OA={fus_agg['oa']['mean']:.4f}+-{fus_agg['oa']['std']:.4f}  "
          f"F1={fus_agg['f1']['mean']:.4f}+-{fus_agg['f1']['std']:.4f}")

    # Save full results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "seed_averaged_results.json")
    with open(path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"\n  Full results saved to {path}")

    return full_results


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run seed-averaged experiments for paper"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds to average over",
    )
    parser.add_argument("--backbone", default="efficientnet_b0",
                        choices=["vgg16", "resnet50", "efficientnet_b0"])
    parser.add_argument("--data-source", default="synthetic",
                        choices=["synthetic", "real"])
    parser.add_argument("--real-dataset", default="eurosat",
                        choices=["eurosat"])
    parser.add_argument("--multimodal-data-source", default="synthetic",
                        choices=["synthetic", "real"])
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--fusion-epochs", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_seed_averaged_experiments(
        seeds=args.seeds,
        device=device,
        backbone=args.backbone,
        data_source=args.data_source,
        real_dataset=args.real_dataset,
        multimodal_data_source=args.multimodal_data_source,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        fusion_epochs=args.fusion_epochs,
    )


if __name__ == "__main__":
    main()
