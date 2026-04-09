"""
Pillar II on real Indian data: Compare ImageNet init vs SimCLR pretraining.

SimCLR pretrains on unlabeled Indian city patches, then fine-tunes for
classification. Compare against ImageNet-init baseline on same data.
"""

import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import BATCH_SIZE, SEED
from main import get_device, set_seed

set_seed(SEED)
device = get_device()
print(f"Device: {device}")

# ── Load Indian data ──
os.environ.setdefault("INDIAN_PATCH_ROOT", "data/indian_cities_locked")
from src.real_data_loaders import IndianCityDataset
from src.pillar2_self_supervised import (
    UnlabelledFromLabelledDataset,
    run_self_supervised_pipeline,
)
from src.dataset import MultispectralAugment
from src.train import progressive_train
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


def build_indian_loaders(batch_size=BATCH_SIZE):
    """Build train/val/test loaders from Indian patches."""
    ds = IndianCityDataset(cities=["Mumbai", "Delhi_NCR", "Bangalore"])
    print(f"  Indian dataset: {len(ds)} samples")

    labels = [s[1] for s in ds.samples]
    indices = list(range(len(ds)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=SEED
    )
    train_labels = [labels[i] for i in train_idx]
    train_idx2, val_idx = train_test_split(
        train_idx, test_size=0.15, stratify=train_labels, random_state=SEED
    )

    from torch.utils.data import Subset
    train_ds = Subset(ds, train_idx2)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


# ── Build loaders ──
train_loader, val_loader, test_loader = build_indian_loaders()
print(f"  train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

results = {}

# ── Experiment 1: ImageNet init -> fine-tune (baseline) ──
print("\n" + "=" * 60)
print("  EXP 1: ImageNet Init -> Fine-tune on Indian data")
print("=" * 60)

t0 = time.time()
_, _, test_metrics_imagenet, total_time = progressive_train(
    backbone_name="efficientnet_b0",
    device=device,
    loaders=(train_loader, val_loader, test_loader),
)
results["imagenet_init"] = {
    "oa": float(test_metrics_imagenet["oa"]),
    "f1": float(test_metrics_imagenet["f1"]),
    "miou": float(test_metrics_imagenet["miou"]),
    "precision": float(test_metrics_imagenet["precision"]),
    "recall": float(test_metrics_imagenet["recall"]),
    "time_min": round((time.time() - t0) / 60, 2),
}
print(f"\n  ImageNet init: OA={test_metrics_imagenet['oa']:.4f} F1={test_metrics_imagenet['f1']:.4f}")

# ── Experiment 2: SimCLR pretrain -> fine-tune ──
print("\n" + "=" * 60)
print("  EXP 2: SimCLR Pretrain -> Fine-tune on Indian data")
print("=" * 60)

t0 = time.time()
# Use train set as unlabeled data for SimCLR
unlabeled_ds = UnlabelledFromLabelledDataset(train_loader.dataset)

classifier, history, test_metrics_simclr = run_self_supervised_pipeline(
    device=device,
    pretrain_epochs=20,
    finetune_epochs=10,
    loaders=(train_loader, val_loader, test_loader),
    unlabeled_dataset=unlabeled_ds,
)
results["simclr_pretrain"] = {
    "oa": float(test_metrics_simclr["oa"]),
    "f1": float(test_metrics_simclr["f1"]),
    "miou": float(test_metrics_simclr["miou"]),
    "precision": float(test_metrics_simclr["precision"]),
    "recall": float(test_metrics_simclr["recall"]),
    "time_min": round((time.time() - t0) / 60, 2),
}
print(f"\n  SimCLR pretrain: OA={test_metrics_simclr['oa']:.4f} F1={test_metrics_simclr['f1']:.4f}")

# ── Save results ──
out_dir = "outputs/research_results"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "pillar2_indian_simclr.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_dir}/pillar2_indian_simclr.json")

# Summary
print("\n" + "=" * 60)
print("  PILLAR II COMPARISON SUMMARY")
print("=" * 60)
for exp_name, m in results.items():
    print(f"  {exp_name:20s} | OA={m['oa']:.4f} | F1={m['f1']:.4f} | mIoU={m['miou']:.4f} | {m['time_min']:.1f} min")
