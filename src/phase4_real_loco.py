"""
Real-data Leave-One-City-Out (LOCO) experiments for the locked 3-city setup.

Runs progressive fine-tuning on the Indian patch subset using:
  - Mumbai
  - Delhi_NCR
  - Bangalore

Each fold trains on two cities and tests on the held-out third city.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    BATCH_SIZE,
    OUTPUT_DIR,
    SEED,
    STAGES,
)
from main import get_device, set_seed
from src.real_data_loaders import IndianCityDataset
from src.train import progressive_train
from src.dataset import MultispectralAugment


LOCKED_CITIES = ["Mumbai", "Delhi_NCR", "Bangalore"]


def _infer_city_from_path(img_path: str, candidate_cities):
    norm = os.path.normpath(img_path)
    for city in candidate_cities:
        if f"{os.sep}{city}{os.sep}" in norm:
            return city
    parts = Path(norm).parts
    for city in candidate_cities:
        if city in parts:
            return city
    raise ValueError(f"Could not infer city from path: {img_path}")


class PatchFileDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = list(samples)  # [(img_path, label, city)]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, city = self.samples[idx]
        img = np.load(img_path).astype(np.float32)
        patch = torch.from_numpy(img)
        if self.transform is not None:
            patch = self.transform(patch)
        return patch, label


def _collect_samples(cities):
    base = IndianCityDataset(cities=cities)
    samples = []
    for img_path, label in base.samples:
        city = _infer_city_from_path(img_path, cities)
        samples.append((img_path, int(label), city))
    return samples


def build_loco_loaders(held_out_city, batch_size=BATCH_SIZE, seed=SEED):
    train_cities = [c for c in LOCKED_CITIES if c != held_out_city]
    all_samples = _collect_samples(LOCKED_CITIES)

    train_samples = [s for s in all_samples if s[2] in train_cities]
    test_samples = [s for s in all_samples if s[2] == held_out_city]

    train_labels = [s[1] for s in train_samples]
    train_idx, val_idx = train_test_split(
        list(range(len(train_samples))),
        test_size=0.18,
        stratify=train_labels,
        random_state=seed,
    )

    train_ds = PatchFileDataset([train_samples[i] for i in train_idx], transform=MultispectralAugment())
    val_ds = PatchFileDataset([train_samples[i] for i in val_idx], transform=None)
    test_ds = PatchFileDataset(test_samples, transform=None)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    meta = {
        "held_out_city": held_out_city,
        "train_cities": train_cities,
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "test_count": len(test_ds),
    }
    return train_loader, val_loader, test_loader, meta


def _resolve_seed(cli_seed=None):
    if cli_seed is not None:
        return int(cli_seed)
    raw = os.environ.get("SEED_OVERRIDE", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid SEED_OVERRIDE={raw!r}") from exc
    return int(SEED)


def run_real_loco(backbone_name, device, epochs_per_stage=5, held_out_only=None, seed=SEED):
    original_epochs = [stage["epochs"] for stage in STAGES]
    for stage in STAGES:
        stage["epochs"] = epochs_per_stage

    results = {}
    try:
        fold_cities = [held_out_only] if held_out_only else LOCKED_CITIES
        for held_out_city in fold_cities:
            print("\n" + "=" * 70)
            print(
                f"  REAL LOCO | backbone={backbone_name} | held-out={held_out_city} | seed={seed}"
            )
            print("=" * 70)
            train_loader, val_loader, test_loader, meta = build_loco_loaders(
                held_out_city,
                seed=seed,
            )
            print(
                f"  Train cities: {', '.join(meta['train_cities'])} | "
                f"train={meta['train_count']} val={meta['val_count']} test={meta['test_count']}"
            )
            _, _, test_metrics, total_time = progressive_train(
                backbone_name=backbone_name,
                device=device,
                loaders=(train_loader, val_loader, test_loader),
            )
            results[held_out_city] = {
                "oa": float(test_metrics["oa"]),
                "precision": float(test_metrics["precision"]),
                "recall": float(test_metrics["recall"]),
                "f1": float(test_metrics["f1"]),
                "miou": float(test_metrics["miou"]),
                "training_time_min": round(total_time / 60.0, 2),
                "train_cities": meta["train_cities"],
                "train_count": meta["train_count"],
                "val_count": meta["val_count"],
                "test_count": meta["test_count"],
            }
    finally:
        for stage, epoch_value in zip(STAGES, original_epochs):
            stage["epochs"] = epoch_value

    avg = {
        "oa": float(np.mean([r["oa"] for r in results.values()])),
        "precision": float(np.mean([r["precision"] for r in results.values()])),
        "recall": float(np.mean([r["recall"] for r in results.values()])),
        "f1": float(np.mean([r["f1"] for r in results.values()])),
        "miou": float(np.mean([r["miou"] for r in results.values()])),
        "training_time_min": float(np.sum([r["training_time_min"] for r in results.values()])),
    }
    return {"backbone": backbone_name, "folds": results, "average": avg}


def save_loco_outputs(payload, seed=SEED):
    out_dir = Path(OUTPUT_DIR) / "research_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone = payload["backbone"]
    suffix = "" if int(seed) == int(SEED) else f"_seed{seed}"

    json_path = out_dir / f"phase4_loco_{backbone}{suffix}.json"
    md_path = out_dir / f"phase4_loco_{backbone}{suffix}.md"

    json_path.write_text(json.dumps(payload, indent=2))

    lines = [
        f"# Phase 4 LOCO - {backbone}",
        "",
        f"Seed: `{seed}`",
        "",
        "| Held-out City | OA | Precision | Recall | F1 | mIoU | Train Time (min) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for city, metrics in payload["folds"].items():
        lines.append(
            f"| {city} | {metrics['oa']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['miou']:.4f} | "
            f"{metrics['training_time_min']:.2f} |"
        )
    avg = payload["average"]
    lines.extend([
        "",
        f"Average OA: `{avg['oa']:.4f}`",
        f"Average F1: `{avg['f1']:.4f}`",
        f"Average mIoU: `{avg['miou']:.4f}`",
        f"Total train time: `{avg['training_time_min']:.2f}` min",
        "",
    ])
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Phase 4 real-data LOCO runner")
    parser.add_argument("--backbone", required=True, choices=[
        "efficientnet_b0", "resnet50", "swin_tiny"
    ])
    parser.add_argument("--epochs-per-stage", type=int, default=5)
    parser.add_argument("--held-out-city", choices=LOCKED_CITIES, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seed = _resolve_seed(args.seed)
    set_seed(seed)
    device = get_device()
    payload = run_real_loco(
        backbone_name=args.backbone,
        device=device,
        epochs_per_stage=args.epochs_per_stage,
        held_out_only=args.held_out_city,
        seed=seed,
    )
    payload["seed"] = seed
    json_path, md_path = save_loco_outputs(payload, seed=seed)
    print(f"\nSaved LOCO JSON to {json_path}")
    print(f"Saved LOCO summary to {md_path}")


if __name__ == "__main__":
    main()
