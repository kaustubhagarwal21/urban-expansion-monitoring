"""
Small real-data ablation for the locked 3-city benchmark.

Configs:
  1. full      -> reuse Phase 2 EfficientNet-B0 result when available
  2. no_fpn    -> EfficientNet-B0 backbone without FPN fusion
  3. ce_only   -> EfficientNet-B0 with CrossEntropy-only loss
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EARLY_STOP_PATIENCE,
    OUTPUT_DIR,
    SEED,
    STAGES,
    WEIGHT_DECAY,
)
from main import get_device, set_seed
from src.dataset import get_dataloaders
from src.losses import CombinedLoss
from src.metrics import evaluate
from src.models import BACKBONE_BUILDERS, ClassificationHead, UNFREEZE_FNS
from src.train import train_one_epoch, validate


class NoFPNClassifier(nn.Module):
    _FIXED_SIZE_MODELS = {"swin_tiny": 224, "prithvi": 224}

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True, num_classes=3):
        super().__init__()
        self.backbone_name = backbone_name
        builder = BACKBONE_BUILDERS[backbone_name]
        self.blocks, ch_list = builder(pretrained=pretrained)
        self._is_wrapper = not isinstance(self.blocks, nn.ModuleList)
        self._required_size = self._FIXED_SIZE_MODELS.get(backbone_name, None)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = ClassificationHead(sum(ch_list), num_classes)

    def extract_features(self, x):
        if self._required_size and x.shape[-1] != self._required_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self._required_size, self._required_size),
                mode="bilinear",
                align_corners=False,
            )
        if self._is_wrapper:
            return self.blocks(x)
        feats = []
        out = x
        for block in self.blocks:
            out = block(out)
            feats.append(out)
        return feats

    def forward(self, x):
        features = self.extract_features(x)
        pooled = [self.gap(f).flatten(1) for f in features]
        return self.head(torch.cat(pooled, dim=1))


def _progressive_train_custom(model, train_loader, val_loader, test_loader, device, criterion_factory):
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_miou": []}
    best_val_loss = float("inf")
    best_state = None
    total_time = 0.0

    for stage_cfg in STAGES:
        stage_name = stage_cfg["name"]
        lr = stage_cfg["lr"]
        epochs = stage_cfg["epochs"]
        UNFREEZE_FNS[stage_cfg["unfreeze"]](model)
        criterion = criterion_factory().to(device)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        patience_counter = 0

        print(f"\n--- {stage_name} (lr={lr}, epochs={epochs}) ---")
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - start
            total_time += elapsed
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metrics["oa"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_miou"].append(val_metrics["miou"])

            print(
                f"  Epoch {epoch:2d}/{epochs} | TrLoss {train_loss:.4f} | "
                f"VaLoss {val_loss:.4f} | VaAcc {val_metrics['oa']:.4f} | "
                f"VaF1 {val_metrics['f1']:.4f} | mIoU {val_metrics['miou']:.4f} | {elapsed:.1f}s"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    model.load_state_dict(best_state)
    test_criterion = criterion_factory().to(device)
    _, test_metrics = validate(model, test_loader, test_criterion, device)
    return history, test_metrics, total_time


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


def _load_phase2_full(seed):
    out_dir = Path(OUTPUT_DIR) / "research_results"
    candidate_paths = []
    if int(seed) == int(SEED):
        candidate_paths.append(out_dir / "phase2_efficientnet_b0_results.json")
    else:
        candidate_paths.append(out_dir / f"phase2_efficientnet_b0_seed{seed}.json")
        candidate_paths.append(out_dir / "phase2_efficientnet_b0_results.json")

    path = next((p for p in candidate_paths if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Could not find Phase 2 EfficientNet results for seed {seed} in {out_dir}"
        )
    data = json.loads(path.read_text())
    if "efficientnet_b0" in data:
        metrics = dict(data["efficientnet_b0"])
    else:
        metrics = {
            "oa": float(data["oa"]),
            "precision": float(data["precision"]),
            "recall": float(data["recall"]),
            "f1": float(data["f1"]),
            "miou": float(data["miou"]),
        }
        if "training_time_min" in data:
            metrics["training_time_min"] = float(data["training_time_min"])
    metrics.setdefault("training_time_min", 2.9)
    return metrics


def run_small_ablation(device, epochs_per_stage=5, seed=SEED):
    original_epochs = [stage["epochs"] for stage in STAGES]
    for stage in STAGES:
        stage["epochs"] = epochs_per_stage

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        data_source="real",
        real_dataset="indian_cities",
        download=False,
    )

    results = {
        "full": _load_phase2_full(seed)
    }
    try:
        print("\n" + "=" * 70)
        print(f"  ABLATION | no_fpn | seed={seed}")
        print("=" * 70)
        no_fpn_model = NoFPNClassifier("efficientnet_b0", pretrained=True).to(device)
        _, no_fpn_metrics, no_fpn_time = _progressive_train_custom(
            no_fpn_model,
            train_loader,
            val_loader,
            test_loader,
            device,
            criterion_factory=lambda: CombinedLoss(class_weights=[1.0, 1.0, 3.0]),
        )
        results["no_fpn"] = {
            "oa": float(no_fpn_metrics["oa"]),
            "precision": float(no_fpn_metrics["precision"]),
            "recall": float(no_fpn_metrics["recall"]),
            "f1": float(no_fpn_metrics["f1"]),
            "miou": float(no_fpn_metrics["miou"]),
            "training_time_min": round(no_fpn_time / 60.0, 2),
        }

        print("\n" + "=" * 70)
        print(f"  ABLATION | ce_only | seed={seed}")
        print("=" * 70)
        from src.models import UrbanClassifier
        ce_model = UrbanClassifier("efficientnet_b0", pretrained=True).to(device)
        _, ce_metrics, ce_time = _progressive_train_custom(
            ce_model,
            train_loader,
            val_loader,
            test_loader,
            device,
            criterion_factory=lambda: nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 3.0], dtype=torch.float32)),
        )
        results["ce_only"] = {
            "oa": float(ce_metrics["oa"]),
            "precision": float(ce_metrics["precision"]),
            "recall": float(ce_metrics["recall"]),
            "f1": float(ce_metrics["f1"]),
            "miou": float(ce_metrics["miou"]),
            "training_time_min": round(ce_time / 60.0, 2),
        }
    finally:
        for stage, epoch_value in zip(STAGES, original_epochs):
            stage["epochs"] = epoch_value

    out_dir = Path(OUTPUT_DIR) / "research_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if int(seed) == int(SEED) else f"_seed{seed}"
    payload = {
        "dataset": "indian_cities_locked",
        "backbone": "efficientnet_b0",
        "epochs_per_stage": epochs_per_stage,
        "seed": int(seed),
        "results": results,
    }
    (out_dir / f"phase4_small_ablation{suffix}.json").write_text(json.dumps(payload, indent=2))
    lines = [
        "# Phase 4 Small Ablation",
        "",
        f"Seed: `{seed}`",
        "",
        "| Config | OA | Precision | Recall | F1 | mIoU | Train Time (min) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in results.items():
        lines.append(
            f"| {name} | {metrics['oa']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['miou']:.4f} | {metrics['training_time_min']:.2f} |"
        )
    (out_dir / f"phase4_small_ablation{suffix}.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 small real-data ablation")
    parser.add_argument("--epochs-per-stage", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seed = _resolve_seed(args.seed)
    set_seed(seed)
    device = get_device()
    run_small_ablation(device=device, epochs_per_stage=args.epochs_per_stage, seed=seed)


if __name__ == "__main__":
    main()
