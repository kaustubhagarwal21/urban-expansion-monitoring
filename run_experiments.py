"""
Research-grade experiment runner for conference submission.

Produces:
  1. Multi-seed backbone comparison (VGG16, ResNet50, EfficientNet-B0) on EuroSAT
  2. Classical baselines (SVM, RF) on the same data
  3. Ablation: with/without FPN, with/without progressive fine-tuning,
     pretrained vs random init
  4. Per-class metrics and confusion matrices
  5. LaTeX-ready tables and JSON results with mean +/- std

Usage:
    python run_experiments.py                         # Full suite (5 seeds, EuroSAT)
    python run_experiments.py --seeds 42 123 456      # Custom seeds
    python run_experiments.py --quick                  # 2 seeds, 3 epochs (smoke test)
    python run_experiments.py --backbone efficientnet_b0  # Single backbone
"""

import argparse
import copy
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import (
    BACKBONES, BATCH_SIZE, CLASS_NAMES, DEFAULT_BACKBONE,
    EARLY_STOP_PATIENCE, FPN_CHANNELS, NUM_CHANNELS, NUM_CLASSES,
    OUTPUT_DIR, SEED, STAGES, WEIGHT_DECAY,
)
from src.baselines import extract_features, run_baselines
from src.dataset import get_dataloaders, MultispectralAugment
from src.losses import CombinedLoss
from src.metrics import evaluate
from src.models import (
    UrbanClassifier, BACKBONE_BUILDERS, ClassificationHead,
    freeze_backbone, UNFREEZE_FNS,
)
from src.train import train_one_epoch, validate, progressive_train

RESULTS_DIR = os.path.join(OUTPUT_DIR, "research_results")
PAPER_METRICS = ["oa", "precision", "recall", "f1", "miou"]


def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("  Using CPU")
    return "cpu"


def _extract(metrics):
    return {k: float(metrics[k]) for k in PAPER_METRICS if k in metrics}


def aggregate(results_list):
    """Aggregate list of metric dicts -> {metric: {mean, std, values}}."""
    if not results_list:
        return {}
    agg = {}
    for key in PAPER_METRICS:
        vals = [r[key] for r in results_list if key in r]
        if vals:
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": vals,
            }
    return agg


# =====================================================================
#  No-FPN Classifier for ablation
# =====================================================================

class NoFPNClassifier(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        builder = BACKBONE_BUILDERS[backbone_name]
        self.blocks, ch_list = builder(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = ClassificationHead(ch_list[-1], NUM_CLASSES)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        pooled = self.gap(out).flatten(1)
        return self.head(pooled)


# Unfreeze helpers that work for both UrbanClassifier and NoFPNClassifier
def _safe_unfreeze(model, key):
    if key == "head":
        for block in model.blocks:
            for p in block.parameters():
                p.requires_grad = False
    elif key == "last_blocks":
        for p in model.blocks[-1].parameters():
            p.requires_grad = True
    elif key == "all":
        for p in model.parameters():
            p.requires_grad = True


# =====================================================================
#  Core training function (returns test metrics)
# =====================================================================

def train_and_evaluate(
    model, loaders, device, stages, use_mixup=True, tag=""
):
    """Train a model with progressive fine-tuning and return test metrics."""
    train_loader, val_loader, test_loader = loaders
    criterion = CombinedLoss().to(device)
    model = model.to(device)

    best_val_loss = float("inf")
    best_state = None
    total_time = 0.0

    for stage_cfg in stages:
        lr = stage_cfg["lr"]
        epochs = stage_cfg["epochs"]
        unfreeze_key = stage_cfg["unfreeze"]

        _safe_unfreeze(model, unfreeze_key)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        patience = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup)
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()
            total_time += time.time() - t0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= EARLY_STOP_PATIENCE:
                    break

    model.load_state_dict(best_state)
    _, test_metrics = validate(model, test_loader, criterion, device)
    test_metrics["train_time_sec"] = total_time

    if tag:
        print(f"    {tag}: OA={test_metrics['oa']:.4f} F1={test_metrics['f1']:.4f} "
              f"mIoU={test_metrics['miou']:.4f} ({total_time:.0f}s)")
    return test_metrics


# =====================================================================
#  Experiment 1: Multi-seed backbone comparison
# =====================================================================

def exp_backbone_comparison(seeds, device, loaders_fn, stages, backbones=None):
    """Compare all backbones across multiple seeds on real data."""
    if backbones is None:
        backbones = BACKBONES
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Backbone Comparison (multi-seed)")
    print("=" * 70)

    results = {b: [] for b in backbones}

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        set_all_seeds(seed)
        loaders = loaders_fn()

        for backbone in backbones:
            model = UrbanClassifier(backbone, pretrained=True)
            metrics = train_and_evaluate(
                model, loaders, device, stages,
                tag=f"{backbone} (seed={seed})"
            )
            results[backbone].append(_extract(metrics))

    aggregated = {b: aggregate(results[b]) for b in BACKBONES}
    return results, aggregated


# =====================================================================
#  Experiment 2: Classical baselines
# =====================================================================

def exp_baselines(seeds, loaders_fn):
    """Run SVM and RF baselines across seeds."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Classical Baselines (SVM, Random Forest)")
    print("=" * 70)

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    svm_results = []
    rf_results = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        set_all_seeds(seed)
        loaders = loaders_fn()
        train_ds = loaders[0].dataset
        test_ds = loaders[2].dataset

        X_train, y_train = extract_features(train_ds, max_samples=5000)
        X_test, y_test = extract_features(test_ds, max_samples=2000)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # SVM
        svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)
        svm.fit(X_train, y_train)
        svm_preds = svm.predict(X_test)
        svm_m = evaluate(y_test.tolist(), svm_preds.tolist(), CLASS_NAMES)
        svm_results.append(_extract(svm_m))
        print(f"    SVM: OA={svm_m['oa']:.4f} F1={svm_m['f1']:.4f} mIoU={svm_m['miou']:.4f}")

        # RF
        rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                    random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_m = evaluate(y_test.tolist(), rf_preds.tolist(), CLASS_NAMES)
        rf_results.append(_extract(rf_m))
        print(f"    RF:  OA={rf_m['oa']:.4f} F1={rf_m['f1']:.4f} mIoU={rf_m['miou']:.4f}")

    return {
        "SVM": {"raw": svm_results, "agg": aggregate(svm_results)},
        "RandomForest": {"raw": rf_results, "agg": aggregate(rf_results)},
    }


# =====================================================================
#  Experiment 3: Ablation study
# =====================================================================

def exp_ablation(seeds, device, loaders_fn, stages):
    """Ablation: FPN, progressive fine-tuning, pretrained init."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Ablation Study")
    print("=" * 70)

    backbone = DEFAULT_BACKBONE
    ablations = {}

    # Full model (reference)
    full_results = []
    # No FPN
    no_fpn_results = []
    # No progressive (train all from start)
    no_prog_results = []
    # Random init (no ImageNet)
    random_init_results = []
    # No Mixup
    no_mixup_results = []

    single_stage = [{"name": "full", "lr": 1e-4, "epochs": sum(s["epochs"] for s in stages), "unfreeze": "all"}]

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        set_all_seeds(seed)
        loaders = loaders_fn()

        # Full model
        model = UrbanClassifier(backbone, pretrained=True)
        m = train_and_evaluate(model, loaders, device, stages, tag="Full model")
        full_results.append(_extract(m))

        # No FPN
        set_all_seeds(seed)
        model = NoFPNClassifier(backbone, pretrained=True).to(device)
        # NoFPNClassifier doesn't have the same unfreeze interface, so train all
        m = train_and_evaluate(model, loaders, device, single_stage, tag="No FPN")
        no_fpn_results.append(_extract(m))

        # No progressive fine-tuning
        set_all_seeds(seed)
        model = UrbanClassifier(backbone, pretrained=True)
        m = train_and_evaluate(model, loaders, device, single_stage, tag="No progressive FT")
        no_prog_results.append(_extract(m))

        # Random init
        set_all_seeds(seed)
        model = UrbanClassifier(backbone, pretrained=False)
        m = train_and_evaluate(model, loaders, device, single_stage, tag="Random init")
        random_init_results.append(_extract(m))

        # No Mixup
        set_all_seeds(seed)
        model = UrbanClassifier(backbone, pretrained=True)
        m = train_and_evaluate(model, loaders, device, stages, use_mixup=False, tag="No Mixup")
        no_mixup_results.append(_extract(m))

    ablations = {
        "Full model (proposed)": {"raw": full_results, "agg": aggregate(full_results)},
        "w/o FPN": {"raw": no_fpn_results, "agg": aggregate(no_fpn_results)},
        "w/o Progressive FT": {"raw": no_prog_results, "agg": aggregate(no_prog_results)},
        "w/o ImageNet pretrain": {"raw": random_init_results, "agg": aggregate(random_init_results)},
        "w/o Mixup": {"raw": no_mixup_results, "agg": aggregate(no_mixup_results)},
    }
    return ablations


# =====================================================================
#  Output: LaTeX table generation
# =====================================================================

def generate_latex_table(backbone_agg, baseline_agg, ablation_agg):
    """Generate LaTeX tables for the paper."""
    lines = []

    # Table 1: Main results
    lines.append("% Table 1: Model Comparison")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Classification performance on EuroSAT (mean $\pm$ std over 5 seeds).}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & OA (\%) & Precision & Recall & F1 & mIoU \\")
    lines.append(r"\midrule")

    # Baselines
    for name, data in baseline_agg.items():
        agg = data["agg"]
        row = f"{name}"
        for m in PAPER_METRICS:
            if m == "oa":
                row += f" & {agg[m]['mean']*100:.1f} $\\pm$ {agg[m]['std']*100:.1f}"
            else:
                row += f" & {agg[m]['mean']:.3f} $\\pm$ {agg[m]['std']:.3f}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\midrule")

    # DL backbones
    best_oa = max(backbone_agg[b]["oa"]["mean"] for b in backbone_agg)
    for backbone in backbone_agg:
        agg = backbone_agg[backbone]
        name = {"vgg16": "VGG16", "resnet50": "ResNet50",
                "efficientnet_b0": "EfficientNet-B0"}[backbone]
        is_best = abs(agg["oa"]["mean"] - best_oa) < 1e-6
        if is_best:
            name = r"\textbf{" + name + "}"
        row = f"{name}"
        for m in PAPER_METRICS:
            val = f"{agg[m]['mean']*100:.1f}" if m == "oa" else f"{agg[m]['mean']:.3f}"
            std = f"{agg[m]['std']*100:.1f}" if m == "oa" else f"{agg[m]['std']:.3f}"
            if is_best:
                row += f" & \\textbf{{{val}}} $\\pm$ {std}"
            else:
                row += f" & {val} $\\pm$ {std}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # Table 2: Ablation
    lines.append("% Table 2: Ablation Study")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation study on EfficientNet-B0 (mean $\pm$ std over 5 seeds).}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Configuration & OA (\%) & F1 & mIoU \\")
    lines.append(r"\midrule")

    for name, data in ablation_agg.items():
        agg = data["agg"]
        oa = f"{agg['oa']['mean']*100:.1f} $\\pm$ {agg['oa']['std']*100:.1f}"
        f1 = f"{agg['f1']['mean']:.3f} $\\pm$ {agg['f1']['std']:.3f}"
        miou = f"{agg['miou']['mean']:.3f} $\\pm$ {agg['miou']['std']:.3f}"
        lines.append(f"{name} & {oa} & {f1} & {miou} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Research experiment runner")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 2024])
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (2 seeds, 3 epochs)")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=BACKBONES, help="Run single backbone only")
    parser.add_argument("--data-source", type=str, default="real",
                        choices=["synthetic", "real"])
    parser.add_argument("--real-dataset", type=str, default="eurosat")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42, 123]
        epochs_per_stage = 3
    else:
        epochs_per_stage = None  # use defaults from config

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()

    # Build stages (possibly with reduced epochs)
    stages = copy.deepcopy(STAGES)
    if epochs_per_stage is not None:
        for s in stages:
            s["epochs"] = epochs_per_stage

    backbones_to_run = BACKBONES
    if args.backbone:
        backbones_to_run = [args.backbone]

    def loaders_fn():
        return get_dataloaders(
            batch_size=BATCH_SIZE,
            data_source=args.data_source,
            real_dataset=args.real_dataset,
            download=args.download,
        )

    print("=" * 70)
    print("  RESEARCH EXPERIMENT SUITE")
    print(f"  Seeds: {args.seeds}")
    print(f"  Data: {args.data_source} ({args.real_dataset})")
    print(f"  Device: {device}")
    print("=" * 70)

    all_outputs = {}
    t_start = time.time()

    # Experiment 1: Backbone comparison
    raw_backbone, agg_backbone = exp_backbone_comparison(
        args.seeds, device, loaders_fn, stages, backbones_to_run
    )
    all_outputs["backbone_comparison"] = {
        b: {"raw": raw_backbone[b], "agg": agg_backbone[b]} for b in backbones_to_run
    }

    # Experiment 2: Baselines
    if not args.skip_baselines:
        baseline_results = exp_baselines(args.seeds, loaders_fn)
        all_outputs["baselines"] = baseline_results
    else:
        baseline_results = {}

    # Experiment 3: Ablation
    if not args.skip_ablation:
        ablation_results = exp_ablation(args.seeds, device, loaders_fn, stages)
        all_outputs["ablation"] = ablation_results
    else:
        ablation_results = {}

    total_time = time.time() - t_start

    # ── Save results ──────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_outputs, f, indent=2, default=str)
    print(f"\n  Results saved to {json_path}")

    # CSV summary
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "mean", "std"])
        for b in backbones_to_run:
            for m in PAPER_METRICS:
                a = agg_backbone[b].get(m, {})
                writer.writerow([b, m, a.get("mean", ""), a.get("std", "")])
        for name, data in baseline_results.items():
            for m in PAPER_METRICS:
                a = data["agg"].get(m, {})
                writer.writerow([name, m, a.get("mean", ""), a.get("std", "")])

    # LaTeX tables
    if baseline_results and ablation_results:
        latex = generate_latex_table(agg_backbone, baseline_results, ablation_results)
        latex_path = os.path.join(RESULTS_DIR, "tables.tex")
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"  LaTeX tables saved to {latex_path}")

    # ── Print summary ─────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY (mean +/- std)")
    print(f"{'='*70}")
    header = f"{'Model':<25} {'OA (%)':>12} {'F1':>12} {'mIoU':>12}"
    print(header)
    print("-" * 65)

    for name, data in baseline_results.items():
        agg = data["agg"]
        oa = f"{agg['oa']['mean']*100:.1f}+/-{agg['oa']['std']*100:.1f}"
        f1 = f"{agg['f1']['mean']:.3f}+/-{agg['f1']['std']:.3f}"
        miou = f"{agg['miou']['mean']:.3f}+/-{agg['miou']['std']:.3f}"
        print(f"{name:<25} {oa:>12} {f1:>12} {miou:>12}")

    print("-" * 65)
    for b in backbones_to_run:
        agg = agg_backbone[b]
        name = {"vgg16": "VGG16", "resnet50": "ResNet50",
                "efficientnet_b0": "EfficientNet-B0"}.get(b, b)
        oa = f"{agg['oa']['mean']*100:.1f}+/-{agg['oa']['std']*100:.1f}"
        f1 = f"{agg['f1']['mean']:.3f}+/-{agg['f1']['std']:.3f}"
        miou = f"{agg['miou']['mean']:.3f}+/-{agg['miou']['std']:.3f}"
        print(f"{name:<25} {oa:>12} {f1:>12} {miou:>12}")

    if ablation_results:
        print("-" * 65)
        print("  Ablation:")
        for name, data in ablation_results.items():
            agg = data["agg"]
            oa = f"{agg['oa']['mean']*100:.1f}+/-{agg['oa']['std']*100:.1f}"
            f1 = f"{agg['f1']['mean']:.3f}+/-{agg['f1']['std']:.3f}"
            miou = f"{agg['miou']['mean']:.3f}+/-{agg['miou']['std']:.3f}"
            print(f"  {name:<23} {oa:>12} {f1:>12} {miou:>12}")

    print(f"\n  Total time: {total_time / 60:.1f} min")
    print(f"  Outputs: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
