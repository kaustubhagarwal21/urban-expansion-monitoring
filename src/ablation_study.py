"""
Ablation Study Framework for Urban Expansion Monitoring.

Systematic evaluation of architectural and training choices:
  1. Backbone comparison (VGG16, ResNet50, EfficientNet-B0)
  2. With / without Feature Pyramid Network (FPN)
  3. Progressive fine-tuning vs full training from start
  4. Loss function ablation (CE, Focal, Dice, combinations)
  5. Mixup augmentation ablation
  6. ImageNet pre-training vs random initialization
  7. Statistical significance (paired t-test)

Results are saved as CSV tables, bar-chart figures, and a JSON summary.
"""

import os, sys, json, copy, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    BACKBONES, DEFAULT_BACKBONE, STAGES, BATCH_SIZE, WEIGHT_DECAY,
    NUM_CHANNELS, NUM_CLASSES, CLASS_NAMES, FPN_CHANNELS,
    FIGURE_DIR, MODEL_DIR, OUTPUT_DIR, SEED,
    LOSS_WEIGHTS, FOCAL_GAMMA, AUGMENT, EARLY_STOP_PATIENCE,
)
from src.models import (
    UrbanClassifier, FPN, ClassificationHead,
    freeze_backbone, UNFREEZE_FNS, BACKBONE_BUILDERS,
)
from src.losses import CombinedLoss, FocalLoss, DiceLoss
from src.dataset import UrbanExpansionDataset, MultispectralAugment, mixup_data
from src.metrics import evaluate
from src.train import train_one_epoch, validate

warnings.filterwarnings("ignore")

# =====================================================================
#  Reduced dataset sizes for ablation speed
# =====================================================================
ABLATION_N_TRAIN = 2000
ABLATION_N_VAL = 500
ABLATION_N_TEST = 500
ABLATION_EPOCHS_PER_STAGE = 5

# Metrics tracked across all experiments
METRIC_KEYS = ["oa", "precision", "recall", "f1", "miou"]


# =====================================================================
#  Lightweight data-loader builder (reduced sizes)
# =====================================================================

def _get_ablation_loaders(batch_size=BATCH_SIZE, seed=SEED, use_augment=True):
    """Return (train, val, test) loaders with reduced sample counts."""
    aug = MultispectralAugment() if use_augment else None
    train_ds = UrbanExpansionDataset(ABLATION_N_TRAIN, transform=aug, seed=seed)
    val_ds = UrbanExpansionDataset(ABLATION_N_VAL, transform=None, seed=seed + 1)
    test_ds = UrbanExpansionDataset(ABLATION_N_TEST, transform=None, seed=seed + 2)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


# =====================================================================
#  No-FPN Classifier (backbone + GAP + head, no pyramid)
# =====================================================================

class NoFPNClassifier(nn.Module):
    """
    Baseline classifier without Feature Pyramid Network.
    Passes input through all backbone blocks sequentially, then applies
    global average pooling on the last feature map followed by the
    classification head.
    """

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        builder = BACKBONE_BUILDERS[backbone_name]
        self.blocks, ch_list = builder(pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Head receives features only from the last backbone block
        self.head = ClassificationHead(ch_list[-1], NUM_CLASSES)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = self.gap(out).flatten(1)
        return self.head(out)


# =====================================================================
#  Core: run a single experiment given a config dict
# =====================================================================

def run_single_experiment(config_dict, device, seed):
    """
    Train a model described by *config_dict* and return test metrics.

    Expected keys in config_dict:
        backbone       : str   - backbone name
        pretrained     : bool  - ImageNet weights
        use_fpn        : bool  - True -> UrbanClassifier, False -> NoFPNClassifier
        progressive    : bool  - progressive fine-tuning stages
        criterion      : nn.Module  - loss function (already instantiated)
        use_mixup      : bool  - mixup augmentation during training
        use_augment    : bool  - spatial augmentation on dataset
        epochs_per_stage : int - epochs per progressive stage
        label          : str   - human-readable experiment label
    """
    # ---- reproducibility ----
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = config_dict["backbone"]
    pretrained = config_dict.get("pretrained", True)
    use_fpn = config_dict.get("use_fpn", True)
    progressive = config_dict.get("progressive", True)
    criterion = config_dict["criterion"].to(device)
    use_mixup = config_dict.get("use_mixup", True)
    use_augment = config_dict.get("use_augment", True)
    epochs_per_stage = config_dict.get("epochs_per_stage", ABLATION_EPOCHS_PER_STAGE)
    label = config_dict.get("label", "experiment")

    # ---- model ----
    if use_fpn:
        model = UrbanClassifier(backbone, pretrained=pretrained).to(device)
    else:
        model = NoFPNClassifier(backbone, pretrained=pretrained).to(device)

    # ---- data ----
    train_loader, val_loader, test_loader = _get_ablation_loaders(
        batch_size=BATCH_SIZE, seed=seed, use_augment=use_augment,
    )

    # ---- training stages ----
    best_val_loss = float("inf")
    best_state = None

    if progressive and use_fpn:
        # Progressive unfreezing over the three configured stages
        stages = [
            {"name": "head_only",    "lr": 1e-3, "epochs": epochs_per_stage, "unfreeze": "head"},
            {"name": "last_blocks",  "lr": 1e-4, "epochs": epochs_per_stage, "unfreeze": "last_blocks"},
            {"name": "full",         "lr": 1e-5, "epochs": epochs_per_stage, "unfreeze": "all"},
        ]
    else:
        # Non-progressive: unfreeze everything from the start
        stages = [
            {"name": "full", "lr": 1e-4, "epochs": epochs_per_stage * 3, "unfreeze": "all"},
        ]

    for stage_cfg in stages:
        # Apply unfreezing (for NoFPNClassifier, unfreeze_all works on any nn.Module)
        if use_fpn:
            UNFREEZE_FNS[stage_cfg["unfreeze"]](model)
        else:
            # For NoFPNClassifier always unfreeze everything that the stage requests
            if stage_cfg["unfreeze"] == "head":
                # Freeze backbone blocks
                for block in model.blocks:
                    for p in block.parameters():
                        p.requires_grad = False
            elif stage_cfg["unfreeze"] == "last_blocks":
                for p in model.blocks[-1].parameters():
                    p.requires_grad = True
            else:
                for p in model.parameters():
                    p.requires_grad = True

        trainable = [p for p in model.parameters() if p.requires_grad]
        if len(trainable) == 0:
            # Safety: ensure at least the head is trainable
            for p in model.head.parameters():
                p.requires_grad = True
            trainable = [p for p in model.parameters() if p.requires_grad]

        optimizer = AdamW(trainable, lr=stage_cfg["lr"], weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=stage_cfg["epochs"])
        patience_counter = 0

        for epoch in range(1, stage_cfg["epochs"] + 1):
            train_one_epoch(model, train_loader, criterion, optimizer, device,
                            use_mixup=use_mixup)
            val_loss, _ = validate(model, val_loader, criterion, device)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    break

    # ---- evaluate on test set ----
    model.load_state_dict(best_state)
    _, test_metrics = validate(model, test_loader, criterion, device)

    return {k: test_metrics[k] for k in METRIC_KEYS}


# =====================================================================
#  Helper: run one experiment over multiple seeds and aggregate
# =====================================================================

def _run_over_seeds(config_dict, device, num_seeds=3):
    """Run experiment with different seeds and return per-seed metrics."""
    all_metrics = {k: [] for k in METRIC_KEYS}
    for i in range(num_seeds):
        seed_i = SEED + i * 100
        metrics = run_single_experiment(config_dict, device, seed_i)
        for k in METRIC_KEYS:
            all_metrics[k].append(metrics[k])
    return all_metrics


def _summarize(all_metrics):
    """Compute mean +/- std for each metric."""
    summary = {}
    for k in METRIC_KEYS:
        vals = np.array(all_metrics[k])
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))
    return summary


# =====================================================================
#  Individual ablation experiments
# =====================================================================

def ablation_backbone(device, num_seeds=3):
    """Experiment 1: Backbone comparison."""
    print("\n" + "=" * 70)
    print("  ABLATION 1: Backbone Comparison")
    print("=" * 70)

    results = {}
    for backbone in BACKBONES:
        print(f"\n  >> {backbone} ({num_seeds} seeds) ...")
        cfg = dict(
            backbone=backbone, pretrained=True, use_fpn=True,
            progressive=True, criterion=CombinedLoss(),
            use_mixup=True, use_augment=True,
            epochs_per_stage=ABLATION_EPOCHS_PER_STAGE,
            label=f"backbone_{backbone}",
        )
        raw = _run_over_seeds(cfg, device, num_seeds)
        results[backbone] = {"raw": raw, **_summarize(raw)}
        print(f"     OA = {results[backbone]['oa_mean']:.4f} +/- {results[backbone]['oa_std']:.4f}  |  "
              f"F1 = {results[backbone]['f1_mean']:.4f} +/- {results[backbone]['f1_std']:.4f}")
    return results


def ablation_fpn(device, num_seeds=3):
    """Experiment 2: With vs without FPN."""
    print("\n" + "=" * 70)
    print("  ABLATION 2: FPN vs No-FPN")
    print("=" * 70)

    results = {}
    for use_fpn, tag in [(True, "with_FPN"), (False, "without_FPN")]:
        print(f"\n  >> {tag} ({num_seeds} seeds) ...")
        cfg = dict(
            backbone=DEFAULT_BACKBONE, pretrained=True, use_fpn=use_fpn,
            progressive=use_fpn,  # progressive only meaningful with FPN
            criterion=CombinedLoss(),
            use_mixup=True, use_augment=True,
            epochs_per_stage=ABLATION_EPOCHS_PER_STAGE,
            label=tag,
        )
        raw = _run_over_seeds(cfg, device, num_seeds)
        results[tag] = {"raw": raw, **_summarize(raw)}
        print(f"     OA = {results[tag]['oa_mean']:.4f} +/- {results[tag]['oa_std']:.4f}  |  "
              f"F1 = {results[tag]['f1_mean']:.4f} +/- {results[tag]['f1_std']:.4f}")
    return results


def ablation_progressive(device, num_seeds=3):
    """Experiment 3: Progressive fine-tuning vs train-all-from-start."""
    print("\n" + "=" * 70)
    print("  ABLATION 3: Progressive Fine-Tuning vs Full Training")
    print("=" * 70)

    results = {}
    for prog, tag in [(True, "progressive"), (False, "non_progressive")]:
        print(f"\n  >> {tag} ({num_seeds} seeds) ...")
        cfg = dict(
            backbone=DEFAULT_BACKBONE, pretrained=True, use_fpn=True,
            progressive=prog, criterion=CombinedLoss(),
            use_mixup=True, use_augment=True,
            epochs_per_stage=ABLATION_EPOCHS_PER_STAGE,
            label=tag,
        )
        raw = _run_over_seeds(cfg, device, num_seeds)
        results[tag] = {"raw": raw, **_summarize(raw)}
        print(f"     OA = {results[tag]['oa_mean']:.4f} +/- {results[tag]['oa_std']:.4f}  |  "
              f"F1 = {results[tag]['f1_mean']:.4f} +/- {results[tag]['f1_std']:.4f}")
    return results


def ablation_loss(device, num_seeds=3):
    """Experiment 4: Loss function ablation."""
    print("\n" + "=" * 70)
    print("  ABLATION 4: Loss Function Ablation")
    print("=" * 70)

    loss_configs = {
        "CE_only":          nn.CrossEntropyLoss(),
        "CE+Focal":         CombinedLoss(w_ce=0.7, w_focal=0.3, w_dice=0.0),
        "CE+Focal+Dice":    CombinedLoss(),  # default weights from config
        "Focal_only":       FocalLoss(gamma=FOCAL_GAMMA),
        "Dice_only":        DiceLoss(),
    }

    results = {}
    for tag, criterion in loss_configs.items():
        print(f"\n  >> {tag} ({num_seeds} seeds) ...")
        cfg = dict(
            backbone=DEFAULT_BACKBONE, pretrained=True, use_fpn=True,
            progressive=True, criterion=criterion,
            use_mixup=True, use_augment=True,
            epochs_per_stage=ABLATION_EPOCHS_PER_STAGE,
            label=f"loss_{tag}",
        )
        raw = _run_over_seeds(cfg, device, num_seeds)
        results[tag] = {"raw": raw, **_summarize(raw)}
        print(f"     OA = {results[tag]['oa_mean']:.4f} +/- {results[tag]['oa_std']:.4f}  |  "
              f"F1 = {results[tag]['f1_mean']:.4f} +/- {results[tag]['f1_std']:.4f}")
    return results


def ablation_mixup(device, num_seeds=3):
    """Experiment 5: With vs without mixup augmentation."""
    print("\n" + "=" * 70)
    print("  ABLATION 5: Mixup Augmentation")
    print("=" * 70)

    results = {}
    for use_mix, tag in [(True, "with_mixup"), (False, "without_mixup")]:
        print(f"\n  >> {tag} ({num_seeds} seeds) ...")
        cfg = dict(
            backbone=DEFAULT_BACKBONE, pretrained=True, use_fpn=True,
            progressive=True, criterion=CombinedLoss(),
            use_mixup=use_mix, use_augment=True,
            epochs_per_stage=ABLATION_EPOCHS_PER_STAGE,
            label=tag,
        )
        raw = _run_over_seeds(cfg, device, num_seeds)
        results[tag] = {"raw": raw, **_summarize(raw)}
        print(f"     OA = {results[tag]['oa_mean']:.4f} +/- {results[tag]['oa_std']:.4f}  |  "
              f"F1 = {results[tag]['f1_mean']:.4f} +/- {results[tag]['f1_std']:.4f}")
    return results


def ablation_pretrain(device, num_seeds=3):
    """Experiment 6: ImageNet pre-training vs random init."""
    print("\n" + "=" * 70)
    print("  ABLATION 6: Pre-training (ImageNet vs Random Init)")
    print("=" * 70)

    results = {}
    for pt, tag in [(True, "pretrained"), (False, "random_init")]:
        print(f"\n  >> {tag} ({num_seeds} seeds) ...")
        cfg = dict(
            backbone=DEFAULT_BACKBONE, pretrained=pt, use_fpn=True,
            progressive=True, criterion=CombinedLoss(),
            use_mixup=True, use_augment=True,
            epochs_per_stage=ABLATION_EPOCHS_PER_STAGE,
            label=tag,
        )
        raw = _run_over_seeds(cfg, device, num_seeds)
        results[tag] = {"raw": raw, **_summarize(raw)}
        print(f"     OA = {results[tag]['oa_mean']:.4f} +/- {results[tag]['oa_std']:.4f}  |  "
              f"F1 = {results[tag]['f1_mean']:.4f} +/- {results[tag]['f1_std']:.4f}")
    return results


# =====================================================================
#  Statistical significance testing
# =====================================================================

def statistical_tests(all_results):
    """
    Experiment 7: Paired t-tests comparing the best model configuration
    (default full pipeline) against each alternative on the F1 metric.

    The 'best' model is taken from the backbone ablation as the
    DEFAULT_BACKBONE entry, which uses all default settings.
    """
    print("\n" + "=" * 70)
    print("  STATISTICAL SIGNIFICANCE (Paired t-test on F1)")
    print("=" * 70)

    # Reference: default backbone with all bells and whistles
    ref_key = DEFAULT_BACKBONE
    ref_f1 = all_results["backbone"][ref_key]["raw"]["f1"]

    comparisons = []

    # Collect all alternative F1 arrays with labels
    # -- other backbones
    for bb in BACKBONES:
        if bb != ref_key:
            comparisons.append((f"backbone: {bb}",
                                all_results["backbone"][bb]["raw"]["f1"]))

    # -- FPN ablation
    comparisons.append(("without FPN",
                        all_results["fpn"]["without_FPN"]["raw"]["f1"]))

    # -- progressive ablation
    comparisons.append(("non-progressive",
                        all_results["progressive"]["non_progressive"]["raw"]["f1"]))

    # -- loss variants (all except default CE+Focal+Dice)
    for tag in all_results["loss"]:
        if tag != "CE+Focal+Dice":
            comparisons.append((f"loss: {tag}",
                                all_results["loss"][tag]["raw"]["f1"]))

    # -- mixup
    comparisons.append(("without mixup",
                        all_results["mixup"]["without_mixup"]["raw"]["f1"]))

    # -- pretraining
    comparisons.append(("random init",
                        all_results["pretrain"]["random_init"]["raw"]["f1"]))

    test_results = []
    for alt_label, alt_f1 in comparisons:
        # Ensure arrays have the same length for paired test
        n = min(len(ref_f1), len(alt_f1))
        if n < 2:
            t_stat, p_val = float("nan"), float("nan")
        else:
            t_stat, p_val = stats.ttest_rel(ref_f1[:n], alt_f1[:n])
        sig = "Yes" if p_val < 0.05 else "No"
        entry = {
            "comparison": f"best ({ref_key}) vs {alt_label}",
            "ref_f1_mean": float(np.mean(ref_f1)),
            "alt_f1_mean": float(np.mean(alt_f1)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_p05": sig,
        }
        test_results.append(entry)
        print(f"  {entry['comparison']:45s}  t={t_stat:+.3f}  p={p_val:.4f}  sig={sig}")

    return test_results


# =====================================================================
#  Output: tables (CSV + printed), charts, JSON
# =====================================================================

def _build_summary_df(all_results):
    """Flatten all ablation results into a single DataFrame."""
    rows = []
    for ablation_name, experiments in all_results.items():
        if ablation_name == "significance":
            continue
        for variant, data in experiments.items():
            row = {"ablation": ablation_name, "variant": variant}
            for k in METRIC_KEYS:
                row[f"{k}_mean"] = data.get(f"{k}_mean", float("nan"))
                row[f"{k}_std"] = data.get(f"{k}_std", float("nan"))
            rows.append(row)
    return pd.DataFrame(rows)


def save_tables(all_results):
    """Print and save CSV summary tables."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = _build_summary_df(all_results)

    # ---- pretty-print ----
    print("\n" + "=" * 70)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 70)

    for ablation_name in ["backbone", "fpn", "progressive", "loss", "mixup", "pretrain"]:
        sub = df[df["ablation"] == ablation_name].copy()
        if sub.empty:
            continue
        print(f"\n--- {ablation_name.upper()} ---")
        display_cols = ["variant"]
        for k in METRIC_KEYS:
            sub[k] = sub.apply(
                lambda r, _k=k: f"{r[f'{_k}_mean']:.4f} +/- {r[f'{_k}_std']:.4f}", axis=1
            )
            display_cols.append(k)
        print(sub[display_cols].to_string(index=False))

    # ---- CSV ----
    csv_path = os.path.join(OUTPUT_DIR, "ablation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV saved to {csv_path}")

    # Significance table
    if "significance" in all_results:
        sig_df = pd.DataFrame(all_results["significance"])
        sig_csv = os.path.join(OUTPUT_DIR, "ablation_significance.csv")
        sig_df.to_csv(sig_csv, index=False)
        print(f"  Significance CSV saved to {sig_csv}")

    return df


def save_charts(all_results):
    """Generate publication-quality ablation bar charts."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    chart_specs = [
        ("backbone",    "Backbone Comparison",         "ablation_backbone.png"),
        ("fpn",         "FPN vs No-FPN",               "ablation_fpn.png"),
        ("progressive", "Progressive Fine-Tuning",     "ablation_progressive.png"),
        ("loss",        "Loss Function Ablation",       "ablation_loss.png"),
        ("mixup",       "Mixup Augmentation",           "ablation_mixup.png"),
        ("pretrain",    "Pre-training Ablation",        "ablation_pretrain.png"),
    ]

    for ablation_key, title, filename in chart_specs:
        data = all_results.get(ablation_key, {})
        if not data:
            continue

        variants = list(data.keys())
        metrics_to_plot = ["oa", "f1", "miou"]
        x = np.arange(len(variants))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(6, len(variants) * 1.5), 5))
        for i, metric in enumerate(metrics_to_plot):
            means = [data[v].get(f"{metric}_mean", 0) for v in variants]
            stds = [data[v].get(f"{metric}_std", 0) for v in variants]
            ax.bar(x + i * width, means, width, yerr=stds,
                   label=metric.upper(), capsize=3, alpha=0.85)

        ax.set_xlabel("Variant", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        path = os.path.join(FIGURE_DIR, filename)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"  Chart saved to {path}")


def save_json(all_results):
    """Save full results dict as JSON (drop non-serializable objects)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(path, "w") as f:
        json.dump(_clean(all_results), f, indent=2)
    print(f"  Full results JSON saved to {path}")


# =====================================================================
#  Main entry point
# =====================================================================

def run_ablation_study(device, num_seeds=3):
    """
    Execute the complete ablation study and produce all outputs.

    Parameters
    ----------
    device : str
        'cuda' or 'cpu'.
    num_seeds : int
        Number of random seeds per experiment for mean/std reporting.

    Returns
    -------
    dict : nested dictionary with all results.
    """
    print("\n" + "#" * 70)
    print("#" + " " * 18 + "ABLATION STUDY" + " " * 18 + "#")
    print("#" + " " * 10 + "Urban Expansion Monitoring" + " " * 10 + "#")
    print("#" * 70)
    print(f"  Device        : {device}")
    print(f"  Seeds         : {num_seeds}")
    print(f"  Train samples : {ABLATION_N_TRAIN}")
    print(f"  Val samples   : {ABLATION_N_VAL}")
    print(f"  Test samples  : {ABLATION_N_TEST}")
    print(f"  Epochs/stage  : {ABLATION_EPOCHS_PER_STAGE}")
    t_start = time.time()

    all_results = {}

    # 1. Backbone comparison
    all_results["backbone"] = ablation_backbone(device, num_seeds)

    # 2. FPN ablation
    all_results["fpn"] = ablation_fpn(device, num_seeds)

    # 3. Progressive fine-tuning ablation
    all_results["progressive"] = ablation_progressive(device, num_seeds)

    # 4. Loss function ablation
    all_results["loss"] = ablation_loss(device, num_seeds)

    # 5. Mixup augmentation ablation
    all_results["mixup"] = ablation_mixup(device, num_seeds)

    # 6. Pre-training ablation
    all_results["pretrain"] = ablation_pretrain(device, num_seeds)

    # 7. Statistical significance
    all_results["significance"] = statistical_tests(all_results)

    # ---- outputs ----
    elapsed = time.time() - t_start
    print(f"\n  Total ablation study time: {elapsed / 60:.1f} minutes")

    save_tables(all_results)
    save_charts(all_results)
    save_json(all_results)

    print("\n" + "#" * 70)
    print("  ABLATION STUDY COMPLETE")
    print("#" * 70)

    return all_results


# =====================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_ablation_study(device, num_seeds=3)
