"""
Experiment Tracker — TensorBoard-based tracking for research reproducibility.

Wraps torch.utils.tensorboard.SummaryWriter with convenience methods
for logging metrics, confusion matrices, embeddings, PR curves, and
hyperparameter sweeps.  Also provides checkpoint save/load, LaTeX table
generation, and multi-experiment comparison utilities.
"""

import os
import sys
import json
import csv
import datetime
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

# ── project imports ─────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


# ============================================================================
#  ExperimentTracker
# ============================================================================

class ExperimentTracker:
    """TensorBoard-backed experiment tracker with checkpoint management."""

    def __init__(self, experiment_name, config_dict=None, log_dir=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{experiment_name}_{timestamp}"

        if log_dir is None:
            self.log_dir = os.path.join(OUTPUT_DIR, "runs", run_name)
        else:
            self.log_dir = os.path.join(log_dir, run_name)

        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.experiment_name = experiment_name
        self.config_dict = config_dict or {}
        self.start_time = datetime.datetime.now()

        # Persist config as JSON
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config_dict, f, indent=2, default=str)

        # Also log config as TensorBoard text for quick in-browser viewing
        config_text = json.dumps(self.config_dict, indent=2, default=str)
        self.writer.add_text("config", f"```json\n{config_text}\n```", 0)

        print(f"[ExperimentTracker] Logging to {self.log_dir}")

    # ── scalar logging ──────────────────────────────────────────────────────

    def log_scalar(self, tag, value, step):
        """Log a single scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars under the same main tag (for run comparison)."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_metrics(self, metrics_dict, step, prefix=""):
        """Log a dictionary of metrics (e.g. oa, f1, miou) as scalars."""
        for key, value in metrics_dict.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.writer.add_scalar(tag, value, step)

    # ── image & figure logging ──────────────────────────────────────────────

    def log_confusion_matrix(self, cm, class_names, step):
        """Render a confusion matrix as a matplotlib figure and log as image."""
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm[i, j]:.0f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        # Convert figure to tensor image (C, H, W) in [0, 1]
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)                       # (H, W, 4)
        image = image[:, :, :3]                       # drop alpha
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        self.writer.add_image("confusion_matrix", image_tensor, step)
        plt.close(fig)

    def log_image(self, tag, image_tensor, step):
        """Log an image tensor (C, H, W) or (H, W, C)."""
        if image_tensor.ndim == 3 and image_tensor.shape[2] in (1, 3, 4):
            # (H, W, C) -> (C, H, W)
            image_tensor = image_tensor.permute(2, 0, 1)
        self.writer.add_image(tag, image_tensor, step)

    # ── model & graph logging ───────────────────────────────────────────────

    def log_model_graph(self, model, input_shape):
        """Log the model's computation graph.

        Parameters
        ----------
        model : torch.nn.Module
        input_shape : tuple  e.g. (1, 6, 256, 256)
        """
        device = next(model.parameters()).device
        dummy = torch.zeros(*input_shape, device=device)
        try:
            self.writer.add_graph(model, dummy)
        except Exception as e:
            print(f"[ExperimentTracker] Could not log model graph: {e}")

    # ── hyperparameter dashboard ────────────────────────────────────────────

    def log_hyperparameters(self, hparam_dict, metric_dict):
        """Log hyper-parameters alongside final metrics for the HParam dashboard."""
        self.writer.add_hparams(hparam_dict, metric_dict)

    # ── embeddings ──────────────────────────────────────────────────────────

    def log_embedding(self, features, labels, label_names, step):
        """Log feature embeddings for the TensorBoard projector.

        Parameters
        ----------
        features : torch.Tensor  (N, D)
        labels : torch.Tensor    (N,)  integer class indices
        label_names : list[str]  human-readable names per class
        step : int
        """
        metadata = [label_names[int(l)] for l in labels]
        self.writer.add_embedding(
            features,
            metadata=metadata,
            global_step=step,
            tag=f"embeddings/step_{step}",
        )

    # ── precision-recall curves ─────────────────────────────────────────────

    def log_pr_curve(self, tag, labels, predictions, step):
        """Log a precision-recall curve.

        Parameters
        ----------
        labels : torch.Tensor (N,)       ground-truth binary labels
        predictions : torch.Tensor (N,)  predicted probabilities
        """
        self.writer.add_pr_curve(tag, labels, predictions, step)

    # ── histograms ──────────────────────────────────────────────────────────

    def log_histogram(self, tag, values, step):
        """Log a histogram (e.g. weight distributions)."""
        self.writer.add_histogram(tag, values, step)

    # ── checkpoint management ───────────────────────────────────────────────

    def save_checkpoint(self, model, optimizer, epoch, metrics, filename=None):
        """Save a training checkpoint.

        Saved keys: model_state_dict, optimizer_state_dict, epoch, metrics,
        config, experiment_name, timestamp.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"
        ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        filepath = os.path.join(ckpt_dir, filename)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": self.config_dict,
            "experiment_name": self.experiment_name,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        torch.save(checkpoint, filepath)
        print(f"[ExperimentTracker] Checkpoint saved to {filepath}")
        return filepath

    def load_checkpoint(self, filepath, model, optimizer=None):
        """Load a checkpoint and restore model (and optionally optimizer) state.

        Returns
        -------
        dict  with keys epoch, metrics, config.
        """
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        info = {
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
            "config": checkpoint.get("config", {}),
        }
        print(f"[ExperimentTracker] Loaded checkpoint from {filepath} (epoch {info['epoch']})")
        return info

    # ── lifecycle ───────────────────────────────────────────────────────────

    def close(self):
        """Flush and close the TensorBoard writer; save a summary JSON."""
        elapsed = datetime.datetime.now() - self.start_time
        summary = {
            "experiment_name": self.experiment_name,
            "log_dir": self.log_dir,
            "start_time": self.start_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
        }
        summary_path = os.path.join(self.log_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.writer.flush()
        self.writer.close()
        print(f"[ExperimentTracker] Closed. Duration: {elapsed}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
#  Helper functions
# ============================================================================

def create_experiment_config(backbone_name=DEFAULT_BACKBONE, **overrides):
    """Build a complete experiment config dict from the global project settings.

    Any key present in *overrides* replaces the default value.
    """
    config = {
        # model
        "backbone": backbone_name,
        "num_channels": NUM_CHANNELS,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "fpn_channels": FPN_CHANNELS,
        # training
        "stages": STAGES,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "loss_weights": LOSS_WEIGHTS,
        "focal_gamma": FOCAL_GAMMA,
        "mixup_alpha": MIXUP_ALPHA,
        "seed": SEED,
        # data
        "patch_size": PATCH_SIZE,
        "total_patches": TOTAL_PATCHES,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "cities": CITIES,
        # paths
        "output_dir": OUTPUT_DIR,
        "model_dir": MODEL_DIR,
        "figure_dir": FIGURE_DIR,
        "log_dir": LOG_DIR,
    }
    config.update(overrides)
    return config


def compare_experiments(experiment_dirs, metric_keys=None):
    """Load results from multiple experiment directories and create comparison
    bar charts saved to FIGURE_DIR.

    Each directory is expected to contain a ``results.json`` file with metric
    values (or a ``summary.json`` with an embedded ``metrics`` key).

    Returns
    -------
    dict  mapping experiment_name -> metrics dict
    """
    if metric_keys is None:
        metric_keys = ["oa", "f1", "miou"]

    all_results = {}
    for exp_dir in experiment_dirs:
        # Try results.json first, then summary.json
        results_path = os.path.join(exp_dir, "results.json")
        summary_path = os.path.join(exp_dir, "summary.json")

        metrics = {}
        if os.path.isfile(results_path):
            with open(results_path) as f:
                metrics = json.load(f)
        elif os.path.isfile(summary_path):
            with open(summary_path) as f:
                data = json.load(f)
                metrics = data.get("metrics", data)

        exp_name = os.path.basename(exp_dir)
        all_results[exp_name] = {k: metrics.get(k, 0.0) for k in metric_keys}

    if not all_results:
        print("[compare_experiments] No results found.")
        return all_results

    # Create grouped bar chart
    exp_names = list(all_results.keys())
    n_metrics = len(metric_keys)
    n_exps = len(exp_names)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_exps, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, exp_name in enumerate(exp_names):
        values = [all_results[exp_name][k] for k in metric_keys]
        offset = (idx - n_exps / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=exp_name)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylabel("Score")
    ax.set_title("Experiment Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in metric_keys])
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    out_path = os.path.join(FIGURE_DIR, "experiment_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare_experiments] Comparison chart saved to {out_path}")

    return all_results


def generate_latex_table(results_dict, caption="", label=""):
    """Generate a booktabs-style LaTeX table from a results dictionary.

    Parameters
    ----------
    results_dict : dict
        {experiment_name: {metric_name: value, ...}, ...}
    caption : str
    label : str

    Returns
    -------
    str  LaTeX source text.
    """
    if not results_dict:
        return ""

    # Collect all metric keys in consistent order
    metric_keys = []
    for metrics in results_dict.values():
        for k in metrics:
            if k not in metric_keys:
                metric_keys.append(k)

    col_spec = "l" + "c" * len(metric_keys)
    header_cells = " & ".join(k.upper() for k in metric_keys)

    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
    ]
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines += [
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"    Experiment & {header_cells} \\\\",
        r"    \midrule",
    ]

    # Find best value per metric for bolding
    best = {}
    for k in metric_keys:
        vals = [m.get(k, -float("inf")) for m in results_dict.values()]
        best[k] = max(vals)

    for exp_name, metrics in results_dict.items():
        cells = []
        for k in metric_keys:
            val = metrics.get(k, 0.0)
            formatted = f"{val:.4f}"
            if val == best[k]:
                formatted = f"\\textbf{{{formatted}}}"
            cells.append(formatted)
        row = " & ".join(cells)
        lines.append(f"    {exp_name} & {row} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_results_csv(results_dict, filepath):
    """Save experiment results to a CSV file.

    Parameters
    ----------
    results_dict : dict
        {experiment_name: {metric_name: value, ...}, ...}
    filepath : str
    """
    if not results_dict:
        return

    metric_keys = []
    for metrics in results_dict.values():
        for k in metrics:
            if k not in metric_keys:
                metric_keys.append(k)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment"] + metric_keys)
        for exp_name, metrics in results_dict.items():
            row = [exp_name] + [metrics.get(k, "") for k in metric_keys]
            writer.writerow(row)
    print(f"[generate_results_csv] Saved to {filepath}")


# ============================================================================
#  Demo
# ============================================================================

if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim

    # ── 1. Build a tiny dummy model ────────────────────────────────────────
    class DummySegModel(nn.Module):
        def __init__(self, in_ch=NUM_CHANNELS, n_cls=NUM_CLASSES):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, n_cls, 1),
            )

        def forward(self, x):
            return self.conv(x)

    model = DummySegModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)

    # ── 2. Create experiment config & tracker ──────────────────────────────
    config = create_experiment_config("efficientnet_b0", lr=1e-3)

    with ExperimentTracker("demo_run", config_dict=config) as tracker:
        # Log model graph
        tracker.log_model_graph(model, input_shape=(1, NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE))

        # ── 3. Fake training loop ──────────────────────────────────────────
        num_epochs = 5
        for epoch in range(1, num_epochs + 1):
            # Simulate metrics that improve over time
            noise = np.random.normal(0, 0.02)
            oa = min(0.60 + 0.07 * epoch + noise, 1.0)
            f1 = min(0.55 + 0.06 * epoch + noise, 1.0)
            miou = min(0.45 + 0.08 * epoch + noise, 1.0)
            loss = max(1.2 - 0.18 * epoch + noise, 0.05)

            # Scalar logging
            tracker.log_scalar("train/loss", loss, epoch)
            tracker.log_metrics({"oa": oa, "f1": f1, "miou": miou}, epoch, prefix="val")

            # Log weight histograms
            for name, param in model.named_parameters():
                tracker.log_histogram(f"weights/{name}", param.data, epoch)

            # Confusion matrix at last epoch
            if epoch == num_epochs:
                cm = np.array([[120, 10, 5],
                               [8, 100, 12],
                               [3, 15, 60]], dtype=float)
                tracker.log_confusion_matrix(cm, CLASS_NAMES, epoch)

                # PR curve (binary example for class 0)
                n_samples = 200
                labels = torch.cat([torch.ones(100), torch.zeros(100)])
                preds = torch.cat([
                    torch.clamp(torch.randn(100) * 0.2 + 0.8, 0, 1),
                    torch.clamp(torch.randn(100) * 0.2 + 0.2, 0, 1),
                ])
                tracker.log_pr_curve("pr/urban", labels, preds, epoch)

            # Save checkpoint
            metrics = {"oa": oa, "f1": f1, "miou": miou, "loss": loss}
            tracker.save_checkpoint(model, optimizer, epoch, metrics)

            print(f"  Epoch {epoch}/{num_epochs}  loss={loss:.4f}  OA={oa:.4f}  F1={f1:.4f}  mIoU={miou:.4f}")

        # ── 4. Hyperparameter logging ──────────────────────────────────────
        tracker.log_hyperparameters(
            hparam_dict={
                "backbone": config["backbone"],
                "batch_size": config["batch_size"],
                "lr": config.get("lr", 1e-3),
                "weight_decay": config["weight_decay"],
            },
            metric_dict={"hparam/oa": oa, "hparam/f1": f1, "hparam/miou": miou},
        )

        # ── 5. Save final results JSON alongside the run ──────────────────
        final_results = {"oa": oa, "f1": f1, "miou": miou}
        results_path = os.path.join(tracker.log_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)

    # ── 6. LaTeX table & CSV export ────────────────────────────────────────
    demo_results = {
        "EfficientNet-B0": {"oa": 0.912, "f1": 0.887, "miou": 0.823},
        "ResNet-50":       {"oa": 0.895, "f1": 0.870, "miou": 0.801},
        "VGG-16":          {"oa": 0.878, "f1": 0.851, "miou": 0.779},
    }

    latex = generate_latex_table(
        demo_results,
        caption="Segmentation results on the Urban Expansion benchmark.",
        label="tab:results",
    )
    print("\n── LaTeX table ──")
    print(latex)

    csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    generate_results_csv(demo_results, csv_path)

    print("\nDone. Launch TensorBoard with:")
    print(f"  tensorboard --logdir \"{os.path.join(OUTPUT_DIR, 'runs')}\"")
