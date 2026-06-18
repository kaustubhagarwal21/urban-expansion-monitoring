"""
Visualization utilities: training curves, confusion matrices,
model comparison bar charts, and urban growth pattern visualizations.
"""

import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


def plot_training_curves(history, backbone_name, save_dir=FIGURE_DIR):
    """Plot loss and accuracy curves over all training stages."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{backbone_name} - Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{backbone_name} - Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1 & mIoU
    axes[2].plot(epochs, history["val_f1"], label="Val F1", linewidth=2)
    axes[2].plot(epochs, history["val_miou"], label="Val mIoU", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_title(f"{backbone_name} - F1 & mIoU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"training_curves_{backbone_name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm, class_names, title, save_dir=FIGURE_DIR):
    """Plot and save a confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    safe_title = title.replace(" ", "_").replace("/", "_")
    path = os.path.join(save_dir, f"cm_{safe_title}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(all_results, save_dir=FIGURE_DIR):
    """
    Bar chart comparing OA, F1, mIoU across all models.
    all_results: dict of {model_name: metrics_dict}
    """
    os.makedirs(save_dir, exist_ok=True)
    model_names = list(all_results.keys())
    metrics_keys = ["oa", "precision", "recall", "f1", "miou"]
    metric_labels = ["OA", "Precision", "Recall", "F1-score", "mIoU"]

    x = np.arange(len(model_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (mk, ml) in enumerate(zip(metrics_keys, metric_labels)):
        vals = [all_results[m][mk] for m in model_names]
        bars = ax.bar(x + i * width, vals, width, label=ml)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Urban Expansion Classification")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_urban_growth(save_dir=FIGURE_DIR):
    """
    Visualize reported urban growth rates across the 5 study cities.
    """
    os.makedirs(save_dir, exist_ok=True)
    cities = list(CITY_GROWTH.keys())
    growth = [CITY_GROWTH[c] * 100 for c in cities]
    colors = ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0", "#00BCD4", "#795548"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    bars = axes[0].bar(cities, growth, color=colors, edgecolor="black", linewidth=0.5)
    for bar, g in zip(bars, growth):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     f"+{g:.0f}%", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Urban Area Increase (%)")
    axes[0].set_title("Urban Expansion 1990-2023")
    axes[0].grid(axis="y", alpha=0.3)

    # Simulated temporal expansion curves
    years = np.arange(1990, 2024)
    for i, city in enumerate(cities):
        rate = CITY_GROWTH[city]
        # Logistic-like growth curve
        mid = 2005 + i * 2
        curve = 100 * (1 + rate * (1 / (1 + np.exp(-0.15 * (years - mid)))))
        axes[1].plot(years, curve, label=city, color=colors[i], linewidth=2)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Relative Urban Area (%)")
    axes[1].set_title("Temporal Urban Growth Trajectories")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "urban_growth_patterns.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_sample_patches(save_dir=FIGURE_DIR):
    """Visualize sample synthetic patches for each class."""
    from src.dataset import generate_patch
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for row, (label, name) in enumerate(enumerate(CLASS_NAMES)):
        for col in range(4):
            patch = generate_patch(label)
            # Show RGB composite (channels 2, 1, 0)
            rgb = np.stack([patch[2], patch[1], patch[0]], axis=-1)
            rgb = np.clip(rgb * 3, 0, 1)  # brightness boost for visibility
            axes[row, col].imshow(rgb)
            axes[row, col].set_title(f"{name} - Sample {col + 1}")
            axes[row, col].axis("off")

    plt.suptitle("Sample Synthetic Multispectral Patches (RGB Composite)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, "sample_patches.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_architecture_diagram(save_dir=FIGURE_DIR):
    """Create a simplified architecture diagram."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", edgecolor="#1565C0",
                     linewidth=2)
    head_style = dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0", edgecolor="#E65100",
                      linewidth=2)
    fpn_style = dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", edgecolor="#2E7D32",
                     linewidth=2)
    out_style = dict(boxstyle="round,pad=0.4", facecolor="#FCE4EC", edgecolor="#C62828",
                     linewidth=2)

    # Input
    ax.text(1, 4, "6-Channel\nMultispectral\nInput\n(256x256x6)", ha="center", va="center",
            fontsize=9, bbox=box_style)

    # Backbone blocks
    for i, (name, stride) in enumerate(zip(["Block 1", "Block 2", "Block 3"],
                                            ["Stride 4", "Stride 8", "Stride 16"])):
        y = 6 - i * 2
        ax.text(4, y, f"{name}\n{stride}", ha="center", va="center", fontsize=9,
                bbox=box_style)
        ax.annotate("", xy=(3.2, y), xytext=(1.9, 4),
                    arrowprops=dict(arrowstyle="->", color="#1565C0"))

    # FPN
    ax.text(7, 4, "Feature\nPyramid\nNetwork", ha="center", va="center", fontsize=10,
            fontweight="bold", bbox=fpn_style)
    for i, y in enumerate([6, 4, 2]):
        ax.annotate("", xy=(6.1, 4), xytext=(4.9, y),
                    arrowprops=dict(arrowstyle="->", color="#2E7D32"))

    # GAP + Head
    ax.text(10, 4, "Global Avg Pool\n+\nClassification\nHead", ha="center", va="center",
            fontsize=9, bbox=head_style)
    ax.annotate("", xy=(9.1, 4), xytext=(7.9, 4),
                arrowprops=dict(arrowstyle="->", color="#E65100", linewidth=2))

    # Output
    ax.text(13, 4, "Urban\nNon-Urban\nTransition", ha="center", va="center",
            fontsize=10, fontweight="bold", bbox=out_style)
    ax.annotate("", xy=(12.1, 4), xytext=(11, 4),
                arrowprops=dict(arrowstyle="->", color="#C62828", linewidth=2))

    # Siamese branch note
    ax.text(8, 7.2, "Siamese Branch: Shared weights encode T1 & T2 → Concatenate → Change Head",
            ha="center", va="center", fontsize=9, style="italic",
            bbox=dict(boxstyle="round", facecolor="#F3E5F5", edgecolor="#7B1FA2"))

    ax.set_title("Urban Expansion Monitoring - Model Architecture", fontsize=14,
                 fontweight="bold", pad=10)
    plt.tight_layout()
    path = os.path.join(save_dir, "architecture_diagram.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
