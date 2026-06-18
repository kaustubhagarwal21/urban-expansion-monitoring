"""
Paper Figure Generator
======================
Generates all figures required for the research paper:
  Fig 1: Architecture diagram (5-pillar framework overview)
  Fig 2: Example multimodal inputs (S2 optical + SAR + labels) for Indian cities
  Fig 3: Model comparison bar chart (all 9 models)
  Fig 4: Cross-city transfer matrix heatmap
  Fig 5: Few-shot adaptation curves
  Fig 6: Urban expansion forecasts with uncertainty bands (Pillar IV)
  Fig 7: Efficiency comparison (params vs accuracy trade-off)
  Fig 8: Per-class performance comparison

Usage:
    python src/paper_figures.py
    python src/paper_figures.py --results outputs/results.json
"""

import os, sys, json, glob
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from configs.config import *


# ═══════════════════════════════════════════════════════
#  Fig 1: Architecture Diagram
# ═══════════════════════════════════════════════════════

def fig_architecture():
    """Generate the 5-pillar framework architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(8.5, 7.5, "The Living Map: Urban Expansion Monitoring Framework",
            ha="center", va="center", fontsize=22, fontweight="bold")

    # Transfer Learning Core (center bar)
    core = FancyBboxPatch((0.8, 3.4), 15.4, 1.0, boxstyle="round,pad=0.15",
                          facecolor="#2c3e50", edgecolor="none")
    ax.add_patch(core)
    ax.text(8.5, 3.9, "Transfer Learning Core (ResNet50 + FPN + Progressive Fine-Tuning)",
            ha="center", va="center", fontsize=16, color="white", fontweight="bold")

    # 5 Pillars (above)
    pillar_info = [
        ("I\nSAR Fusion", "#e74c3c"),
        ("II\nSelf-Supervised", "#e67e22"),
        ("III\nHigh-Res", "#f1c40f"),
        ("IV\nPredictive", "#27ae60"),
        ("V\nReal-Time", "#3498db"),
    ]
    for i, (label, color) in enumerate(pillar_info):
        x = 1.0 + i * 3.1
        box = FancyBboxPatch((x, 4.9), 2.6, 1.8, boxstyle="round,pad=0.12",
                             facecolor=color, edgecolor="none", alpha=0.85)
        ax.add_patch(box)
        ax.text(x + 1.3, 5.8, label, ha="center", va="center",
                fontsize=15, color="white", fontweight="bold")
        ax.annotate("", xy=(x + 1.3, 4.4), xytext=(x + 1.3, 4.9),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.5))

    # Data sources (below)
    data_sources = [
        "Sentinel-2\n(10m, 2017-2023)", "Sentinel-1 SAR\n(10m, 2017-2023)",
        "ESA WorldCover\n(Labels, 10m)", "Landsat 5/7/8/9\n(30m, 1990-2023)\n+ Census/RBI",
        "Sentinel-2\n(Near-Real-Time)"
    ]
    for i, src in enumerate(data_sources):
        x = 1.0 + i * 3.1
        box = FancyBboxPatch((x, 0.5), 2.6, 1.5, boxstyle="round,pad=0.12",
                             facecolor="#ecf0f1", edgecolor="#bdc3c7", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + 1.3, 1.25, src, ha="center", va="center", fontsize=12)
        ax.annotate("", xy=(x + 1.3, 3.4), xytext=(x + 1.3, 2.0),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=2))

    # Outputs (right side)
    out_box = FancyBboxPatch((16.8, 3.6), 2.8, 3.8, boxstyle="round,pad=0.15",
                             facecolor="#f8f9fa", edgecolor="#2c3e50", linewidth=2)
    ax.add_patch(out_box)
    ax.text(18.2, 7.0, "Outputs", fontsize=16, fontweight="bold",
            ha="center", va="center", color="#2c3e50")
    outputs = ["Urban Maps", "Change Detection", "Sprawl Forecasts",
               "Encroachment Alerts", "GradCAM Maps"]
    for i, out in enumerate(outputs):
        ax.text(18.2, 6.3 - i * 0.6, out, fontsize=13, ha="center", color="#2c3e50")

    # Arrow from core to outputs box
    ax.annotate("", xy=(16.8, 3.9), xytext=(16.2, 3.9),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.5))

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig1_architecture.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Fig 1 saved: {path}")


# ═══════════════════════════════════════════════════════
#  Fig 3: Model Comparison Bar Chart
# ═══════════════════════════════════════════════════════

def fig_model_comparison(results_path=None):
    """Generate model comparison bar chart with all metrics."""
    if results_path is None:
        results_path = os.path.join(OUTPUT_DIR, "results.json")
    if not os.path.exists(results_path):
        print(f"  [SKIP] No results at {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    models = list(results.keys())
    metrics = ["oa", "f1", "miou"]
    metric_labels = ["Overall Accuracy", "F1-Score", "mIoU"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = sns.color_palette("husl", len(models))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m].get(metric, 0) for m in models]
        bars = axes[idx].barh(models, values, color=colors)
        axes[idx].set_xlabel(label, fontsize=12)
        axes[idx].set_xlim(0, 1.0)
        axes[idx].invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, values):
            axes[idx].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                          f"{val:.3f}", va="center", fontsize=9)

    plt.suptitle("Model Performance Comparison (EuroSAT)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURE_DIR, "fig3_model_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Fig 3 saved: {path}")


# ═══════════════════════════════════════════════════════
#  Fig 7: Efficiency vs Accuracy Trade-off
# ═══════════════════════════════════════════════════════

def fig_efficiency_accuracy(results_path=None, efficiency_path=None):
    """Scatter plot: params vs accuracy with latency as bubble size."""
    if results_path is None:
        results_path = os.path.join(OUTPUT_DIR, "results.json")
    if efficiency_path is None:
        efficiency_path = os.path.join(OUTPUT_DIR, "efficiency_benchmark.json")

    if not os.path.exists(results_path) or not os.path.exists(efficiency_path):
        print("  [SKIP] Need both results.json and efficiency_benchmark.json")
        return

    with open(results_path) as f:
        results = json.load(f)
    with open(efficiency_path) as f:
        efficiency = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    for model_name in efficiency:
        if model_name not in results or "error" in efficiency[model_name]:
            continue
        eff = efficiency[model_name]
        res = results[model_name]

        params = eff["total_params_m"]
        oa = res.get("oa", 0)
        latency = eff.get("latency_ms", 10)

        ax.scatter(params, oa, s=latency * 10, alpha=0.7, edgecolors="black", linewidths=0.5)
        ax.annotate(model_name, (params, oa), textcoords="offset points",
                   xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Parameters (M)", fontsize=12)
    ax.set_ylabel("Overall Accuracy", fontsize=12)
    ax.set_title("Efficiency vs. Accuracy Trade-off\n(bubble size = inference latency)",
                fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig7_efficiency_accuracy.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Fig 7 saved: {path}")


# ═══════════════════════════════════════════════════════
#  Fig 8: Per-Class Performance Heatmap
# ═══════════════════════════════════════════════════════

def fig_per_class_performance(results_path=None):
    """Heatmap of per-class F1 scores across models."""
    if results_path is None:
        results_path = os.path.join(OUTPUT_DIR, "results.json")
    if not os.path.exists(results_path):
        print(f"  [SKIP] No results at {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Try to extract per-class metrics from reports
    # For now, show overall metrics per model as a heatmap
    models = list(results.keys())
    metrics = ["oa", "precision", "recall", "f1", "miou"]
    metric_labels = ["OA", "Precision", "Recall", "F1", "mIoU"]

    data = np.array([[results[m].get(metric, 0) for metric in metrics] for m in models])

    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(models) * 0.6)))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=metric_labels, yticklabels=models,
                vmin=0.5, vmax=1.0, ax=ax)
    ax.set_title("Model Performance Heatmap", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig8_performance_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Fig 8 saved: {path}")


# ═══════════════════════════════════════════════════════
#  LaTeX Table: Main Results
# ═══════════════════════════════════════════════════════

def generate_latex_table(results_path=None):
    """Generate LaTeX table for main results."""
    if results_path is None:
        results_path = os.path.join(OUTPUT_DIR, "results.json")
    if not os.path.exists(results_path):
        print(f"  [SKIP] No results at {results_path}")
        return ""

    with open(results_path) as f:
        results = json.load(f)

    # Find best model
    best_model = max(results.keys(), key=lambda m: results[m].get("oa", 0))

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Model performance comparison on EuroSAT benchmark (15 epochs).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & OA (\%) & Precision & Recall & F1 & mIoU \\",
        r"\midrule",
    ]

    # Group: baselines first, then DL
    baselines = [m for m in results if m in ("SVM", "Random_Forest", "RandomForest")]
    dl_models = [m for m in results if m not in baselines]

    for name in baselines + dl_models:
        r = results[name]
        oa = r.get("oa", 0)
        prec = r.get("precision", 0)
        rec = r.get("recall", 0)
        f1 = r.get("f1", 0)
        miou = r.get("miou", 0)

        # Bold best
        bold = name == best_model
        fmt = lambda v: f"\\textbf{{{v:.3f}}}" if bold else f"{v:.3f}"

        display_name = name.replace("_", r"\_")
        lines.append(
            f"{display_name} & {fmt(oa*100 if oa < 1 else oa)} & {fmt(prec)} & "
            f"{fmt(rec)} & {fmt(f1)} & {fmt(miou)} \\\\"
        )

        if name == baselines[-1] and baselines:
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    # Save
    path = os.path.join(OUTPUT_DIR, "table_main_results.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"  LaTeX table saved: {path}")

    return latex


# ═══════════════════════════════════════════════════════
#  Generate All
# ═══════════════════════════════════════════════════════

def generate_all_figures():
    """Generate all paper figures and tables."""
    print(f"\n{'='*60}")
    print(f"  GENERATING PAPER FIGURES AND TABLES")
    print(f"{'='*60}")

    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig_architecture()
    fig_model_comparison()
    fig_per_class_performance()
    fig_efficiency_accuracy()
    generate_latex_table()

    print(f"\n  All figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    generate_all_figures()
