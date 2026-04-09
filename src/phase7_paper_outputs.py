"""
Phase 7 paper-output generator for the locked 3-city Indian pipeline.

Creates compact tables and figures from:
  - Phase 2 benchmark results
  - Phase 5.5 multi-seed summary (when available)
  - Phase 4 LOCO and ablation outputs
  - Phase 5 pillar outputs
  - Efficiency benchmark (when available)
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import FIGURE_DIR, OUTPUT_DIR


RESULTS_DIR = Path(OUTPUT_DIR) / "research_results"
GEE_DIR = Path(r"G:\My Drive\urban_expansion_india")
LOCKED_CITIES = ["Mumbai", "Delhi_NCR", "Bangalore"]
BENCHMARK_MODELS = ["SVM", "RandomForest", "resnet50", "efficientnet_b0", "swin_tiny", "mobilenet_v3_small"]
LOCO_MODELS = ["efficientnet_b0", "resnet50", "swin_tiny"]
ABLATION_CONFIGS = ["full", "no_fpn", "ce_only"]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _maybe_load_json(path):
    return _load_json(path) if os.path.exists(path) else None


def _metric_string(agg):
    if not agg:
        return ""
    mean = agg.get("mean")
    std = agg.get("std")
    if mean is None:
        return ""
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def _single_or_multi_benchmark():
    multi = _maybe_load_json(RESULTS_DIR / "multi_seed_summary.json")
    if multi:
        rows = []
        for model in BENCHMARK_MODELS:
            section = multi["baselines"].get(model) or multi["benchmark"].get(model)
            agg = section.get("aggregated", {}) if section else {}
            rows.append({
                "model": model,
                "oa": _metric_string(agg.get("oa")),
                "f1": _metric_string(agg.get("f1")),
                "miou": _metric_string(agg.get("miou")),
                "oa_value": agg.get("oa", {}).get("mean"),
            })
        return rows, True

    summary = _load_json(RESULTS_DIR / "phase2_benchmark_summary.json")
    rows = []
    for item in summary["models"]:
        rows.append({
            "model": item["model"],
            "oa": f"{item['oa']:.4f}",
            "f1": f"{item['f1']:.4f}",
            "miou": f"{item['miou']:.4f}",
            "oa_value": float(item["oa"]),
        })
    return rows, False


def _single_or_multi_loco():
    multi = _maybe_load_json(RESULTS_DIR / "multi_seed_summary.json")
    if multi:
        rows = []
        for model in LOCO_MODELS:
            agg = multi["loco"].get(model, {}).get("aggregated", {})
            rows.append({
                "model": model,
                "oa": _metric_string(agg.get("oa")),
                "f1": _metric_string(agg.get("f1")),
                "miou": _metric_string(agg.get("miou")),
            })
        return rows, True

    rows = []
    for model in LOCO_MODELS:
        data = _load_json(RESULTS_DIR / f"phase4_loco_{model}.json")
        avg = data["average"]
        rows.append({
            "model": model,
            "oa": f"{avg['oa']:.4f}",
            "f1": f"{avg['f1']:.4f}",
            "miou": f"{avg['miou']:.4f}",
        })
    return rows, False


def _single_or_multi_ablation():
    multi = _maybe_load_json(RESULTS_DIR / "multi_seed_summary.json")
    if multi:
        rows = []
        for config_name in ABLATION_CONFIGS:
            agg = multi["ablation"].get(config_name, {}).get("aggregated", {})
            rows.append({
                "config": config_name,
                "oa": _metric_string(agg.get("oa")),
                "f1": _metric_string(agg.get("f1")),
                "miou": _metric_string(agg.get("miou")),
            })
        return rows, True

    data = _load_json(RESULTS_DIR / "phase4_small_ablation.json")
    rows = []
    for config_name, metrics in data["results"].items():
        rows.append({
            "config": config_name,
            "oa": f"{metrics['oa']:.4f}",
            "f1": f"{metrics['f1']:.4f}",
            "miou": f"{metrics['miou']:.4f}",
        })
    return rows, False


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def _include_label_agreement():
    raw = os.environ.get("PHASE7_INCLUDE_LABEL_AGREEMENT", "").strip().lower()
    return raw in {"1", "true", "yes"}


def _label_agreement_rows():
    try:
        import rasterio
    except ImportError:
        return []

    rows = []
    for city in LOCKED_CITIES:
        wc_path = GEE_DIR / f"Labels_WC_{city}.tif"
        dw_path = GEE_DIR / f"Labels_DW_{city}.tif"
        if not (wc_path.exists() and dw_path.exists()):
            continue

        with rasterio.open(wc_path) as src_wc, rasterio.open(dw_path) as src_dw:
            wc = src_wc.read(1)
            dw = src_dw.read(1)

        h = min(wc.shape[0], dw.shape[0])
        w = min(wc.shape[1], dw.shape[1])
        wc = wc[:h, :w].astype(np.int64)
        dw = dw[:h, :w].astype(np.int64)
        valid = np.isfinite(wc) & np.isfinite(dw) & np.isin(wc, [0, 1, 2]) & np.isin(dw, [0, 1, 2])
        if not valid.any():
            continue

        wc = wc[valid]
        dw = dw[valid]
        agreement = float(np.mean(wc == dw))
        total = int(valid.sum())

        confusion = np.zeros((3, 3), dtype=np.int64)
        for i in range(3):
            for j in range(3):
                confusion[i, j] = int(np.sum((wc == i) & (dw == j)))

        row_marginals = confusion.sum(axis=1)
        col_marginals = confusion.sum(axis=0)
        pe = float(np.sum(row_marginals * col_marginals)) / float(total * total) if total else 0.0
        kappa = (agreement - pe) / (1 - pe) if total and abs(1 - pe) > 1e-8 else 0.0

        rows.append({
            "city": city,
            "agreement": f"{agreement:.4f}",
            "kappa": f"{kappa:.4f}",
            "n_pixels": total,
            "wc_urban_pct": f"{_safe_div(np.sum(wc == 0), total):.4f}",
            "dw_urban_pct": f"{_safe_div(np.sum(dw == 0), total):.4f}",
            "wc_transition_pct": f"{_safe_div(np.sum(wc == 2), total):.4f}",
            "dw_transition_pct": f"{_safe_div(np.sum(dw == 2), total):.4f}",
            "disagreement_pct": f"{1.0 - agreement:.4f}",
        })
    return rows


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_tables():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    benchmark_rows, benchmark_is_multi = _single_or_multi_benchmark()
    loco_rows, loco_is_multi = _single_or_multi_loco()
    ablation_rows, ablation_is_multi = _single_or_multi_ablation()

    pillar1_optical = _maybe_load_json(RESULTS_DIR / "pillar1_optical_only_baseline.json")
    pillar1_sar = _maybe_load_json(RESULTS_DIR / "pillar1_indian_sar_fusion.json")
    pillar2 = _maybe_load_json(RESULTS_DIR / "pillar2_indian_simclr.json")
    efficiency = _maybe_load_json(RESULTS_DIR / "efficiency_benchmark.json") or _maybe_load_json(Path(OUTPUT_DIR) / "efficiency_benchmark.json")
    label_rows = _label_agreement_rows() if _include_label_agreement() else []

    write_csv(
        RESULTS_DIR / "table1_main_benchmark.csv",
        ["model", "oa", "f1", "miou"],
        benchmark_rows,
    )
    write_csv(
        RESULTS_DIR / "table2_loco.csv",
        ["model", "oa", "f1", "miou"],
        loco_rows,
    )
    write_csv(
        RESULTS_DIR / "table3_ablation.csv",
        ["config", "oa", "f1", "miou"],
        ablation_rows,
    )

    pillar_rows = []
    if pillar1_optical:
        pillar_rows.append({
            "experiment": "pillar1_optical_only",
            "oa": f"{pillar1_optical['test_metrics']['oa']:.4f}",
            "f1": f"{pillar1_optical['test_metrics']['f1']:.4f}",
            "miou": f"{pillar1_optical['test_metrics']['miou']:.4f}",
        })
    if pillar1_sar:
        pillar_rows.append({
            "experiment": "pillar1_optical_plus_sar",
            "oa": f"{pillar1_sar['test_metrics']['oa']:.4f}",
            "f1": f"{pillar1_sar['test_metrics']['f1']:.4f}",
            "miou": f"{pillar1_sar['test_metrics']['miou']:.4f}",
        })
    if pillar2:
        for name, metrics in pillar2.items():
            pillar_rows.append({
                "experiment": f"pillar2_{name}",
                "oa": f"{metrics['oa']:.4f}",
                "f1": f"{metrics['f1']:.4f}",
                "miou": f"{metrics['miou']:.4f}",
            })
    if pillar_rows:
        write_csv(
            RESULTS_DIR / "table4_pillars.csv",
            ["experiment", "oa", "f1", "miou"],
            pillar_rows,
        )

    if efficiency:
        eff_rows = []
        for model_name, metrics in efficiency.items():
            if "error" in metrics:
                continue
            eff_rows.append({
                "model": model_name,
                "params_m": f"{metrics['total_params_m']:.2f}",
                "gflops": f"{metrics['gflops']:.2f}",
                "latency_ms": f"{metrics['latency_ms']:.2f}",
                "peak_gpu_mb": f"{metrics['peak_gpu_mb']:.1f}",
            })
        write_csv(
            RESULTS_DIR / "table5_efficiency.csv",
            ["model", "params_m", "gflops", "latency_ms", "peak_gpu_mb"],
            eff_rows,
        )
    if label_rows:
        write_csv(
            RESULTS_DIR / "table6_label_agreement.csv",
            [
                "city",
                "agreement",
                "kappa",
                "n_pixels",
                "wc_urban_pct",
                "dw_urban_pct",
                "wc_transition_pct",
                "dw_transition_pct",
                "disagreement_pct",
            ],
            label_rows,
        )

    md_lines = [
        "# Phase 7 Tables",
        "",
        f"- Main benchmark source: {'multi-seed' if benchmark_is_multi else 'single-seed'}",
        f"- LOCO source: {'multi-seed' if loco_is_multi else 'single-seed'}",
        f"- Ablation source: {'multi-seed' if ablation_is_multi else 'single-seed'}",
        "",
        "Generated CSV files:",
        "- table1_main_benchmark.csv",
        "- table2_loco.csv",
        "- table3_ablation.csv",
    ]
    if pillar_rows:
        md_lines.append("- table4_pillars.csv")
    if efficiency:
        md_lines.append("- table5_efficiency.csv")
    if label_rows:
        md_lines.append("- table6_label_agreement.csv")
    (RESULTS_DIR / "phase7_tables_summary.md").write_text("\n".join(md_lines) + "\n")


def generate_figures():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    benchmark_rows, _ = _single_or_multi_benchmark()
    label_rows = _label_agreement_rows() if _include_label_agreement() else []

    # Figure 1: main benchmark bar chart
    models = [row["model"] for row in benchmark_rows]
    oa_values = [float(row["oa_value"]) for row in benchmark_rows]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, oa_values, color="#3A6EA5")
    plt.ylabel("Overall Accuracy")
    plt.title("Phase 7: Main Benchmark")
    plt.xticks(rotation=25, ha="right")
    for bar, value in zip(bars, oa_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(FIGURE_DIR) / "phase7_benchmark_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 2: LOCO heatmap from single-seed fold values
    loco_matrix = []
    for model in LOCO_MODELS:
        data = _load_json(RESULTS_DIR / f"phase4_loco_{model}.json")
        loco_matrix.append([data["folds"][city]["oa"] for city in LOCKED_CITIES])
    loco_matrix = np.array(loco_matrix)
    plt.figure(figsize=(6, 4))
    plt.imshow(loco_matrix, cmap="YlOrRd", vmin=0.5, vmax=1.0)
    plt.xticks(range(len(LOCKED_CITIES)), LOCKED_CITIES)
    plt.yticks(range(len(LOCO_MODELS)), LOCO_MODELS)
    for i in range(loco_matrix.shape[0]):
        for j in range(loco_matrix.shape[1]):
            plt.text(j, i, f"{loco_matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(label="OA")
    plt.title("Phase 7: LOCO Transfer Heatmap")
    plt.tight_layout()
    plt.savefig(Path(FIGURE_DIR) / "phase7_loco_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 3: ablation bar chart
    ablation = _load_json(RESULTS_DIR / "phase4_small_ablation.json")
    names = list(ablation["results"].keys())
    f1_values = [ablation["results"][name]["f1"] for name in names]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, f1_values, color="#DA8A67")
    plt.ylabel("F1")
    plt.title("Phase 7: Small Ablation")
    for bar, value in zip(bars, f1_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(Path(FIGURE_DIR) / "phase7_ablation_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    if label_rows:
        cities = [row["city"] for row in label_rows]
        agreements = [float(row["agreement"]) for row in label_rows]
        plt.figure(figsize=(6, 4))
        bars = plt.bar(cities, agreements, color="#5E8C61")
        plt.ylabel("Agreement")
        plt.ylim(0.0, 1.0)
        plt.title("Phase 7: WorldCover vs Dynamic World Agreement")
        for bar, value in zip(bars, agreements):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(Path(FIGURE_DIR) / "phase7_label_agreement_bar.png", dpi=200, bbox_inches="tight")
        plt.close()


def main():
    generate_tables()
    generate_figures()
    print(f"Phase 7 tables/figures generated under {RESULTS_DIR} and {FIGURE_DIR}")


if __name__ == "__main__":
    main()
