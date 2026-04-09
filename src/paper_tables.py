"""
Paper table export: LaTeX and CSV formatters for experiment results.

Reads seed-averaged results (from seed_runner) or single-run results
(from paper_experiment) and outputs publication-ready tables.

Usage:
    python -m src.paper_tables                           # from seed_averaged_results.json
    python -m src.paper_tables --source single            # from paper_experiment results.json
    python -m src.paper_tables --format latex csv         # both formats
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import OUTPUT_DIR


RESULTS_DIR = os.path.join(OUTPUT_DIR, "paper_experiment")

METRIC_NAMES = ["oa", "precision", "recall", "f1", "miou"]
METRIC_DISPLAY = {
    "oa": "OA",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "miou": "mIoU",
}

EXPERIMENT_DISPLAY = {
    "supervised_optical": "Supervised (Optical)",
    "self_supervised_optical": "Self-Supervised (Optical)",
    "multimodal_fusion": "Multimodal Fusion (SAR+Optical)",
}


# =========================================================================
# Data loading
# =========================================================================

def load_seed_averaged_results(path: str = None) -> Optional[dict]:
    """Load seed-averaged results JSON."""
    if path is None:
        path = os.path.join(RESULTS_DIR, "seed_averaged_results.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_single_results(path: str = None) -> Optional[dict]:
    """Load single-run paper experiment results JSON."""
    if path is None:
        path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


# =========================================================================
# Table 1: Main performance comparison (seed-averaged)
# =========================================================================

def format_mean_std(mean: float, std: float, bold: bool = False) -> str:
    """Format as 'mean +/- std' for LaTeX, with optional bold."""
    val = f"{mean:.4f} $\\pm$ {std:.4f}"
    if bold:
        val = f"\\textbf{{{mean:.4f}}} $\\pm$ {std:.4f}"
    return val


def table1_seed_averaged_latex(results: dict) -> str:
    """
    Generate LaTeX for Table 1: Main performance comparison.

    Expected input: seed_averaged_results.json
    """
    experiments = ["supervised_optical", "self_supervised_optical", "multimodal_fusion"]

    # Find best mean for each metric to bold it
    best = {}
    for metric in METRIC_NAMES:
        best_val = -1
        best_exp = None
        for exp in experiments:
            agg = results.get(exp, {}).get("aggregated", {})
            m = agg.get(metric, {}).get("mean", 0)
            if m > best_val:
                best_val = m
                best_exp = exp
        best[metric] = best_exp

    # Build LaTeX
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Main performance comparison (mean $\\pm$ std over "
                 f"{len(results.get('config', {}).get('seeds', []))} seeds).}}")
    lines.append("\\label{tab:main_results}")

    col_spec = "l" + "c" * len(METRIC_NAMES)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header = "Method & " + " & ".join(METRIC_DISPLAY[m] for m in METRIC_NAMES) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for exp in experiments:
        display = EXPERIMENT_DISPLAY.get(exp, exp)
        agg = results.get(exp, {}).get("aggregated", {})
        cells = []
        for metric in METRIC_NAMES:
            m_data = agg.get(metric, {})
            mean = m_data.get("mean", 0)
            std = m_data.get("std", 0)
            is_best = best.get(metric) == exp
            cells.append(format_mean_std(mean, std, bold=is_best))
        row = f"{display} & " + " & ".join(cells) + " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def table1_single_run_latex(results: dict) -> str:
    """
    Generate LaTeX for Table 1 from single-run results (no std).
    """
    experiments = ["supervised_optical", "self_supervised_optical", "multimodal_fusion"]

    best = {}
    for metric in METRIC_NAMES:
        best_val = -1
        best_exp = None
        for exp in experiments:
            m = results.get(exp, {}).get(metric, 0)
            if m > best_val:
                best_val = m
                best_exp = exp
        best[metric] = best_exp

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Main performance comparison (single run).}")
    lines.append("\\label{tab:main_results}")

    col_spec = "l" + "c" * len(METRIC_NAMES)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header = "Method & " + " & ".join(METRIC_DISPLAY[m] for m in METRIC_NAMES) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for exp in experiments:
        display = EXPERIMENT_DISPLAY.get(exp, exp)
        cells = []
        for metric in METRIC_NAMES:
            val = results.get(exp, {}).get(metric, 0)
            is_best = best.get(metric) == exp
            if is_best:
                cells.append(f"\\textbf{{{val:.4f}}}")
            else:
                cells.append(f"{val:.4f}")
        row = f"{display} & " + " & ".join(cells) + " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =========================================================================
# Table 4: Efficiency and deployment
# =========================================================================

def table_efficiency_latex(results: dict, seed_averaged: bool = True) -> str:
    """
    Generate LaTeX for Table 4: Efficiency comparison.
    Shows training time per experiment.
    """
    experiments = ["supervised_optical", "self_supervised_optical", "multimodal_fusion"]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Training efficiency comparison.}")
    lines.append("\\label{tab:efficiency}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\toprule")
    lines.append("Method & Training Time (s) \\\\")
    lines.append("\\midrule")

    for exp in experiments:
        display = EXPERIMENT_DISPLAY.get(exp, exp)
        if seed_averaged:
            agg = results.get(exp, {}).get("aggregated", {})
            t = agg.get("time_sec", {})
            if isinstance(t, dict):
                mean = t.get("mean", 0)
                std = t.get("std", 0)
                lines.append(f"{display} & {mean:.1f} $\\pm$ {std:.1f} \\\\")
            else:
                lines.append(f"{display} & --- \\\\")
        else:
            t = results.get(exp, {}).get("time_sec", 0)
            lines.append(f"{display} & {t:.1f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


# =========================================================================
# CSV export
# =========================================================================

def export_csv_seed_averaged(results: dict, path: str = None) -> str:
    """Export seed-averaged results as CSV."""
    if path is None:
        path = os.path.join(RESULTS_DIR, "paper_table1.csv")

    experiments = ["supervised_optical", "self_supervised_optical", "multimodal_fusion"]
    header = ["method"] + [f"{m}_mean" for m in METRIC_NAMES] + [f"{m}_std" for m in METRIC_NAMES] + ["time_sec_mean"]

    rows = []
    for exp in experiments:
        display = EXPERIMENT_DISPLAY.get(exp, exp)
        agg = results.get(exp, {}).get("aggregated", {})
        row = [display]
        for metric in METRIC_NAMES:
            row.append(f"{agg.get(metric, {}).get('mean', 0):.4f}")
        for metric in METRIC_NAMES:
            row.append(f"{agg.get(metric, {}).get('std', 0):.4f}")
        row.append(f"{agg.get('time_sec', {}).get('mean', 0):.1f}")
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  CSV saved to {path}")
    return path


def export_csv_single(results: dict, path: str = None) -> str:
    """Export single-run results as CSV."""
    if path is None:
        path = os.path.join(RESULTS_DIR, "paper_table1_single.csv")

    experiments = ["supervised_optical", "self_supervised_optical", "multimodal_fusion"]
    header = ["method"] + METRIC_NAMES + ["time_sec"]

    rows = []
    for exp in experiments:
        display = EXPERIMENT_DISPLAY.get(exp, exp)
        data = results.get(exp, {})
        row = [display]
        for metric in METRIC_NAMES:
            row.append(f"{data.get(metric, 0):.4f}")
        row.append(f"{data.get('time_sec', 0):.1f}")
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  CSV saved to {path}")
    return path


# =========================================================================
# Per-seed detail CSV (for supplementary material)
# =========================================================================

def export_per_seed_csv(results: dict, path: str = None) -> str:
    """Export per-seed detailed results (all seeds, all experiments)."""
    if path is None:
        path = os.path.join(RESULTS_DIR, "per_seed_results.csv")

    experiments = ["supervised_optical", "self_supervised_optical", "multimodal_fusion"]
    header = ["experiment", "seed"] + METRIC_NAMES + ["time_sec"]

    rows = []
    for exp in experiments:
        display = EXPERIMENT_DISPLAY.get(exp, exp)
        per_seed = results.get(exp, {}).get("per_seed", [])
        for entry in per_seed:
            row = [display, entry.get("seed", "")]
            for metric in METRIC_NAMES:
                row.append(f"{entry.get(metric, 0):.4f}")
            row.append(f"{entry.get('time_sec', 0):.1f}")
            rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  Per-seed CSV saved to {path}")
    return path


# =========================================================================
# Main export orchestrator
# =========================================================================

def export_all_tables(
    source: str = "auto",
    formats: List[str] = None,
) -> Dict[str, str]:
    """
    Export all paper tables.

    Args:
        source: "seed_averaged", "single", or "auto" (try seed_averaged first)
        formats: list of "latex", "csv" (default both)

    Returns:
        Dict mapping output name to file path.
    """
    if formats is None:
        formats = ["latex", "csv"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    outputs = {}

    seed_data = load_seed_averaged_results()
    single_data = load_single_results()

    if source == "auto":
        source = "seed_averaged" if seed_data else "single"

    if source == "seed_averaged" and seed_data is None:
        print("  No seed-averaged results found. Run src/seed_runner.py first.")
        return outputs

    if source == "single" and single_data is None:
        print("  No single-run results found. Run paper_experiment.py first.")
        return outputs

    print("=" * 60)
    print(f"  PAPER TABLE EXPORT (source: {source})")
    print("=" * 60)

    if source == "seed_averaged":
        data = seed_data

        if "latex" in formats:
            # Table 1: Main comparison
            latex = table1_seed_averaged_latex(data)
            path = os.path.join(RESULTS_DIR, "table1_main.tex")
            with open(path, "w") as f:
                f.write(latex)
            print(f"  LaTeX Table 1 saved to {path}")
            outputs["table1_latex"] = path

            # Table 4: Efficiency
            eff_latex = table_efficiency_latex(data, seed_averaged=True)
            path = os.path.join(RESULTS_DIR, "table4_efficiency.tex")
            with open(path, "w") as f:
                f.write(eff_latex)
            print(f"  LaTeX Table 4 saved to {path}")
            outputs["table4_latex"] = path

        if "csv" in formats:
            outputs["table1_csv"] = export_csv_seed_averaged(data)
            outputs["per_seed_csv"] = export_per_seed_csv(data)

    else:
        data = single_data

        if "latex" in formats:
            latex = table1_single_run_latex(data)
            path = os.path.join(RESULTS_DIR, "table1_main.tex")
            with open(path, "w") as f:
                f.write(latex)
            print(f"  LaTeX Table 1 saved to {path}")
            outputs["table1_latex"] = path

            eff_latex = table_efficiency_latex(data, seed_averaged=False)
            path = os.path.join(RESULTS_DIR, "table4_efficiency.tex")
            with open(path, "w") as f:
                f.write(eff_latex)
            print(f"  LaTeX Table 4 saved to {path}")
            outputs["table4_latex"] = path

        if "csv" in formats:
            outputs["table1_csv"] = export_csv_single(data)

    print(f"\n  Exported {len(outputs)} files.")
    return outputs


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Export paper tables")
    parser.add_argument(
        "--source", default="auto",
        choices=["auto", "seed_averaged", "single"],
        help="Which results to use",
    )
    parser.add_argument(
        "--format", nargs="+", default=["latex", "csv"],
        choices=["latex", "csv"],
        help="Output formats",
    )
    args = parser.parse_args()

    export_all_tables(source=args.source, formats=args.format)


if __name__ == "__main__":
    main()
