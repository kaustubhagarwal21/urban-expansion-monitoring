"""
Aggregate Phase 5.5 multi-seed outputs for the locked 3-city pipeline.

Builds:
  - outputs/research_results/multi_seed_summary.json
  - outputs/research_results/multi_seed_summary.md
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import OUTPUT_DIR


RESULTS_DIR = Path(OUTPUT_DIR) / "research_results"
SEEDS = [42, 123, 7]
BENCHMARK_BACKBONES = [
    "efficientnet_b0",
    "resnet50",
    "swin_tiny",
    "mobilenet_v3_small",
]
LOCO_BACKBONES = ["efficientnet_b0", "resnet50", "swin_tiny"]
BASELINES = ["SVM", "RandomForest"]
ABLATION_CONFIGS = ["full", "no_fpn", "ce_only"]


def _mean_std(values):
    if not values:
        return None
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "values": [float(values[0])]}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
        "values": [float(v) for v in values],
    }


def _load_json(path):
    return json.loads(Path(path).read_text())


def _benchmark_seed_path(backbone, seed):
    if seed == 42:
        legacy_name = {
            "efficientnet_b0": "phase2_efficientnet_b0_results.json",
            "resnet50": "phase2_resnet50_results.json",
            "swin_tiny": "phase2_swin_tiny_results.json",
            "mobilenet_v3_small": "phase2_mobilenet_results.json",
        }[backbone]
        return RESULTS_DIR / legacy_name
    return RESULTS_DIR / f"phase2_{backbone}_seed{seed}.json"


def _extract_benchmark_metrics(backbone, seed):
    path = _benchmark_seed_path(backbone, seed)
    if not path.exists():
        return None
    data = _load_json(path)
    if backbone in data:
        metrics = data[backbone]
    else:
        metrics = data
    return {
        "oa": float(metrics["oa"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "miou": float(metrics["miou"]),
        "training_time_min": float(metrics.get("training_time_min", 0.0)),
    }


def _extract_baseline_metrics(model_name, seed):
    if seed == 42:
        for path in [
            RESULTS_DIR / "phase2_resnet50_results.json",
            RESULTS_DIR / "phase2_efficientnet_b0_results.json",
            RESULTS_DIR / "phase2_swin_tiny_results.json",
            RESULTS_DIR / "phase2_mobilenet_results.json",
        ]:
            if path.exists():
                data = _load_json(path)
                if model_name in data:
                    metrics = data[model_name]
                    return {
                        "oa": float(metrics["oa"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "f1": float(metrics["f1"]),
                        "miou": float(metrics["miou"]),
                    }
        return None

    path = RESULTS_DIR / f"phase2_baselines_seed{seed}.json"
    if not path.exists():
        return None
    data = _load_json(path)
    metrics = data["results"][model_name]
    return {
        "oa": float(metrics["oa"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "miou": float(metrics["miou"]),
    }


def _extract_loco_metrics(backbone, seed):
    suffix = "" if seed == 42 else f"_seed{seed}"
    path = RESULTS_DIR / f"phase4_loco_{backbone}{suffix}.json"
    if not path.exists():
        return None
    data = _load_json(path)
    avg = data["average"]
    return {
        "oa": float(avg["oa"]),
        "precision": float(avg["precision"]),
        "recall": float(avg["recall"]),
        "f1": float(avg["f1"]),
        "miou": float(avg["miou"]),
        "training_time_min": float(avg.get("training_time_min", 0.0)),
    }


def _extract_ablation_metrics(config_name, seed):
    suffix = "" if seed == 42 else f"_seed{seed}"
    path = RESULTS_DIR / f"phase4_small_ablation{suffix}.json"
    if not path.exists():
        return None
    data = _load_json(path)
    metrics = data["results"][config_name]
    return {
        "oa": float(metrics["oa"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "miou": float(metrics["miou"]),
        "training_time_min": float(metrics.get("training_time_min", 0.0)),
    }


def _aggregate_metric_dict(entries):
    if not entries:
        return {}
    metrics = {}
    for key in entries[0].keys():
        values = [entry[key] for entry in entries if key in entry]
        metrics[key] = _mean_std(values)
    return metrics


def build_summary():
    summary = {
        "seeds": SEEDS,
        "benchmark": {},
        "baselines": {},
        "loco": {},
        "ablation": {},
        "single_seed_only": {},
    }

    for backbone in BENCHMARK_BACKBONES:
        per_seed = []
        for seed in SEEDS:
            metrics = _extract_benchmark_metrics(backbone, seed)
            if metrics is not None:
                per_seed.append({"seed": seed, **metrics})
        summary["benchmark"][backbone] = {
            "per_seed": per_seed,
            "aggregated": _aggregate_metric_dict(per_seed),
        }

    for model_name in BASELINES:
        per_seed = []
        for seed in SEEDS:
            metrics = _extract_baseline_metrics(model_name, seed)
            if metrics is not None:
                per_seed.append({"seed": seed, **metrics})
        summary["baselines"][model_name] = {
            "per_seed": per_seed,
            "aggregated": _aggregate_metric_dict(per_seed),
        }

    for backbone in LOCO_BACKBONES:
        per_seed = []
        for seed in SEEDS:
            metrics = _extract_loco_metrics(backbone, seed)
            if metrics is not None:
                per_seed.append({"seed": seed, **metrics})
        summary["loco"][backbone] = {
            "per_seed": per_seed,
            "aggregated": _aggregate_metric_dict(per_seed),
        }

    for config_name in ABLATION_CONFIGS:
        per_seed = []
        for seed in SEEDS:
            metrics = _extract_ablation_metrics(config_name, seed)
            if metrics is not None:
                per_seed.append({"seed": seed, **metrics})
        summary["ablation"][config_name] = {
            "per_seed": per_seed,
            "aggregated": _aggregate_metric_dict(per_seed),
        }

    for name in [
        "pillar1_optical_only_baseline",
        "pillar1_indian_sar_fusion",
        "pillar2_indian_simclr",
        "phase4_levir_siamese_efficientnet_b0",
    ]:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            summary["single_seed_only"][name] = _load_json(path)

    return summary


def write_markdown(summary):
    lines = [
        "# Phase 5.5 Multi-Seed Summary",
        "",
        f"Seeds: `{', '.join(str(s) for s in summary['seeds'])}`",
        "",
        "## Main Benchmark",
        "",
        "| Model | OA | F1 | mIoU |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model_name in BASELINES + BENCHMARK_BACKBONES:
        section = summary["baselines"].get(model_name) or summary["benchmark"].get(model_name)
        agg = section.get("aggregated", {}) if section else {}
        oa = agg.get("oa")
        f1 = agg.get("f1")
        miou = agg.get("miou")
        if oa and f1 and miou:
            lines.append(
                f"| {model_name} | {oa['mean']:.4f} ± {oa['std']:.4f} | "
                f"{f1['mean']:.4f} ± {f1['std']:.4f} | "
                f"{miou['mean']:.4f} ± {miou['std']:.4f} |"
            )

    lines.extend([
        "",
        "## LOCO",
        "",
        "| Model | OA | F1 | mIoU |",
        "| --- | ---: | ---: | ---: |",
    ])
    for backbone in LOCO_BACKBONES:
        agg = summary["loco"][backbone]["aggregated"]
        oa = agg.get("oa")
        f1 = agg.get("f1")
        miou = agg.get("miou")
        if oa and f1 and miou:
            lines.append(
                f"| {backbone} | {oa['mean']:.4f} ± {oa['std']:.4f} | "
                f"{f1['mean']:.4f} ± {f1['std']:.4f} | "
                f"{miou['mean']:.4f} ± {miou['std']:.4f} |"
            )

    lines.extend([
        "",
        "## Ablation",
        "",
        "| Config | OA | F1 | mIoU |",
        "| --- | ---: | ---: | ---: |",
    ])
    for config_name in ABLATION_CONFIGS:
        agg = summary["ablation"][config_name]["aggregated"]
        oa = agg.get("oa")
        f1 = agg.get("f1")
        miou = agg.get("miou")
        if oa and f1 and miou:
            lines.append(
                f"| {config_name} | {oa['mean']:.4f} ± {oa['std']:.4f} | "
                f"{f1['mean']:.4f} ± {f1['std']:.4f} | "
                f"{miou['mean']:.4f} ± {miou['std']:.4f} |"
            )

    return "\n".join(lines) + "\n"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    json_path = RESULTS_DIR / "multi_seed_summary.json"
    md_path = RESULTS_DIR / "multi_seed_summary.md"
    json_path.write_text(json.dumps(summary, indent=2))
    md_path.write_text(write_markdown(summary))
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
