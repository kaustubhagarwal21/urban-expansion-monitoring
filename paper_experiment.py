"""
Paper-focused experiment runner for conference submission.

Compares three publishable configurations:
  1. Supervised optical baseline (EfficientNet-B0 + FPN on EuroSAT)
  2. Self-supervised optical pretraining + fine-tuning
  3. Multimodal SAR-optical fusion (synthetic fallback if So2Sat unavailable)

Also runs classical baselines (SVM, RF) on the same data split.

Outputs:
  - outputs/paper_experiment/results.json    (all metrics per seed)
  - outputs/paper_experiment/results.csv     (summary table)
  - outputs/paper_experiment/summary.txt     (human-readable)

Usage:
    python paper_experiment.py                          # Full (5 seeds, EuroSAT)
    python paper_experiment.py --quick                  # 2 seeds, 3 epochs
    python paper_experiment.py --skip-ssl --skip-fusion # Supervised only
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import (
    OUTPUT_DIR, SEED, BATCH_SIZE, STAGES, DEFAULT_BACKBONE, CLASS_NAMES,
)
from src.dataset import get_dataloaders
from src.baselines import extract_features
from src.metrics import evaluate
from src.train import progressive_train


PAPER_DIR = os.path.join(OUTPUT_DIR, "paper_experiment")
PAPER_METRICS = ["oa", "precision", "recall", "f1", "miou"]


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _clean(metrics):
    return {k: float(metrics[k]) for k in PAPER_METRICS if k in metrics}


def _agg(results_list):
    agg = {}
    for k in PAPER_METRICS:
        vals = [r[k] for r in results_list if k in r]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def save_results(results, seeds):
    os.makedirs(PAPER_DIR, exist_ok=True)

    # JSON
    json_path = os.path.join(PAPER_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV
    csv_path = os.path.join(PAPER_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "metric", "mean", "std", "per_seed_values"])
        for exp_name, exp_data in results.items():
            agg = exp_data.get("aggregated", {})
            for m in PAPER_METRICS:
                if m in agg:
                    writer.writerow([
                        exp_name, m,
                        f"{agg[m]['mean']:.4f}",
                        f"{agg[m]['std']:.4f}",
                        str(agg[m].get("values", [])),
                    ])

    # Human-readable summary
    summary_path = os.path.join(PAPER_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("PAPER EXPERIMENT RESULTS\n")
        f.write(f"Seeds: {seeds}\n")
        f.write("=" * 70 + "\n\n")

        fmt = "{:<30} {:>10} {:>10} {:>10}\n"
        f.write(fmt.format("Experiment", "OA (%)", "F1", "mIoU"))
        f.write("-" * 65 + "\n")

        for exp_name, exp_data in results.items():
            agg = exp_data.get("aggregated", {})
            if "oa" in agg:
                oa = f"{agg['oa']['mean']*100:.1f}+/-{agg['oa']['std']*100:.1f}"
                f1 = f"{agg['f1']['mean']:.3f}+/-{agg['f1']['std']:.3f}"
                miou = f"{agg['miou']['mean']:.3f}+/-{agg['miou']['std']:.3f}"
                f.write(fmt.format(exp_name, oa, f1, miou))

    print(f"  Saved: {json_path}")
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {summary_path}")


def run_paper_experiment(
    seeds=(42, 123, 456, 789, 2024),
    backbone=DEFAULT_BACKBONE,
    data_source="real",
    real_dataset="eurosat",
    stages=None,
    skip_ssl=False,
    skip_fusion=False,
    skip_baselines=False,
    download=False,
):
    device = get_device()
    if stages is None:
        stages = copy.deepcopy(STAGES)

    print("=" * 70)
    print("  PAPER EXPERIMENT SUITE")
    print(f"  Seeds: {list(seeds)}")
    print(f"  Backbone: {backbone}")
    print(f"  Data: {data_source} ({real_dataset})")
    print(f"  Device: {device}")
    print("=" * 70)

    results = {}

    # ── 1. Supervised optical ─────────────────────────
    print("\n--- Experiment: Supervised Optical ---")
    sup_raw = []
    for seed in seeds:
        set_all_seeds(seed)
        loaders = get_dataloaders(
            batch_size=BATCH_SIZE, data_source=data_source,
            real_dataset=real_dataset, download=download,
        )
        _, _, metrics, t = progressive_train(
            backbone_name=backbone, device=device, loaders=loaders,
        )
        clean = _clean(metrics)
        clean["time_sec"] = t
        sup_raw.append(clean)
        print(f"  seed={seed}: OA={clean['oa']:.4f} F1={clean['f1']:.4f} mIoU={clean['miou']:.4f}")

    results["Supervised Optical"] = {
        "per_seed": sup_raw, "aggregated": _agg(sup_raw),
    }

    # ── 2. Classical baselines ────────────────────────
    if not skip_baselines:
        print("\n--- Experiment: Classical Baselines ---")
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        svm_raw, rf_raw = [], []
        for seed in seeds:
            set_all_seeds(seed)
            loaders = get_dataloaders(
                batch_size=BATCH_SIZE, data_source=data_source,
                real_dataset=real_dataset, download=download,
            )
            X_tr, y_tr = extract_features(loaders[0].dataset, max_samples=5000)
            X_te, y_te = extract_features(loaders[2].dataset, max_samples=2000)
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)
            svm.fit(X_tr, y_tr)
            svm_m = _clean(evaluate(y_te.tolist(), svm.predict(X_te).tolist(), CLASS_NAMES))
            svm_raw.append(svm_m)

            rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                        random_state=seed, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            rf_m = _clean(evaluate(y_te.tolist(), rf.predict(X_te).tolist(), CLASS_NAMES))
            rf_raw.append(rf_m)

            print(f"  seed={seed}: SVM OA={svm_m['oa']:.4f} | RF OA={rf_m['oa']:.4f}")

        results["SVM"] = {"per_seed": svm_raw, "aggregated": _agg(svm_raw)}
        results["Random Forest"] = {"per_seed": rf_raw, "aggregated": _agg(rf_raw)}

    # ── 3. Self-supervised ────────────────────────────
    if not skip_ssl:
        print("\n--- Experiment: Self-Supervised + Fine-tune ---")
        from src.pillar2_self_supervised import run_self_supervised_pipeline
        ssl_raw = []
        for seed in seeds:
            set_all_seeds(seed)
            loaders = get_dataloaders(
                batch_size=BATCH_SIZE, data_source=data_source,
                real_dataset=real_dataset, download=download,
            )
            _, _, metrics = run_self_supervised_pipeline(
                device=device,
                pretrain_epochs=stages[0]["epochs"],
                finetune_epochs=stages[0]["epochs"],
                loaders=loaders,
                unlabeled_dataset=loaders[0].dataset,
            )
            clean = _clean(metrics)
            ssl_raw.append(clean)
            print(f"  seed={seed}: OA={clean['oa']:.4f} F1={clean['f1']:.4f}")

        results["Self-Supervised"] = {
            "per_seed": ssl_raw, "aggregated": _agg(ssl_raw),
        }

    # ── 4. Multimodal fusion ─────────────────────────
    if not skip_fusion:
        print("\n--- Experiment: Multimodal Fusion ---")
        from src.pillar1_sar_fusion import train_fusion
        fusion_raw = []
        for seed in seeds:
            set_all_seeds(seed)
            _, _, metrics = train_fusion(
                device=device, epochs=stages[0]["epochs"],
                data_source="synthetic",  # So2Sat not available
            )
            clean = _clean(metrics)
            fusion_raw.append(clean)
            print(f"  seed={seed}: OA={clean['oa']:.4f} F1={clean['f1']:.4f}")

        results["Multimodal Fusion"] = {
            "per_seed": fusion_raw, "aggregated": _agg(fusion_raw),
        }

    save_results(results, list(seeds))

    # Print summary
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY (mean +/- std)")
    print(f"{'='*70}")
    for name, data in results.items():
        agg = data["aggregated"]
        if "oa" in agg:
            print(f"  {name:<30} OA={agg['oa']['mean']*100:.1f}+/-{agg['oa']['std']*100:.1f}%  "
                  f"F1={agg['f1']['mean']:.3f}+/-{agg['f1']['std']:.3f}  "
                  f"mIoU={agg['miou']['mean']:.3f}+/-{agg['miou']['std']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Paper experiment suite")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    parser.add_argument("--quick", action="store_true", help="2 seeds, 3 epochs")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--data-source", default="real", choices=["synthetic", "real"])
    parser.add_argument("--real-dataset", default="eurosat")
    parser.add_argument("--skip-ssl", action="store_true")
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    stages = copy.deepcopy(STAGES)
    if args.quick:
        args.seeds = [42, 123]
        for s in stages:
            s["epochs"] = 3

    run_paper_experiment(
        seeds=args.seeds,
        backbone=args.backbone,
        data_source=args.data_source,
        real_dataset=args.real_dataset,
        stages=stages,
        skip_ssl=args.skip_ssl,
        skip_fusion=args.skip_fusion,
        skip_baselines=args.skip_baselines,
        download=args.download,
    )


if __name__ == "__main__":
    main()
