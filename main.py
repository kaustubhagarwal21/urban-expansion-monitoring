"""
Urban Expansion Monitoring Using Transfer Learning on Historical Satellite Imagery
==================================================================================

Main entry point. Runs the full pipeline:

BASE PIPELINE:
  1. Generate visualizations (sample patches, architecture, growth patterns)
  2. Train classical ML baselines (SVM, Random Forest)
  3. Train deep learning models (VGG16, ResNet50, EfficientNet-B0) with progressive fine-tuning
  4. Train Siamese change-detection network
  5. Compare all models and generate result plots

EXTENDED PILLARS (The Living Map):
  6. Pillar I   — Multi-Modal Fusion (Optical + SAR)
  7. Pillar II  — Self-Supervised Pre-Training (SimCLR)
  8. Pillar III — High-Resolution Analysis (Sub-Metre)
  9. Pillar IV  — Predictive Socio-Economic Modelling (LSTM + Attention)
 10. Pillar V   — Real-Time Monitoring & Alert System

Usage:
    python main.py                          # Full pipeline (base + all pillars)
    python main.py --base-only              # Only base pipeline (steps 1-5)
    python main.py --pillars-only           # Only extended pillars (steps 6-10)
    python main.py --pillar 1 3 5           # Run specific pillars
    python main.py --backbone efficientnet_b0
    python main.py --baselines-only
    python main.py --visualize-only
    python main.py --epochs-override 2      # Quick test with 2 epochs per stage
"""

import argparse, os, sys, time, torch, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs.config import *
from src.dataset import get_dataloaders
from src.train import progressive_train, train_siamese
from src.baselines import run_baselines
from src.visualize import (plot_training_curves, plot_confusion_matrix,
                           plot_model_comparison, plot_urban_growth,
                           plot_sample_patches, plot_architecture_diagram)


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("  Using CPU (training will be slower)")
    return device


def run_base_pipeline(args, device):
    """Steps 1-5: Base pipeline (visualizations, baselines, DL models, Siamese, comparison)."""
    all_results = {}
    data_loaders = None

    if args.data_source == "real":
        data_loaders = get_dataloaders(
            batch_size=BATCH_SIZE,
            data_source="real",
            real_dataset=args.real_dataset,
            download=args.download_real_data,
        )

    # ── Step 1: Visualizations ─────────────────────────
    print("\n[1/5] Generating visualizations...")
    plot_sample_patches()
    plot_architecture_diagram()
    plot_urban_growth()

    if args.visualize_only:
        print("\nDone (visualize-only mode).")
        return None

    # ── Step 2: Classical Baselines ────────────────────
    print("\n[2/5] Training classical baselines...")
    if data_loaders is not None:
        # Use the same train/test splits from real data
        baseline_results = run_baselines(
            train_dataset=data_loaders[0].dataset,
            test_dataset=data_loaders[2].dataset,
        )
    else:
        baseline_results = run_baselines()

    if args.baselines_only:
        print("\nDone (baselines-only mode).")
        return None

    for name, metrics in baseline_results.items():
        all_results[name] = metrics

    # ── Step 3: Deep Learning Models ───────────────────
    backbones = [args.backbone] if args.backbone else BACKBONES
    dl_histories = {}
    dl_times = {}

    for i, backbone_name in enumerate(backbones):
        print(f"\n[3/5] Training backbone {i + 1}/{len(backbones)}: {backbone_name}")
        model, history, test_metrics, total_time = progressive_train(
            backbone_name=backbone_name, device=device, loaders=data_loaders
        )
        all_results[backbone_name] = test_metrics
        dl_histories[backbone_name] = history
        dl_times[backbone_name] = total_time

        plot_training_curves(history, backbone_name)
        plot_confusion_matrix(
            test_metrics["confusion_matrix"],
            CLASS_NAMES,
            f"Confusion Matrix - {backbone_name}",
        )

    # ── Step 4: Siamese Change Detection ───────────────
    if not args.skip_siamese:
        if args.data_source == "real":
            print("\n[4/5] Skipping Siamese training in real-data mode (no paired real change dataset wired yet).")
        else:
            print("\n[4/5] Training Siamese change detector...")
            siamese_backbone = args.backbone or DEFAULT_BACKBONE
            train_siamese(backbone_name=siamese_backbone, device=device,
                          epochs=args.epochs_override or 20)
    else:
        print("\n[4/5] Skipping Siamese training.")

    # ── Step 5: Model Comparison ───────────────────────
    print("\n[5/5] Generating comparison plots...")
    plot_model_comparison(all_results)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  BASE PIPELINE RESULTS")
    print(f"{'='*70}")
    header = f"{'Model':<20} {'OA':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'mIoU':>8}"
    print(header)
    print("-" * 70)
    for name, m in all_results.items():
        row = (f"{name:<20} {m['oa']:>8.4f} {m['precision']:>8.4f} "
               f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['miou']:>8.4f}")
        print(row)
    print("-" * 70)

    if dl_times:
        print(f"\n  Training Times:")
        for name, t in dl_times.items():
            print(f"    {name}: {t / 60:.1f} min")

    return all_results, dl_histories, dl_times


def run_pillar1(device, epochs, args):
    """Pillar I: Multi-Modal Fusion (Optical + SAR)."""
    from src.pillar1_sar_fusion import train_fusion
    fusion_source = "real" if args.data_source == "real" and args.real_dataset == "so2sat" else "synthetic"
    model, history, test_metrics = train_fusion(
        device=device, epochs=epochs, data_source=fusion_source
    )
    return {"model": model, "history": history, "metrics": test_metrics}


def run_pillar2(device, epochs, args):
    """Pillar II: Self-Supervised Pre-Training."""
    from src.pillar2_self_supervised import run_self_supervised_pipeline
    ssl_loaders = None
    unlabeled_dataset = None

    if args.data_source == "real" and args.real_dataset == "eurosat":
        ssl_loaders = get_dataloaders(
            batch_size=BATCH_SIZE,
            data_source="real",
            real_dataset="eurosat",
            download=args.download_real_data,
        )
        unlabeled_dataset = ssl_loaders[0].dataset

    classifier, history, test_metrics = run_self_supervised_pipeline(
        device=device, pretrain_epochs=epochs, finetune_epochs=epochs,
        loaders=ssl_loaders, unlabeled_dataset=unlabeled_dataset,
    )
    return {"model": classifier, "history": history, "metrics": test_metrics}


def run_pillar3(device, epochs):
    """Pillar III: High-Resolution Analysis."""
    from src.pillar3_high_resolution import train_high_res
    model, history, test_metrics = train_high_res(device=device, epochs=epochs)
    return {"model": model, "history": history, "metrics": test_metrics}


def run_pillar4(device, epochs):
    """Pillar IV: Predictive Socio-Economic Modelling."""
    from src.pillar4_predictive import train_predictor
    model, history, forecasts, pred_metrics = train_predictor(
        device=device, epochs=epochs
    )
    return {"model": model, "history": history, "forecasts": forecasts, "metrics": pred_metrics}


def run_pillar5(device, epochs):
    """Pillar V: Real-Time Monitoring & Alert System."""
    from src.pillar5_realtime import train_realtime_detector, simulate_monitoring, benchmark_latency
    model, history, alert_engine, change_acc = train_realtime_detector(
        device=device, epochs=epochs
    )
    # Benchmark latency
    latency_stats = benchmark_latency(model, device)
    return {
        "model": model, "history": history,
        "alert_engine": alert_engine, "change_acc": change_acc,
        "latency": latency_stats,
    }


PILLAR_RUNNERS = {
    1: ("Pillar I:   Multi-Modal Fusion (Optical + SAR)", run_pillar1),
    2: ("Pillar II:  Self-Supervised Pre-Training", run_pillar2),
    3: ("Pillar III: High-Resolution Analysis", run_pillar3),
    4: ("Pillar IV:  Predictive Modelling (LSTM + Attention)", run_pillar4),
    5: ("Pillar V:   Real-Time Monitoring & Alerts", run_pillar5),
}


def run_pillars(pillars_to_run, device, epochs, args):
    """Run specified pillars and collect results."""
    pillar_results = {}

    for num in pillars_to_run:
        name, runner = PILLAR_RUNNERS[num]
        print(f"\n{'='*60}")
        print(f"  [{num}/5] {name}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            result = runner(device, epochs, args) if num in [1, 2] else runner(device, epochs)
            elapsed = time.time() - t0
            result["time"] = elapsed
            pillar_results[num] = result
            print(f"\n  Pillar {num} completed in {elapsed / 60:.1f} min")
        except Exception as e:
            print(f"\n  Pillar {num} FAILED: {e}")
            import traceback
            traceback.print_exc()
            pillar_results[num] = {"error": str(e)}

    return pillar_results


def print_pillar_summary(pillar_results):
    """Print summary of all pillar results."""
    print(f"\n{'='*70}")
    print(f"  EXTENDED PILLARS — SUMMARY")
    print(f"{'='*70}")

    for num in sorted(pillar_results.keys()):
        name = PILLAR_RUNNERS[num][0]
        result = pillar_results[num]

        if "error" in result:
            print(f"\n  {name}")
            print(f"    Status: FAILED — {result['error']}")
            continue

        print(f"\n  {name}")
        elapsed = result.get("time", 0)
        print(f"    Time: {elapsed / 60:.1f} min")

        metrics = result.get("metrics", {})
        if "oa" in metrics:
            print(f"    OA={metrics['oa']:.4f} | F1={metrics['f1']:.4f} | mIoU={metrics['miou']:.4f}")
        elif "mae" in metrics:
            print(f"    MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | R²={metrics['r2']:.4f}")

        if "change_acc" in result:
            print(f"    Change Detection Acc: {result['change_acc']:.4f}")

        if "latency" in result and result["latency"]:
            lat = result["latency"]
            print(f"    Inference Latency: {lat.get('mean_ms', 0):.1f} ms (mean)")

        if "forecasts" in result:
            print(f"    Forecasts generated for {len(result['forecasts'])} cities")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Urban Expansion Monitoring — The Living Map")
    parser.add_argument("--backbone", type=str, default=None,
                        choices=BACKBONES, help="Train single backbone only")
    parser.add_argument("--baselines-only", action="store_true",
                        help="Run only classical ML baselines")
    parser.add_argument("--visualize-only", action="store_true",
                        help="Generate visualizations only")
    parser.add_argument("--skip-siamese", action="store_true",
                        help="Skip Siamese change-detection training")
    parser.add_argument("--base-only", action="store_true",
                        help="Run only the base pipeline (steps 1-5)")
    parser.add_argument("--pillars-only", action="store_true",
                        help="Run only the extended pillars (steps 6-10)")
    parser.add_argument("--pillar", type=int, nargs="+", choices=[1, 2, 3, 4, 5],
                        help="Run specific pillars (e.g., --pillar 1 3 5)")
    parser.add_argument("--epochs-override", type=int, default=None,
                        help="Override epochs per stage (for quick testing)")
    parser.add_argument("--data-source", type=str, default=DATA_SOURCE,
                        choices=["synthetic", "real"],
                        help="Choose whether to train on synthetic or supported real data")
    parser.add_argument("--real-dataset", type=str, default=REAL_DATASET,
                        choices=["eurosat", "so2sat", "spacenet", "indian_cities", "so2sat_classification"],
                        help="Real dataset to use when --data-source real")
    parser.add_argument("--download-real-data", action="store_true",
                        help="Allow supported real datasets to be downloaded when possible")
    parser.add_argument("--cross-city", action="store_true",
                        help="Run cross-city generalization experiments (LOCO)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study (multi-seed)")
    parser.add_argument("--ablation-seeds", type=int, default=3,
                        help="Number of seeds for ablation study")
    args = parser.parse_args()

    set_seed()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("  URBAN EXPANSION MONITORING — THE LIVING MAP")
    print("  Transfer Learning on Historical Satellite Imagery")
    print("  Study Area: Indian Metropolitan Regions")
    print(f"  Cities: {', '.join(CITIES)}")
    print("=" * 60)

    device = get_device()
    pillar_epochs = args.epochs_override or 10

    # Override base pipeline epochs
    if args.epochs_override:
        for s in STAGES:
            s["epochs"] = args.epochs_override

    global_start = time.time()
    all_results = {}

    # ── Base Pipeline ────────────────────────────────────
    if not args.pillars_only:
        base_out = run_base_pipeline(args, device)
        if base_out is not None:
            all_results, dl_histories, dl_times = base_out

        if args.visualize_only or args.baselines_only:
            return

    # ── Extended Pillars ─────────────────────────────────
    if not args.base_only and not args.visualize_only and not args.baselines_only:
        pillars_to_run = args.pillar if args.pillar else [1, 2, 3, 4, 5]
        pillar_results = run_pillars(pillars_to_run, device, pillar_epochs, args)
        print_pillar_summary(pillar_results)

        # Add pillar classification results to comparison
        pillar_class_names = {1: "Fusion (Opt+SAR)", 2: "Self-Supervised", 3: "High-Res"}
        for num in [1, 2, 3]:
            if num in pillar_results and "metrics" in pillar_results[num] and "oa" in pillar_results[num].get("metrics", {}):
                all_results[pillar_class_names[num]] = pillar_results[num]["metrics"]

    # ── Cross-City Generalization ──────────────────────────
    if args.cross_city:
        print(f"\n{'='*60}")
        print("  CROSS-CITY GENERALIZATION EXPERIMENTS")
        print(f"{'='*60}")
        from src.cross_city_generalization import run_cross_city_experiments
        cross_city_results = run_cross_city_experiments(device)

    # ── Ablation Study ──────────────────────────────────
    if args.ablation:
        print(f"\n{'='*60}")
        print("  ABLATION STUDY")
        print(f"{'='*60}")
        from src.ablation_study import run_ablation_study
        ablation_results = run_ablation_study(device, num_seeds=args.ablation_seeds)

    # ── Final Summary ────────────────────────────────────
    total_time = time.time() - global_start

    if all_results:
        # Save all results
        results_path = os.path.join(OUTPUT_DIR, "results.json")
        serializable = {}
        for name, m in all_results.items():
            serializable[name] = {
                k: float(v) for k, v in m.items()
                if isinstance(v, (int, float, np.floating))
            }
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results saved to {results_path}")

        # Final comparison table
        print(f"\n{'='*70}")
        print(f"  FINAL RESULTS — ALL MODELS")
        print(f"{'='*70}")
        header = f"{'Model':<25} {'OA':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'mIoU':>8}"
        print(header)
        print("-" * 73)
        for name, m in all_results.items():
            if "oa" in m:
                row = (f"{name:<25} {m['oa']:>8.4f} {m.get('precision', 0):>8.4f} "
                       f"{m.get('recall', 0):>8.4f} {m['f1']:>8.4f} {m['miou']:>8.4f}")
                print(row)
        print("-" * 73)

    print(f"\n  Total pipeline time: {total_time / 60:.1f} min")
    print(f"  Figures saved to {FIGURE_DIR}")
    print(f"  Models saved to {MODEL_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
