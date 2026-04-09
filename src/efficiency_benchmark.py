"""
Efficiency Benchmarking for All Models
=======================================
Generates Table 4 for the paper: params, FLOPs, latency, GPU memory per model.

Usage:
    python src/efficiency_benchmark.py
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from configs.config import *
from src.models import UrbanClassifier, BACKBONE_BUILDERS

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


ACTIVE_EFFICIENCY_MODELS = [
    "resnet50",
    "efficientnet_b0",
    "swin_tiny",
    "mobilenet_v3_small",
]


def _resolve_model_names(model_names=None):
    if model_names:
        return list(model_names)
    raw = os.environ.get("EFFICIENCY_MODELS", "").strip()
    if raw:
        parsed = [name.strip() for name in raw.split(",") if name.strip()]
        if parsed:
            return parsed
    return list(ACTIVE_EFFICIENCY_MODELS)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, input_size, device="cpu"):
    """Estimate FLOPs using fvcore or fallback to parameter-based estimate."""
    x = torch.randn(1, NUM_CHANNELS, input_size, input_size).to(device)
    model = model.to(device)
    model.eval()

    if HAS_FVCORE:
        flops = FlopCountAnalysis(model, x)
        return flops.total()

    # Fallback: estimate from forward pass time ratio
    # (rough but better than nothing)
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 2  # Very rough: ~2 FLOPs per parameter per forward pass


def benchmark_latency(model, device, input_size=256, n_warmup=10, n_runs=50):
    """Measure inference latency (ms) and throughput (patches/sec)."""
    model = model.to(device)
    model.eval()

    x = torch.randn(1, NUM_CHANNELS, input_size, input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "throughput": float(1000 / np.mean(times)),  # patches/sec
    }


def measure_gpu_memory(model, device, input_size=256, batch_size=16):
    """Measure peak GPU memory during forward + backward pass."""
    if device != "cuda":
        return {"peak_mb": 0, "model_mb": 0}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = model.to(device)
    model.train()

    # Model memory
    model_mem = torch.cuda.memory_allocated() / 1e6

    x = torch.randn(batch_size, NUM_CHANNELS, input_size, input_size).to(device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)

    # Forward + backward
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    del x, y, logits, loss
    torch.cuda.empty_cache()

    return {
        "peak_mb": float(peak_mem),
        "model_mb": float(model_mem),
        "training_mb": float(peak_mem - model_mem),
    }


def run_full_benchmark(device="cuda", model_names=None):
    """Benchmark all models and generate comparison table."""
    model_names = _resolve_model_names(model_names)
    print(f"\n{'='*70}")
    print(f"  EFFICIENCY BENCHMARKING")
    print(f"{'='*70}")
    print(f"  Models: {', '.join(model_names)}")

    results = {}

    for name in model_names:
        if name not in BACKBONE_BUILDERS:
            print(f"\n  Skipping unknown model: {name}")
            continue
        print(f"\n  Benchmarking {name}...")
        try:
            model = UrbanClassifier(name, pretrained=False)
            input_size = PATCH_SIZE

            # Parameters
            total_params, trainable_params = count_parameters(model)

            # FLOPs
            flops = estimate_flops(model, input_size, device="cpu")

            # Latency
            latency = benchmark_latency(model, device, input_size)

            # GPU Memory
            memory = measure_gpu_memory(model, device, input_size)

            results[name] = {
                "total_params": total_params,
                "total_params_m": total_params / 1e6,
                "trainable_params_m": trainable_params / 1e6,
                "flops": flops,
                "gflops": flops / 1e9,
                "latency_ms": latency["mean_ms"],
                "latency_std_ms": latency["std_ms"],
                "throughput": latency["throughput"],
                "peak_gpu_mb": memory["peak_mb"],
                "model_gpu_mb": memory["model_mb"],
            }

            print(f"    Params: {total_params/1e6:.1f}M | "
                  f"GFLOPs: {flops/1e9:.2f} | "
                  f"Latency: {latency['mean_ms']:.1f}ms | "
                  f"GPU: {memory['peak_mb']:.0f}MB")

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    FAILED: {e}")
            results[name] = {"error": str(e)}

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"  {'Model':<22} {'Params(M)':>10} {'GFLOPs':>8} {'Latency(ms)':>12} "
          f"{'Throughput':>12} {'GPU Peak(MB)':>13}")
    print(f"  {'-'*85}")

    for name, r in results.items():
        if "error" in r:
            continue
        print(f"  {name:<22} {r['total_params_m']:>10.1f} {r['gflops']:>8.2f} "
              f"{r['latency_ms']:>9.1f} +/- {r['latency_std_ms']:.1f} "
              f"{r['throughput']:>10.1f}/s {r['peak_gpu_mb']:>11.0f}")

    print(f"  {'-'*85}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "efficiency_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    research_out_path = os.path.join(OUTPUT_DIR, "research_results", "efficiency_benchmark.json")
    os.makedirs(os.path.dirname(research_out_path), exist_ok=True)
    with open(research_out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Research copy saved to {research_out_path}")

    # Generate figure
    _plot_efficiency(results)

    return results


def _plot_efficiency(results):
    """Generate efficiency comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return

    names = list(valid.keys())
    params = [valid[n]["total_params_m"] for n in names]
    latencies = [valid[n]["latency_ms"] for n in names]
    memories = [valid[n]["peak_gpu_mb"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Bar chart: parameters
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    axes[0].barh(names, params, color=colors)
    axes[0].set_xlabel("Parameters (M)")
    axes[0].set_title("Model Size")
    axes[0].invert_yaxis()

    # Bar chart: latency
    axes[1].barh(names, latencies, color=colors)
    axes[1].set_xlabel("Inference Latency (ms)")
    axes[1].set_title("Inference Speed")
    axes[1].invert_yaxis()

    # Bar chart: GPU memory
    axes[2].barh(names, memories, color=colors)
    axes[2].set_xlabel("Peak GPU Memory (MB)")
    axes[2].set_title("Memory Usage")
    axes[2].invert_yaxis()

    plt.suptitle("Model Efficiency Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(FIGURE_DIR, exist_ok=True)
    fig_path = os.path.join(FIGURE_DIR, "efficiency_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved to {fig_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_full_benchmark(device)
