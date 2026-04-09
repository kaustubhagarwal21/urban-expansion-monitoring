"""
Run the remaining locked local pipeline from Phase 5.5 through Phase 8.

This script is intentionally sequential and skip-aware:
  - it skips outputs that already exist
  - it uses the locked 3-city environment
  - it aggregates after each major chunk
  - it finishes by running Phase 6, Phase 7, and Phase 8 utilities
"""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "research_results"
MODELS_DIR = ROOT / "outputs" / "models"
LOCKED_CITIES = ["Mumbai", "Delhi_NCR", "Bangalore"]
BENCHMARK_BACKBONES = ["efficientnet_b0", "mobilenet_v3_small", "resnet50", "swin_tiny"]
LOCO_BACKBONES = ["efficientnet_b0", "resnet50", "swin_tiny"]
SEEDS = [123, 7]


def _env():
    env = os.environ.copy()
    env.setdefault("INDIAN_PATCH_ROOT", "data/indian_cities_locked")
    env.setdefault("INDIAN_CITIES_FILTER", ",".join(LOCKED_CITIES))
    env.setdefault("INTEGRATION_USE_LOCKED_SELECTION", "1")
    env.setdefault("INTEGRATION_CITY_FILTER", ",".join(LOCKED_CITIES))
    env.setdefault("INTEGRATION_YEAR_FILTER", "2019,2021,2023")
    env.setdefault("INTEGRATION_SEASON_FILTER", "pre_monsoon")
    env.setdefault("INTEGRATION_INCLUDE_S2", "1")
    env.setdefault("INTEGRATION_INCLUDE_LANDSAT", "1")
    return env


def _run(cmd, env):
    print("\n" + "=" * 80)
    print("[RUN]", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _exists(*parts):
    return (ROOT.joinpath(*parts)).exists()


def _bench_path(backbone, seed):
    return RESULTS_DIR / f"phase2_{backbone}_seed{seed}.json"


def _baseline_path(seed):
    return RESULTS_DIR / f"phase2_baselines_seed{seed}.json"


def _loco_path(backbone, seed):
    return RESULTS_DIR / f"phase4_loco_{backbone}_seed{seed}.json"


def _ablation_path(seed):
    return RESULTS_DIR / f"phase4_small_ablation_seed{seed}.json"


def main():
    py = sys.executable
    env = _env()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 5.5: benchmark + baselines
    for seed in SEEDS:
        if not _baseline_path(seed).exists():
            _run([py, "src/phase55_benchmark_seed.py", "--baselines", "--seed", str(seed)], env)

        for backbone in BENCHMARK_BACKBONES:
            if not _bench_path(backbone, seed).exists():
                _run(
                    [
                        py,
                        "src/phase55_benchmark_seed.py",
                        "--backbone",
                        backbone,
                        "--seed",
                        str(seed),
                        "--epochs-per-stage",
                        "5",
                    ],
                    env,
                )

        for backbone in LOCO_BACKBONES:
            if not _loco_path(backbone, seed).exists():
                _run(
                    [
                        py,
                        "src/phase4_real_loco.py",
                        "--backbone",
                        backbone,
                        "--seed",
                        str(seed),
                        "--epochs-per-stage",
                        "5",
                    ],
                    env,
                )

        if not _ablation_path(seed).exists():
            _run(
                [
                    py,
                    "src/phase4_small_ablation.py",
                    "--seed",
                    str(seed),
                    "--epochs-per-stage",
                    "5",
                ],
                env,
            )

        _run([py, "src/phase55_aggregate.py"], env)
        _run([py, "src/phase7_paper_outputs.py"], env)

    # Phase 6
    integration_summary = ROOT / "outputs" / "integration" / "pipeline_summary.json"
    if not integration_summary.exists():
        _run([py, "src/phase6_locked_integration.py", "--full"], env)

    # Phase 7
    efficiency_path = RESULTS_DIR / "efficiency_benchmark.json"
    if not efficiency_path.exists():
        _run([py, "src/efficiency_benchmark.py"], env)
    gradcam_path = ROOT / "outputs" / "figures" / "phase7_real_gradcam.png"
    if not gradcam_path.exists():
        _run([py, "src/phase7_real_gradcam.py"], env)
    _run([py, "src/phase7_paper_outputs.py"], env)

    # Phase 8
    _run([py, "src/phase8_paper_scaffold.py"], env)


if __name__ == "__main__":
    main()
