"""
Local Phase 5.5 queue runner.

Runs the extra-seed benchmark/LOCO/ablation steps with the correct locked
environment inside Python, avoiding fragile shell quoting.
"""

import argparse
import os
import subprocess
import sys


LOCKED_CITIES = ["Mumbai", "Delhi_NCR", "Bangalore"]


def _base_env():
    env = os.environ.copy()
    env.setdefault("INDIAN_PATCH_ROOT", "data/indian_cities_locked")
    env.setdefault("INDIAN_CITIES_FILTER", ",".join(LOCKED_CITIES))
    return env


def _run(cmd, env):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5.5 steps locally in sequence.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backbone", choices=["efficientnet_b0", "resnet50", "swin_tiny", "mobilenet_v3_small"], required=True)
    parser.add_argument("--epochs-per-stage", type=int, default=5)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-loco", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    env = _base_env()
    py = sys.executable

    if not args.skip_benchmark:
        _run(
            [
                py,
                "src/phase55_benchmark_seed.py",
                "--backbone",
                args.backbone,
                "--seed",
                str(args.seed),
                "--epochs-per-stage",
                str(args.epochs_per_stage),
            ],
            env,
        )

    if not args.skip_loco and args.backbone in {"efficientnet_b0", "resnet50", "swin_tiny"}:
        _run(
            [
                py,
                "src/phase4_real_loco.py",
                "--backbone",
                args.backbone,
                "--epochs-per-stage",
                str(args.epochs_per_stage),
                "--seed",
                str(args.seed),
            ],
            env,
        )

    if not args.skip_ablation and args.backbone == "efficientnet_b0":
        _run(
            [
                py,
                "src/phase4_small_ablation.py",
                "--epochs-per-stage",
                str(args.epochs_per_stage),
                "--seed",
                str(args.seed),
            ],
            env,
        )


if __name__ == "__main__":
    main()
