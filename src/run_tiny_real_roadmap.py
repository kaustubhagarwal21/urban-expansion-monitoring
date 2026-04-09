"""
Tiny real-data roadmap runner.

Purpose:
- Finish a working end-to-end pass of the CLAUDE.md roadmap quickly.
- Use a very small real-data subset so training completes on an RTX 4070 Laptop GPU.
- Preserve earlier EuroSAT checkpoints before Indian-data runs overwrite them.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import MODEL_DIR, OUTPUT_DIR


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
LOG_PATH = ROOT / "outputs" / "tiny_real_roadmap.log"
SUMMARY_PATH = ROOT / "outputs" / "tiny_real_roadmap_summary.json"
EUROSAT_BACKUP_DIR = ROOT / "outputs" / "models" / "eurosat_phase_a_backup"
INDIAN_BACKUP_DIR = ROOT / "outputs" / "models" / "indian_tiny_run"

TINY_ENV = {
    "INDIAN_CITIES_FILTER": "Mumbai,Delhi_NCR",
    "INDIAN_MAX_PATCHES_PER_CITY": "50",
    "INTEGRATION_CITY_FILTER": "Mumbai,Delhi_NCR",
    "INTEGRATION_MAX_GEOTIFFS": "2",
}


def log(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name, args, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    log(f"START {name}: {' '.join(args)}")
    started = time.time()
    result = subprocess.run(args, cwd=ROOT, env=merged_env)
    elapsed = time.time() - started
    log(f"END   {name}: returncode={result.returncode} elapsed_min={elapsed / 60:.2f}")
    return {
        "name": name,
        "args": args,
        "returncode": result.returncode,
        "elapsed_sec": elapsed,
    }


def backup_checkpoints(dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in Path(MODEL_DIR).glob("*_best.pth"):
        dst = dest_dir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    log(f"Backed up {len(copied)} checkpoints to {dest_dir}")
    return copied


def main():
    summary = {
        "mode": "tiny_real_data",
        "env": TINY_ENV,
        "steps": [],
    }

    log("Tiny real-data roadmap started")
    summary["eurosat_backup"] = backup_checkpoints(EUROSAT_BACKUP_DIR)

    summary["steps"].append(
        run_step(
            "tiny_real_base_training",
            [
                PYTHON,
                "main.py",
                "--base-only",
                "--data-source",
                "real",
                "--real-dataset",
                "indian_cities",
                "--backbone",
                "efficientnet_b0",
                "--skip-siamese",
                "--epochs-override",
                "1",
            ],
            env=TINY_ENV,
        )
    )

    summary["indian_backup"] = backup_checkpoints(INDIAN_BACKUP_DIR)

    summary["steps"].append(
        run_step(
            "efficiency_benchmark",
            [PYTHON, "src/efficiency_benchmark.py"],
            env=TINY_ENV,
        )
    )

    summary["steps"].append(
        run_step(
            "integration_full",
            [PYTHON, "src/integration_pipeline.py", "--full", "--backbone", "efficientnet_b0"],
            env=TINY_ENV,
        )
    )

    summary["steps"].append(
        run_step(
            "paper_figures",
            [PYTHON, "src/paper_figures.py"],
            env=TINY_ENV,
        )
    )

    summary["steps"].append(
        run_step(
            "explainability",
            [
                PYTHON,
                "src/explainability.py",
                "--backbone",
                "efficientnet_b0",
                "--checkpoint",
                str(Path(MODEL_DIR) / "efficientnet_b0_best.pth"),
                "--num-samples",
                "100",
            ],
            env=TINY_ENV,
        )
    )

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
