"""
Phase 8 paper scaffold generator.

Creates a working markdown skeleton for:
  - novelty statement
  - reproducibility section
  - current result highlights
  - remaining paper fill-ins (SOTA comparison, temporal validation)
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import OUTPUT_DIR


RESULTS_DIR = Path(OUTPUT_DIR) / "research_results"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _maybe_load_json(path):
    return _load_json(path) if os.path.exists(path) else None


def main():
    phase2 = _maybe_load_json(RESULTS_DIR / "phase2_benchmark_summary.json")
    multi_seed = _maybe_load_json(RESULTS_DIR / "multi_seed_summary.json")
    loco_eff = _maybe_load_json(RESULTS_DIR / "phase4_loco_efficientnet_b0.json")
    loco_res = _maybe_load_json(RESULTS_DIR / "phase4_loco_resnet50.json")
    loco_swin = _maybe_load_json(RESULTS_DIR / "phase4_loco_swin_tiny.json")
    ablation = _maybe_load_json(RESULTS_DIR / "phase4_small_ablation.json")
    pillar1 = _maybe_load_json(RESULTS_DIR / "pillar1_indian_sar_fusion.json")
    pillar2 = _maybe_load_json(RESULTS_DIR / "pillar2_indian_simclr.json")
    levir = _maybe_load_json(RESULTS_DIR / "phase4_levir_siamese_efficientnet_b0.json")

    lines = [
        "# Phase 8 Paper Scaffold",
        "",
        "## Working Contribution Statement",
        "",
        "This paper presents a research-grade urban expansion monitoring pipeline for Indian metropolitan regions that combines transfer-learning-based land-cover classification, cross-city generalization analysis, multimodal optical+SAR comparison, self-supervised pretraining analysis, predictive sprawl modelling, and alert-oriented downstream integration.",
        "",
        "## Draft Novelty Statement",
        "",
        "1. We build a locked 3-city Indian benchmark (`Mumbai`, `Delhi_NCR`, `Bangalore`) from real Google Earth Engine exports rather than relying only on synthetic or non-Indian proxy datasets.",
        "2. We evaluate a balanced model suite spanning classical baselines (`SVM`, `Random Forest`), efficient CNNs (`EfficientNet-B0`, `MobileNetV3-Small`), a standard CNN baseline (`ResNet50`), and a transformer (`Swin-Tiny`).",
        "3. We move beyond in-city accuracy by measuring leave-one-city-out transfer, which exposes meaningful domain shift across Indian urban morphologies.",
        "4. We connect the classification stage to downstream forecasting and alerting, preserving an end-to-end operational story rather than a disconnected benchmark-only paper.",
        "",
        "## Current Result Highlights",
        "",
    ]

    if phase2:
        best = max(phase2["models"], key=lambda item: item["oa"])
        lines.extend([
            f"- Best Phase 2 benchmark model so far: `{best['model']}` with `OA={best['oa']:.4f}`, `F1={best['f1']:.4f}`, `mIoU={best['miou']:.4f}`.",
            f"- Benchmark dataset: `{phase2['dataset']}` with `{phase2['total_patches']}` total patches across `{', '.join(phase2['cities'])}`.",
        ])
    if multi_seed:
        lines.append("- Multi-seed aggregation is available and should be used for the final camera-ready tables.")
    else:
        lines.append("- Multi-seed aggregation is still in progress; current paper tables are using single-seed fallback values.")
    if loco_res and loco_eff and loco_swin:
        lines.extend([
            f"- LOCO averages: `ResNet50 OA={loco_res['average']['oa']:.4f}`, `EfficientNet-B0 OA={loco_eff['average']['oa']:.4f}`, `Swin-Tiny OA={loco_swin['average']['oa']:.4f}`.",
        ])
    if ablation:
        lines.extend([
            f"- Small ablation currently shows `full F1={ablation['results']['full']['f1']:.4f}`, `no_fpn F1={ablation['results']['no_fpn']['f1']:.4f}`, `ce_only F1={ablation['results']['ce_only']['f1']:.4f}`.",
        ])
    if pillar1:
        lines.extend([
            f"- Pillar I optical+SAR result: `OA={pillar1['test_metrics']['oa']:.4f}`, `F1={pillar1['test_metrics']['f1']:.4f}`, `mIoU={pillar1['test_metrics']['miou']:.4f}`.",
        ])
    if pillar2:
        lines.extend([
            f"- Pillar II comparison: `ImageNet-init OA={pillar2['imagenet_init']['oa']:.4f}` vs `SimCLR OA={pillar2['simclr_pretrain']['oa']:.4f}`.",
        ])
    if levir:
        lines.extend([
            f"- External LEVIR-CD Siamese benchmark: `OA={levir['oa']:.4f}`, `F1={levir['f1']:.4f}`.",
        ])

    lines.extend([
        "",
        "## Reproducibility Section Draft",
        "",
        "### Code + Environment",
        "- OS: Windows 11",
        "- Primary local GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM)",
        "- Python runtime used for training: Python 3.11 with PyTorch 2.5.1+cu121",
        "- Core entry points: `main.py`, `src/phase4_real_loco.py`, `src/phase4_small_ablation.py`, `src/phase4_levir_cd.py`, `src/integration_pipeline.py`",
        "",
        "### Data",
        "- Locked Indian patch dataset: `data/indian_cities_locked`",
        "- External change benchmark: `data/levir_cd`",
        "- Raw satellite source directory: `G:/My Drive/urban_expansion_india`",
        "",
        "### Output Artifacts",
        "- Phase 2 benchmark summary: `outputs/research_results/phase2_benchmark_summary.json`",
        "- Phase 4 LOCO: `outputs/research_results/phase4_loco_*.json`",
        "- Phase 4 ablation: `outputs/research_results/phase4_small_ablation.json`",
        "- Pillar results: `outputs/research_results/pillar1_*.json`, `outputs/research_results/pillar2_indian_simclr.json`",
        "- Phase 7 tables: `outputs/research_results/table*.csv`",
        "",
        "## Remaining Paper Fill-ins",
        "",
        "1. Add published SOTA comparison numbers and citations for the target venue.",
        "2. Replace any single-seed table entries with `mean ± std` once Phase 5.5 completes.",
        "3. Add Phase 6 forecast/alert quantitative summaries after integration outputs are generated.",
        "4. Add temporal validation discussion comparing forecasted 2023 expansion against observed 2023 satellite-derived urban area.",
        "",
        "## Targeted Story Arc",
        "",
        "1. Real Indian benchmark performance establishes the core classifier comparison.",
        "2. LOCO shows transfer difficulty and domain shift across Indian cities.",
        "3. Pillar I and II test whether additional modality or pretraining meaningfully help.",
        "4. LEVIR-CD and downstream forecasting/alerts demonstrate that the system is not benchmark-only.",
        "",
    ])

    out_path = RESULTS_DIR / "phase8_paper_scaffold.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved paper scaffold to {out_path}")


if __name__ == "__main__":
    main()
