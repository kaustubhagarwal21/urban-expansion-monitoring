"""
Prepare the locked 3-city Indian subset for the current research pipeline.

This script follows the agreed shortlist in README.md:
  - Cities: Mumbai, Delhi_NCR, Bangalore
  - Main classifier imagery: 2021 / 2019 / 2023 Sentinel-2 pre-monsoon
  - Labels: WorldCover (primary), Dynamic World (secondary inventory)
  - SAR and Landsat anchors are tracked in the inventory but not patch-extracted here

Outputs:
  - Clean patch dataset under data/indian_cities_locked/
  - Manifest for the extracted subset
  - 3-city split metadata for benchmark + LOCO
  - Extraction summary under outputs/
"""

import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DATA_DIR, OUTPUT_DIR, PATCH_SIZE
from src.download_data import extract_patches_from_geotiff, build_dataset_manifest


GEE_DIR = Path(r"G:\My Drive\urban_expansion_india")
LOCKED_DATASET_DIR = Path(DATA_DIR) / "indian_cities_locked"
SUMMARY_PATH = Path(OUTPUT_DIR) / "locked_three_city_extraction_summary.json"

LOCKED_PLAN = {
    "Mumbai": {
        "label_candidates": ["Labels_WC_Mumbai.tif", "Labels_DW_Mumbai.tif"],
        "secondary_labels": ["Labels_DW_Mumbai.tif"],
        "classifier_images": [
            "S2_Mumbai_2021_pre_monsoon.tif",
            "S2_Mumbai_2019_pre_monsoon.tif",
            "S2_Mumbai_2023_pre_monsoon.tif",
        ],
        "optional_temporal_images": ["S2_Mumbai_2017_pre_monsoon.tif"],
        "sar_files": ["SAR_Mumbai_2023_post_monsoon.tif"],
        "landsat_files": [
            "LS_Mumbai_1990_pre_monsoon.tif",
            "LS_Mumbai_2000_pre_monsoon.tif",
            "LS_Mumbai_2010_pre_monsoon.tif",
            "LS_Mumbai_2023_pre_monsoon.tif",
        ],
    },
    "Delhi_NCR": {
        "label_candidates": ["Labels_WC_Delhi_NCR.tif", "Labels_DW_Delhi_NCR.tif"],
        "secondary_labels": ["Labels_DW_Delhi_NCR.tif"],
        "classifier_images": [
            "S2_Delhi_NCR_2021_pre_monsoon.tif",
            "S2_Delhi_NCR_2019_pre_monsoon.tif",
            "S2_Delhi_NCR_2023_pre_monsoon.tif",
        ],
        "optional_temporal_images": ["S2_Delhi_NCR_2017_pre_monsoon.tif"],
        "sar_files": ["SAR_Delhi_NCR_2023_pre_monsoon.tif"],
        "landsat_files": [
            "LS_Delhi_NCR_1990_pre_monsoon.tif",
            "LS_Delhi_NCR_2000_pre_monsoon.tif",
            "LS_Delhi_NCR_2010_pre_monsoon.tif",
            "LS_Delhi_NCR_2023_pre_monsoon.tif",
        ],
    },
    "Bangalore": {
        "label_candidates": ["Labels_WC_Bangalore.tif", "Labels_DW_Bangalore.tif"],
        "secondary_labels": ["Labels_DW_Bangalore.tif"],
        "classifier_images": [
            "S2_Bangalore_2021_pre_monsoon.tif",
            "S2_Bangalore_2019_pre_monsoon.tif",
            "S2_Bangalore_2023_pre_monsoon.tif",
        ],
        "optional_temporal_images": ["S2_Bangalore_2017_pre_monsoon.tif"],
        "sar_files": ["SAR_Bangalore_2023_post_monsoon.tif"],
        "landsat_files": [
            "LS_Bangalore_1990_pre_monsoon.tif",
            "LS_Bangalore_2005_pre_monsoon.tif",
            "LS_Bangalore_2010_pre_monsoon.tif",
            "LS_Bangalore_2023_pre_monsoon.tif",
        ],
    },
}


def _pick_first_existing(base_dir: Path, candidates):
    for name in candidates:
        path = base_dir / name
        if path.exists():
            return path
    return None


def _gather_city_manifests(city_root: Path):
    entries = []
    for manifest_path in city_root.rglob("*_manifest.json"):
        if manifest_path.name.endswith("_manifest.json") and manifest_path.stem in {
            "Mumbai_manifest",
            "Delhi_NCR_manifest",
            "Bangalore_manifest",
        }:
            continue
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries.extend(json.load(f))
    return entries


def _write_locked_splits(output_dir: Path, cities):
    splits = {
        "main_benchmark": {
            "cities": list(cities),
            "split_strategy": "stratified_random_over_all_patches",
            "note": (
                "Main benchmark uses all three cities together with stratified "
                "train/val/test splits inside the training pipeline."
            ),
        },
        "geographic_holdout_validation": {
            "train_cities": ["Mumbai", "Delhi_NCR"],
            "val_cities": ["Bangalore"],
            "test_cities": [],
            "note": "Optional city-holdout validation metadata for quick manual checks.",
        },
        "loco": {
            f"loco_{city}": {
                "train_cities": [other for other in cities if other != city],
                "val_cities": [],
                "test_cities": [city],
            }
            for city in cities
        },
    }
    path = output_dir / "locked_three_city_splits.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    return path, splits


def extract_immediate_subset(patch_size: int = PATCH_SIZE, include_optional_temporal: bool = False):
    if not GEE_DIR.exists():
        raise FileNotFoundError(f"GEE directory not found: {GEE_DIR}")

    LOCKED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    extraction_summary = {
        "gee_dir": str(GEE_DIR),
        "output_dir": str(LOCKED_DATASET_DIR),
        "patch_size": patch_size,
        "include_optional_temporal": include_optional_temporal,
        "cities": {},
    }

    for city, cfg in LOCKED_PLAN.items():
        label_path = _pick_first_existing(GEE_DIR, cfg["label_candidates"])
        dynamic_world_path = _pick_first_existing(GEE_DIR, cfg["secondary_labels"])
        selected_images = list(cfg["classifier_images"])
        if include_optional_temporal:
            selected_images.extend(cfg["optional_temporal_images"])

        city_summary = {
            "label_path": str(label_path) if label_path else None,
            "secondary_label_path": str(dynamic_world_path) if dynamic_world_path else None,
            "processed_images": [],
            "missing_images": [],
            "ancillary_inventory": {
                "sar": {},
                "landsat": {},
                "optional_temporal": {},
            },
            "patches_output_dir": None,
        }

        for name in cfg["sar_files"]:
            city_summary["ancillary_inventory"]["sar"][name] = (GEE_DIR / name).exists()
        for name in cfg["landsat_files"]:
            city_summary["ancillary_inventory"]["landsat"][name] = (GEE_DIR / name).exists()
        for name in cfg["optional_temporal_images"]:
            city_summary["ancillary_inventory"]["optional_temporal"][name] = (GEE_DIR / name).exists()

        if label_path is None:
            print(f"[SKIP] {city}: no labels available yet")
            extraction_summary["cities"][city] = city_summary
            continue

        city_out_dir = LOCKED_DATASET_DIR / city / "patches"
        city_out_dir.mkdir(parents=True, exist_ok=True)
        city_summary["patches_output_dir"] = str(city_out_dir)

        for image_name in selected_images:
            image_path = GEE_DIR / image_name
            if not image_path.exists():
                city_summary["missing_images"].append(image_name)
                print(f"[MISS] {city}: {image_name}")
                continue

            image_stem = image_path.stem
            image_out_dir = city_out_dir / image_stem
            image_out_dir.mkdir(parents=True, exist_ok=True)
            image_manifest_path = image_out_dir / f"{image_stem}_manifest.json"

            if image_manifest_path.exists():
                with open(image_manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                print(f"[SKIP] {city}: {image_name} already extracted ({len(manifest)} patches)")
            else:
                print(f"[EXTRACT] {city}: {image_name}")
                manifest = extract_patches_from_geotiff(
                    str(image_path),
                    str(label_path),
                    str(image_out_dir),
                    city_name=city,
                    patch_size=patch_size,
                )
                for entry in manifest:
                    entry["city"] = city
                    entry["source_city"] = city
                    entry["image"] = image_name
                    entry["label_source"] = label_path.name

                with open(image_manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

            city_summary["processed_images"].append(
                {
                    "image": image_name,
                    "patches": len(manifest),
                    "manifest_path": str(image_manifest_path),
                }
            )

        combined_manifest = _gather_city_manifests(city_out_dir)
        manifest_path = city_out_dir / f"{city}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(combined_manifest, f, indent=2)

        city_summary["manifest_path"] = str(manifest_path)
        city_summary["total_patches"] = len(combined_manifest)
        extraction_summary["cities"][city] = city_summary

    global_manifest = build_dataset_manifest(str(LOCKED_DATASET_DIR), output_path=str(LOCKED_DATASET_DIR / "manifest.json"))
    splits_path, splits = _write_locked_splits(LOCKED_DATASET_DIR, LOCKED_PLAN.keys())

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "extraction": extraction_summary,
                "global_manifest_path": str(LOCKED_DATASET_DIR / "manifest.json"),
                "global_splits_path": str(splits_path),
                "total_patches": global_manifest.get("total_patches", 0),
                "cities": global_manifest.get("cities", []),
                "split_keys": list(splits.keys()),
            },
            f,
            indent=2,
        )
    print(f"[DONE] Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    extract_immediate_subset()
