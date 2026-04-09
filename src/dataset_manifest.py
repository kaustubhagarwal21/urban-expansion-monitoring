"""
Geographic split metadata and dataset manifests.

Provides reproducible city-level splits for cross-city evaluation,
and exports dataset manifests (JSON) recording exact sample counts,
class distributions, and split assignments for paper reproducibility.
"""

import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    CITIES, CITY_BOUNDS, CITY_GROWTH, CITY_DESCRIPTIONS,
    PATCHES_PER_CITY, CLASS_DISTRIBUTION, CLASS_NAMES,
    NUM_CLASSES, SEED, OUTPUT_DIR, DATA_DIR,
    TRAIN_RATIO, VAL_RATIO,
)


# =========================================================================
# City-level geographic splits
# =========================================================================

# Default split protocol for paper experiments:
#   Train: 4 cities, Val: 1 city, Test: 2 held-out cities
SPLIT_CONFIGS = {
    "default": {
        "train": ["Mumbai", "Delhi_NCR", "Bangalore", "Pune"],
        "val": ["Ahmedabad"],
        "test": ["Hyderabad", "Chennai"],
        "description": "Default geographic split for cross-city evaluation",
    },
    "south_holdout": {
        "train": ["Mumbai", "Delhi_NCR", "Pune", "Ahmedabad"],
        "val": ["Hyderabad"],
        "test": ["Bangalore", "Chennai"],
        "description": "Hold out southern cities to test north-south transfer",
    },
    "mega_holdout": {
        "train": ["Bangalore", "Hyderabad", "Chennai", "Pune", "Ahmedabad"],
        "val": ["Mumbai"],
        "test": ["Delhi_NCR"],
        "description": "Train on smaller cities, test on mega-cities",
    },
    "loco_mumbai": {
        "train": ["Delhi_NCR", "Bangalore", "Hyderabad", "Chennai", "Pune", "Ahmedabad"],
        "val": [],
        "test": ["Mumbai"],
        "description": "Leave-One-City-Out: Mumbai",
    },
    "loco_delhi": {
        "train": ["Mumbai", "Bangalore", "Hyderabad", "Chennai", "Pune", "Ahmedabad"],
        "val": [],
        "test": ["Delhi_NCR"],
        "description": "Leave-One-City-Out: Delhi NCR",
    },
    "loco_bangalore": {
        "train": ["Mumbai", "Delhi_NCR", "Hyderabad", "Chennai", "Pune", "Ahmedabad"],
        "val": [],
        "test": ["Bangalore"],
        "description": "Leave-One-City-Out: Bangalore",
    },
}


def get_city_metadata() -> Dict[str, dict]:
    """Return structured metadata for each study city."""
    meta = {}
    for city in CITIES:
        meta[city] = {
            "name": city,
            "bounds": CITY_BOUNDS[city],
            "growth_multiplier": CITY_GROWTH[city],
            "description": CITY_DESCRIPTIONS[city],
            "patches": PATCHES_PER_CITY,
        }
    return meta


def get_geographic_split(
    split_name: str = "default",
) -> Dict[str, List[str]]:
    """
    Return the train/val/test city lists for a named split config.

    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    if split_name not in SPLIT_CONFIGS:
        raise ValueError(
            f"Unknown split: {split_name}. "
            f"Available: {list(SPLIT_CONFIGS.keys())}"
        )
    cfg = SPLIT_CONFIGS[split_name]
    return {
        "train": list(cfg["train"]),
        "val": list(cfg["val"]),
        "test": list(cfg["test"]),
    }


def generate_all_loco_splits() -> Dict[str, Dict[str, List[str]]]:
    """Generate Leave-One-City-Out splits for every city."""
    splits = {}
    for city in CITIES:
        key = f"loco_{city.lower()}"
        others = [c for c in CITIES if c != city]
        splits[key] = {
            "train": others,
            "val": [],
            "test": [city],
        }
    return splits


# =========================================================================
# Dataset indexing by city (for synthetic datasets)
# =========================================================================

def get_city_indices(dataset: Dataset) -> Dict[str, List[int]]:
    """
    Group dataset sample indices by city assignment.

    Works with UrbanExpansionDataset which has a .cities attribute
    mapping each sample index to a city index.
    """
    if not hasattr(dataset, "cities"):
        raise AttributeError(
            "Dataset does not have a .cities attribute. "
            "City-level splitting requires UrbanExpansionDataset."
        )
    city_indices: Dict[str, List[int]] = {c: [] for c in CITIES}
    for idx, city_idx in enumerate(dataset.cities):
        city_name = CITIES[city_idx]
        city_indices[city_name].append(idx)
    return city_indices


def split_dataset_by_cities(
    dataset: Dataset,
    split_name: str = "default",
) -> Tuple[Subset, Subset, Subset]:
    """
    Split a dataset into train/val/test Subsets based on geographic city assignment.

    Returns:
        (train_subset, val_subset, test_subset)
    """
    geo = get_geographic_split(split_name)
    city_idx_map = get_city_indices(dataset)

    train_indices, val_indices, test_indices = [], [], []
    for city in geo["train"]:
        train_indices.extend(city_idx_map.get(city, []))
    for city in geo["val"]:
        val_indices.extend(city_idx_map.get(city, []))
    for city in geo["test"]:
        test_indices.extend(city_idx_map.get(city, []))

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


# =========================================================================
# Few-shot target adaptation splits
# =========================================================================

def make_fewshot_split(
    dataset: Dataset,
    target_city: str,
    k_shots: int = 10,
    seed: int = SEED,
) -> Tuple[Subset, Subset]:
    """
    Create a few-shot adaptation split for a target city.

    From the target city's samples, randomly select k_shots per class
    for fine-tuning and use the rest for evaluation.

    Returns:
        (fewshot_train_subset, eval_subset)
    """
    city_indices = get_city_indices(dataset)
    target_indices = city_indices.get(target_city, [])
    if not target_indices:
        raise ValueError(f"No samples found for city: {target_city}")

    # Group by label
    label_groups: Dict[int, List[int]] = {c: [] for c in range(NUM_CLASSES)}
    for idx in target_indices:
        _, label = dataset[idx] if not isinstance(dataset, Subset) else dataset.dataset[idx]
        label = int(label) if hasattr(label, 'item') else label
        label_groups[label].append(idx)

    rng = random.Random(seed)
    fewshot_indices = []
    eval_indices = []

    for cls in range(NUM_CLASSES):
        pool = label_groups[cls]
        rng.shuffle(pool)
        k = min(k_shots, len(pool))
        fewshot_indices.extend(pool[:k])
        eval_indices.extend(pool[k:])

    return Subset(dataset, fewshot_indices), Subset(dataset, eval_indices)


# =========================================================================
# Manifest generation
# =========================================================================

def generate_manifest(
    dataset: Dataset,
    split_name: str = "default",
    dataset_name: str = "synthetic",
    extra_info: Optional[dict] = None,
) -> dict:
    """
    Generate a complete dataset manifest for reproducibility.

    The manifest records:
      - Dataset name and type
      - Geographic split configuration
      - Per-split sample counts
      - Per-split class distributions
      - City metadata
      - Random seed
    """
    geo = get_geographic_split(split_name)

    manifest = {
        "dataset_name": dataset_name,
        "split_config": split_name,
        "split_description": SPLIT_CONFIGS.get(split_name, {}).get("description", ""),
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "target_distribution": CLASS_DISTRIBUTION,
        "geographic_split": geo,
        "city_metadata": get_city_metadata(),
        "splits": {},
    }

    # If dataset supports city-based splitting, compute per-split stats
    if hasattr(dataset, "cities"):
        train_sub, val_sub, test_sub = split_dataset_by_cities(dataset, split_name)
        for name, subset in [("train", train_sub), ("val", val_sub), ("test", test_sub)]:
            indices = subset.indices if hasattr(subset, "indices") else []
            labels = []
            for idx in indices:
                item = dataset[idx]
                labels.append(int(item[1]))
            counts = Counter(labels)
            manifest["splits"][name] = {
                "num_samples": len(indices),
                "class_counts": {CLASS_NAMES[k]: v for k, v in sorted(counts.items())},
                "cities": geo[name],
            }
    else:
        # Generic dataset: just record total size
        manifest["splits"]["total"] = {
            "num_samples": len(dataset),
        }

    if extra_info:
        manifest["extra"] = extra_info

    return manifest


def save_manifest(manifest: dict, filename: str = "dataset_manifest.json"):
    """Save manifest to the outputs directory."""
    out_dir = os.path.join(OUTPUT_DIR, "paper_experiment")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Manifest saved to {path}")
    return path


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    print("Available geographic split configs:")
    for name, cfg in SPLIT_CONFIGS.items():
        print(f"  {name:20s} — {cfg['description']}")
        print(f"    Train: {cfg['train']}")
        print(f"    Val:   {cfg['val']}")
        print(f"    Test:  {cfg['test']}")
        print()

    print("\nCity metadata:")
    for city, meta in get_city_metadata().items():
        print(f"  {city}: growth={meta['growth_multiplier']}x, bounds={meta['bounds']}")

    print("\nGenerating manifest with synthetic dataset...")
    from src.dataset import UrbanExpansionDataset
    ds = UrbanExpansionDataset(num_patches=1000, seed=SEED)
    manifest = generate_manifest(ds, split_name="default", dataset_name="synthetic")
    save_manifest(manifest)
    print("Done.")
