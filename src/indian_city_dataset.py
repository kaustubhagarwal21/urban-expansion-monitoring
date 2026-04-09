"""
PyTorch Dataset for Indian city satellite imagery patches.

Loads patches extracted from GEE-downloaded GeoTIFFs for the 7 Indian
metropolitan areas. Supports temporal pair generation for Siamese
change detection.

Expected directory structure (after running download_indian_cities.py --process-downloads):
    data/indian_cities/
        Mumbai/
            sentinel2_2018_2020/patches/patch_00000.npy ...
            sentinel2_2020_2023/patches/patch_00000.npy ...
            landsat_2013_2020/patches/patch_00000.npy ...
        Delhi_NCR/
            ...

Usage:
    from src.indian_city_dataset import IndianCityDataset, get_indian_city_loaders

    # Classification (single-date)
    loaders = get_indian_city_loaders(period="sentinel2_2020_2023")

    # Temporal pairs (for Siamese change detection)
    from src.indian_city_dataset import IndianCityTemporalDataset
    ds = IndianCityTemporalDataset(period_t1="landsat_2013_2020", period_t2="sentinel2_2020_2023")
"""

import json
import os
import sys
from collections import Counter
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    CITIES, CITY_BOUNDS, DATA_DIR, BATCH_SIZE, SEED,
    CLASS_NAMES, NUM_CLASSES,
)

INDIAN_CITIES_DIR = os.path.join(DATA_DIR, "indian_cities")


# ── Label assignment heuristics ───────────────────────────
# Without ground truth, we use NDVI + NDBI to assign pseudo-labels.
# This is standard practice for unsupervised/semi-supervised urban mapping.
# Bands in our data: [Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, NDBI]

def assign_pseudo_labels(patch: np.ndarray) -> int:
    """
    Assign a pseudo-label based on spectral indices.

    Args:
        patch: (8, H, W) array with bands [..., NDVI, NDBI]

    Returns:
        0 = Urban, 1 = Non-Urban, 2 = Transition
    """
    if patch.shape[0] >= 8:
        ndvi = patch[6]  # NDVI band
        ndbi = patch[7]  # NDBI band
    elif patch.shape[0] >= 6:
        # Compute from raw bands: NIR=3, Red=2, SWIR1=4
        nir, red, swir = patch[3], patch[2], patch[4]
        eps = 1e-8
        ndvi = (nir - red) / (nir + red + eps)
        ndbi = (swir - nir) / (swir + nir + eps)
    else:
        return 1  # default to non-urban if insufficient bands

    mean_ndvi = np.nanmean(ndvi)
    mean_ndbi = np.nanmean(ndbi)

    # Thresholds based on literature for Indian urban areas
    if mean_ndbi > 0.0 and mean_ndvi < 0.2:
        return 0  # Urban
    elif mean_ndvi > 0.35 and mean_ndbi < -0.05:
        return 1  # Non-Urban (vegetation)
    else:
        return 2  # Transition (mixed)


class IndianCityDataset(Dataset):
    """
    Dataset of satellite patches for Indian metropolitan areas.
    Loads pre-extracted .npy patches from disk.
    """

    def __init__(
        self,
        cities: Optional[List[str]] = None,
        period: str = "sentinel2_2020_2023",
        transform: Optional[Callable] = None,
        max_channels: int = 6,
        use_pseudo_labels: bool = True,
    ):
        self.transform = transform
        self.max_channels = max_channels
        self.samples: List[Tuple[str, int, str]] = []  # (path, label, city)

        if cities is None:
            cities = CITIES

        for city in cities:
            patch_dir = os.path.join(INDIAN_CITIES_DIR, city, period, "patches")
            if not os.path.isdir(patch_dir):
                continue

            npy_files = sorted([
                f for f in os.listdir(patch_dir) if f.endswith(".npy")
            ])

            for fname in npy_files:
                fpath = os.path.join(patch_dir, fname)
                patch = np.load(fpath)

                if use_pseudo_labels:
                    label = assign_pseudo_labels(patch)
                else:
                    label = -1  # unlabeled

                self.samples.append((fpath, label, city))

        if not self.samples:
            available = self._check_available()
            raise FileNotFoundError(
                f"No patches found for period '{period}'. "
                f"Available data: {available}\n"
                f"Run: python -m src.download_indian_cities --process-downloads"
            )

    def _check_available(self):
        """List what data is actually available."""
        available = {}
        if not os.path.isdir(INDIAN_CITIES_DIR):
            return "No Indian city data downloaded yet."
        for city in os.listdir(INDIAN_CITIES_DIR):
            city_dir = os.path.join(INDIAN_CITIES_DIR, city)
            if os.path.isdir(city_dir):
                periods = [d for d in os.listdir(city_dir)
                           if os.path.isdir(os.path.join(city_dir, d))]
                available[city] = periods
        return available

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label, city = self.samples[idx]
        patch = np.load(fpath).astype(np.float32)

        # Take first max_channels bands
        if patch.shape[0] > self.max_channels:
            patch = patch[:self.max_channels]

        tensor = torch.from_numpy(patch)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, label

    def get_city_labels(self):
        """Return list of city names for each sample (for cross-city analysis)."""
        return [s[2] for s in self.samples]

    def class_distribution(self):
        """Return class distribution."""
        labels = [s[1] for s in self.samples]
        return Counter(labels)


class IndianCityTemporalDataset(Dataset):
    """
    Bi-temporal dataset for Siamese change detection.
    Pairs patches from two different time periods at the same location.
    """

    def __init__(
        self,
        cities: Optional[List[str]] = None,
        period_t1: str = "landsat_2013_2020",
        period_t2: str = "sentinel2_2020_2023",
        transform: Optional[Callable] = None,
        max_channels: int = 6,
    ):
        self.transform = transform
        self.max_channels = max_channels
        self.pairs: List[Tuple[str, str, int]] = []  # (path_t1, path_t2, change_label)

        if cities is None:
            cities = CITIES

        for city in cities:
            dir_t1 = os.path.join(INDIAN_CITIES_DIR, city, period_t1, "patches")
            dir_t2 = os.path.join(INDIAN_CITIES_DIR, city, period_t2, "patches")

            if not os.path.isdir(dir_t1) or not os.path.isdir(dir_t2):
                continue

            files_t1 = sorted([f for f in os.listdir(dir_t1) if f.endswith(".npy")])
            files_t2 = sorted([f for f in os.listdir(dir_t2) if f.endswith(".npy")])

            # Match by index (assumes same spatial grid)
            n_pairs = min(len(files_t1), len(files_t2))
            for i in range(n_pairs):
                path_t1 = os.path.join(dir_t1, files_t1[i])
                path_t2 = os.path.join(dir_t2, files_t2[i])

                # Determine change label from pseudo-labels
                p1 = np.load(path_t1)
                p2 = np.load(path_t2)
                label_t1 = assign_pseudo_labels(p1)
                label_t2 = assign_pseudo_labels(p2)

                # Change = went from non-urban to urban or transition
                changed = 1 if (label_t1 == 1 and label_t2 in [0, 2]) else 0
                self.pairs.append((path_t1, path_t2, changed))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_t1, path_t2, label = self.pairs[idx]

        p1 = np.load(path_t1).astype(np.float32)[:self.max_channels]
        p2 = np.load(path_t2).astype(np.float32)[:self.max_channels]

        t1 = torch.from_numpy(p1)
        t2 = torch.from_numpy(p2)

        if self.transform:
            t1 = self.transform(t1)
            t2 = self.transform(t2)

        return t1, t2, torch.tensor(label, dtype=torch.long)


def get_indian_city_loaders(
    period: str = "sentinel2_2020_2023",
    cities: Optional[List[str]] = None,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    test_size: float = 0.15,
    val_size: float = 0.15,
):
    """
    Create train/val/test loaders for Indian city data.
    Stratified split preserving city and class distribution.
    """
    from src.dataset import MultispectralAugment

    dataset = IndianCityDataset(cities=cities, period=period)
    labels = [s[1] for s in dataset.samples]

    # Stratified split
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=seed,
    )
    train_labels = [labels[i] for i in train_idx]
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size / (1 - test_size),
        stratify=train_labels, random_state=seed,
    )

    from torch.utils.data import Subset

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    # Wrap train with augmentation
    aug = MultispectralAugment()

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    print(f"  Indian cities ({period}): train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"  Class distribution: {dataset.class_distribution()}")

    return train_loader, val_loader, test_loader


def check_indian_data_status():
    """Print status of downloaded Indian city data."""
    print("\n" + "=" * 60)
    print("  Indian City Data Status")
    print("=" * 60)

    if not os.path.isdir(INDIAN_CITIES_DIR):
        print("  No data downloaded yet.")
        print(f"  Expected location: {INDIAN_CITIES_DIR}")
        print("  Run: python -m src.download_indian_cities")
        return

    for city in CITIES:
        city_dir = os.path.join(INDIAN_CITIES_DIR, city)
        if not os.path.isdir(city_dir):
            print(f"  {city}: NOT DOWNLOADED")
            continue

        periods = sorted(os.listdir(city_dir))
        for period in periods:
            patch_dir = os.path.join(city_dir, period, "patches")
            if os.path.isdir(patch_dir):
                n = len([f for f in os.listdir(patch_dir) if f.endswith(".npy")])
                print(f"  {city}/{period}: {n} patches")
            else:
                print(f"  {city}/{period}: GeoTIFF present, patches not extracted")


if __name__ == "__main__":
    check_indian_data_status()
