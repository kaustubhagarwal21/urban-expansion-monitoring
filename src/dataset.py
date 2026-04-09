"""
Synthetic satellite dataset generator and PyTorch Dataset classes.

Generates realistic 6-channel multispectral patches simulating
Landsat / Sentinel-2 imagery for Urban, Non-Urban, and Transition classes.
"""

import os, math, random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *


# ── Spectral profiles (mean reflectance per channel) ───
#   Channels: Blue, Green, Red, NIR, SWIR1, SWIR2
#   Profiles have overlapping ranges to prevent trivial separation.
SPECTRAL_PROFILES = {
    "urban":      {"mean": np.array([0.14, 0.16, 0.18, 0.22, 0.25, 0.20]),
                   "std":  np.array([0.04, 0.04, 0.05, 0.06, 0.06, 0.05])},
    "non_urban":  {"mean": np.array([0.07, 0.10, 0.08, 0.38, 0.18, 0.12]),
                   "std":  np.array([0.04, 0.04, 0.04, 0.10, 0.06, 0.05])},
    "transition": {"mean": np.array([0.11, 0.13, 0.14, 0.30, 0.22, 0.16]),
                   "std":  np.array([0.05, 0.05, 0.05, 0.08, 0.06, 0.05])},
}


def _make_texture(h, w, scale=32):
    """Generate smooth spatial texture using bilinear-upsampled noise."""
    small = np.random.rand(max(1, h // scale), max(1, w // scale)).astype(np.float32)
    img = Image.fromarray((small * 255).astype(np.uint8))
    img = img.resize((w, h), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def generate_patch(label: int, patch_size: int = PATCH_SIZE,
                   num_channels: int = NUM_CHANNELS) -> np.ndarray:
    """
    Generate a single synthetic multispectral patch with realistic spectral
    variability and overlapping class distributions to avoid trivial separation.

    Args:
        label: 0=Urban, 1=Non-Urban, 2=Transition
        patch_size: spatial dimension
        num_channels: number of spectral channels

    Returns:
        patch of shape (num_channels, patch_size, patch_size), values in [0, 1]
    """
    h = w = patch_size
    keys = ["urban", "non_urban", "transition"]
    profile = SPECTRAL_PROFILES[keys[label]]
    # Per-patch spectral shift: sample a base from a Gaussian around the class mean
    base = np.array([
        np.random.normal(profile["mean"][c], profile["std"][c])
        for c in range(num_channels)
    ], dtype=np.float32)

    patch = np.zeros((num_channels, h, w), dtype=np.float32)
    for c in range(num_channels):
        texture = _make_texture(h, w, scale=random.choice([16, 32, 64]))
        noise = np.random.normal(0, 0.04, (h, w)).astype(np.float32)
        patch[c] = base[c] + 0.12 * (texture - 0.5) + noise

    # For transition patches add mixed urban/vegetation blocks
    if label == 2:
        n_blocks = random.randint(3, 8)
        for _ in range(n_blocks):
            bh, bw = random.randint(16, 64), random.randint(16, 64)
            y0 = random.randint(0, h - bh)
            x0 = random.randint(0, w - bw)
            src_key = "urban" if random.random() > 0.5 else "non_urban"
            src_prof = SPECTRAL_PROFILES[src_key]
            for c in range(num_channels):
                block_val = np.random.normal(src_prof["mean"][c], src_prof["std"][c])
                patch[c, y0:y0 + bh, x0:x0 + bw] = block_val + np.random.normal(0, 0.03, (bh, bw))

    # For urban patches add grid-like structures (roads / buildings)
    if label == 0:
        spacing = random.choice([32, 48, 64])
        for i in range(0, h, spacing):
            thickness = random.randint(1, 3)
            patch[:, max(0, i - thickness):i + thickness, :] *= 0.8
            patch[:, :, max(0, i - thickness):i + thickness] *= 0.8

    return np.clip(patch, 0.0, 1.0)


def generate_temporal_pair(label: int, changed: bool):
    """
    Generate a bi-temporal pair for Siamese change detection.

    Returns:
        (patch_t1, patch_t2, change_label)
        change_label: 1 if expansion occurred, 0 otherwise
    """
    if changed:
        # t1 is non-urban, t2 is urban or transition
        patch_t1 = generate_patch(1)
        patch_t2 = generate_patch(random.choice([0, 2]))
        return patch_t1, patch_t2, 1
    else:
        lbl = random.choice([0, 1])
        patch_t1 = generate_patch(lbl)
        patch_t2 = generate_patch(lbl)
        return patch_t1, patch_t2, 0


# ── Dataset classes ────────────────────────────────────

class UrbanExpansionDataset(Dataset):
    """Single-date classification dataset."""

    def __init__(self, num_patches: int, transform=None, seed=SEED):
        super().__init__()
        self.num_patches = num_patches
        self.transform = transform
        rng = np.random.RandomState(seed)

        # Sample labels following class distribution
        self.labels = rng.choice(
            NUM_CLASSES,
            size=num_patches,
            p=CLASS_DISTRIBUTION,
        ).tolist()

        # Pre-generate city assignments
        self.cities = rng.choice(len(CITIES), size=num_patches).tolist()

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        label = self.labels[idx]
        patch = generate_patch(label)
        patch = torch.from_numpy(patch)

        if self.transform:
            patch = self.transform(patch)

        return patch, label


class SyntheticDataset(UrbanExpansionDataset):
    """Backward-compatible alias used by real-data fallback code."""
    pass


class SiameseChangeDataset(Dataset):
    """Bi-temporal dataset for Siamese change detection."""

    def __init__(self, num_pairs: int, change_ratio=0.5, seed=SEED):
        super().__init__()
        self.num_pairs = num_pairs
        rng = np.random.RandomState(seed)
        self.changed = (rng.rand(num_pairs) < change_ratio).tolist()

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        t1, t2, lbl = generate_temporal_pair(0, self.changed[idx])
        return (torch.from_numpy(t1), torch.from_numpy(t2),
                torch.tensor(lbl, dtype=torch.long))


# ── Augmentation ───────────────────────────────────────

class MultispectralAugment:
    """Random augmentations for multispectral patches."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (C, H, W)
        if random.random() > 0.5:
            x = torch.flip(x, [1])       # vertical flip
        if random.random() > 0.5:
            x = torch.flip(x, [2])       # horizontal flip
        k = random.randint(0, 3)
        x = torch.rot90(x, k, [1, 2])    # random 90-degree rotation
        # color jitter (per-channel brightness shift)
        if random.random() > 0.5:
            jitter = torch.randn(x.shape[0], 1, 1) * 0.03
            x = (x + jitter).clamp(0, 1)
        # gaussian noise
        if random.random() > 0.5:
            noise = torch.randn_like(x) * 0.02
            x = (x + noise).clamp(0, 1)
        return x


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Mixup augmentation (Zhang et al., 2018)."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ── Data loaders ───────────────────────────────────────

def get_synthetic_dataloaders(batch_size=BATCH_SIZE):
    """Create synthetic train / val / test dataloaders."""
    n_train = int(TOTAL_PATCHES * TRAIN_RATIO)
    n_val = int(TOTAL_PATCHES * VAL_RATIO)
    n_test = TOTAL_PATCHES - n_train - n_val

    aug = MultispectralAugment() if AUGMENT else None

    train_ds = UrbanExpansionDataset(n_train, transform=aug, seed=SEED)
    val_ds = UrbanExpansionDataset(n_val, transform=None, seed=SEED + 1)
    test_ds = UrbanExpansionDataset(n_test, transform=None, seed=SEED + 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_dataloaders(
    batch_size=BATCH_SIZE,
    data_source=DATA_SOURCE,
    real_dataset=REAL_DATASET,
    download=ALLOW_REAL_DATA_DOWNLOAD,
    num_workers=0,
):
    """
    Unified loader entry point.

    Args:
        batch_size: loader batch size
        data_source: "synthetic" or "real"
        real_dataset: real dataset name when data_source == "real"
        download: allow downloading datasets that support it
        num_workers: worker count for real-data loaders
    """
    if data_source == "synthetic":
        return get_synthetic_dataloaders(batch_size=batch_size)

    if data_source != "real":
        raise ValueError(f"Unsupported data_source: {data_source}")

    from src.real_data_loaders import RealDataManager

    manager = RealDataManager(batch_size=batch_size, num_workers=num_workers, seed=SEED)

    if real_dataset == "eurosat":
        return manager.get_eurosat_loaders(download=download)

    if real_dataset == "so2sat":
        raise ValueError(
            "So2Sat is multimodal and does not match the single-stream classifier "
            "interface. Use the paper experiment runner or multimodal fusion pipeline."
        )

    if real_dataset == "spacenet":
        raise ValueError(
            "SpaceNet is a segmentation dataset and does not match the current "
            "classification training pipeline."
        )

    if real_dataset == "indian_cities":
        return manager.get_indian_city_loaders()

    if real_dataset == "so2sat_classification":
        return manager.get_so2sat_classification_loaders()

    raise ValueError(f"Unsupported real_dataset: {real_dataset}")


def get_data_loaders(*args, **kwargs):
    """Backward-compatible alias."""
    return get_dataloaders(*args, **kwargs)


def get_siamese_loaders(num_pairs=3000, batch_size=BATCH_SIZE):
    """Create Siamese change-detection dataloaders."""
    n_train = int(num_pairs * TRAIN_RATIO)
    n_val = int(num_pairs * VAL_RATIO)
    n_test = num_pairs - n_train - n_val

    train_ds = SiameseChangeDataset(n_train, seed=SEED + 10)
    val_ds = SiameseChangeDataset(n_val, seed=SEED + 11)
    test_ds = SiameseChangeDataset(n_test, seed=SEED + 12)

    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))


if __name__ == "__main__":
    # Quick sanity check
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, Labels: {y}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")
