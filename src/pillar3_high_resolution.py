"""
Pillar III: High-Resolution Analysis
======================================
Extends the framework to ingest high-resolution commercial satellite imagery
(e.g., WorldView, PlanetScope). By dropping below the sub-metre threshold,
we move from monitoring broad urban zones to extracting fine-grained,
individual infrastructure features.

Resolution levels: 30m (historical) -> 10m (current) -> sub-metre (proposed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os, sys, time, copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.models import UrbanClassifier, FPN, ClassificationHead
from src.losses import CombinedLoss
from src.metrics import evaluate
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ═══════════════════════════════════════════════════════
#  High-Resolution Data Generation
# ═══════════════════════════════════════════════════════

# Finer-grained classes for sub-metre analysis
HR_CLASSES = [
    "Residential",    # Individual houses, apartments
    "Commercial",     # Offices, shops, malls
    "Industrial",     # Factories, warehouses
    "Road",           # Road infrastructure
    "Vegetation",     # Parks, trees, green areas
    "Bare_Soil",      # Construction sites, empty land
]
HR_NUM_CLASSES = len(HR_CLASSES)
HR_DISTRIBUTION = [0.25, 0.15, 0.10, 0.15, 0.20, 0.15]

# Spectral profiles for high-res 4-band imagery (R, G, B, NIR)
HR_SPECTRAL = {
    "Residential":  np.array([0.18, 0.16, 0.14, 0.20]),
    "Commercial":   np.array([0.22, 0.20, 0.18, 0.15]),
    "Industrial":   np.array([0.25, 0.22, 0.20, 0.12]),
    "Road":         np.array([0.15, 0.14, 0.13, 0.10]),
    "Vegetation":   np.array([0.06, 0.10, 0.05, 0.50]),
    "Bare_Soil":    np.array([0.20, 0.18, 0.16, 0.22]),
}


def generate_hr_patch(label, patch_size=256):
    """
    Generate sub-metre resolution synthetic patch (4 bands: R,G,B,NIR).
    Higher detail than standard Landsat/Sentinel patches.
    """
    h = w = patch_size
    class_name = HR_CLASSES[label]
    base = HR_SPECTRAL[class_name]
    n_channels = 4

    patch = np.zeros((n_channels, h, w), dtype=np.float32)

    for c in range(n_channels):
        # Finer texture for high-res
        texture = _hr_texture(h, w, scale=8)
        noise = np.random.normal(0, 0.015, (h, w)).astype(np.float32)
        patch[c] = base[c] + 0.06 * (texture - 0.5) + noise

    # Add class-specific fine structures
    if class_name == "Residential":
        _add_buildings(patch, h, w, size_range=(8, 20), count=random.randint(10, 25))
    elif class_name == "Commercial":
        _add_buildings(patch, h, w, size_range=(20, 50), count=random.randint(3, 8))
    elif class_name == "Industrial":
        _add_buildings(patch, h, w, size_range=(30, 80), count=random.randint(2, 5))
    elif class_name == "Road":
        _add_roads(patch, h, w)
    elif class_name == "Vegetation":
        _add_canopy(patch, h, w)

    return np.clip(patch, 0.0, 1.0)


def _hr_texture(h, w, scale=8):
    """High-frequency spatial texture for sub-metre imagery."""
    from PIL import Image
    small = np.random.rand(max(1, h // scale), max(1, w // scale)).astype(np.float32)
    img = Image.fromarray((small * 255).astype(np.uint8))
    img = img.resize((w, h), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def _add_buildings(patch, h, w, size_range=(8, 20), count=15):
    """Add individual building footprints."""
    for _ in range(count):
        bh = random.randint(*size_range)
        bw = random.randint(*size_range)
        y0 = random.randint(0, h - bh)
        x0 = random.randint(0, w - bw)
        # Buildings have higher reflectance, lower NIR
        patch[0:3, y0:y0+bh, x0:x0+bw] += 0.08
        patch[3, y0:y0+bh, x0:x0+bw] -= 0.05
        # Add shadow
        sy = min(y0 + bh + 3, h)
        sx = min(x0 + bw + 3, w)
        patch[:, y0+bh:sy, x0:sx] *= 0.7


def _add_roads(patch, h, w):
    """Add road-like linear structures."""
    n_roads = random.randint(2, 5)
    for _ in range(n_roads):
        thickness = random.randint(3, 8)
        if random.random() > 0.5:
            y = random.randint(thickness, h - thickness)
            patch[:, y-thickness:y+thickness, :] = np.array([0.15, 0.14, 0.13, 0.10])[:, None]
        else:
            x = random.randint(thickness, w - thickness)
            patch[:, :, x-thickness:x+thickness] = np.array([0.15, 0.14, 0.13, 0.10])[:, None]


def _add_canopy(patch, h, w):
    """Add tree canopy patterns."""
    n_trees = random.randint(20, 50)
    for _ in range(n_trees):
        r = random.randint(3, 12)
        cy, cx = random.randint(r, h-r), random.randint(r, w-r)
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = (x*x + y*y) <= r*r
        patch[3, mask] += 0.15  # Boost NIR for vegetation
        patch[0, mask] -= 0.03
        patch[2, mask] -= 0.02


# ═══════════════════════════════════════════════════════
#  High-Resolution Dataset
# ═══════════════════════════════════════════════════════

class HighResDataset(Dataset):
    """Sub-metre resolution satellite imagery dataset."""

    def __init__(self, num_patches, seed=SEED):
        super().__init__()
        self.num_patches = num_patches
        rng = np.random.RandomState(seed)
        self.labels = rng.choice(HR_NUM_CLASSES, size=num_patches, p=HR_DISTRIBUTION).tolist()

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        label = self.labels[idx]
        patch = generate_hr_patch(label, patch_size=256)
        return torch.from_numpy(patch), label


# ═══════════════════════════════════════════════════════
#  High-Resolution Model
# ═══════════════════════════════════════════════════════

class HighResClassifier(nn.Module):
    """
    Fine-grained urban infrastructure classifier for sub-metre imagery.
    Uses a deeper feature extraction pipeline to capture individual
    buildings, roads, and vegetation patterns.
    """

    def __init__(self, in_channels=4, num_classes=HR_NUM_CLASSES):
        super().__init__()
        # Custom encoder for 4-band high-res imagery
        self.features = nn.Sequential(
            # Block 1: stride 2
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # Block 2: stride 4
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # Block 3: stride 8
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            # Block 4: stride 16
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            # Block 5: stride 32
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        # Detail-preserving skip connection path
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=4, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=4, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        main_feat = self.gap(self.features(x)).flatten(1)
        detail_feat = self.detail_branch(x).flatten(1)
        combined = torch.cat([main_feat, detail_feat], dim=1)
        return self.head(combined)


# ═══════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════

def train_high_res(device="cuda", epochs=10):
    """Train the high-resolution analysis model."""
    print(f"\n{'='*60}")
    print("  PILLAR III: High-Resolution Analysis (Sub-Metre)")
    print(f"{'='*60}")

    n_train = int(TOTAL_PATCHES * TRAIN_RATIO)
    n_val = int(TOTAL_PATCHES * VAL_RATIO)
    n_test = TOTAL_PATCHES - n_train - n_val

    train_ds = HighResDataset(n_train, seed=SEED + 40)
    val_ds = HighResDataset(n_val, seed=SEED + 41)
    test_ds = HighResDataset(n_test, seed=SEED + 42)

    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    model = HighResClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels, val_loss_total, val_n = [], [], 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_total += loss.item() * y.size(0)
                val_n += y.size(0)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_loss = val_loss_total / val_n
        history["train_loss"].append(running_loss / total)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch:2d}/{epochs} | TrLoss {running_loss/total:.4f} | "
              f"VaLoss {val_loss:.4f} | VaAcc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(y.numpy())

    test_metrics = evaluate(test_labels, test_preds, HR_CLASSES)
    print(f"\n  High-Res Test Results: OA={test_metrics['oa']:.4f} | "
          f"F1={test_metrics['f1']:.4f} | mIoU={test_metrics['miou']:.4f}")
    print(f"  Classes: {HR_CLASSES}")

    save_path = os.path.join(MODEL_DIR, "pillar3_high_res.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_state, save_path)
    print(f"  Model saved to {save_path}")

    return model, history, test_metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_high_res(device, epochs=5)
