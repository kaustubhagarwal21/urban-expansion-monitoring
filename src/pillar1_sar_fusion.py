"""
Pillar I: Multi-Modal Fusion (Optical + SAR)
=============================================
Fuses optical satellite imagery (Landsat/Sentinel-2) with Synthetic Aperture
Radar (SAR) data for all-weather, continuous urban monitoring.

SAR penetrates cloud cover, providing structural information regardless of
atmospheric conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os, sys, time, copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.models import UrbanClassifier, FPN, ClassificationHead, _expand_first_conv
from src.losses import CombinedLoss
from src.metrics import evaluate
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ═══════════════════════════════════════════════════════
#  SAR Data Generation
# ═══════════════════════════════════════════════════════

SAR_PROFILES = {
    # SAR backscatter profiles: [VV, VH] polarization
    "urban":      np.array([0.35, 0.20]),   # High backscatter (buildings reflect)
    "non_urban":  np.array([0.10, 0.05]),   # Low backscatter (vegetation absorbs)
    "transition": np.array([0.22, 0.12]),   # Mixed signal
}


def generate_sar_patch(label, patch_size=128):
    """
    Generate synthetic SAR patch with 2 channels (VV, VH polarization).
    Simulates Sentinel-1 C-band SAR characteristics.
    """
    h = w = patch_size
    profiles = list(SAR_PROFILES.values())
    base = profiles[label]

    patch = np.zeros((2, h, w), dtype=np.float32)

    for c in range(2):
        # SAR has speckle noise (multiplicative)
        texture = np.random.exponential(1.0, (h, w)).astype(np.float32)
        patch[c] = base[c] * texture * 0.3

    # Urban areas have strong double-bounce returns
    if label == 0:
        spacing = random.choice([24, 32, 48])
        for i in range(0, h, spacing):
            t = random.randint(1, 3)
            patch[:, max(0, i-t):i+t, :] *= 2.5
            patch[:, :, max(0, i-t):i+t] *= 2.5

    # Transition zones have mixed patterns
    if label == 2:
        n_blocks = random.randint(3, 6)
        for _ in range(n_blocks):
            bh, bw = random.randint(16, 48), random.randint(16, 48)
            y0 = random.randint(0, h - bh)
            x0 = random.randint(0, w - bw)
            src = SAR_PROFILES["urban"] if random.random() > 0.5 else SAR_PROFILES["non_urban"]
            for c in range(2):
                speckle = np.random.exponential(1.0, (bh, bw)).astype(np.float32)
                patch[c, y0:y0+bh, x0:x0+bw] = src[c] * speckle * 0.3

    return np.clip(patch, 0.0, 1.0)


# ═══════════════════════════════════════════════════════
#  Multi-Modal Dataset (Optical + SAR)
# ═══════════════════════════════════════════════════════

class MultiModalDataset(Dataset):
    """Dataset returning paired optical + SAR patches."""

    def __init__(self, num_patches, seed=SEED):
        super().__init__()
        self.num_patches = num_patches
        rng = np.random.RandomState(seed)
        self.labels = rng.choice(NUM_CLASSES, size=num_patches, p=CLASS_DISTRIBUTION).tolist()

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        from src.dataset import generate_patch
        label = self.labels[idx]
        optical = generate_patch(label, patch_size=128, num_channels=NUM_CHANNELS)
        sar = generate_sar_patch(label, patch_size=128)
        return (torch.from_numpy(optical), torch.from_numpy(sar), label)


class So2SatFusionAdapter(Dataset):
    """Adapt So2Sat samples to the fusion model's 6-channel optical and 2-channel SAR format."""

    def __init__(self, base_dataset, image_size=128):
        super().__init__()
        self.base_dataset = base_dataset
        self.image_size = image_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        optical, sar, label = self.base_dataset[idx]

        optical = optical.float()
        sar = sar.float()

        if optical.ndim != 3 or sar.ndim != 3:
            raise ValueError("Expected So2Sat tensors with shape (C, H, W)")

        optical = optical[:NUM_CHANNELS]
        sar = sar[:2]

        optical = F.interpolate(
            optical.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        sar = F.interpolate(
            sar.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        optical = optical.clamp(0, 1)
        sar = sar.clamp(0, 1)

        return optical, sar, int(label)


class IndianSARFusionDataset(Dataset):
    """Dataset loading paired optical + SAR .npy patches from Indian cities."""

    def __init__(self, samples, image_size=128):
        super().__init__()
        self.samples = samples  # [(optical_path, sar_path, label)]
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        opt_path, sar_path, label = self.samples[idx]
        optical = torch.from_numpy(np.load(opt_path).astype(np.float32))  # (6, 256, 256)
        sar = torch.from_numpy(np.load(sar_path).astype(np.float32))      # (2, 256, 256)

        # Resize to fusion model input size
        optical = F.interpolate(
            optical.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        sar = F.interpolate(
            sar.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        return optical, sar, label


def get_multimodal_loaders(data_source="synthetic", batch_size=BATCH_SIZE):
    """Return loaders for synthetic multimodal data or real So2Sat data."""
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=True)

    if data_source == "synthetic":
        n_train = int(TOTAL_PATCHES * TRAIN_RATIO)
        n_val = int(TOTAL_PATCHES * VAL_RATIO)
        n_test = TOTAL_PATCHES - n_train - n_val

        train_ds = MultiModalDataset(n_train, seed=SEED + 20)
        val_ds = MultiModalDataset(n_val, seed=SEED + 21)
        test_ds = MultiModalDataset(n_test, seed=SEED + 22)
        return (
            DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw),
        )

    if data_source == "real":
        from src.real_data_loaders import RealDataManager

        manager = RealDataManager(batch_size=batch_size, num_workers=0, seed=SEED)
        if not manager.is_available("so2sat"):
            print("  So2Sat not available locally. Falling back to synthetic multimodal data.")
            return get_multimodal_loaders(data_source="synthetic", batch_size=batch_size)

        train_loader, val_loader, test_loader = manager.get_so2sat_loaders()

        train_ds = So2SatFusionAdapter(train_loader.dataset)
        val_ds = So2SatFusionAdapter(val_loader.dataset)
        test_ds = So2SatFusionAdapter(test_loader.dataset)
        return (
            DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw),
        )

    if data_source == "indian_sar":
        from src.real_data_loaders import _env_indian_patch_root
        from sklearn.model_selection import train_test_split

        patch_root = _env_indian_patch_root() or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "indian_cities_locked"
        )
        cities = ["Mumbai", "Delhi_NCR", "Bangalore"]

        samples = []  # (optical_path, sar_path, label)
        for city in cities:
            sar_dir = os.path.join(patch_root, city, "sar_patches")
            if not os.path.isdir(sar_dir):
                print(f"  No SAR patches for {city}, skipping")
                continue
            # Find all SAR patches and match to optical
            for fname in sorted(os.listdir(sar_dir)):
                if not fname.endswith("_sar.npy"):
                    continue
                patch_id = fname.replace("_sar.npy", "")
                sar_path = os.path.join(sar_dir, fname)
                # Find matching optical + label (search all source subdirs)
                patches_dir = os.path.join(patch_root, city, "patches")
                opt_path = lbl_path = None
                for subdir in os.listdir(patches_dir):
                    sub = os.path.join(patches_dir, subdir)
                    if not os.path.isdir(sub):
                        continue
                    candidate_img = os.path.join(sub, f"{patch_id}_img.npy")
                    candidate_lbl = os.path.join(sub, f"{patch_id}_lbl.npy")
                    if os.path.isfile(candidate_img) and os.path.isfile(candidate_lbl):
                        opt_path = candidate_img
                        lbl_path = candidate_lbl
                        break
                if opt_path is None:
                    continue
                lbl_data = np.load(lbl_path)
                label = int(np.bincount(lbl_data.flatten().astype(int)).argmax())
                samples.append((opt_path, sar_path, label))

        print(f"  Indian SAR fusion: {len(samples)} paired samples across {len(cities)} cities")

        if len(samples) == 0:
            print("  No paired samples found. Falling back to synthetic.")
            return get_multimodal_loaders(data_source="synthetic", batch_size=batch_size)

        labels = [s[2] for s in samples]
        train_idx, test_idx = train_test_split(
            list(range(len(samples))), test_size=0.2,
            stratify=labels, random_state=SEED
        )
        train_labels = [labels[i] for i in train_idx]
        train_idx2, val_idx = train_test_split(
            train_idx, test_size=0.15,
            stratify=train_labels, random_state=SEED
        )

        train_ds = IndianSARFusionDataset([samples[i] for i in train_idx2])
        val_ds = IndianSARFusionDataset([samples[i] for i in val_idx])
        test_ds = IndianSARFusionDataset([samples[i] for i in test_idx])

        print(f"  Splits: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        return (
            DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw),
        )

    raise ValueError(f"Unsupported multimodal data_source: {data_source}")


# ═══════════════════════════════════════════════════════
#  Fusion Architectures
# ═══════════════════════════════════════════════════════

class SAREncoder(nn.Module):
    """Lightweight CNN encoder for 2-channel SAR data."""

    def __init__(self, out_channels_list=(24, 80, 192)):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels_list[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels_list[0]), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels_list[0], out_channels_list[1], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels_list[1]), nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(out_channels_list[1], out_channels_list[2], 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels_list[2]), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        return [f1, f2, f3]


class CrossModalAttention(nn.Module):
    """Cross-attention module to fuse optical and SAR features."""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 4, 1)
        self.key = nn.Conv2d(channels, channels // 4, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, optical_feat, sar_feat):
        # Ensure same spatial size
        if optical_feat.shape[2:] != sar_feat.shape[2:]:
            sar_feat = F.interpolate(sar_feat, size=optical_feat.shape[2:],
                                     mode="bilinear", align_corners=False)

        q = self.query(optical_feat)
        k = self.key(sar_feat)
        v = self.value(sar_feat)

        B, C, H, W = q.shape
        attn = torch.bmm(q.view(B, C, -1).permute(0, 2, 1),
                         k.view(B, C, -1))
        attn = F.softmax(attn / (C ** 0.5), dim=-1)
        attended = torch.bmm(v.view(B, v.shape[1], -1),
                            attn.permute(0, 2, 1)).view(B, v.shape[1], H, W)

        combined = torch.cat([optical_feat, attended], dim=1)
        gate = self.gate(combined)
        return optical_feat * gate + attended * (1 - gate)


class MultiModalFusionNet(nn.Module):
    """
    Full multi-modal fusion network combining optical and SAR streams.

    Architecture:
        Optical -> EfficientNet Encoder -> FPN ─┐
                                                 ├─> Cross-Modal Attention -> Classification
        SAR -> SAR Encoder -> FPN ──────────────┘
    """

    def __init__(self, backbone_name="efficientnet_b0", pretrained=True):
        super().__init__()
        # Optical stream (reuse existing backbone)
        self.optical_encoder = UrbanClassifier(backbone_name, pretrained)

        # SAR stream
        self.sar_encoder = SAREncoder(out_channels_list=(24, 80, 192))
        self.sar_fpn = FPN([24, 80, 192], FPN_CHANNELS)

        # Cross-modal attention at each FPN level
        self.cross_attn = nn.ModuleList([
            CrossModalAttention(FPN_CHANNELS) for _ in range(3)
        ])

        # Fused classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = ClassificationHead(FPN_CHANNELS * 3, NUM_CLASSES)

    def forward(self, optical, sar):
        # Optical stream
        opt_features = self.optical_encoder.extract_features(optical)
        opt_fpn = self.optical_encoder.fpn(opt_features)

        # SAR stream
        sar_features = self.sar_encoder(sar)
        sar_fpn = self.sar_fpn(sar_features)

        # Cross-modal fusion at each scale
        fused = []
        for i in range(3):
            f = self.cross_attn[i](opt_fpn[i], sar_fpn[i])
            fused.append(f)

        # Pool and classify
        pooled = [self.gap(f).flatten(1) for f in fused]
        combined = torch.cat(pooled, dim=1)
        return self.head(combined)


# ═══════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════

def train_fusion(device="cuda", epochs=10, loaders=None, data_source="synthetic"):
    """Train the multi-modal fusion network."""
    print(f"\n{'='*60}")
    print("  PILLAR I: Multi-Modal Fusion (Optical + SAR)")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = loaders or get_multimodal_loaders(
        data_source=data_source, batch_size=BATCH_SIZE
    )

    model = MultiModalFusionNet("efficientnet_b0", pretrained=True).to(device)
    criterion = CombinedLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running_loss, correct, total = 0, 0, 0
        for optical, sar, y in train_loader:
            optical, sar = optical.to(device), sar.to(device)
            y = y.to(device)
            logits = model(optical, sar)
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
            for optical, sar, y in val_loader:
                optical, sar = optical.to(device), sar.to(device)
                y = y.to(device)
                logits = model(optical, sar)
                loss = criterion(logits, y)
                val_loss_total += loss.item() * y.size(0)
                val_n += y.size(0)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_metrics = evaluate(val_labels, val_preds, CLASS_NAMES)
        val_loss = val_loss_total / val_n
        history["train_loss"].append(running_loss / total)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["oa"])
        history["val_f1"].append(val_metrics["f1"])

        print(f"  Epoch {epoch:2d}/{epochs} | TrLoss {running_loss/total:.4f} | "
              f"VaLoss {val_loss:.4f} | VaAcc {val_metrics['oa']:.4f} | "
              f"VaF1 {val_metrics['f1']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for optical, sar, y in test_loader:
            optical, sar = optical.to(device), sar.to(device)
            logits = model(optical, sar)
            test_preds.extend(logits.argmax(1).cpu().numpy())
            test_labels.extend(y.numpy())

    test_metrics = evaluate(test_labels, test_preds, CLASS_NAMES)
    print(f"\n  Fusion Test Results: OA={test_metrics['oa']:.4f} | "
          f"F1={test_metrics['f1']:.4f} | mIoU={test_metrics['miou']:.4f}")

    save_path = os.path.join(MODEL_DIR, "pillar1_fusion.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_state, save_path)
    print(f"  Model saved to {save_path}")

    return model, history, test_metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_fusion(device=device, epochs=5)
