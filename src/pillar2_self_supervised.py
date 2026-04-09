"""
Pillar II: Self-Supervised Pre-Training
========================================
Implements contrastive learning (SimCLR-style) for multi-spectral satellite
imagery. The model learns domain-specific representations from unlabelled
Earth observation data, eliminating reliance on human-labelled datasets.

Approach: Contrastive pre-training on augmented views of satellite patches,
then fine-tuning for urban classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os, sys, time, copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.models import UrbanClassifier, ClassificationHead
from src.losses import CombinedLoss
from src.metrics import evaluate
from src.dataset import generate_patch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ═══════════════════════════════════════════════════════
#  Contrastive Augmentations for Satellite Imagery
# ═══════════════════════════════════════════════════════

class SatelliteContrastiveAugment:
    """
    Generate two augmented views of the same satellite patch
    for contrastive learning.
    """

    def __call__(self, x):
        """x: (C, H, W) tensor"""
        view1 = self._augment(x.clone())
        view2 = self._augment(x.clone())
        return view1, view2

    def _augment(self, x):
        # Random horizontal flip
        if random.random() > 0.5:
            x = torch.flip(x, [2])
        # Random vertical flip
        if random.random() > 0.5:
            x = torch.flip(x, [1])
        # Random 90-degree rotation
        k = random.randint(0, 3)
        x = torch.rot90(x, k, [1, 2])
        # Random channel-wise brightness jitter
        jitter = 1.0 + torch.randn(x.shape[0], 1, 1) * 0.1
        x = (x * jitter).clamp(0, 1)
        # Random Gaussian blur (simulates atmospheric variation)
        if random.random() > 0.5:
            noise = torch.randn_like(x) * 0.03
            x = (x + noise).clamp(0, 1)
        # Random crop and resize
        if random.random() > 0.5:
            C, H, W = x.shape
            crop_size = random.randint(int(H * 0.7), H)
            y0 = random.randint(0, H - crop_size)
            x0 = random.randint(0, W - crop_size)
            x = x[:, y0:y0+crop_size, x0:x0+crop_size]
            x = F.interpolate(x.unsqueeze(0), size=(H, W), mode='bilinear',
                            align_corners=False).squeeze(0)
        return x


# ═══════════════════════════════════════════════════════
#  Self-Supervised Dataset (Unlabelled)
# ═══════════════════════════════════════════════════════

class UnlabelledSatelliteDataset(Dataset):
    """
    Simulates a large corpus of unlabelled satellite patches.
    Returns two augmented views for contrastive learning.
    """

    def __init__(self, num_patches, seed=SEED):
        super().__init__()
        self.num_patches = num_patches
        self.augment = SatelliteContrastiveAugment()
        rng = np.random.RandomState(seed)
        # Labels exist for generation but are NOT used in training
        self.labels = rng.choice(NUM_CLASSES, size=num_patches, p=CLASS_DISTRIBUTION).tolist()

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        label = self.labels[idx]
        patch = generate_patch(label, patch_size=128, num_channels=NUM_CHANNELS)
        patch = torch.from_numpy(patch)
        view1, view2 = self.augment(patch)
        return view1, view2


class UnlabelledFromLabelledDataset(Dataset):
    """Build an SSL dataset from any labelled dataset returning image-first tuples."""

    def __init__(self, base_dataset):
        super().__init__()
        self.base_dataset = base_dataset
        self.augment = SatelliteContrastiveAugment()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if isinstance(item, (tuple, list)):
            patch = item[0]
        else:
            patch = item

        if not isinstance(patch, torch.Tensor):
            patch = torch.as_tensor(patch)

        view1, view2 = self.augment(patch.float())
        return view1, view2


# ═══════════════════════════════════════════════════════
#  Projection Head (SimCLR-style)
# ═══════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, in_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class SelfSupervisedModel(nn.Module):
    """
    Self-supervised contrastive learning model.
    Uses the same backbone as the main classifier but with a projection head
    instead of a classification head.
    """

    def __init__(self, backbone_name="efficientnet_b0", pretrained=False):
        super().__init__()
        # Use backbone WITHOUT ImageNet pretrained weights
        # to demonstrate learning from scratch on satellite data
        self.encoder = UrbanClassifier(backbone_name, pretrained=pretrained)
        feat_dim = FPN_CHANNELS * 3
        self.projection = ProjectionHead(feat_dim, 256, 128)

    def forward(self, x):
        feat = self.encoder.get_feature_vector(x)
        proj = self.projection(feat)
        return feat, proj


# ═══════════════════════════════════════════════════════
#  NT-Xent Loss (Normalized Temperature-scaled Cross-Entropy)
# ═══════════════════════════════════════════════════════

class NTXentLoss(nn.Module):
    """Contrastive loss for self-supervised learning (SimCLR)."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
        loss = F.cross_entropy(sim, labels)
        return loss


# ═══════════════════════════════════════════════════════
#  Pre-Training + Fine-Tuning Pipeline
# ═══════════════════════════════════════════════════════

def pretrain_self_supervised(device="cuda", epochs=10, unlabeled_dataset=None):
    """Phase 1: Contrastive pre-training on unlabelled data."""
    print(f"\n  Phase 1: Contrastive Pre-Training ({epochs} epochs)...")

    dataset = unlabeled_dataset or UnlabelledSatelliteDataset(TOTAL_PATCHES, seed=SEED + 30)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=0, pin_memory=True)

    model = SelfSupervisedModel("efficientnet_b0", pretrained=False).to(device)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0, 0
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)
            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        print(f"    Epoch {epoch:2d}/{epochs} | Loss: {total_loss/n_batches:.4f}")

    return model


def finetune_from_pretrained(pretrained_model, device="cuda", epochs=10, loaders=None):
    """Phase 2: Fine-tune the pre-trained encoder for classification."""
    print(f"\n  Phase 2: Fine-Tuning for Classification ({epochs} epochs)...")

    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader = loaders or get_dataloaders(batch_size=BATCH_SIZE)

    # Transfer the pre-trained encoder weights
    classifier = pretrained_model.encoder
    # Replace the head
    feat_dim = FPN_CHANNELS * 3
    classifier.head = ClassificationHead(feat_dim, NUM_CLASSES).to(device)
    classifier = classifier.to(device)

    criterion = CombinedLoss().to(device)
    optimizer = AdamW(classifier.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        classifier.train()
        running_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = classifier(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        # Validate
        classifier.eval()
        val_preds, val_labels, val_loss_total, val_n = [], [], 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = classifier(x)
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

        print(f"    Epoch {epoch:2d}/{epochs} | TrLoss {running_loss/total:.4f} | "
              f"VaAcc {val_metrics['oa']:.4f} | VaF1 {val_metrics['f1']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(classifier.state_dict())

    # Test
    classifier.load_state_dict(best_state)
    classifier.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = classifier(x).argmax(1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(y.numpy())

    test_metrics = evaluate(test_labels, test_preds, CLASS_NAMES)
    return classifier, history, test_metrics


def run_self_supervised_pipeline(
    device="cuda",
    pretrain_epochs=10,
    finetune_epochs=10,
    loaders=None,
    unlabeled_dataset=None,
):
    """Full self-supervised pre-training + fine-tuning pipeline."""
    print(f"\n{'='*60}")
    print("  PILLAR II: Self-Supervised Pre-Training")
    print(f"{'='*60}")

    # Phase 1: Pre-train
    pretrained_model = pretrain_self_supervised(
        device, pretrain_epochs, unlabeled_dataset=unlabeled_dataset
    )

    # Phase 2: Fine-tune
    classifier, history, test_metrics = finetune_from_pretrained(
        pretrained_model, device, finetune_epochs, loaders=loaders
    )

    print(f"\n  Self-Supervised Test Results: OA={test_metrics['oa']:.4f} | "
          f"F1={test_metrics['f1']:.4f} | mIoU={test_metrics['miou']:.4f}")

    save_path = os.path.join(MODEL_DIR, "pillar2_self_supervised.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(classifier.state_dict(), save_path)
    print(f"  Model saved to {save_path}")

    return classifier, history, test_metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_self_supervised_pipeline(device, pretrain_epochs=5, finetune_epochs=5)
