"""
Phase 4 real-data Siamese change detection on LEVIR-CD.

This adapts the current image-level Siamese change detector to LEVIR-CD by
collapsing each pair's binary mask into an image-level change / no-change label:
  - 0 = no changed pixels
  - 1 = at least one changed pixel
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import BATCH_SIZE, DATA_DIR, MODEL_DIR, OUTPUT_DIR, SEED, WEIGHT_DECAY
from main import get_device, set_seed
from src.losses import ChangeLoss
from src.metrics import evaluate
from src.models import SiameseChangeDetector


def _rgb_to_6ch(img_rgb: np.ndarray) -> np.ndarray:
    arr = np.array(Image.fromarray(img_rgb).resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    eps = 1e-7
    ndvi_like = (g - r) / (g + r + eps)
    brightness = np.mean(arr, axis=2)
    try:
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(brightness, size=5)
        local_sq_mean = uniform_filter(brightness ** 2, size=5)
        texture = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0.0))
    except ImportError:
        dy = np.diff(brightness, axis=0, prepend=brightness[:1, :])
        dx = np.diff(brightness, axis=1, prepend=brightness[:, :1])
        texture = np.sqrt(dx ** 2 + dy ** 2)
    patch = np.stack([r, g, b, ndvi_like, brightness, texture], axis=0)
    return patch.astype(np.float32)


class LEVIRChangeDataset(Dataset):
    def __init__(self, split: str):
        self.root = Path(DATA_DIR) / "levir_cd" / split
        self.a_dir = self.root / "A"
        self.b_dir = self.root / "B"
        self.label_dir = self.root / "label"
        if not self.a_dir.is_dir():
            raise FileNotFoundError(f"LEVIR-CD split not found: {self.root}")
        self.files = sorted([p.name for p in self.a_dir.glob("*.png")])
        if not self.files:
            raise RuntimeError(f"No LEVIR images found in {self.a_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_a = np.array(Image.open(self.a_dir / name).convert("RGB"), dtype=np.uint8)
        img_b = np.array(Image.open(self.b_dir / name).convert("RGB"), dtype=np.uint8)
        label_mask = np.array(Image.open(self.label_dir / name).convert("L"), dtype=np.uint8)

        x1 = torch.from_numpy(_rgb_to_6ch(img_a))
        x2 = torch.from_numpy(_rgb_to_6ch(img_b))
        y = 1 if np.any(label_mask > 0) else 0
        return x1, x2, torch.tensor(y, dtype=torch.long)


def _make_loaders(batch_size=BATCH_SIZE):
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    return (
        DataLoader(LEVIRChangeDataset("train"), shuffle=True, **kw),
        DataLoader(LEVIRChangeDataset("val"), shuffle=False, **kw),
        DataLoader(LEVIRChangeDataset("test"), shuffle=False, **kw),
    )


def _eval_change(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            preds = model(x1, x2).argmax(1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    metrics = evaluate(y_true, y_pred, ["NoChange", "Change"])
    return metrics


def run_levir_siamese(backbone_name="efficientnet_b0", epochs=10):
    set_seed(SEED)
    device = get_device()
    train_loader, val_loader, test_loader = _make_loaders()

    model = SiameseChangeDetector(backbone_name=backbone_name, pretrained=True).to(device)
    criterion = ChangeLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    best_state = None
    total_time = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        for x1, x2, y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            logits = model(x1, x2)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        total_time += time.time() - start
        val_metrics = _eval_change(model, val_loader, device)
        train_acc = correct / max(total, 1)
        train_loss = total_loss / max(total, 1)
        print(
            f"Epoch {epoch:2d}/{epochs} | Loss {train_loss:.4f} | "
            f"TrainAcc {train_acc:.4f} | ValAcc {val_metrics['oa']:.4f} | "
            f"ValF1 {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = _eval_change(model, test_loader, device)

    out_dir = Path(OUTPUT_DIR) / "research_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"phase4_levir_siamese_{backbone_name}.pth"
    json_path = out_dir / f"phase4_levir_siamese_{backbone_name}.json"
    torch.save(best_state, model_path)

    payload = {
        "dataset": "LEVIR-CD",
        "backbone": backbone_name,
        "epochs": epochs,
        "training_time_min": round(total_time / 60.0, 2),
        "oa": float(test_metrics["oa"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "f1": float(test_metrics["f1"]),
        "miou": float(test_metrics["miou"]),
        "report": test_metrics["report"],
        "model_path": str(model_path),
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print("\nLEVIR-CD Siamese results")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("report", "model_path")}, indent=2))
    print(f"Saved model to {model_path}")
    print(f"Saved results to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 LEVIR-CD Siamese runner")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run_levir_siamese(backbone_name=args.backbone, epochs=args.epochs)


if __name__ == "__main__":
    main()
