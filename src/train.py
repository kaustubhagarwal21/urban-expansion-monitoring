"""
Training pipeline with progressive fine-tuning and cosine-annealing LR.
"""

import os, sys, time, copy, torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.models import (UrbanClassifier, SiameseChangeDetector,
                        freeze_backbone, UNFREEZE_FNS)
from src.losses import CombinedLoss, ChangeLoss
from src.dataset import get_dataloaders, get_siamese_loaders, mixup_data
from src.metrics import evaluate


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if use_mixup and AUGMENT:
            x, y_a, y_b, lam = mixup_data(x, y)
            logits = model(x)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            preds = logits.argmax(1)
            correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()

        total += y.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * y.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    n = len(all_labels)
    metrics = evaluate(all_labels, all_preds, CLASS_NAMES)
    return running_loss / n, metrics


def progressive_train(backbone_name=DEFAULT_BACKBONE, device="cuda", loaders=None):
    """
    Three-stage progressive fine-tuning.
    """
    print(f"\n{'='*60}")
    print(f"  Training {backbone_name.upper()} with progressive fine-tuning")
    print(f"{'='*60}")

    model = UrbanClassifier(backbone_name, pretrained=True).to(device)
    # Upweight Transition class (class 2) to address severe class imbalance
    criterion = CombinedLoss(class_weights=[1.0, 1.0, 3.0]).to(device)
    train_loader, val_loader, test_loader = loaders or get_dataloaders()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "val_f1": [], "val_miou": []}
    best_val_loss = float("inf")
    best_state = None
    total_time = 0.0

    for stage_cfg in STAGES:
        stage_name = stage_cfg["name"]
        lr = stage_cfg["lr"]
        epochs = stage_cfg["epochs"]
        unfreeze_key = stage_cfg["unfreeze"]

        print(f"\n--- {stage_name} (lr={lr}, epochs={epochs}) ---")

        # Apply unfreezing strategy
        UNFREEZE_FNS[unfreeze_key](model)

        # Optimizer and scheduler for this stage
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            scheduler.step()
            elapsed = time.time() - t0
            total_time += elapsed

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_metrics["oa"])
            history["val_f1"].append(val_metrics["f1"])
            history["val_miou"].append(val_metrics["miou"])

            print(f"  Epoch {epoch:2d}/{epochs} | "
                  f"TrLoss {train_loss:.4f} | VaLoss {val_loss:.4f} | "
                  f"VaAcc {val_metrics['oa']:.4f} | VaF1 {val_metrics['f1']:.4f} | "
                  f"mIoU {val_metrics['miou']:.4f} | {elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS ({backbone_name})")
    print(f"{'='*60}")
    print(f"  OA:        {test_metrics['oa']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  mIoU:      {test_metrics['miou']:.4f}")
    print(f"  Total training time: {total_time / 60:.1f} min")
    print(f"\n{test_metrics['report']}")

    # Save model
    save_path = os.path.join(MODEL_DIR, f"{backbone_name}_best.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_state, save_path)
    print(f"  Model saved to {save_path}")

    return model, history, test_metrics, total_time


# ═══════════════════════════════════════════════════════
#  Siamese Change-Detection Training
# ═══════════════════════════════════════════════════════

def train_siamese(backbone_name=DEFAULT_BACKBONE, device="cuda", epochs=20):
    """Train the Siamese change-detection branch."""
    print(f"\n{'='*60}")
    print(f"  Training Siamese Change Detector ({backbone_name})")
    print(f"{'='*60}")

    model = SiameseChangeDetector(backbone_name, pretrained=True).to(device)
    criterion = ChangeLoss().to(device)
    train_loader, val_loader, test_loader = get_siamese_loaders()

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
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

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                preds = model(x1, x2).argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        print(f"  Epoch {epoch:2d}/{epochs} | Loss {total_loss / total:.4f} | "
              f"TrainAcc {correct / total:.4f} | ValAcc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            preds = model(x1, x2).argmax(1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)
    print(f"\n  Siamese Test Accuracy: {test_correct / test_total:.4f}")

    save_path = os.path.join(MODEL_DIR, f"siamese_{backbone_name}.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_state, save_path)
    return model
