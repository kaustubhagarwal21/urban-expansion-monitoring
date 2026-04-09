"""
Cross-City Generalization and Domain Adaptation Experiments.

Evaluates how well urban expansion models transfer across Indian cities:
  1. Leave-One-City-Out (LOCO) generalization
  2. Leave-Two-Cities-Out (paired coastal / northern / southern cities)
  3. Few-Shot Adaptation with K labelled target samples
  4. Domain Shift Analysis via Maximum Mean Discrepancy (MMD)
  5. Cross-City Transfer Matrix (N x N accuracy)

City-specific spectral characteristics simulate real geographic variation:
  - Coastal cities (Mumbai, Chennai): higher NIR moisture signature
  - Arid/inland cities (Delhi_NCR, Ahmedabad): dust/haze spectral shift
  - Plateau cities (Bangalore, Hyderabad, Pune): moderate baseline
  - Growth rates influence transition-class frequency
"""

import os
import sys
import copy
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.models import UrbanClassifier
from src.losses import CombinedLoss
from src.dataset import generate_patch, SPECTRAL_PROFILES
from src.metrics import evaluate
from src.train import train_one_epoch, validate


# ═══════════════════════════════════════════════════════════
#  City-Specific Spectral Profiles
# ═══════════════════════════════════════════════════════════

# Per-city spectral shift vectors (additive, per channel)
# Channels: Blue, Green, Red, NIR, SWIR1, SWIR2
CITY_SPECTRAL_SHIFTS = {
    # Coastal cities: higher blue/green (water proximity), suppressed SWIR
    "Mumbai":    np.array([ 0.02,  0.02,  0.00, -0.02,  0.00, -0.03], dtype=np.float32),
    "Chennai":   np.array([ 0.03,  0.02,  0.01, -0.03,  0.00, -0.04], dtype=np.float32),
    # Arid / northern plains: dust haze raises visible bands, lower NIR vegetation
    "Delhi_NCR": np.array([ 0.03,  0.03,  0.04, -0.04,  0.02,  0.02], dtype=np.float32),
    "Ahmedabad": np.array([ 0.02,  0.02,  0.03, -0.03,  0.03,  0.03], dtype=np.float32),
    # Deccan plateau / IT corridor: moderate, slightly greener
    "Bangalore": np.array([-0.01, -0.01, -0.02,  0.04, -0.01, -0.01], dtype=np.float32),
    "Hyderabad": np.array([ 0.00,  0.00, -0.01,  0.03, -0.01,  0.00], dtype=np.float32),
    "Pune":      np.array([-0.01,  0.00, -0.01,  0.02,  0.00, -0.01], dtype=np.float32),
}

# Per-city texture scale preferences (simulating urban morphology differences)
CITY_TEXTURE_SCALES = {
    "Mumbai":     [16, 24],    # dense, fine-grained urban fabric
    "Delhi_NCR":  [32, 48, 64],  # sprawling, coarser patterns
    "Bangalore":  [24, 32],
    "Hyderabad":  [24, 32, 48],
    "Chennai":    [16, 24, 32],
    "Pune":       [24, 32],
    "Ahmedabad":  [32, 48],
}


def _city_class_distribution(city_name):
    """
    Adjust class distribution based on city growth rate.
    Higher growth -> more transition patches, fewer non-urban.
    """
    growth = CITY_GROWTH.get(city_name, 3.0)
    # Normalize growth to [0, 1] range across cities
    min_g = min(CITY_GROWTH.values())
    max_g = max(CITY_GROWTH.values())
    norm_g = (growth - min_g) / (max_g - min_g + 1e-8)

    # Base: [0.45, 0.40, 0.15]
    urban = 0.42 + 0.06 * norm_g       # faster growth = slightly more urban
    transition = 0.12 + 0.10 * norm_g   # faster growth = more transition
    non_urban = 1.0 - urban - transition
    non_urban = max(non_urban, 0.10)
    total = urban + non_urban + transition
    return [urban / total, non_urban / total, transition / total]


def generate_city_patch(label, city_name, patch_size=PATCH_SIZE,
                        num_channels=NUM_CHANNELS):
    """
    Generate a synthetic multispectral patch with city-specific spectral shifts
    and texture characteristics.
    """
    base_patch = generate_patch(label, patch_size, num_channels)

    # Apply city-specific spectral shift
    shift = CITY_SPECTRAL_SHIFTS.get(city_name, np.zeros(num_channels, dtype=np.float32))
    for c in range(min(num_channels, len(shift))):
        base_patch[c] += shift[c]

    # Add city-specific noise variance (arid cities have more spectral noise)
    if city_name in ("Delhi_NCR", "Ahmedabad"):
        noise = np.random.normal(0, 0.015, base_patch.shape).astype(np.float32)
        base_patch += noise
    elif city_name in ("Mumbai", "Chennai"):
        # Coastal moisture adds slight correlation between bands
        moisture = np.random.normal(0, 0.01, (1, patch_size, patch_size)).astype(np.float32)
        base_patch += moisture

    return np.clip(base_patch, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════
#  City-Specific Dataset
# ═══════════════════════════════════════════════════════════

class CitySpecificDataset(Dataset):
    """
    Dataset that generates patches with city-specific spectral characteristics.

    Each city gets a different spectral shift, texture scale, and class
    distribution (informed by CITY_GROWTH).
    """

    def __init__(self, city_name, num_patches, seed=SEED, transform=None):
        super().__init__()
        self.city_name = city_name
        self.num_patches = num_patches
        self.transform = transform

        rng = np.random.RandomState(seed)
        dist = _city_class_distribution(city_name)
        self.labels = rng.choice(NUM_CLASSES, size=num_patches, p=dist).tolist()
        self.cities = [city_name] * num_patches

        # Store seeds for reproducible patch generation
        self._patch_seeds = rng.randint(0, 2**31, size=num_patches).tolist()

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        label = self.labels[idx]
        # Set per-patch random state for reproducibility
        np.random.seed(self._patch_seeds[idx])
        random.seed(self._patch_seeds[idx])

        patch = generate_city_patch(label, self.city_name)
        patch = torch.from_numpy(patch)

        if self.transform:
            patch = self.transform(patch)

        return patch, label


class MultiCityDataset(Dataset):
    """
    Combines CitySpecificDataset instances from multiple cities into one dataset.
    """

    def __init__(self, city_names, patches_per_city, seed=SEED, transform=None):
        super().__init__()
        self.datasets = []
        self.city_names = list(city_names)
        for i, city in enumerate(city_names):
            ds = CitySpecificDataset(city, patches_per_city,
                                     seed=seed + i * 1000, transform=transform)
            self.datasets.append(ds)

        # Build flat index mapping
        self._lengths = [len(ds) for ds in self.datasets]
        self._cumulative = np.cumsum([0] + self._lengths)

    def __len__(self):
        return sum(self._lengths)

    def __getitem__(self, idx):
        # Find which sub-dataset this idx belongs to
        for i, ds in enumerate(self.datasets):
            if idx < self._cumulative[i + 1]:
                local_idx = idx - self._cumulative[i]
                return ds[local_idx]
        raise IndexError(f"Index {idx} out of range for MultiCityDataset of size {len(self)}")


# ═══════════════════════════════════════════════════════════
#  Helper: quick model training
# ═══════════════════════════════════════════════════════════

TRAIN_PATCHES_PER_CITY = 500
TEST_PATCHES_PER_CITY = 150


def _build_model(device, pretrained=True):
    """Build a fresh UrbanClassifier on the given device."""
    model = UrbanClassifier(DEFAULT_BACKBONE, pretrained=pretrained).to(device)
    return model


def _train_model(model, train_loader, device, epochs=5, lr=1e-3):
    """Train a model for a fixed number of epochs. Returns trained model."""
    criterion = CombinedLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=False
        )
        scheduler.step()

        if train_loss < best_loss:
            best_loss = train_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f"    Epoch {epoch}/{epochs} | Loss {train_loss:.4f} | Acc {train_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _evaluate_model(model, test_loader, device):
    """Evaluate model and return metrics dict."""
    criterion = CombinedLoss().to(device)
    _, metrics = validate(model, test_loader, criterion, device)
    return metrics


# ═══════════════════════════════════════════════════════════
#  Experiment 1: Leave-One-City-Out (LOCO)
# ═══════════════════════════════════════════════════════════

def run_loco_experiment(device, epochs=5):
    """
    Leave-One-City-Out cross-validation.
    For each city, train on the remaining 6, test on the held-out city.
    Returns dict mapping city -> test metrics.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT: Leave-One-City-Out (LOCO) Generalization")
    print("=" * 70)

    results = {}

    for held_out in CITIES:
        print(f"\n  --- Held-out city: {held_out} ---")
        train_cities = [c for c in CITIES if c != held_out]

        # Build datasets
        train_ds = MultiCityDataset(train_cities, TRAIN_PATCHES_PER_CITY, seed=SEED)
        test_ds = CitySpecificDataset(held_out, TEST_PATCHES_PER_CITY, seed=SEED + 999)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=True)

        # Train and evaluate
        model = _build_model(device)
        model = _train_model(model, train_loader, device, epochs=epochs)
        metrics = _evaluate_model(model, test_loader, device)

        results[held_out] = metrics
        print(f"  {held_out}: OA={metrics['oa']:.4f}, F1={metrics['f1']:.4f}, "
              f"mIoU={metrics['miou']:.4f}")

        # Free memory
        del model, train_ds, test_ds
        torch.cuda.empty_cache() if device == "cuda" else None

    # Summary
    print(f"\n{'='*70}")
    print("  LOCO SUMMARY")
    print(f"{'='*70}")
    oas = [results[c]["oa"] for c in CITIES]
    f1s = [results[c]["f1"] for c in CITIES]
    print(f"  Mean OA:  {np.mean(oas):.4f} +/- {np.std(oas):.4f}")
    print(f"  Mean F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # Plot
    _plot_loco_results(results)

    return results


def _plot_loco_results(results):
    """Bar chart of per-city LOCO accuracy and F1."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    cities = list(results.keys())
    oas = [results[c]["oa"] for c in cities]
    f1s = [results[c]["f1"] for c in cities]
    mious = [results[c]["miou"] for c in cities]

    x = np.arange(len(cities))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, oas, width, label="Overall Accuracy", color="#2196F3")
    bars2 = ax.bar(x, f1s, width, label="Weighted F1", color="#4CAF50")
    bars3 = ax.bar(x + width, mious, width, label="Mean IoU", color="#FF9800")

    ax.set_xlabel("Held-Out City", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Leave-One-City-Out Generalization Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    # Add mean line
    mean_oa = np.mean(oas)
    ax.axhline(y=mean_oa, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(len(cities) - 0.5, mean_oa + 0.01, f"Mean OA={mean_oa:.3f}",
            color="red", fontsize=9, ha="right")

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "loco_cross_city_generalization.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


# ═══════════════════════════════════════════════════════════
#  Experiment 1b: Leave-Two-Cities-Out
# ═══════════════════════════════════════════════════════════

def run_leave_two_out_experiment(device, epochs=5):
    """
    Leave-Two-Cities-Out with geographically meaningful pairs:
      - (Mumbai, Chennai): both coastal
      - (Delhi_NCR, Ahmedabad): both northern / arid
      - (Bangalore, Hyderabad): both IT-driven southern plateau
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT: Leave-Two-Cities-Out")
    print("=" * 70)

    pairs = [
        ("Mumbai", "Chennai"),        # coastal
        ("Delhi_NCR", "Ahmedabad"),   # northern
        ("Bangalore", "Hyderabad"),   # IT-driven southern
    ]
    pair_labels = ["Coastal", "Northern", "Southern IT"]
    results = {}

    for (c1, c2), pair_label in zip(pairs, pair_labels):
        print(f"\n  --- Held-out: {c1} & {c2} ({pair_label}) ---")
        train_cities = [c for c in CITIES if c not in (c1, c2)]

        train_ds = MultiCityDataset(train_cities, TRAIN_PATCHES_PER_CITY, seed=SEED)
        test_ds1 = CitySpecificDataset(c1, TEST_PATCHES_PER_CITY, seed=SEED + 999)
        test_ds2 = CitySpecificDataset(c2, TEST_PATCHES_PER_CITY, seed=SEED + 998)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)
        test_loader1 = DataLoader(test_ds1, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=True)
        test_loader2 = DataLoader(test_ds2, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=True)

        model = _build_model(device)
        model = _train_model(model, train_loader, device, epochs=epochs)

        m1 = _evaluate_model(model, test_loader1, device)
        m2 = _evaluate_model(model, test_loader2, device)

        results[(c1, c2)] = {
            "pair_label": pair_label,
            c1: m1,
            c2: m2,
            "mean_oa": (m1["oa"] + m2["oa"]) / 2,
            "mean_f1": (m1["f1"] + m2["f1"]) / 2,
        }

        print(f"  {c1}: OA={m1['oa']:.4f}, F1={m1['f1']:.4f}")
        print(f"  {c2}: OA={m2['oa']:.4f}, F1={m2['f1']:.4f}")
        print(f"  Pair mean OA: {results[(c1, c2)]['mean_oa']:.4f}")

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    return results


# ═══════════════════════════════════════════════════════════
#  Experiment 2: Few-Shot Adaptation
# ═══════════════════════════════════════════════════════════

def run_few_shot_adaptation(device, epochs=3):
    """
    Train on 6 cities, then fine-tune with K={5,10,25,50,100} labelled
    samples from the held-out city. Reports adaptation curve.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT: Few-Shot Domain Adaptation")
    print("=" * 70)

    k_values = [5, 10, 25, 50, 100]
    # Use a subset of cities for tractability
    target_cities = ["Mumbai", "Delhi_NCR", "Bangalore"]
    results = {city: {} for city in target_cities}

    for target_city in target_cities:
        print(f"\n  --- Target city: {target_city} ---")
        source_cities = [c for c in CITIES if c != target_city]

        # Train base model on source cities
        print(f"  Training base model on {len(source_cities)} source cities...")
        train_ds = MultiCityDataset(source_cities, TRAIN_PATCHES_PER_CITY, seed=SEED)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)

        base_model = _build_model(device)
        base_model = _train_model(base_model, train_loader, device, epochs=epochs + 2)
        base_state = copy.deepcopy(base_model.state_dict())

        # Test set for the target city
        test_ds = CitySpecificDataset(target_city, TEST_PATCHES_PER_CITY, seed=SEED + 999)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=True)

        # Zero-shot baseline (no target data)
        zero_shot_metrics = _evaluate_model(base_model, test_loader, device)
        results[target_city][0] = zero_shot_metrics
        print(f"  K=0 (zero-shot): OA={zero_shot_metrics['oa']:.4f}")

        # Few-shot fine-tuning for each K
        for k in k_values:
            print(f"  Fine-tuning with K={k} target samples...")
            # Create tiny fine-tuning dataset
            ft_ds = CitySpecificDataset(target_city, k, seed=SEED + k)
            ft_loader = DataLoader(ft_ds, batch_size=min(k, BATCH_SIZE), shuffle=True,
                                   num_workers=0, pin_memory=True)

            # Reset to base model
            ft_model = _build_model(device)
            ft_model.load_state_dict(base_state)

            # Fine-tune with lower learning rate
            ft_model = _train_model(ft_model, ft_loader, device,
                                    epochs=epochs, lr=1e-4)

            metrics = _evaluate_model(ft_model, test_loader, device)
            results[target_city][k] = metrics
            print(f"    K={k}: OA={metrics['oa']:.4f}, F1={metrics['f1']:.4f}")

            del ft_model
            torch.cuda.empty_cache() if device == "cuda" else None

        del base_model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Plot
    _plot_few_shot_curves(results, k_values)

    return results


def _plot_few_shot_curves(results, k_values):
    """Plot few-shot adaptation curves: K vs accuracy for each target city."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {"Mumbai": "#E53935", "Delhi_NCR": "#1E88E5", "Bangalore": "#43A047"}
    markers = {"Mumbai": "o", "Delhi_NCR": "s", "Bangalore": "^"}
    all_k = [0] + k_values

    # OA subplot
    ax = axes[0]
    for city, city_results in results.items():
        oas = [city_results[k]["oa"] for k in all_k]
        ax.plot(all_k, oas, marker=markers.get(city, "o"), linewidth=2,
                markersize=8, label=city, color=colors.get(city, None))
    ax.set_xlabel("Number of Target Samples (K)", fontsize=12)
    ax.set_ylabel("Overall Accuracy", fontsize=12)
    ax.set_title("Few-Shot Adaptation: Accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(all_k)

    # F1 subplot
    ax = axes[1]
    for city, city_results in results.items():
        f1s = [city_results[k]["f1"] for k in all_k]
        ax.plot(all_k, f1s, marker=markers.get(city, "o"), linewidth=2,
                markersize=8, label=city, color=colors.get(city, None))
    ax.set_xlabel("Number of Target Samples (K)", fontsize=12)
    ax.set_ylabel("Weighted F1 Score", fontsize=12)
    ax.set_title("Few-Shot Adaptation: F1 Score", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(all_k)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "few_shot_adaptation_curves.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


# ═══════════════════════════════════════════════════════════
#  Experiment 3: Cross-City Transfer Matrix
# ═══════════════════════════════════════════════════════════

def run_transfer_matrix(device, epochs=5):
    """
    Compute N x N transfer matrix: entry (i, j) = OA of model trained on
    city i, tested on city j.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT: Cross-City Transfer Matrix")
    print("=" * 70)

    n = len(CITIES)
    matrix_oa = np.zeros((n, n), dtype=np.float64)
    matrix_f1 = np.zeros((n, n), dtype=np.float64)

    # Pre-build test loaders
    test_loaders = {}
    for j, city_j in enumerate(CITIES):
        test_ds = CitySpecificDataset(city_j, TEST_PATCHES_PER_CITY, seed=SEED + 500 + j)
        test_loaders[city_j] = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=0, pin_memory=True)

    for i, city_i in enumerate(CITIES):
        print(f"\n  Training on {city_i}...")
        train_ds = CitySpecificDataset(city_i, TRAIN_PATCHES_PER_CITY, seed=SEED + i)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)

        model = _build_model(device)
        model = _train_model(model, train_loader, device, epochs=epochs)

        for j, city_j in enumerate(CITIES):
            metrics = _evaluate_model(model, test_loaders[city_j], device)
            matrix_oa[i, j] = metrics["oa"]
            matrix_f1[i, j] = metrics["f1"]
            print(f"    {city_i} -> {city_j}: OA={metrics['oa']:.4f}")

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Plot
    _plot_transfer_matrix(matrix_oa, "Overall Accuracy", "transfer_matrix_oa.png")
    _plot_transfer_matrix(matrix_f1, "Weighted F1", "transfer_matrix_f1.png")

    return {"oa": matrix_oa, "f1": matrix_f1}


def _plot_transfer_matrix(matrix, metric_name, filename):
    """Heatmap of the N x N transfer matrix."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(len(CITIES)))
    ax.set_yticks(range(len(CITIES)))
    ax.set_xticklabels(CITIES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CITIES, fontsize=10)
    ax.set_xlabel("Test City", fontsize=12)
    ax.set_ylabel("Train City", fontsize=12)
    ax.set_title(f"Cross-City Transfer Matrix ({metric_name})",
                 fontsize=14, fontweight="bold")

    # Annotate cells
    for i in range(len(CITIES)):
        for j in range(len(CITIES)):
            val = matrix[i, j]
            color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if i == j else "normal")

    # Highlight diagonal
    for k in range(len(CITIES)):
        rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                              edgecolor="blue", linewidth=2)
        ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name, fontsize=11)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


# ═══════════════════════════════════════════════════════════
#  Experiment 4: Domain Shift Analysis (MMD)
# ═══════════════════════════════════════════════════════════

def _gaussian_kernel(x, y, sigma=1.0):
    """Compute Gaussian RBF kernel between batches x and y."""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1)  # (n, 1, d)
    y = y.unsqueeze(0)  # (1, m, d)

    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)

    return torch.exp(-torch.sum((tiled_x - tiled_y) ** 2, dim=2) / (2 * sigma ** 2))


def compute_mmd(x, y, sigma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets of feature vectors.
    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    """
    xx = _gaussian_kernel(x, x, sigma)
    yy = _gaussian_kernel(y, y, sigma)
    xy = _gaussian_kernel(x, y, sigma)

    n = x.size(0)
    m = y.size(0)

    mmd_sq = (xx.sum() / (n * n)) + (yy.sum() / (m * m)) - 2 * (xy.sum() / (n * m))
    return torch.clamp(mmd_sq, min=0.0).sqrt().item()


@torch.no_grad()
def _extract_city_features(model, city_name, device, num_patches=200):
    """Extract feature vectors for patches from a specific city."""
    model.eval()
    ds = CitySpecificDataset(city_name, num_patches, seed=SEED + 777)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

    features_list = []
    for x, _ in loader:
        x = x.to(device)
        feat = model.get_feature_vector(x)
        features_list.append(feat.cpu())

    return torch.cat(features_list, dim=0)


def run_domain_analysis(model, device):
    """
    Compute pairwise MMD distance between all city pairs.
    Produces a symmetric distance matrix and heatmap.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT: Domain Shift Analysis (MMD)")
    print("=" * 70)

    model.eval()
    n = len(CITIES)

    # Extract features for all cities
    print("  Extracting features per city...")
    city_features = {}
    for city in CITIES:
        city_features[city] = _extract_city_features(model, city, device, num_patches=200)
        print(f"    {city}: {city_features[city].shape}")

    # Compute pairwise MMD
    print("  Computing pairwise MMD distances...")
    mmd_matrix = np.zeros((n, n), dtype=np.float64)

    # Use multiple sigma values for a more robust MMD estimate
    sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]

    for i in range(n):
        for j in range(i + 1, n):
            # Multi-kernel MMD: average over multiple bandwidths
            mmd_vals = []
            for sigma in sigmas:
                mmd_val = compute_mmd(city_features[CITIES[i]],
                                      city_features[CITIES[j]], sigma=sigma)
                mmd_vals.append(mmd_val)
            avg_mmd = np.mean(mmd_vals)
            mmd_matrix[i, j] = avg_mmd
            mmd_matrix[j, i] = avg_mmd
            print(f"    MMD({CITIES[i]}, {CITIES[j]}) = {avg_mmd:.4f}")

    # Plot
    _plot_domain_distance_matrix(mmd_matrix)

    return mmd_matrix


def _plot_domain_distance_matrix(mmd_matrix):
    """Heatmap of pairwise MMD domain distances."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = np.max(mmd_matrix) if np.max(mmd_matrix) > 0 else 1.0
    im = ax.imshow(mmd_matrix, cmap="Blues", vmin=0.0, vmax=vmax, aspect="equal")

    ax.set_xticks(range(len(CITIES)))
    ax.set_yticks(range(len(CITIES)))
    ax.set_xticklabels(CITIES, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(CITIES, fontsize=10)
    ax.set_xlabel("City", fontsize=12)
    ax.set_ylabel("City", fontsize=12)
    ax.set_title("Inter-City Domain Distance (Multi-Kernel MMD)",
                 fontsize=14, fontweight="bold")

    # Annotate cells
    for i in range(len(CITIES)):
        for j in range(len(CITIES)):
            val = mmd_matrix[i, j]
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MMD Distance", fontsize=11)

    # Add geographic grouping annotations
    # Coastal: Mumbai(0), Chennai(4); Northern: Delhi(1), Ahmedabad(6)
    # Southern IT: Bangalore(2), Hyderabad(3)
    groups = [
        ([0, 4], "Coastal", "#E53935"),
        ([1, 6], "Northern", "#1E88E5"),
        ([2, 3], "Southern IT", "#43A047"),
    ]
    for indices, label, color in groups:
        for idx in indices:
            rect_h = plt.Rectangle((-0.5, idx - 0.5), len(CITIES), 1,
                                   fill=False, edgecolor=color,
                                   linewidth=1.5, linestyle="--")
            ax.add_patch(rect_h)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "domain_distance_mmd_matrix.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {path}")


# ═══════════════════════════════════════════════════════════
#  Run All Experiments
# ═══════════════════════════════════════════════════════════

def run_cross_city_experiments(device):
    """
    Execute the full cross-city generalization experiment suite:
      1. Leave-One-City-Out (LOCO)
      2. Leave-Two-Cities-Out
      3. Few-Shot Adaptation
      4. Cross-City Transfer Matrix
      5. Domain Shift Analysis (MMD)
    """
    print("\n" + "#" * 70)
    print("#  CROSS-CITY GENERALIZATION EXPERIMENT SUITE")
    print("#" * 70)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    all_results = {}

    # 1. LOCO
    loco_results = run_loco_experiment(device, epochs=5)
    all_results["loco"] = loco_results

    # 2. Leave-Two-Cities-Out
    l2co_results = run_leave_two_out_experiment(device, epochs=5)
    all_results["leave_two_out"] = l2co_results

    # 3. Few-Shot Adaptation
    fewshot_results = run_few_shot_adaptation(device, epochs=3)
    all_results["few_shot"] = fewshot_results

    # 4. Transfer Matrix
    transfer_results = run_transfer_matrix(device, epochs=5)
    all_results["transfer_matrix"] = transfer_results

    # 5. Domain Analysis (use a freshly trained multi-city model)
    print("\n  Training multi-city model for domain analysis...")
    multi_ds = MultiCityDataset(CITIES, TRAIN_PATCHES_PER_CITY, seed=SEED)
    multi_loader = DataLoader(multi_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    domain_model = _build_model(device)
    domain_model = _train_model(domain_model, multi_loader, device, epochs=5)
    mmd_matrix = run_domain_analysis(domain_model, device)
    all_results["mmd_matrix"] = mmd_matrix

    # Save summary
    _save_summary(all_results)

    print("\n" + "#" * 70)
    print("#  ALL CROSS-CITY EXPERIMENTS COMPLETE")
    print(f"#  Figures saved to: {FIGURE_DIR}")
    print("#" * 70)

    return all_results


def _save_summary(all_results):
    """Save a text summary of all results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "cross_city_summary.txt")

    with open(path, "w") as f:
        f.write("CROSS-CITY GENERALIZATION EXPERIMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # LOCO
        f.write("1. Leave-One-City-Out (LOCO)\n")
        f.write("-" * 40 + "\n")
        if "loco" in all_results:
            for city in CITIES:
                m = all_results["loco"][city]
                f.write(f"  {city:15s}: OA={m['oa']:.4f}  F1={m['f1']:.4f}  "
                        f"mIoU={m['miou']:.4f}\n")
            oas = [all_results["loco"][c]["oa"] for c in CITIES]
            f.write(f"  {'Mean':15s}: OA={np.mean(oas):.4f} +/- {np.std(oas):.4f}\n")

        # Leave-Two-Out
        f.write("\n2. Leave-Two-Cities-Out\n")
        f.write("-" * 40 + "\n")
        if "leave_two_out" in all_results:
            for key, val in all_results["leave_two_out"].items():
                c1, c2 = key
                f.write(f"  {val['pair_label']:15s} ({c1}, {c2}): "
                        f"Mean OA={val['mean_oa']:.4f}\n")

        # Few-Shot
        f.write("\n3. Few-Shot Adaptation\n")
        f.write("-" * 40 + "\n")
        if "few_shot" in all_results:
            for city, city_res in all_results["few_shot"].items():
                f.write(f"  {city}:\n")
                for k in sorted(city_res.keys()):
                    f.write(f"    K={k:3d}: OA={city_res[k]['oa']:.4f}  "
                            f"F1={city_res[k]['f1']:.4f}\n")

        # Transfer Matrix
        f.write("\n4. Transfer Matrix (OA)\n")
        f.write("-" * 40 + "\n")
        if "transfer_matrix" in all_results:
            mat = all_results["transfer_matrix"]["oa"]
            header = "          " + "  ".join(f"{c:>10s}" for c in CITIES) + "\n"
            f.write(header)
            for i, city in enumerate(CITIES):
                row = f"  {city:8s}" + "  ".join(f"{mat[i,j]:10.4f}" for j in range(len(CITIES)))
                f.write(row + "\n")

        f.write(f"\nResults saved at {path}\n")

    print(f"  [Saved] {path}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    run_cross_city_experiments(device)
