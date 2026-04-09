"""
Classical ML baselines: SVM and Random Forest.
Works with both synthetic and real (EuroSAT) datasets.
Extracts hand-crafted spectral features and evaluates on the same test split.
"""

import os, sys, time, numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.metrics import evaluate


def extract_features(dataset, max_samples=None):
    """
    Extract hand-crafted spectral features per patch:
    - Per-channel mean, std, min, max  (6 channels x 4 = 24 features)
    - NDVI-like and NDBI-like index stats (4 features)
    Total = 28 features per patch.
    """
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    X, y = [], []
    for i in range(n):
        item = dataset[i]
        # Handle both (patch, label) and (patch, sar, label) formats
        if len(item) == 2:
            patch, label = item
        else:
            patch, _, label = item[0], item[1], item[-1]
        arr = patch.numpy() if hasattr(patch, 'numpy') else np.array(patch)
        feats = []
        for c in range(arr.shape[0]):
            ch = arr[c]
            feats.extend([ch.mean(), ch.std(), ch.min(), ch.max()])
        # Derived indices from channels 3 (NIR) and 2 (Red)
        nir, red = arr[min(3, arr.shape[0]-1)], arr[min(2, arr.shape[0]-1)]
        ndvi = (nir - red) / (nir + red + 1e-8)
        feats.extend([ndvi.mean(), ndvi.std()])
        # Built-up index from channels 4 (SWIR1) and 3 (NIR)
        swir = arr[min(4, arr.shape[0]-1)]
        nir2 = arr[min(3, arr.shape[0]-1)]
        ndbi = (swir - nir2) / (swir + nir2 + 1e-8)
        feats.extend([ndbi.mean(), ndbi.std()])
        X.append(feats)
        y.append(int(label))
    return np.array(X, dtype=np.float32), np.array(y)


def run_baselines(train_dataset=None, test_dataset=None, max_train=5000, max_test=2000):
    """
    Train and evaluate SVM and Random Forest baselines.

    Args:
        train_dataset: PyTorch Dataset (if None, uses synthetic)
        test_dataset: PyTorch Dataset (if None, uses synthetic)
        max_train: max samples for training (SVM is slow on large sets)
        max_test: max samples for testing
    """
    print(f"\n{'='*60}")
    print("  Classical ML Baselines (SVM & Random Forest)")
    print(f"{'='*60}")

    if train_dataset is None or test_dataset is None:
        from src.dataset import UrbanExpansionDataset
        n_train = int(TOTAL_PATCHES * TRAIN_RATIO)
        n_test = TOTAL_PATCHES - n_train - int(TOTAL_PATCHES * VAL_RATIO)
        train_dataset = UrbanExpansionDataset(n_train, seed=SEED)
        test_dataset = UrbanExpansionDataset(n_test, seed=SEED + 2)

    max_train = min(max_train, len(train_dataset))
    max_test = min(max_test, len(test_dataset))

    print(f"  Extracting train features ({max_train} samples)...")
    X_train, y_train = extract_features(train_dataset, max_train)
    print(f"  Extracting test features ({max_test} samples)...")
    X_test, y_test = extract_features(test_dataset, max_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}

    # ── SVM ────────────────────────────────────────────
    print("\n  Training SVM...")
    t0 = time.time()
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
    svm.fit(X_train, y_train)
    svm_time = time.time() - t0
    svm_preds = svm.predict(X_test)
    svm_metrics = evaluate(y_test.tolist(), svm_preds.tolist(), CLASS_NAMES)
    results["SVM"] = svm_metrics
    print(f"  SVM OA: {svm_metrics['oa']:.4f} | F1: {svm_metrics['f1']:.4f} | "
          f"mIoU: {svm_metrics['miou']:.4f} | Time: {svm_time:.1f}s")
    print(svm_metrics['report'])

    # ── Random Forest ──────────────────────────────────
    print("  Training Random Forest...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=SEED,
                                n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_preds = rf.predict(X_test)
    rf_metrics = evaluate(y_test.tolist(), rf_preds.tolist(), CLASS_NAMES)
    results["RandomForest"] = rf_metrics
    print(f"  RF  OA: {rf_metrics['oa']:.4f} | F1: {rf_metrics['f1']:.4f} | "
          f"mIoU: {rf_metrics['miou']:.4f} | Time: {rf_time:.1f}s")
    print(rf_metrics['report'])

    return results


if __name__ == "__main__":
    run_baselines()
