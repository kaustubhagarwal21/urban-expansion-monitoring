"""
Improved SVM baseline with feature engineering, normalization, PCA, and grid search.
Goal: Push SVM accuracy above published 91.01% (Chamoli 2024).
"""
import sys, os, json, time, warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['INDIAN_PATCH_ROOT'] = 'data/indian_cities_locked'

from src.real_data_loaders import IndianCityDataset

SEEDS = [42, 123, 7]
CITIES = ['Mumbai', 'Delhi_NCR', 'Bangalore']


def compute_spectral_indices(patch_6ch):
    """Compute NDVI, NDBI, NDWI from 6-band patch (B2,B3,B4,B8,B11,B12)."""
    eps = 1e-8
    B2, B3, B4, B8, B11, B12 = [patch_6ch[i] for i in range(6)]

    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    ndbi = (B11 - B8) / (B11 + B8 + eps)
    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    # SAVI = ((NIR - Red) / (NIR + Red + 0.5)) * 1.5
    savi = ((B8 - B4) / (B8 + B4 + 0.5)) * 1.5
    # BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
    bsi = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + eps)

    return ndvi, ndbi, ndwi, savi, bsi


def extract_features(patch_6ch):
    """Extract rich feature vector from a 6-channel patch."""
    features = []

    # 1. Per-band statistics (6 bands x 4 stats = 24 features)
    for i in range(6):
        band = patch_6ch[i]
        features.extend([band.mean(), band.std(), np.percentile(band, 25), np.percentile(band, 75)])

    # 2. Spectral indices statistics (5 indices x 4 stats = 20 features)
    ndvi, ndbi, ndwi, savi, bsi = compute_spectral_indices(patch_6ch)
    for idx in [ndvi, ndbi, ndwi, savi, bsi]:
        features.extend([idx.mean(), idx.std(), np.percentile(idx, 25), np.percentile(idx, 75)])

    # 3. Texture features - GLCM-like (variance of local differences)
    for i in range(6):
        band = patch_6ch[i]
        # Horizontal gradient magnitude
        h_diff = np.abs(np.diff(band, axis=1)).mean()
        # Vertical gradient magnitude
        v_diff = np.abs(np.diff(band, axis=0)).mean()
        features.extend([h_diff, v_diff])

    # 4. Band ratios (key ones)
    eps = 1e-8
    B2, B3, B4, B8, B11, B12 = [patch_6ch[i].mean() for i in range(6)]
    features.append(B8 / (B4 + eps))  # NIR/Red
    features.append(B11 / (B8 + eps))  # SWIR/NIR
    features.append(B4 / (B3 + eps))   # Red/Green
    features.append(B12 / (B11 + eps)) # SWIR2/SWIR1

    return np.array(features, dtype=np.float32)


def load_dataset_features(ds, indices):
    """Load patches and extract engineered features."""
    X, y = [], []
    for i in indices:
        img, label = ds[i]
        patch = img.numpy()  # (6, 256, 256)
        feat = extract_features(patch)
        X.append(feat)
        y.append(label)
    return np.array(X), np.array(y)


def load_dataset_flat(ds, indices):
    """Load patches as flattened raw pixels (original approach)."""
    X, y = [], []
    for i in indices:
        img, label = ds[i]
        X.append(img.numpy().flatten())
        y.append(label)
    return np.array(X), np.array(y)


def run_improved_svm(seed):
    """Run improved SVM + RF for one seed."""
    print(f"\n{'='*60}")
    print(f"  SEED = {seed}")
    print(f"{'='*60}")

    np.random.seed(seed)

    ds = IndianCityDataset(cities=CITIES)
    labels = [s[1] for s in ds.samples]
    indices = list(range(len(ds)))

    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=seed)
    train_labels = [labels[i] for i in train_idx]
    train_idx2, val_idx = train_test_split(train_idx, test_size=0.15, stratify=train_labels, random_state=seed)

    print(f"Train: {len(train_idx2)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # --- Engineered features ---
    print("Extracting engineered features...")
    t0 = time.time()
    X_train, y_train = load_dataset_features(ds, train_idx2)
    X_val, y_val = load_dataset_features(ds, val_idx)
    X_test, y_test = load_dataset_features(ds, test_idx)
    feat_time = time.time() - t0
    print(f"Feature extraction: {feat_time:.1f}s, feature dim: {X_train.shape[1]}")

    # Combine train+val for final training (use val for grid search)
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    results = {}

    # --- 1. Improved SVM with StandardScaler + PCA + Grid Search ---
    print("\n--- SVM with Feature Engineering + Scaler + PCA + Grid Search ---")
    t0 = time.time()

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.99, random_state=seed)),  # keep 99% variance
        ('svm', SVC(random_state=seed))
    ])

    param_grid = {
        'svm__C': [1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01],
        'svm__kernel': ['rbf'],
    }

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=0)
    grid.fit(X_trainval, y_trainval)

    svm_pred = grid.predict(X_test)
    svm_oa = accuracy_score(y_test, svm_pred) * 100
    svm_f1 = f1_score(y_test, svm_pred, average='macro')
    svm_time = time.time() - t0

    print(f"Best params: {grid.best_params_}")
    print(f"SVM (improved): OA={svm_oa:.2f}%, F1={svm_f1:.4f}, time={svm_time:.1f}s")
    print(classification_report(y_test, svm_pred, target_names=['Urban', 'Non-Urban', 'Transition']))

    results['svm_improved'] = {
        'oa': round(svm_oa, 2),
        'f1': round(float(svm_f1), 4),
        'best_params': {k: str(v) for k, v in grid.best_params_.items()},
        'pca_components': int(grid.best_estimator_.named_steps['pca'].n_components_),
        'time_sec': round(svm_time, 1),
    }

    # --- 2. Improved RF with engineered features ---
    print("\n--- Random Forest with Feature Engineering ---")
    t0 = time.time()

    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=2,
                                       n_jobs=1, random_state=seed))
    ])
    rf_pipe.fit(X_trainval, y_trainval)
    rf_pred = rf_pipe.predict(X_test)
    rf_oa = accuracy_score(y_test, rf_pred) * 100
    rf_f1 = f1_score(y_test, rf_pred, average='macro')
    rf_time = time.time() - t0

    print(f"RF (improved): OA={rf_oa:.2f}%, F1={rf_f1:.4f}, time={rf_time:.1f}s")

    results['rf_improved'] = {
        'oa': round(rf_oa, 2),
        'f1': round(float(rf_f1), 4),
        'time_sec': round(rf_time, 1),
    }

    return results


def main():
    all_results = {}

    for seed in SEEDS:
        all_results[f'seed_{seed}'] = run_improved_svm(seed)

    # Aggregate
    print(f"\n{'='*60}")
    print("  AGGREGATED RESULTS (mean +/- std over 3 seeds)")
    print(f"{'='*60}")

    summary = {}
    for method in ['svm_improved', 'rf_improved']:
        oas = [all_results[f'seed_{s}'][method]['oa'] for s in SEEDS]
        f1s = [all_results[f'seed_{s}'][method]['f1'] for s in SEEDS]
        summary[method] = {
            'oa_mean': round(float(np.mean(oas)), 2),
            'oa_std': round(float(np.std(oas)), 2),
            'f1_mean': round(float(np.mean(f1s)), 4),
            'f1_std': round(float(np.std(f1s)), 4),
            'per_seed_oa': {str(s): oas[i] for i, s in enumerate(SEEDS)},
        }
        print(f"{method:25s}: OA = {summary[method]['oa_mean']:.2f} +/- {summary[method]['oa_std']:.2f}%  "
              f"F1 = {summary[method]['f1_mean']:.4f} +/- {summary[method]['f1_std']:.4f}")

    # Compare with published
    print(f"\n--- Comparison ---")
    print(f"Published SVM (Chamoli 2024):     91.01%")
    print(f"Our original SVM (Table 1):       89.2 +/- 0.4%")
    print(f"Our improved SVM:                 {summary['svm_improved']['oa_mean']:.2f} +/- {summary['svm_improved']['oa_std']:.2f}%")
    print(f"Our improved RF:                  {summary['rf_improved']['oa_mean']:.2f} +/- {summary['rf_improved']['oa_std']:.2f}%")
    print(f"Our DL best (ResNet50):           97.5 +/- 0.2%")

    beaten = summary['svm_improved']['oa_mean'] > 91.01
    print(f"\nBeaten published 91.01%? {'YES' if beaten else 'NO'}")

    # Save
    output = {
        'seeds': SEEDS,
        'per_seed': all_results,
        'summary': summary,
        'published_svm_chamoli2024': 91.01,
        'our_dl_best_resnet50': 97.5,
        'beaten_published': beaten,
    }

    out_path = 'outputs/research_results/improved_svm_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
