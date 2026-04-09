"""Run all remaining Phase 5.5 multi-seed experiments."""
import sys, os, json, time, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["INDIAN_PATCH_ROOT"] = "data/indian_cities_locked"

from main import get_device, set_seed
from configs.config import BATCH_SIZE, STAGES
from src.real_data_loaders import IndianCityDataset
from src.train import progressive_train
from src.phase4_real_loco import run_real_loco, save_loco_outputs
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

device = get_device()
OUT = "outputs/research_results"
os.makedirs(OUT, exist_ok=True)

def build_loaders(seed):
    ds = IndianCityDataset(cities=["Mumbai", "Delhi_NCR", "Bangalore"])
    labels = [s[1] for s in ds.samples]
    idx = list(range(len(ds)))
    tr, te = train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
    tr2, va = train_test_split(tr, test_size=0.15, stratify=[labels[i] for i in tr], random_state=seed)
    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    return (
        DataLoader(Subset(ds, tr2), shuffle=True, **kw),
        DataLoader(Subset(ds, va), shuffle=False, **kw),
        DataLoader(Subset(ds, te), shuffle=False, **kw),
    )

def run_benchmark(backbone, seed):
    tag = f"phase2_{backbone}_seed{seed}.json"
    if os.path.exists(os.path.join(OUT, tag)):
        print(f"  SKIP {tag} (exists)")
        return
    set_seed(seed)
    loaders = build_loaders(seed)
    _, _, m, t = progressive_train(backbone_name=backbone, device=device, loaders=loaders)
    r = {"backbone": backbone, "seed": seed,
         "oa": float(m["oa"]), "f1": float(m["f1"]), "miou": float(m["miou"]),
         "precision": float(m["precision"]), "recall": float(m["recall"]),
         "training_time_min": round(t/60, 2)}
    json.dump(r, open(os.path.join(OUT, tag), "w"), indent=2)
    print(f"  DONE {tag}: OA={r['oa']:.4f}")

def run_loco(backbone, seed):
    tag = f"phase4_loco_{backbone}_seed{seed}.json"
    if os.path.exists(os.path.join(OUT, tag)):
        print(f"  SKIP {tag} (exists)")
        return
    # Temporarily override SEED in config
    import configs.config as cfg
    orig = cfg.SEED
    cfg.SEED = seed
    set_seed(seed)
    payload = run_real_loco(backbone_name=backbone, device=device, epochs_per_stage=5)
    cfg.SEED = orig
    # Save with seed suffix
    payload["seed"] = seed
    json.dump(payload, open(os.path.join(OUT, tag), "w"), indent=2)
    # MD
    md_tag = tag.replace(".json", ".md")
    lines = [f"# LOCO {backbone} seed={seed}", "",
             "| City | OA | F1 | mIoU |", "| --- | ---: | ---: | ---: |"]
    for city, m in payload["folds"].items():
        lines.append(f"| {city} | {m['oa']:.4f} | {m['f1']:.4f} | {m['miou']:.4f} |")
    a = payload["average"]
    lines.append(f"\nAvg OA={a['oa']:.4f} F1={a['f1']:.4f} mIoU={a['miou']:.4f}")
    open(os.path.join(OUT, md_tag), "w").write("\n".join(lines))
    print(f"  DONE {tag}: avg OA={a['oa']:.4f}")

def run_ablation(seed):
    tag = f"phase4_small_ablation_seed{seed}.json"
    if os.path.exists(os.path.join(OUT, tag)):
        print(f"  SKIP {tag} (exists)")
        return
    set_seed(seed)
    loaders = build_loaders(seed)
    results = {}
    configs_list = [
        ("full", {}),
        ("no_fpn", {"use_fpn": False}),
        ("ce_only", {"loss_type": "ce_only"}),
    ]
    for name, overrides in configs_list:
        print(f"    Ablation {name} seed={seed}...")
        from src.train import progressive_train as pt
        _, _, m, t = pt(backbone_name="efficientnet_b0", device=device, loaders=loaders)
        results[name] = {"oa": float(m["oa"]), "f1": float(m["f1"]), "miou": float(m["miou"]),
                         "training_time_min": round(t/60, 2)}
    out = {"seed": seed, "backbone": "efficientnet_b0", "results": results}
    json.dump(out, open(os.path.join(OUT, tag), "w"), indent=2)
    print(f"  DONE {tag}")

# ── Remaining runs ──
BACKBONES = ["efficientnet_b0", "resnet50", "swin_tiny", "mobilenet_v3_small"]

print("=" * 60)
print("  PHASE 5.5: Remaining multi-seed experiments")
print("=" * 60)

# Seed 123 remaining
print("\n--- Seed 123: LOCO Swin-Tiny ---")
run_loco("swin_tiny", 123)

print("\n--- Seed 123: Ablation ---")
run_ablation(123)

# Seed 7: all
print("\n--- Seed 7: Benchmarks ---")
for bb in BACKBONES:
    run_benchmark(bb, 7)

print("\n--- Seed 7: Baselines (SVM/RF) ---")
tag7 = "phase2_baselines_seed7.json"
if os.path.exists(os.path.join(OUT, tag7)):
    print(f"  SKIP {tag7} (exists)")
else:
    print("  Running SVM/RF seed=7...")
    set_seed(7)
    ds = IndianCityDataset(cities=["Mumbai", "Delhi_NCR", "Bangalore"])
    labels = [s[1] for s in ds.samples]
    idx = list(range(len(ds)))
    tr, te = train_test_split(idx, test_size=0.2, stratify=labels, random_state=7)
    tr2, _ = train_test_split(tr, test_size=0.15, stratify=[labels[i] for i in tr], random_state=7)
    def load_flat(idxs):
        X, y = [], []
        for i in idxs:
            img, lbl = ds[i]
            X.append(img.numpy().flatten())
            y.append(lbl)
        return np.array(X), np.array(y)
    X_tr, y_tr = load_flat(tr2)
    X_te, y_te = load_flat(te)
    from src.baselines import run_baselines
    res = run_baselines(X_tr, y_tr, X_te, y_te)
    json.dump({"seed": 7, "results": res}, open(os.path.join(OUT, tag7), "w"), indent=2)
    print(f"  DONE {tag7}")

print("\n--- Seed 7: LOCO (3 models) ---")
for bb in ["efficientnet_b0", "resnet50", "swin_tiny"]:
    run_loco(bb, 7)

print("\n--- Seed 7: Ablation ---")
run_ablation(7)

print("\n" + "=" * 60)
print("  PHASE 5.5 COMPLETE")
print("=" * 60)
