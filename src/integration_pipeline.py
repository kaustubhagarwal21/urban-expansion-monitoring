"""
End-to-End Integration Pipeline
================================
Connects the 5 pillars into a single coherent system:

  GEE Satellite Data → Base Pipeline (classify patches)
       → Urban area time series per city per year
       → Pillar IV (predict future expansion)
       → Pillar V (alert on encroachment into protected zones)

Usage:
    python src/integration_pipeline.py --classify     # Step 1: Classify all GeoTIFFs
    python src/integration_pipeline.py --timeseries   # Step 2: Build urban area time series
    python src/integration_pipeline.py --predict       # Step 3: Run Pillar IV with real data
    python src/integration_pipeline.py --alerts        # Step 4: Run Pillar V alerts
    python src/integration_pipeline.py --full          # All steps
"""

import os, sys, json, glob, time
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *
from src.models import UrbanClassifier

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GEE_DIR = r"G:\My Drive\urban_expansion_india"
INTEGRATION_DIR = os.path.join(OUTPUT_DIR, "integration")
LOCKED_INTEGRATION_CITIES = ["Mumbai", "Delhi_NCR", "Bangalore"]
LOCKED_INTEGRATION_S2_YEARS = {"2019", "2021", "2023"}
LOCKED_LANDSAT_ANCHORS = {
    "Mumbai": {"1990", "2000", "2010", "2023"},
    "Delhi_NCR": {"1990", "2000", "2010", "2023"},
    "Bangalore": {"1990", "2005", "2010", "2023"},
}
LOCKED_INTEGRATION_SEASON = "pre_monsoon"


def _env_city_filter(default_cities):
    raw = os.environ.get("INTEGRATION_CITY_FILTER", "").strip()
    if not raw:
        return list(default_cities)
    allowed = {city.strip() for city in raw.split(",") if city.strip()}
    return [city for city in default_cities if city in allowed]


def _env_max_geotiffs():
    raw = os.environ.get("INTEGRATION_MAX_GEOTIFFS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        return None


def _env_year_filter():
    raw = os.environ.get("INTEGRATION_YEAR_FILTER", "").strip()
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _env_season_filter():
    raw = os.environ.get("INTEGRATION_SEASON_FILTER", "").strip()
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _use_locked_selection():
    raw = os.environ.get("INTEGRATION_USE_LOCKED_SELECTION", "").strip().lower()
    if not raw:
        return True
    return raw not in {"0", "false", "no"}


def _env_bool(name, default=True):
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no"}


def _env_optional_s2_years():
    raw = os.environ.get("INTEGRATION_OPTIONAL_S2_YEARS", "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _parse_geotiff_metadata(filename):
    stem = filename.replace(".tif", "")
    for city in sorted(CITIES, key=len, reverse=True):
        prefix = f"S2_{city}_"
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            parts = remainder.split("_")
            if not parts:
                return "S2", city, "unknown", "unknown"
            year = parts[0]
            season = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
            return "S2", city, year, season
    for city in sorted(CITIES, key=len, reverse=True):
        prefix = f"LS_{city}_"
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            parts = remainder.split("_")
            if not parts:
                return "LS", city, "unknown", "unknown"
            year = parts[0]
            season = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
            return "LS", city, year, season
    for city in sorted(CITIES, key=len, reverse=True):
        prefix = f"SAR_{city}_"
        if stem.startswith(prefix):
            remainder = stem[len(prefix):]
            parts = remainder.split("_")
            if not parts:
                return "SAR", city, "unknown", "unknown"
            year = parts[0]
            season = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
            return "SAR", city, year, season
    parts = stem.split("_")
    source = parts[0] if parts else "unknown"
    city = parts[1] if len(parts) > 1 else "unknown"
    year = parts[2] if len(parts) > 2 else "unknown"
    season = "_".join(parts[3:]) if len(parts) > 3 else "unknown"
    return source, city, year, season


def _select_s2_files(all_files, allowed_cities):
    year_filter = _env_year_filter()
    season_filter = _env_season_filter()
    locked_mode = _use_locked_selection() and year_filter is None and season_filter is None

    if locked_mode:
        year_filter = set(LOCKED_INTEGRATION_S2_YEARS) | _env_optional_s2_years()
        season_filter = {LOCKED_INTEGRATION_SEASON}

    selected = []
    for fpath in all_files:
        fname = os.path.basename(fpath)
        source, city, year, season = _parse_geotiff_metadata(fname)
        if source != "S2":
            continue
        if city not in allowed_cities:
            continue
        if year_filter and year not in year_filter:
            continue
        if season_filter and season not in season_filter:
            continue
        selected.append((city, year, season, fpath))

    selected.sort(key=lambda item: (allowed_cities.index(item[0]), item[1], item[2], item[3]))
    return [item[3] for item in selected]


def _select_landsat_files(all_files, allowed_cities):
    locked_mode = _use_locked_selection()
    selected = []
    for fpath in all_files:
        fname = os.path.basename(fpath)
        source, city, year, season = _parse_geotiff_metadata(fname)
        if source != "LS":
            continue
        if city not in allowed_cities:
            continue
        if locked_mode:
            target_years = LOCKED_LANDSAT_ANCHORS.get(city, set())
            if year not in target_years:
                continue
            if season != LOCKED_INTEGRATION_SEASON:
                continue
        selected.append((city, year, season, fpath))
    selected.sort(key=lambda item: (allowed_cities.index(item[0]), item[1], item[2], item[3]))
    return [item[3] for item in selected]


def _resolve_integration_backbone(requested_backbone=None):
    if requested_backbone:
        return requested_backbone

    authoritative_path = os.path.join(OUTPUT_DIR, "research_results", "table1_authoritative.json")
    if os.path.exists(authoritative_path):
        try:
            data = json.load(open(authoritative_path))
            candidates = []
            for model_name, metrics in data.items():
                if model_name in {"SVM", "RandomForest"}:
                    continue
                oa = metrics.get("oa_mean")
                if oa is not None:
                    candidates.append((float(oa), model_name))
            if candidates:
                chosen = max(candidates)[1]
                print(f"  Auto-selected best authoritative backbone: {chosen}")
                return chosen
        except Exception as exc:
            print(f"  [WARN] Could not auto-select backbone from authoritative table: {exc}")

    multi_seed_path = os.path.join(OUTPUT_DIR, "research_results", "multi_seed_summary.json")
    if os.path.exists(multi_seed_path):
        try:
            data = json.load(open(multi_seed_path))
            candidates = []
            for model_name, section in data.get("benchmark", {}).items():
                agg = section.get("aggregated", {})
                oa = agg.get("oa", {}).get("mean")
                if oa is not None:
                    candidates.append((float(oa), model_name))
            if candidates:
                chosen = max(candidates)[1]
                print(f"  Auto-selected best multi-seed backbone: {chosen}")
                return chosen
        except Exception as exc:
            print(f"  [WARN] Could not auto-select backbone from multi-seed summary: {exc}")

    summary_path = os.path.join(OUTPUT_DIR, "research_results", "phase2_benchmark_summary.json")
    if os.path.exists(summary_path):
        try:
            data = json.load(open(summary_path))
            models = data.get("models", [])
            dl_models = [m for m in models if m.get("model") not in {"SVM", "RandomForest"}]
            if dl_models:
                best = max(dl_models, key=lambda m: m.get("oa", 0.0))
                chosen = best.get("model")
                if chosen:
                    print(f"  Auto-selected best Phase 2 backbone: {chosen}")
                    return chosen
        except Exception as exc:
            print(f"  [WARN] Could not auto-select backbone from Phase 2 summary: {exc}")

    return DEFAULT_BACKBONE


# ═══════════════════════════════════════════════════════
#  Step 1: Classify GeoTIFFs with trained model
# ═══════════════════════════════════════════════════════

def classify_geotiff(model, image_path, device, patch_size=256, batch_size=32):
    """
    Classify a GeoTIFF using sliding window and return per-pixel predictions.

    Returns:
        prediction_map: (H, W) array with class predictions (0=Urban, 1=Non-Urban, 2=Transition)
        urban_fraction: fraction of pixels classified as Urban or Transition
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio required: pip install rasterio")

    with rasterio.open(image_path) as src:
        img_data = src.read().astype(np.float32)  # (bands, H, W)
        n_bands, H, W = img_data.shape
        transform = src.transform
        crs = src.crs

    # Normalize to [0, 1]
    for b in range(n_bands):
        band = img_data[b]
        valid = np.isfinite(band) & (band > 0)
        if valid.any():
            p2, p98 = np.percentile(band[valid], [2, 98])
            if p98 > p2:
                img_data[b] = np.clip((band - p2) / (p98 - p2), 0, 1)
    img_data = np.nan_to_num(img_data, nan=0.0)

    # Ensure 6 channels (pad or trim)
    if n_bands < NUM_CHANNELS:
        pad = np.zeros((NUM_CHANNELS - n_bands, H, W), dtype=np.float32)
        img_data = np.concatenate([img_data, pad], axis=0)
    elif n_bands > NUM_CHANNELS:
        img_data = img_data[:NUM_CHANNELS]

    # Sliding window classification
    prediction_map = np.full((H, W), -1, dtype=np.int8)
    count_map = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    stride = patch_size // 2  # 50% overlap for smoother boundaries

    patches = []
    coords = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img_data[:, y:y+patch_size, x:x+patch_size]
            if np.isfinite(patch).mean() < 0.5:
                continue
            patches.append(patch)
            coords.append((y, x))

    if not patches:
        return prediction_map, 0.0

    # Batch inference
    model.eval()
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = torch.tensor(np.array(patches[i:i+batch_size])).to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            for j, (y, x) in enumerate(coords[i:i+batch_size]):
                for c in range(NUM_CLASSES):
                    count_map[c, y:y+patch_size, x:x+patch_size] += probs[j, c]

    # Argmax over accumulated probabilities
    valid = count_map.sum(axis=0) > 0
    prediction_map[valid] = count_map[:, valid].argmax(axis=0)

    # Urban fraction (Urban=0, Transition=2 both count as urbanized)
    classified = (prediction_map >= 0)
    if classified.any():
        urban_pixels = ((prediction_map == 0) | (prediction_map == 2)) & classified
        urban_fraction = urban_pixels.sum() / classified.sum()
    else:
        urban_fraction = 0.0

    return prediction_map, float(urban_fraction)


def classify_all_geotiffs(backbone_name=None, device="cuda"):
    """
    Classify the locked Sentinel-2 and Landsat GeoTIFFs from the GEE download.
    Returns dict: {city: {year_season: urban_fraction}}
    """
    print(f"\n{'='*60}")
    backbone_name = _resolve_integration_backbone(backbone_name)
    print(f"  Step 1: Classifying GeoTIFFs with {backbone_name}")
    print(f"{'='*60}")

    model_path = os.path.join(MODEL_DIR, f"{backbone_name}_best.pth")
    if not os.path.exists(model_path):
        print(f"  [WARN] No trained model found at {model_path}")
        print(f"  Training a quick model first...")
        from src.train import progressive_train
        from src.dataset import get_dataloaders
        loaders = get_dataloaders(batch_size=BATCH_SIZE, data_source="real", real_dataset="eurosat")
        model, _, _, _ = progressive_train(backbone_name, device, loaders)
    else:
        model = UrbanClassifier(backbone_name, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"  Loaded model from {model_path}")

    default_cities = LOCKED_INTEGRATION_CITIES if _use_locked_selection() else CITIES
    allowed_cities = _env_city_filter(default_cities)
    max_geotiffs = _env_max_geotiffs()

    include_s2 = _env_bool("INTEGRATION_INCLUDE_S2", default=True)
    include_landsat = _env_bool("INTEGRATION_INCLUDE_LANDSAT", default=True)

    selected_files = []
    s2_files = sorted(glob.glob(os.path.join(GEE_DIR, "S2_*.tif")))
    if include_s2:
        selected_files.extend(_select_s2_files(s2_files, allowed_cities))
    ls_files = sorted(glob.glob(os.path.join(GEE_DIR, "LS_*.tif")))
    if include_landsat:
        selected_files.extend(_select_landsat_files(ls_files, allowed_cities))

    if max_geotiffs is not None:
        selected_files = selected_files[:max_geotiffs]
    if not selected_files:
        print(f"  [WARN] No GeoTIFFs found in {GEE_DIR}")
        return {}

    print(f"  Cities: {allowed_cities}")
    print(f"  GeoTIFFs selected: {len(selected_files)}")

    results = defaultdict(dict)
    os.makedirs(INTEGRATION_DIR, exist_ok=True)
    inventory = {"backbone": backbone_name, "selected_files": [], "source_counts": {}}

    for fpath in selected_files:
        fname = os.path.basename(fpath)
        source, city, year, season = _parse_geotiff_metadata(fname)

        key = f"{source}_{year}_{season}"
        print(f"  Classifying {fname}...", end=" ", flush=True)

        t0 = time.time()
        pred_map, urban_frac = classify_geotiff(model, fpath, device)
        elapsed = time.time() - t0

        results[city][key] = {
            "urban_fraction": urban_frac,
            "file": fname,
            "source": source,
            "year": int(year) if str(year).isdigit() else year,
            "season": season,
            "time_sec": elapsed,
        }
        inventory["selected_files"].append({
            "city": city,
            "source": source,
            "year": year,
            "season": season,
            "file": fname,
        })
        print(f"urban={urban_frac:.3f} ({elapsed:.1f}s)")

    # Save results
    out_path = os.path.join(INTEGRATION_DIR, "classification_results.json")
    with open(out_path, "w") as f:
        json.dump(dict(results), f, indent=2)
    inventory["source_counts"] = dict(Counter(item["source"] for item in inventory["selected_files"]))
    inventory_path = os.path.join(INTEGRATION_DIR, "classification_inventory.json")
    with open(inventory_path, "w") as f:
        json.dump(inventory, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print(f"  Inventory saved to {inventory_path}")

    return dict(results)


# ═══════════════════════════════════════════════════════
#  Step 2: Build Urban Area Time Series
# ═══════════════════════════════════════════════════════

def build_urban_timeseries(classification_results=None):
    """
    Convert classification results to urban area time series per city.
    Uses city bounding boxes to estimate actual area in sq km.
    """
    print(f"\n{'='*60}")
    print(f"  Step 2: Building Urban Area Time Series")
    print(f"{'='*60}")

    if classification_results is None:
        results_path = os.path.join(INTEGRATION_DIR, "classification_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                classification_results = json.load(f)
        else:
            print("  [ERROR] No classification results found. Run --classify first.")
            return {}

    timeseries = {}
    source_details = {}
    default_cities = LOCKED_INTEGRATION_CITIES if _use_locked_selection() else CITIES
    for city in _env_city_filter(default_cities):
        if city not in classification_results:
            continue

        # Estimate total city area from bounding box (in sq km)
        bounds = CITY_BOUNDS[city]
        lat_mid = (bounds[1] + bounds[3]) / 2
        lon_span = bounds[2] - bounds[0]
        lat_span = bounds[3] - bounds[1]
        # Approximate: 1 degree lat ~ 111 km, 1 degree lon ~ 111*cos(lat) km
        import math
        area_km2 = (lon_span * 111 * math.cos(math.radians(lat_mid))) * (lat_span * 111)

        city_data = classification_results[city]
        yearly = defaultdict(lambda: defaultdict(list))

        for key, info in city_data.items():
            if isinstance(info, dict):
                source = info.get("source")
                year = info.get("year")
                if source and isinstance(year, int):
                    yearly[year][source].append(float(info["urban_fraction"]))
                    continue
            parts = key.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                yearly[int(parts[1])][parts[0]].append(float(info["urban_fraction"]))

        ts = {}
        city_sources = {}
        for year in sorted(yearly.keys()):
            source_observations = yearly[year]
            preferred_source = "LS" if "LS" in source_observations else sorted(source_observations.keys())[0]
            avg_urban_frac = np.mean(source_observations[preferred_source])
            city_sources[str(year)] = {
                source: {
                    "urban_fraction_mean": float(np.mean(values)),
                    "n_observations": len(values),
                }
                for source, values in sorted(source_observations.items())
            }
            ts[year] = {
                "urban_fraction": float(avg_urban_frac),
                "urban_area_km2": float(avg_urban_frac * area_km2),
                "total_area_km2": float(area_km2),
                "n_observations": int(sum(len(values) for values in source_observations.values())),
                "selected_source": preferred_source,
                "available_sources": sorted(source_observations.keys()),
            }

        timeseries[city] = ts
        source_details[city] = city_sources
        print(f"  {city}: {len(ts)} years, latest urban area = "
              f"{ts[max(ts.keys())]['urban_area_km2']:.1f} sq km")

    # Save
    out_path = os.path.join(INTEGRATION_DIR, "urban_timeseries.json")
    with open(out_path, "w") as f:
        json.dump(timeseries, f, indent=2)
    details_path = os.path.join(INTEGRATION_DIR, "urban_timeseries_sources.json")
    with open(details_path, "w") as f:
        json.dump(source_details, f, indent=2)
    print(f"\n  Time series saved to {out_path}")
    print(f"  Source details saved to {details_path}")

    return timeseries


# ═══════════════════════════════════════════════════════
#  Step 3: Feed into Pillar IV (Predictive Model)
# ═══════════════════════════════════════════════════════

def run_prediction_with_real_data(device="cuda", epochs=50):
    """
    Re-run Pillar IV using real satellite-derived urban area data
    instead of synthetic observations.
    """
    print(f"\n{'='*60}")
    print(f"  Step 3: Predictive Modelling with Real Satellite Data")
    print(f"{'='*60}")

    ts_path = os.path.join(INTEGRATION_DIR, "urban_timeseries.json")
    if not os.path.exists(ts_path):
        print("  [WARN] No time series data. Using existing Pillar IV (synthetic observations).")
        from src.pillar4_predictive import train_predictor
        return train_predictor(device=device, epochs=epochs)

    with open(ts_path) as f:
        timeseries = json.load(f)

    # Convert time series to format Pillar IV expects
    print(f"  Loaded real satellite time series for {len(timeseries)} cities")
    for city, ts in timeseries.items():
        years = sorted(ts.keys(), key=int)
        print(f"    {city}: {years[0]}-{years[-1]}, "
              f"urban area {ts[years[0]]['urban_area_km2']:.0f} -> {ts[years[-1]]['urban_area_km2']:.0f} sq km")

    # Run Pillar IV with satellite observations merged into the historical series
    from src.pillar4_predictive import train_predictor
    model, history, forecasts, metrics = train_predictor(
        device=device, epochs=epochs,
        satellite_timeseries=timeseries,
    )

    return model, history, forecasts, metrics


# ═══════════════════════════════════════════════════════
#  Step 4: Feed predictions into Pillar V (Alerts)
# ═══════════════════════════════════════════════════════

def run_alerts_with_predictions(device="cuda", epochs=10):
    """
    Use Pillar IV expansion predictions to identify zones at risk of
    encroachment into protected areas, and run Pillar V alert engine.
    """
    print(f"\n{'='*60}")
    print(f"  Step 4: Alert System with Predicted Expansion Zones")
    print(f"{'='*60}")

    # Load forecasts from Pillar IV
    forecast_path = os.path.join(OUTPUT_DIR, "pillar4_forecasts.json")
    if os.path.exists(forecast_path):
        with open(forecast_path) as f:
            forecasts = json.load(f)
        print(f"  Loaded forecasts for {len(forecasts)} cities")
        for city, fc in forecasts.items():
            if isinstance(fc, dict) and "predictions" in fc:
                preds = fc["predictions"]
                if preds:
                    last_year = list(preds.keys())[-1] if isinstance(preds, dict) else "?"
                    print(f"    {city}: forecast to {last_year}")
    else:
        print("  [WARN] No forecasts found. Running Pillar V standalone.")
        forecasts = None

    # Run Pillar V
    from src.pillar5_realtime import train_realtime_detector, simulate_monitoring, benchmark_latency
    model, history, alert_engine, change_acc = train_realtime_detector(
        device=device, epochs=epochs
    )

    # Simulate monitoring with forecasts
    alert_engine = simulate_monitoring(
        model,
        device,
        num_observations=300,
    )
    report = alert_engine.generate_report()
    alerts = alert_engine.alerts

    # Benchmark latency
    latency = benchmark_latency(model, device)

    print(f"\n  Change Detection Accuracy: {change_acc:.4f}")
    print(f"  Alerts generated: {len(alerts)}")
    print(f"  Mean latency: {latency.get('mean_ms', 0):.1f} ms")

    return {
        "model": model, "change_acc": change_acc,
        "alerts": len(alerts), "latency": latency,
        "report": report,
        "forecasts_available": forecasts is not None,
    }


# ═══════════════════════════════════════════════════════
#  Full Pipeline
# ═══════════════════════════════════════════════════════

def run_full_pipeline(device="cuda"):
    """Run the complete integration pipeline."""
    print("\n" + "="*70)
    print("  INTEGRATION PIPELINE — THE LIVING MAP")
    print("  Classification -> Time Series -> Prediction -> Alerts")
    print("="*70)

    t0 = time.time()

    # Step 1: Classify satellite imagery
    classification_results = classify_all_geotiffs(device=device)

    # Step 2: Build time series
    timeseries = build_urban_timeseries(classification_results)

    # Step 3: Predictive modelling
    pred_results = run_prediction_with_real_data(device=device)

    # Step 4: Alert system
    alert_results = run_alerts_with_predictions(device=device)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Integration pipeline completed in {elapsed/60:.1f} min")
    print(f"{'='*70}")

    payload = {
        "classification": classification_results,
        "timeseries": timeseries,
        "prediction": pred_results,
        "alerts": alert_results,
    }
    os.makedirs(INTEGRATION_DIR, exist_ok=True)
    summary_path = os.path.join(INTEGRATION_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Integration summary saved to {summary_path}")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Integration Pipeline")
    parser.add_argument("--classify", action="store_true", help="Classify GeoTIFFs")
    parser.add_argument("--timeseries", action="store_true", help="Build urban time series")
    parser.add_argument("--predict", action="store_true", help="Run Pillar IV with real data")
    parser.add_argument("--alerts", action="store_true", help="Run Pillar V alerts")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--backbone", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.full:
        run_full_pipeline(device)
    elif args.classify:
        classify_all_geotiffs(args.backbone, device)
    elif args.timeseries:
        build_urban_timeseries()
    elif args.predict:
        run_prediction_with_real_data(device)
    elif args.alerts:
        run_alerts_with_predictions(device)
    else:
        print("Specify --classify, --timeseries, --predict, --alerts, or --full")
