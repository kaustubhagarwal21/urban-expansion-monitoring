"""
Research-Grade Satellite Data Download Pipeline for Indian Cities
=================================================================
Downloads and processes Sentinel-2, Sentinel-1 SAR, and Landsat data
for 7 Indian metropolitan areas using Google Earth Engine (GEE).

Features:
  - Cloud masking (SCL band for S2, QA_PIXEL for Landsat)
  - Seasonal compositing (pre-monsoon: Jan-Mar, post-monsoon: Oct-Dec)
  - Speckle filtering for SAR (focal median)
  - Patch extraction (64x64 or 128x128 at native resolution)
  - Auto-labeling from ESA WorldCover 2021 and Google Dynamic World
  - Multi-temporal pair generation for change detection
  - Landsat-Sentinel band harmonization
  - Dataset manifest with metadata (city, date, coords, source, label)

Prerequisites:
    pip install earthengine-api geemap geopandas rasterio tqdm
    earthengine authenticate

Alternative (no GEE):
    Uses EuroSAT (free labelled Sentinel-2) as proxy dataset.
"""

import os, sys, json, zipfile, math, glob, time
import numpy as np
from datetime import datetime

# Fix Windows cp1252 encoding for Unicode characters in print statements
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *

try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False

# ═══════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════

# Seasonal windows (India-specific: avoid monsoon Jun-Sep)
SEASONS = {
    "pre_monsoon":  ("01-01", "03-31"),   # Jan-Mar: dry, clear
    "post_monsoon": ("10-01", "12-31"),   # Oct-Dec: post-rain, clear
}

# Sentinel-2 bands for 6-channel input
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
S2_SCALE = 10  # metres

# Sentinel-1 SAR config
SAR_BANDS = ["VV", "VH"]
SAR_SCALE = 10

# Landsat band mapping (harmonized to match S2 order: Blue,Green,Red,NIR,SWIR1,SWIR2)
LANDSAT_CONFIGS = {
    "L5": {
        "collection": "LANDSAT/LT05/C02/T1_L2",
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        "qa_band": "QA_PIXEL",
        "scale_factor": 0.0000275, "offset": -0.2,
        "years": (1990, 2011),
    },
    "L7": {
        "collection": "LANDSAT/LE07/C02/T1_L2",
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        "qa_band": "QA_PIXEL",
        "scale_factor": 0.0000275, "offset": -0.2,
        "years": (1999, 2023),
    },
    "L8": {
        "collection": "LANDSAT/LC08/C02/T1_L2",
        "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "qa_band": "QA_PIXEL",
        "scale_factor": 0.0000275, "offset": -0.2,
        "years": (2013, 2023),
    },
    "L9": {
        "collection": "LANDSAT/LC09/C02/T1_L2",
        "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "qa_band": "QA_PIXEL",
        "scale_factor": 0.0000275, "offset": -0.2,
        "years": (2022, 2025),
    },
}

# Harmonized band names for output
HARMONIZED_BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

# Patch extraction config
PATCH_SIZES = [64, 128]  # pixels
DEFAULT_PATCH_SIZE = 64

# Label mapping from ESA WorldCover 2021
# https://esa-worldcover.org/en
WORLDCOVER_LABEL_MAP = {
    10: 1,   # Tree cover → Non-Urban
    20: 1,   # Shrubland → Non-Urban
    30: 1,   # Grassland → Non-Urban
    40: 1,   # Cropland → Non-Urban (or Transition if near urban)
    50: 0,   # Built-up → Urban
    60: 1,   # Bare / sparse vegetation → Non-Urban
    70: 1,   # Snow and ice → Non-Urban
    80: 1,   # Permanent water → Non-Urban
    90: 1,   # Herbaceous wetland → Non-Urban
    95: 1,   # Mangroves → Non-Urban
    100: 1,  # Moss and lichen → Non-Urban
}

# Dynamic World class mapping
DYNAMIC_WORLD_LABEL_MAP = {
    0: 1,   # water → Non-Urban
    1: 1,   # trees → Non-Urban
    2: 1,   # grass → Non-Urban
    3: 1,   # flooded_vegetation → Non-Urban
    4: 1,   # crops → Non-Urban
    5: 1,   # shrub_and_scrub → Non-Urban
    6: 0,   # built → Urban
    7: 1,   # bare → Non-Urban
    8: 1,   # snow_and_ice → Non-Urban
}


# ═══════════════════════════════════════════════════════
#  GEE Initialization
# ═══════════════════════════════════════════════════════

def initialize_gee(project=None):
    """Initialize Google Earth Engine with authentication."""
    if not EE_AVAILABLE:
        print("  [ERROR] earthengine-api not installed.")
        print("          pip install earthengine-api geemap")
        return False
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print("  ✓ Google Earth Engine initialized.")
        return True
    except Exception as e:
        print(f"  [ERROR] GEE init failed: {e}")
        print("  Run: earthengine authenticate")
        return False


# ═══════════════════════════════════════════════════════
#  Cloud Masking Functions
# ═══════════════════════════════════════════════════════

def mask_s2_clouds(image):
    """
    Cloud mask for Sentinel-2 using Scene Classification Layer (SCL).
    Masks: cloud shadow (3), cloud medium (8), cloud high (9), cirrus (10).
    Keeps: vegetation, bare soil, water, snow.
    """
    scl = image.select("SCL")
    mask = (scl.neq(3)   # Cloud shadow
            .And(scl.neq(8))   # Cloud medium probability
            .And(scl.neq(9))   # Cloud high probability
            .And(scl.neq(10))) # Thin cirrus
    return image.updateMask(mask).select(S2_BANDS).divide(10000)


def mask_landsat_clouds(image, config):
    """
    Cloud mask for Landsat using QA_PIXEL band.
    Masks cloud, cloud shadow, and cirrus bits.
    """
    qa = image.select(config["qa_band"])
    # Bit 3: cloud shadow, Bit 4: cloud, Bit 5: cloud confidence
    cloud_shadow = qa.bitwiseAnd(1 << 3).eq(0)
    cloud = qa.bitwiseAnd(1 << 4).eq(0)
    mask = cloud_shadow.And(cloud)

    # Apply scaling
    bands = config["bands"]
    scaled = (image.select(bands)
              .multiply(config["scale_factor"])
              .add(config["offset"])
              .clamp(0, 1))
    return scaled.updateMask(mask).rename(HARMONIZED_BANDS)


# ═══════════════════════════════════════════════════════
#  Sentinel-2 Pipeline
# ═══════════════════════════════════════════════════════

def get_s2_composite(city_name, year, season="pre_monsoon", max_cloud_pct=20):
    """
    Get cloud-masked seasonal Sentinel-2 composite for an Indian city.

    Args:
        city_name: one of CITIES
        year: integer year (2015-2023)
        season: "pre_monsoon" or "post_monsoon"
        max_cloud_pct: maximum cloud cover percentage filter

    Returns:
        ee.Image: 6-band composite clipped to city ROI
    """
    bounds = CITY_BOUNDS[city_name]
    roi = ee.Geometry.Rectangle(bounds)
    start_md, end_md = SEASONS[season]
    date_start = f"{year}-{start_md}"
    date_end = f"{year}-{end_md}"

    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(roi)
                  .filterDate(date_start, date_end)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
                  .map(mask_s2_clouds))

    composite = collection.median().clip(roi)
    return composite, roi


def download_s2_city(city_name, years=None, seasons=None, output_dir=None):
    """
    Download Sentinel-2 seasonal composites for a city across multiple years.

    Exports to Google Drive (GEE limitation for large exports).
    """
    years = years or list(range(2017, 2024))
    seasons = seasons or list(SEASONS.keys())
    output_dir = output_dir or os.path.join(RAW_DATA_DIR, city_name, "sentinel2")

    tasks = []
    bounds = CITY_BOUNDS[city_name]
    roi = ee.Geometry.Rectangle(bounds)

    for year in years:
        for season in seasons:
            composite, _ = get_s2_composite(city_name, year, season)
            desc = f"S2_{city_name}_{year}_{season}"

            task = ee.batch.Export.image.toDrive(
                image=composite,
                description=desc,
                folder="urban_expansion_india",
                region=roi,
                scale=S2_SCALE,
                maxPixels=1e9,
                fileFormat="GeoTIFF",
            )
            task.start()
            tasks.append({"task": task, "desc": desc})
            print(f"    Export: {desc}")

    return tasks


# ═══════════════════════════════════════════════════════
#  Sentinel-1 SAR Pipeline
# ═══════════════════════════════════════════════════════

def get_sar_composite(city_name, year, season="pre_monsoon"):
    """
    Get speckle-filtered Sentinel-1 SAR composite.

    Processing:
    - Filter IW mode, VV+VH polarization
    - Prefer ascending orbit for geometry consistency, but fall back to
      descending when a city/season has no ascending coverage
    - Focal median speckle filter (3x3)
    - Seasonal composite
    """
    bounds = CITY_BOUNDS[city_name]
    roi = ee.Geometry.Rectangle(bounds)
    start_md, end_md = SEASONS[season]
    date_start = f"{year}-{start_md}"
    date_end = f"{year}-{end_md}"

    base_collection = (ee.ImageCollection("COPERNICUS/S1_GRD")
                       .filterBounds(roi)
                       .filterDate(date_start, date_end)
                       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                       .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                       .filter(ee.Filter.eq("instrumentMode", "IW"))
                       .select(SAR_BANDS))

    ascending = base_collection.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
    descending = base_collection.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))

    asc_count = ascending.size().getInfo()
    desc_count = descending.size().getInfo()

    if asc_count > 0:
        collection = ascending
        orbit_pass = "ASCENDING"
        image_count = asc_count
    elif desc_count > 0:
        collection = descending
        orbit_pass = "DESCENDING"
        image_count = desc_count
    else:
        return None, roi, {"orbit_pass": None, "image_count": 0}

    # Speckle filter: focal median 3x3
    def speckle_filter(image):
        vv = image.select("VV").focal_median(3, "square", "pixels")
        vh = image.select("VH").focal_median(3, "square", "pixels")
        return vv.addBands(vh).copyProperties(image, image.propertyNames())

    filtered = collection.map(speckle_filter)
    composite = filtered.median().clip(roi)
    return composite, roi, {"orbit_pass": orbit_pass, "image_count": image_count}


def download_sar_city(city_name, years=None, seasons=None):
    """Download Sentinel-1 SAR composites for a city."""
    years = years or list(range(2017, 2024))
    seasons = seasons or list(SEASONS.keys())

    tasks = []
    bounds = CITY_BOUNDS[city_name]
    roi = ee.Geometry.Rectangle(bounds)

    for year in years:
        for season in seasons:
            composite, _, meta = get_sar_composite(city_name, year, season)
            desc = f"SAR_{city_name}_{year}_{season}"

            if composite is None:
                print(f"    Skip: {desc} (no Sentinel-1 VV+VH IW coverage)")
                continue

            task = ee.batch.Export.image.toDrive(
                image=composite,
                description=desc,
                folder="urban_expansion_india",
                region=roi,
                scale=SAR_SCALE,
                maxPixels=1e9,
                fileFormat="GeoTIFF",
            )
            task.start()
            tasks.append({"task": task, "desc": desc})
            print(
                f"    Export: {desc} "
                f"[{meta['orbit_pass']}, {meta['image_count']} images]"
            )

    return tasks


# ═══════════════════════════════════════════════════════
#  Landsat Historical Pipeline
# ═══════════════════════════════════════════════════════

def get_landsat_composite(city_name, year, season="pre_monsoon"):
    """
    Get cloud-masked Landsat composite, auto-selecting sensor by year.
    Output bands harmonized to: Blue, Green, Red, NIR, SWIR1, SWIR2.
    """
    bounds = CITY_BOUNDS[city_name]
    roi = ee.Geometry.Rectangle(bounds)
    start_md, end_md = SEASONS[season]
    date_start = f"{year}-{start_md}"
    date_end = f"{year}-{end_md}"

    # Select appropriate Landsat sensor
    if year <= 2011:
        cfg = LANDSAT_CONFIGS["L5"]
    elif year <= 2012:
        cfg = LANDSAT_CONFIGS["L7"]
    elif year <= 2021:
        cfg = LANDSAT_CONFIGS["L8"]
    else:
        cfg = LANDSAT_CONFIGS["L9"]

    collection = (ee.ImageCollection(cfg["collection"])
                  .filterBounds(roi)
                  .filterDate(date_start, date_end)
                  .filter(ee.Filter.lt("CLOUD_COVER", 20)))

    def apply_mask(img):
        return mask_landsat_clouds(img, cfg)

    masked = collection.map(apply_mask)
    composite = masked.median().clip(roi)
    return composite, roi


def download_landsat_city(city_name, years=None, seasons=None):
    """Download historical Landsat composites for a city."""
    years = years or [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023]
    seasons = seasons or ["pre_monsoon"]

    tasks = []
    bounds = CITY_BOUNDS[city_name]
    roi = ee.Geometry.Rectangle(bounds)

    for year in years:
        for season in seasons:
            composite, _ = get_landsat_composite(city_name, year, season)
            desc = f"LS_{city_name}_{year}_{season}"

            task = ee.batch.Export.image.toDrive(
                image=composite,
                description=desc,
                folder="urban_expansion_india",
                region=roi,
                scale=30,
                maxPixels=1e9,
                fileFormat="GeoTIFF",
            )
            task.start()
            tasks.append({"task": task, "desc": desc})
            print(f"    Export: {desc}")

    return tasks


# ═══════════════════════════════════════════════════════
#  Auto-Labeling from LULC Maps
# ═══════════════════════════════════════════════════════

def get_worldcover_labels(city_name):
    """
    Get ESA WorldCover 2021 labels (10m resolution).
    Maps to: 0=Urban, 1=Non-Urban, 2=Transition (edge of urban).
    """
    bounds = CITY_BOUNDS[city_name]
    roi = ee.Geometry.Rectangle(bounds)

    worldcover = (ee.ImageCollection("ESA/WorldCover/v200")
                   .filterBounds(roi)
                   .mosaic()
                   .clip(roi))

    # Remap to our 3-class schema
    # Built-up (50) → Urban (0), everything else → Non-Urban (1)
    urban_mask = worldcover.eq(50)

    # Transition: pixels within 100m of urban boundary but not urban
    # Morphological dilation of urban mask
    urban_dilated = urban_mask.focal_max(radius=100, units="meters")
    transition_mask = urban_dilated.And(urban_mask.Not())

    # Combine: 0=Urban, 1=Non-Urban, 2=Transition
    labels = (ee.Image(1)  # default Non-Urban
              .where(urban_mask, 0)
              .where(transition_mask, 2)
              .clip(roi)
              .rename("label"))

    return labels, roi


def get_dynamic_world_labels(city_name, year):
    """
    Get Google Dynamic World labels (10m, near-real-time LULC).
    Available 2015-present. More temporal coverage than WorldCover.
    """
    bounds = CITY_BOUNDS[city_name]
    roi = ee.Geometry.Rectangle(bounds)

    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterBounds(roi)
          .filterDate(f"{year}-01-01", f"{year}-12-31")
          .select("label"))

    # Mode composite (most frequent class per pixel)
    mode_label = dw.mode().clip(roi)

    # Remap: built (6) → Urban (0), rest → Non-Urban (1)
    urban_mask = mode_label.eq(6)
    urban_dilated = urban_mask.focal_max(radius=100, units="meters")
    transition_mask = urban_dilated.And(urban_mask.Not())

    labels = (ee.Image(1)
              .where(urban_mask, 0)
              .where(transition_mask, 2)
              .clip(roi)
              .rename("label"))

    return labels, roi


def download_labels_city(city_name, label_source="worldcover"):
    """Export labels for a city to Google Drive."""
    bounds = CITY_BOUNDS[city_name]
    roi = ee.Geometry.Rectangle(bounds)

    if label_source == "worldcover":
        labels, _ = get_worldcover_labels(city_name)
        desc = f"Labels_WC_{city_name}"
    else:
        labels, _ = get_dynamic_world_labels(city_name, 2021)
        desc = f"Labels_DW_{city_name}"

    task = ee.batch.Export.image.toDrive(
        image=labels.toInt8(),
        description=desc,
        folder="urban_expansion_india",
        region=roi,
        scale=10,
        maxPixels=1e9,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"    Export: {desc}")
    return task


# ═══════════════════════════════════════════════════════
#  Patch Extraction (Local — runs after GeoTIFFs are downloaded)
# ═══════════════════════════════════════════════════════

def extract_patches_from_geotiff(
    image_path, label_path, output_dir, city_name,
    patch_size=DEFAULT_PATCH_SIZE, stride=None, min_valid_pct=0.8
):
    """
    Extract training patches from downloaded GeoTIFF files.

    Args:
        image_path: path to satellite GeoTIFF (6-band)
        label_path: path to label GeoTIFF (1-band: 0/1/2)
        output_dir: where to save .npy patches
        city_name: city identifier for manifest
        patch_size: patch dimension in pixels
        stride: step size (defaults to patch_size for no overlap)
        min_valid_pct: minimum fraction of valid (non-NaN) pixels

    Returns:
        list of patch metadata dicts
    """
    try:
        import rasterio
    except ImportError:
        print("  [ERROR] rasterio not installed. pip install rasterio")
        return []

    stride = stride or patch_size
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    with rasterio.open(image_path) as src_img, rasterio.open(label_path) as src_lbl:
        img_data = src_img.read()   # (bands, H, W)
        lbl_data = src_lbl.read(1)  # (H, W)
        transform = src_img.transform

        n_bands, H, W = img_data.shape
        patch_id = 0

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                img_patch = img_data[:, y:y+patch_size, x:x+patch_size]
                lbl_patch = lbl_data[y:y+patch_size, x:x+patch_size]

                # Skip patches with too many NaN/nodata pixels
                valid_mask = np.isfinite(img_patch).all(axis=0)
                if valid_mask.mean() < min_valid_pct:
                    continue

                # Replace NaN with 0
                img_patch = np.nan_to_num(img_patch, nan=0.0).astype(np.float32)

                # Determine majority label
                unique, counts = np.unique(lbl_patch[valid_mask], return_counts=True)
                if len(unique) == 0:
                    continue
                majority_label = int(unique[counts.argmax()])

                # Get geographic coordinates of patch center
                cx, cy = rasterio.transform.xy(transform, y + patch_size // 2, x + patch_size // 2)

                # Save
                fname = f"{city_name}_{patch_id:06d}"
                np.save(os.path.join(output_dir, f"{fname}_img.npy"), img_patch)
                np.save(os.path.join(output_dir, f"{fname}_lbl.npy"), lbl_patch.astype(np.int8))

                manifest.append({
                    "id": fname,
                    "city": city_name,
                    "label": majority_label,
                    "label_name": CLASS_NAMES[majority_label],
                    "lat": cy,
                    "lon": cx,
                    "patch_size": patch_size,
                    "source": os.path.basename(image_path),
                    "valid_pct": float(valid_mask.mean()),
                    "urban_pct": float((lbl_patch == 0).mean()),
                    "transition_pct": float((lbl_patch == 2).mean()),
                })
                patch_id += 1

    print(f"    Extracted {patch_id} patches from {os.path.basename(image_path)}")
    return manifest


def extract_temporal_pairs(
    image_t1_path, image_t2_path, label_t1_path, label_t2_path,
    output_dir, city_name, patch_size=DEFAULT_PATCH_SIZE,
):
    """
    Extract bi-temporal patch pairs for Siamese change detection.
    A pair is labeled 'changed' if the majority label differs between t1 and t2.
    """
    try:
        import rasterio
    except ImportError:
        print("  [ERROR] rasterio not installed.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    with (rasterio.open(image_t1_path) as s1,
          rasterio.open(image_t2_path) as s2,
          rasterio.open(label_t1_path) as l1,
          rasterio.open(label_t2_path) as l2):

        img1 = s1.read()
        img2 = s2.read()
        lbl1 = l1.read(1)
        lbl2 = l2.read(1)

        _, H, W = img1.shape
        H = min(H, img2.shape[1])
        W = min(W, img2.shape[2])
        pair_id = 0

        for y in range(0, H - patch_size + 1, patch_size):
            for x in range(0, W - patch_size + 1, patch_size):
                p1 = img1[:, y:y+patch_size, x:x+patch_size]
                p2 = img2[:, y:y+patch_size, x:x+patch_size]
                lb1 = lbl1[y:y+patch_size, x:x+patch_size]
                lb2 = lbl2[y:y+patch_size, x:x+patch_size]

                # Skip invalid
                if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
                    continue

                p1 = np.nan_to_num(p1, nan=0.0).astype(np.float32)
                p2 = np.nan_to_num(p2, nan=0.0).astype(np.float32)

                maj1 = int(np.bincount(lb1.flatten()).argmax())
                maj2 = int(np.bincount(lb2.flatten()).argmax())
                changed = int(maj1 != maj2)

                fname = f"{city_name}_pair_{pair_id:06d}"
                np.save(os.path.join(output_dir, f"{fname}_t1.npy"), p1)
                np.save(os.path.join(output_dir, f"{fname}_t2.npy"), p2)

                manifest.append({
                    "id": fname,
                    "city": city_name,
                    "label_t1": maj1,
                    "label_t2": maj2,
                    "changed": changed,
                    "expansion": int(maj1 == 1 and maj2 == 0),  # Non-Urban → Urban
                })
                pair_id += 1

    print(f"    Extracted {pair_id} temporal pairs for {city_name}")
    return manifest


# ═══════════════════════════════════════════════════════
#  Dataset Manifest Builder
# ═══════════════════════════════════════════════════════

def build_dataset_manifest(patch_dir, output_path=None):
    """
    Scan extracted patches and build a consolidated manifest JSON.
    Useful for creating train/val/test splits by city.
    """
    output_path = output_path or os.path.join(PROCESSED_DATA_DIR, "manifest.json")
    manifest_files = glob.glob(os.path.join(patch_dir, "**", "*_manifest.json"), recursive=True)

    all_entries = []
    for mf in manifest_files:
        base = os.path.basename(mf)
        # Skip previously consolidated manifests to avoid double-counting
        if base == os.path.basename(output_path):
            continue
        if base in {"Mumbai_manifest.json", "Delhi_NCR_manifest.json", "Bangalore_manifest.json"}:
            continue
        with open(mf) as f:
            all_entries.extend(json.load(f))

    # Summary stats
    cities = set(e["city"] for e in all_entries)
    labels = [e["label"] for e in all_entries]
    from collections import Counter
    label_counts = Counter(labels)

    summary = {
        "total_patches": len(all_entries),
        "cities": sorted(cities),
        "label_distribution": {CLASS_NAMES[k]: v for k, v in sorted(label_counts.items())},
        "created": datetime.now().isoformat(),
        "patches": all_entries,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Dataset Manifest: {output_path}")
    print(f"    Total patches: {len(all_entries)}")
    for city in sorted(cities):
        city_count = sum(1 for e in all_entries if e["city"] == city)
        print(f"    {city:15s}: {city_count}")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"    {CLASS_NAMES[lbl]:15s}: {cnt} ({cnt/len(all_entries)*100:.1f}%)")

    return summary


def create_city_splits(manifest_path=None, output_path=None):
    """
    Create geographic train/val/test splits (LOCO-ready).

    Default split:
      Train: Mumbai, Delhi_NCR, Bangalore, Hyderabad, Pune (5 cities)
      Val:   Ahmedabad (1 city)
      Test:  Chennai (1 city)

    Also generates 7 LOCO folds for cross-city generalization.
    """
    manifest_path = manifest_path or os.path.join(PROCESSED_DATA_DIR, "manifest.json")
    output_path = output_path or os.path.join(PROCESSED_DATA_DIR, "splits.json")

    with open(manifest_path) as f:
        manifest = json.load(f)

    patches = manifest["patches"]

    # Default geographic split
    default_split = {
        "train_cities": ["Mumbai", "Delhi_NCR", "Bangalore", "Hyderabad", "Pune"],
        "val_cities": ["Ahmedabad"],
        "test_cities": ["Chennai"],
    }

    # LOCO folds: leave each city out for testing
    loco_folds = {}
    for held_out in CITIES:
        remaining = [c for c in CITIES if c != held_out]
        loco_folds[f"loco_{held_out}"] = {
            "train_cities": remaining[:5],
            "val_cities": [remaining[5]] if len(remaining) > 5 else [remaining[-1]],
            "test_cities": [held_out],
        }

    splits = {
        "default": default_split,
        "loco": loco_folds,
        "created": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"  Splits saved to {output_path}")
    return splits


# ═══════════════════════════════════════════════════════
#  Master Download Orchestrator
# ═══════════════════════════════════════════════════════

def download_all_cities(
    sensors=("sentinel2", "sentinel1", "landsat"),
    s2_years=None, sar_years=None, ls_years=None,
    include_labels=True,
):
    """
    Download all satellite data for all 7 Indian cities.

    This submits GEE export tasks to Google Drive. After tasks complete,
    download GeoTIFFs from Drive and run extract_patches_from_geotiff().

    Args:
        sensors: which sensors to download
        s2_years: Sentinel-2 years (default 2017-2023)
        sar_years: SAR years (default 2017-2023)
        ls_years: Landsat years (default key historical years)
        include_labels: also export WorldCover/DynamicWorld labels
    """
    if not initialize_gee():
        print("\n  Falling back to EuroSAT dataset...")
        download_eurosat()
        return

    s2_years = s2_years or list(range(2017, 2024))
    sar_years = sar_years or list(range(2017, 2024))
    ls_years = ls_years or [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2023]

    all_tasks = []

    print(f"\n{'='*60}")
    print("  Downloading Satellite Data for Indian Metropolitan Areas")
    print(f"  Cities: {', '.join(CITIES)}")
    print(f"  Sensors: {', '.join(sensors)}")
    print(f"{'='*60}")

    for city in CITIES:
        print(f"\n  ── {city} ──")

        if "sentinel2" in sensors:
            print(f"  Sentinel-2 ({s2_years[0]}-{s2_years[-1]}):")
            tasks = download_s2_city(city, years=s2_years)
            all_tasks.extend(tasks)

        if "sentinel1" in sensors:
            print(f"  Sentinel-1 SAR ({sar_years[0]}-{sar_years[-1]}):")
            tasks = download_sar_city(city, years=sar_years)
            all_tasks.extend(tasks)

        if "landsat" in sensors:
            print(f"  Landsat ({ls_years[0]}-{ls_years[-1]}):")
            tasks = download_landsat_city(city, years=ls_years)
            all_tasks.extend(tasks)

        if include_labels:
            print(f"  Labels:")
            download_labels_city(city, "worldcover")
            download_labels_city(city, "dynamic_world")

    print(f"\n{'='*60}")
    print(f"  {len(all_tasks)} GEE export tasks submitted")
    print(f"  Data will appear in Google Drive: 'urban_expansion_india/'")
    print(f"  After download, run: python src/download_data.py --extract")
    print(f"{'='*60}")

    return all_tasks


def check_task_status():
    """Check status of running GEE export tasks."""
    if not initialize_gee():
        return
    tasks = ee.batch.Task.list()
    running = [t for t in tasks if t.state in ("READY", "RUNNING")]
    completed = [t for t in tasks if t.state == "COMPLETED"]
    failed = [t for t in tasks if t.state == "FAILED"]

    print(f"\n  GEE Task Status:")
    print(f"    Running:   {len(running)}")
    print(f"    Completed: {len(completed)}")
    print(f"    Failed:    {len(failed)}")

    for t in running[:10]:
        print(f"    [{t.state:8s}] {t.config.get('description', 'unknown')}")
    for t in failed[:5]:
        print(f"    [FAILED] {t.config.get('description', 'unknown')}: {t.status.get('error_message', '')}")


# ═══════════════════════════════════════════════════════
#  EuroSAT Fallback (Free Labelled Data)
# ═══════════════════════════════════════════════════════

def download_eurosat():
    """Download EuroSAT RGB dataset (~90MB, 27k Sentinel-2 patches)."""
    import urllib.request

    eurosat_dir = os.path.join(DATA_DIR, "eurosat")
    os.makedirs(eurosat_dir, exist_ok=True)

    url = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
    zip_path = os.path.join(eurosat_dir, "EuroSAT_RGB.zip")

    if os.path.exists(os.path.join(eurosat_dir, "EuroSAT_RGB")):
        print("  EuroSAT already downloaded.")
        return eurosat_dir

    print(f"  Downloading EuroSAT dataset (~90MB)...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(eurosat_dir)
        os.remove(zip_path)
        print(f"  EuroSAT saved to {eurosat_dir}")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Using synthetic data as fallback.")

    return eurosat_dir


# ═══════════════════════════════════════════════════════
#  ISRO Bhuvan Helper
# ═══════════════════════════════════════════════════════

def print_bhuvan_instructions():
    """Print instructions for ISRO Bhuvan portal data."""
    print(f"\n{'='*60}")
    print("  ISRO Bhuvan Data Download Instructions")
    print(f"{'='*60}")
    print("""
  For high-resolution Indian satellite data:
    1. Register at: https://bhuvan.nrsc.gov.in/
    2. Go to: Bhuvan > Data Download > Satellite Data
    3. Select sensor:
       - Cartosat-2: 0.65m (Pillar III high-res)
       - ResourceSat-2 LISS-IV: 5.8m
       - ResourceSat-2 LISS-III: 23.5m
    4. Select AOI for study cities
    5. Place GeoTIFFs in: data/raw/{city_name}/

  City bounding boxes:""")
    for city, bounds in CITY_BOUNDS.items():
        print(f"    {city:15s}: {bounds}")


# ═══════════════════════════════════════════════════════
#  GEE JavaScript Code Editor Script
# ═══════════════════════════════════════════════════════

def generate_gee_script():
    """
    Generate a GEE JavaScript script for the Code Editor.
    Fallback if Python API auth is problematic.
    """
    script_path = os.path.join(DATA_DIR, "gee_download_script.js")
    os.makedirs(DATA_DIR, exist_ok=True)

    lines = [
        "// Urban Expansion Monitoring — GEE Download Script",
        "// Paste into https://code.earthengine.google.com/",
        "// Cloud-masked seasonal composites for 7 Indian cities",
        "",
        "// Cloud mask function for Sentinel-2 (SCL band)",
        "function maskS2Clouds(image) {",
        "  var scl = image.select('SCL');",
        "  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));",
        "  return image.updateMask(mask).select(['B2','B3','B4','B8','B11','B12']).divide(10000);",
        "}",
        "",
        "// SAR speckle filter",
        "function speckleFilter(image) {",
        "  return image.select('VV').focal_median(3,'square','pixels')",
        "    .addBands(image.select('VH').focal_median(3,'square','pixels'));",
        "}",
        "",
    ]

    for city, bounds in CITY_BOUNDS.items():
        lines.append(f"// ── {city} ──")
        lines.append(f"var roi_{city} = ee.Geometry.Rectangle({list(bounds)});")
        lines.append("")

        # Sentinel-2 pre-monsoon 2023
        lines.append(f"var s2_{city} = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')")
        lines.append(f"  .filterBounds(roi_{city}).filterDate('2023-01-01','2023-03-31')")
        lines.append(f"  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))")
        lines.append(f"  .map(maskS2Clouds).median().clip(roi_{city});")
        lines.append(f"Export.image.toDrive({{image:s2_{city},description:'S2_{city}_2023_pre_monsoon',")
        lines.append(f"  folder:'urban_expansion_india',region:roi_{city},scale:10,maxPixels:1e9}});")
        lines.append("")

        # SAR
        lines.append(f"var sar_{city} = ee.ImageCollection('COPERNICUS/S1_GRD')")
        lines.append(f"  .filterBounds(roi_{city}).filterDate('2023-01-01','2023-03-31')")
        lines.append(f"  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))")
        lines.append(f"  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))")
        lines.append(f"  .filter(ee.Filter.eq('instrumentMode','IW'))")
        lines.append(f"  .select(['VV','VH']).map(speckleFilter).median().clip(roi_{city});")
        lines.append(f"Export.image.toDrive({{image:sar_{city},description:'SAR_{city}_2023_pre_monsoon',")
        lines.append(f"  folder:'urban_expansion_india',region:roi_{city},scale:10,maxPixels:1e9}});")
        lines.append("")

        # WorldCover labels
        lines.append(f"var wc_{city} = ee.Image('ESA/WorldCover/v200').clip(roi_{city});")
        lines.append(f"var urban_{city} = wc_{city}.eq(50);")
        lines.append(f"var trans_{city} = urban_{city}.focal_max(100,'circle','meters').and(urban_{city}.not());")
        lines.append(f"var labels_{city} = ee.Image(1).where(urban_{city},0).where(trans_{city},2).clip(roi_{city});")
        lines.append(f"Export.image.toDrive({{image:labels_{city}.toInt8(),description:'Labels_{city}',")
        lines.append(f"  folder:'urban_expansion_india',region:roi_{city},scale:10,maxPixels:1e9}});")
        lines.append("")
        lines.append("")

    script = "\n".join(lines)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  GEE Code Editor script saved to {script_path}")
    print("  Paste into https://code.earthengine.google.com/ and click Run.")
    return script_path


# ═══════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download satellite data for Indian cities")
    parser.add_argument("--download", action="store_true", help="Start GEE downloads")
    parser.add_argument("--extract", action="store_true", help="Extract patches from GeoTIFFs")
    parser.add_argument("--gee-script", action="store_true", help="Generate GEE Code Editor script")
    parser.add_argument("--status", action="store_true", help="Check GEE task status")
    parser.add_argument("--eurosat", action="store_true", help="Download EuroSAT fallback")
    parser.add_argument("--sensors", nargs="+", default=["sentinel2", "sentinel1", "landsat"])
    parser.add_argument("--cities", nargs="+", default=CITIES)
    parser.add_argument("--patch-size", type=int, default=64)
    args = parser.parse_args()

    print("Urban Expansion Monitoring — Data Pipeline")
    print("=" * 50)

    if args.gee_script:
        generate_gee_script()
    elif args.download:
        download_all_cities(sensors=args.sensors)
    elif args.status:
        check_task_status()
    elif args.eurosat:
        download_eurosat()
    elif args.extract:
        print("\n  Patch Extraction Mode")
        print("  Looking for GeoTIFFs in data/raw/...")
        for city in args.cities:
            raw_dir = os.path.join(RAW_DATA_DIR, city)
            if not os.path.isdir(raw_dir):
                print(f"    {city}: no data found in {raw_dir}")
                continue
            img_files = glob.glob(os.path.join(raw_dir, "**", "S2_*.tif"), recursive=True)
            lbl_files = glob.glob(os.path.join(raw_dir, "**", "Labels_*.tif"), recursive=True)
            if img_files and lbl_files:
                out_dir = os.path.join(PROCESSED_DATA_DIR, city, "patches")
                manifest = extract_patches_from_geotiff(
                    img_files[0], lbl_files[0], out_dir, city, args.patch_size
                )
                mpath = os.path.join(out_dir, f"{city}_manifest.json")
                with open(mpath, "w") as f:
                    json.dump(manifest, f, indent=2)
    else:
        print("\n1. Generating GEE download script...")
        generate_gee_script()
        print("\n2. Bhuvan instructions:")
        print_bhuvan_instructions()
        print("\n3. To download via GEE API: python src/download_data.py --download")
        print("   To extract patches:       python src/download_data.py --extract")
        print("   To check task status:     python src/download_data.py --status")
