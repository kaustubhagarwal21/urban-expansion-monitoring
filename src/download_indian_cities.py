"""
Download Sentinel-2 and Landsat imagery for Indian metropolitan areas
using Google Earth Engine (GEE).

Prerequisites:
    1. pip install earthengine-api
    2. Authenticate: earthengine authenticate
    3. (Optional) pip install geemap  for interactive visualization

This script downloads cloud-free composite imagery for 7 Indian cities
across multiple time periods, producing analysis-ready patches for the
urban expansion monitoring framework.

Usage:
    python -m src.download_indian_cities                    # Download all cities
    python -m src.download_indian_cities --city Mumbai      # Single city
    python -m src.download_indian_cities --sensor sentinel2 # Sentinel-2 only
    python -m src.download_indian_cities --dry-run          # Preview without downloading

Output structure:
    data/indian_cities/
        Mumbai/
            sentinel2_2020_2023/
                patches/          # 64x64 GeoTIFF patches
                composite.tif     # Full mosaic
                metadata.json
            landsat_2010_2015/
                ...
        Delhi_NCR/
            ...
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import CITIES, CITY_BOUNDS, CITY_DESCRIPTIONS, DATA_DIR

# ── GEE Configuration ────────────────────────────────────

# Time periods for multi-temporal analysis
DOWNLOAD_PERIODS = {
    "landsat_1990_2000": {
        "sensor": "landsat",
        "collection": "LANDSAT/LT05/C02/T1_L2",  # Landsat 5 TM
        "start": "1990-01-01",
        "end": "2000-12-31",
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        "band_names": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "scale": 30,
    },
    "landsat_2000_2010": {
        "sensor": "landsat",
        "collection": "LANDSAT/LE07/C02/T1_L2",  # Landsat 7 ETM+
        "start": "2000-01-01",
        "end": "2010-12-31",
        "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
        "band_names": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "scale": 30,
    },
    "landsat_2013_2020": {
        "sensor": "landsat",
        "collection": "LANDSAT/LC08/C02/T1_L2",  # Landsat 8 OLI
        "start": "2013-04-01",
        "end": "2020-12-31",
        "bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "band_names": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "scale": 30,
    },
    "sentinel2_2018_2020": {
        "sensor": "sentinel2",
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "start": "2018-01-01",
        "end": "2020-12-31",
        "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
        "band_names": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "scale": 10,
    },
    "sentinel2_2020_2023": {
        "sensor": "sentinel2",
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "start": "2020-01-01",
        "end": "2023-12-31",
        "bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
        "band_names": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
        "scale": 10,
    },
}

OUTPUT_BASE = os.path.join(DATA_DIR, "indian_cities")


def check_gee_auth():
    """Check if GEE is authenticated and initialize."""
    try:
        import ee
        ee.Initialize()
        print("  GEE authenticated successfully.")
        return True
    except ImportError:
        print("  ERROR: earthengine-api not installed.")
        print("  Install with: pip install earthengine-api")
        return False
    except Exception as e:
        print(f"  ERROR: GEE authentication failed: {e}")
        print("  Run: earthengine authenticate")
        return False


def mask_clouds_sentinel2(image):
    """Mask clouds in Sentinel-2 SR imagery using QA60 band."""
    import ee
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)


def mask_clouds_landsat(image):
    """Mask clouds in Landsat Collection 2 Level 2 imagery."""
    import ee
    qa = image.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 3).eq(0)  # cloud
    shadow = qa.bitwiseAnd(1 << 4).eq(0)  # cloud shadow
    # Apply scaling factors for Collection 2
    optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    return optical.updateMask(cloud.And(shadow))


def create_composite(city: str, period_key: str, period_cfg: dict):
    """Create a cloud-free median composite for a city and time period."""
    import ee

    bounds = CITY_BOUNDS[city]
    roi = ee.Geometry.Rectangle(bounds)

    collection = ee.ImageCollection(period_cfg["collection"]) \
        .filterBounds(roi) \
        .filterDate(period_cfg["start"], period_cfg["end"])

    if period_cfg["sensor"] == "sentinel2":
        collection = collection.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        collection = collection.map(mask_clouds_sentinel2)
    else:
        collection = collection.map(mask_clouds_landsat)

    composite = collection.select(period_cfg["bands"]).median().clip(roi)

    # Add NDVI and NDBI
    nir_band = period_cfg["bands"][3]
    red_band = period_cfg["bands"][2]
    swir_band = period_cfg["bands"][4]

    ndvi = composite.normalizedDifference([nir_band, red_band]).rename("NDVI")
    ndbi = composite.normalizedDifference([swir_band, nir_band]).rename("NDBI")
    composite = composite.addBands([ndvi, ndbi])

    n_images = collection.size().getInfo()

    return composite, roi, n_images


def export_composite(
    composite, roi, city: str, period_key: str,
    scale: int, dry_run: bool = False,
):
    """Export composite to Google Drive or local (via GEE export)."""
    import ee

    out_dir = os.path.join(OUTPUT_BASE, city, period_key)
    os.makedirs(out_dir, exist_ok=True)

    description = f"{city}_{period_key}"

    if dry_run:
        print(f"    [DRY RUN] Would export {description} at {scale}m resolution")
        return None

    # Export to Google Drive
    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=description,
        folder="urban_expansion_india",
        region=roi,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e9,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"    Export started: {description} (task ID: {task.id})")

    # Save metadata
    meta = {
        "city": city,
        "period": period_key,
        "scale": scale,
        "bounds": CITY_BOUNDS[city],
        "description": CITY_DESCRIPTIONS.get(city, ""),
        "task_id": task.id,
        "status": "RUNNING",
    }
    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return task


def download_city(city: str, periods: dict, dry_run: bool = False):
    """Download all periods for a single city."""
    print(f"\n  City: {city}")
    print(f"  Description: {CITY_DESCRIPTIONS.get(city, 'N/A')}")
    print(f"  Bounds: {CITY_BOUNDS[city]}")

    tasks = []
    for period_key, period_cfg in periods.items():
        print(f"\n    Period: {period_key}")
        try:
            composite, roi, n_images = create_composite(city, period_key, period_cfg)
            print(f"    Available images: {n_images}")

            if n_images == 0:
                print(f"    WARNING: No images found for {city} in {period_key}")
                continue

            task = export_composite(
                composite, roi, city, period_key,
                scale=period_cfg["scale"],
                dry_run=dry_run,
            )
            if task:
                tasks.append(task)
        except Exception as e:
            print(f"    ERROR: {e}")

    return tasks


def check_export_status(tasks):
    """Check status of all running export tasks."""
    import ee
    print("\n  Export Task Status:")
    for task in tasks:
        status = ee.data.getTaskStatus(task.id)[0]
        state = status.get("state", "UNKNOWN")
        print(f"    {task.id}: {state}")


def create_patches_from_geotiff(
    tiff_path: str, patch_size: int = 64, overlap: float = 0.25
):
    """
    Extract patches from a downloaded GeoTIFF composite.
    Call this after downloading the GeoTIFFs from Google Drive.

    Args:
        tiff_path: path to the GeoTIFF file
        patch_size: spatial size of each patch
        overlap: fractional overlap between patches

    Returns:
        List of (patch_array, geo_transform) tuples
    """
    try:
        import rasterio
    except ImportError:
        print("  Install rasterio: pip install rasterio")
        return []

    patches = []
    stride = int(patch_size * (1 - overlap))

    with rasterio.open(tiff_path) as src:
        data = src.read()  # (bands, H, W)
        n_bands, h, w = data.shape
        transform = src.transform

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = data[:, y:y + patch_size, x:x + patch_size]

                # Skip patches with too many NaN/nodata values
                valid_fraction = np.isfinite(patch).mean()
                if valid_fraction < 0.8:
                    continue

                # Replace NaN with 0
                patch = np.nan_to_num(patch, nan=0.0)

                # Compute patch geo-location
                patch_transform = rasterio.transform.from_bounds(
                    transform.c + x * transform.a,
                    transform.f + (y + patch_size) * transform.e,
                    transform.c + (x + patch_size) * transform.a,
                    transform.f + y * transform.e,
                    patch_size, patch_size,
                )
                patches.append((patch.astype(np.float32), patch_transform))

    return patches


def process_downloaded_geotiffs(city: str):
    """
    Process downloaded GeoTIFFs into patches ready for the model.
    Run this after downloading composites from Google Drive.

    Expected input: data/indian_cities/{city}/{period}/composite.tif
    Output: data/indian_cities/{city}/{period}/patches/*.npy
    """
    city_dir = os.path.join(OUTPUT_BASE, city)
    if not os.path.isdir(city_dir):
        print(f"  No data found for {city}")
        return

    for period_dir in sorted(os.listdir(city_dir)):
        period_path = os.path.join(city_dir, period_dir)
        if not os.path.isdir(period_path):
            continue

        # Look for GeoTIFF files
        tiff_files = [f for f in os.listdir(period_path) if f.endswith(".tif")]
        if not tiff_files:
            print(f"  No GeoTIFF found in {period_path}")
            continue

        tiff_path = os.path.join(period_path, tiff_files[0])
        print(f"\n  Processing: {tiff_path}")

        patches = create_patches_from_geotiff(tiff_path, patch_size=64)
        print(f"  Extracted {len(patches)} patches")

        # Save patches
        patch_dir = os.path.join(period_path, "patches")
        os.makedirs(patch_dir, exist_ok=True)

        for i, (patch, transform) in enumerate(patches):
            np.save(os.path.join(patch_dir, f"patch_{i:05d}.npy"), patch)

        # Save manifest
        manifest = {
            "city": city,
            "period": period_dir,
            "n_patches": len(patches),
            "patch_size": 64,
            "source_tiff": tiff_files[0],
        }
        with open(os.path.join(period_path, "patch_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  Saved {len(patches)} patches to {patch_dir}")


# =========================================================================
#  Alternative: Download via SentinelHub or direct Copernicus API
# =========================================================================

def print_alternative_download_instructions():
    """Print instructions for alternative download methods."""
    msg = """
    ================================================================
    Alternative Download Methods for Indian City Satellite Imagery
    ================================================================

    METHOD 1: Google Earth Engine (Recommended)
    -------------------------------------------
    1. pip install earthengine-api
    2. earthengine authenticate
    3. python -m src.download_indian_cities
    4. Download exported GeoTIFFs from Google Drive
    5. python -m src.download_indian_cities --process-downloads

    METHOD 2: Copernicus Open Access Hub (Sentinel-2)
    -------------------------------------------------
    1. Register at https://scihub.copernicus.eu/
    2. pip install sentinelsat
    3. Use the bounding boxes from configs/config.py:
       Mumbai:    [72.75, 18.85, 73.05, 19.30]
       Delhi_NCR: [76.85, 28.40, 77.45, 28.85]
       Bangalore: [77.45, 12.85, 77.75, 13.15]
       etc.

    METHOD 3: USGS EarthExplorer (Landsat)
    ----------------------------------------
    1. Register at https://earthexplorer.usgs.gov/
    2. Search for Landsat 5/7/8 Collection 2 Level 2
    3. Use the same bounding boxes as above
    4. Download Surface Reflectance products

    METHOD 4: Google Earth Engine Code Editor (No Python)
    -----------------------------------------------------
    1. Go to https://code.earthengine.google.com/
    2. Paste the JavaScript from: src/gee_export_script.js
    3. Click Run -> exports appear in Tasks tab
    4. Start each export task -> files go to Google Drive

    After downloading, run:
        python -m src.download_indian_cities --process-downloads
    to extract patches for the model.
    ================================================================
    """
    print(msg)
    return msg


# =========================================================================
#  GEE JavaScript export script (for Code Editor)
# =========================================================================

GEE_JS_SCRIPT = """
// ============================================================
// Urban Expansion Monitoring - Indian Cities
// Google Earth Engine Code Editor Script
// ============================================================
// Paste this into https://code.earthengine.google.com/
// Then click "Run" and start export tasks from the Tasks tab.

var cities = {
  'Mumbai':    ee.Geometry.Rectangle([72.75, 18.85, 73.05, 19.30]),
  'Delhi_NCR': ee.Geometry.Rectangle([76.85, 28.40, 77.45, 28.85]),
  'Bangalore': ee.Geometry.Rectangle([77.45, 12.85, 77.75, 13.15]),
  'Hyderabad': ee.Geometry.Rectangle([78.30, 17.30, 78.60, 17.55]),
  'Chennai':   ee.Geometry.Rectangle([80.15, 12.95, 80.35, 13.20]),
  'Pune':      ee.Geometry.Rectangle([73.75, 18.45, 73.95, 18.65]),
  'Ahmedabad': ee.Geometry.Rectangle([72.50, 22.95, 72.70, 23.15]),
};

// Sentinel-2 cloud masking
function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

// Landsat 8 cloud masking (Collection 2)
function maskL8clouds(image) {
  var qa = image.select('QA_PIXEL');
  var cloud = qa.bitwiseAnd(1 << 3).eq(0);
  var shadow = qa.bitwiseAnd(1 << 4).eq(0);
  var optical = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  return optical.updateMask(cloud.and(shadow));
}

// Export function
function exportCity(cityName, roi, collection, bands, start, end, scale, maskFn) {
  var filtered = collection
    .filterBounds(roi)
    .filterDate(start, end)
    .map(maskFn);

  var composite = filtered.select(bands).median().clip(roi);

  // Add NDVI
  var ndvi = composite.normalizedDifference([bands[3], bands[2]]).rename('NDVI');
  var ndbi = composite.normalizedDifference([bands[4], bands[3]]).rename('NDBI');
  composite = composite.addBands([ndvi, ndbi]);

  var desc = cityName + '_' + start.slice(0,4) + '_' + end.slice(0,4);

  Export.image.toDrive({
    image: composite,
    description: desc,
    folder: 'urban_expansion_india',
    region: roi,
    scale: scale,
    crs: 'EPSG:4326',
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
  });

  print(desc + ': ' + filtered.size().getInfo() + ' images');
}

// Sentinel-2 collection
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30));
var s2Bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'];

// Landsat 8 collection
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2');
var l8Bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];

// Export all cities for recent periods
for (var city in cities) {
  var roi = cities[city];

  // Sentinel-2: 2018-2020
  exportCity(city, roi, s2, s2Bands, '2018-01-01', '2020-12-31', 10, maskS2clouds);

  // Sentinel-2: 2020-2023
  exportCity(city, roi, s2, s2Bands, '2020-01-01', '2023-12-31', 10, maskS2clouds);

  // Landsat 8: 2013-2020
  exportCity(city, roi, l8, l8Bands, '2013-04-01', '2020-12-31', 30, maskL8clouds);
}

print('All export tasks created. Check the Tasks tab to start them.');
"""


def save_gee_js_script():
    """Save the GEE JavaScript export script."""
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gee_export_script.js"
    )
    with open(script_path, "w") as f:
        f.write(GEE_JS_SCRIPT)
    print(f"  GEE JavaScript saved to: {script_path}")
    print("  Paste into https://code.earthengine.google.com/ and click Run.")
    return script_path


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 / Landsat imagery for Indian cities via GEE"
    )
    parser.add_argument("--city", type=str, default=None, choices=CITIES,
                        help="Download for a single city only")
    parser.add_argument("--sensor", type=str, default=None,
                        choices=["sentinel2", "landsat"],
                        help="Download only one sensor type")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview exports without actually starting them")
    parser.add_argument("--process-downloads", action="store_true",
                        help="Process already-downloaded GeoTIFFs into patches")
    parser.add_argument("--save-js", action="store_true",
                        help="Save GEE JavaScript export script (no Python GEE needed)")
    parser.add_argument("--instructions", action="store_true",
                        help="Print alternative download instructions")
    args = parser.parse_args()

    print("=" * 60)
    print("  Indian City Satellite Imagery Downloader")
    print("=" * 60)

    if args.instructions:
        print_alternative_download_instructions()
        return

    if args.save_js:
        save_gee_js_script()
        return

    if args.process_downloads:
        cities_to_process = [args.city] if args.city else CITIES
        for city in cities_to_process:
            process_downloaded_geotiffs(city)
        return

    # Filter periods by sensor if requested
    periods = DOWNLOAD_PERIODS.copy()
    if args.sensor:
        periods = {k: v for k, v in periods.items() if v["sensor"] == args.sensor}

    # Check GEE authentication
    if not check_gee_auth():
        print("\n  Falling back to JavaScript export script...")
        save_gee_js_script()
        print_alternative_download_instructions()
        return

    # Download
    cities_to_download = [args.city] if args.city else CITIES
    all_tasks = []

    for city in cities_to_download:
        tasks = download_city(city, periods, dry_run=args.dry_run)
        all_tasks.extend(tasks)

    if all_tasks and not args.dry_run:
        print(f"\n  Started {len(all_tasks)} export tasks.")
        print("  Check Google Drive folder 'urban_expansion_india' for results.")
        print("  After download, run: python -m src.download_indian_cities --process-downloads")

    # Also save the JS script as backup
    save_gee_js_script()


if __name__ == "__main__":
    main()
