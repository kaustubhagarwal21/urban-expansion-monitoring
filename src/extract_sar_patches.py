"""
Extract SAR patches from GeoTIFFs at the same locations as existing optical patches.

For each city, reads the manifest to get lat/lon of each optical patch,
then extracts the corresponding 256x256 SAR patch (2 channels: VV, VH)
from the SAR GeoTIFF.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import rasterio
    from rasterio.transform import rowcol
except ImportError:
    print("rasterio required: pip install rasterio")
    sys.exit(1)


# Map city -> SAR file (best available)
SAR_FILES = {
    "Mumbai": "SAR_Mumbai_2023_post_monsoon.tif",
    "Delhi_NCR": "SAR_Delhi_NCR_2023_pre_monsoon.tif",
    "Bangalore": "SAR_Bangalore_2023_post_monsoon.tif",
}

GEE_DIR = r"G:\My Drive\urban_expansion_india"
PATCH_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "data", "indian_cities_locked")
PATCH_SIZE = 256


def extract_city_sar(city, dry_run=False):
    """Extract SAR patches for a single city."""
    sar_file = os.path.join(GEE_DIR, SAR_FILES[city])
    if not os.path.isfile(sar_file):
        print(f"  SKIP {city}: SAR file not found: {sar_file}")
        return 0

    # Load manifest
    manifest_path = os.path.join(PATCH_ROOT, city, "patches", f"{city}_manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"  SKIP {city}: manifest not found: {manifest_path}")
        return 0

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"  {city}: {len(manifest)} patches, SAR={SAR_FILES[city]}")

    # Open SAR GeoTIFF
    sar_ds = rasterio.open(sar_file)
    sar_data = sar_ds.read()  # (2, H, W)
    transform = sar_ds.transform

    # Output directory for SAR patches
    out_dir = os.path.join(PATCH_ROOT, city, "sar_patches")
    os.makedirs(out_dir, exist_ok=True)

    extracted = 0
    skipped = 0
    for entry in manifest:
        patch_id = entry["id"]
        lat = entry["lat"]
        lon = entry["lon"]

        # Convert lat/lon to pixel row/col
        row, col = rowcol(transform, lon, lat)

        # Check bounds (center the patch)
        half = PATCH_SIZE // 2
        r0 = row - half
        c0 = col - half
        r1 = r0 + PATCH_SIZE
        c1 = c0 + PATCH_SIZE

        if r0 < 0 or c0 < 0 or r1 > sar_data.shape[1] or c1 > sar_data.shape[2]:
            skipped += 1
            continue

        sar_patch = sar_data[:, r0:r1, c0:c1].astype(np.float32)  # (2, 256, 256)

        # Replace NaN with -30 dB (very low backscatter)
        sar_patch = np.nan_to_num(sar_patch, nan=-30.0)
        # SAR values are in dB. Clip to [-30, 0] and normalize to [0, 1]
        sar_patch = np.clip(sar_patch, -30.0, 0.0)
        sar_patch = (sar_patch + 30.0) / 30.0  # [-30,0] -> [0,1]

        if not dry_run:
            out_path = os.path.join(out_dir, f"{patch_id}_sar.npy")
            np.save(out_path, sar_patch)

        extracted += 1

    sar_ds.close()
    print(f"  {city}: extracted={extracted}, skipped={skipped} (out of bounds)")
    return extracted


def main():
    print("Extracting SAR patches from GeoTIFFs...")
    total = 0
    for city in SAR_FILES:
        total += extract_city_sar(city)
    print(f"\nDone. Total SAR patches extracted: {total}")


if __name__ == "__main__":
    main()
