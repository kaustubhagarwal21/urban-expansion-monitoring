"""
Data loaders for real satellite imagery datasets:
  - EuroSAT (Sentinel-2 RGB land-use classification)
  - So2Sat LCZ42 (Sentinel-1 + Sentinel-2 local climate zones)
  - SpaceNet (high-resolution building footprint segmentation)

Each dataset is mapped to the project's 3-class schema:
  0 = Urban, 1 = Non-Urban, 2 = Transition
"""

import os, sys, random, warnings, zipfile, shutil, logging
from collections import Counter, defaultdict
from typing import Optional, Tuple, Dict, List, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import *

# Optional heavy imports --------------------------------------------------
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

logger = logging.getLogger(__name__)


def _env_city_filter(default_cities):
    raw = os.environ.get("INDIAN_CITIES_FILTER", "").strip()
    if not raw:
        return list(default_cities)
    allowed = {city.strip() for city in raw.split(",") if city.strip()}
    return [city for city in default_cities if city in allowed]


def _env_max_patches_per_city():
    raw = os.environ.get("INDIAN_MAX_PATCHES_PER_CITY", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        logger.warning("Invalid INDIAN_MAX_PATCHES_PER_CITY=%s; ignoring", raw)
        return None


def _env_indian_patch_root():
    raw = os.environ.get("INDIAN_PATCH_ROOT", "").strip()
    return raw or None

# =========================================================================
# EuroSAT mapping
# =========================================================================
EUROSAT_CLASS_MAP = {
    # Urban (class 0)
    "Residential":   0,
    "Industrial":    0,
    "Highway":       0,
    # Non-Urban (class 1)
    "Forest":                1,
    "HerbaceousVegetation":  1,
    "AnnualCrop":            1,
    "PermanentCrop":         1,
    "Pasture":               1,
    "River":                 1,
    "SeaLake":               1,
}

# So2Sat LCZ-to-project mapping (LCZ codes 1-17)
# LCZ 1-3: Compact high/mid/low-rise  -> Urban
# LCZ 4-6: Open high/mid/low-rise     -> Urban
# LCZ 7: Lightweight low-rise         -> Urban
# LCZ 8: Large low-rise               -> Urban
# LCZ 9: Sparsely built               -> Transition
# LCZ 10: Heavy industry              -> Urban
# LCZ 11-17: natural land covers      -> Non-Urban
SO2SAT_LCZ_MAP = {
    1: 0, 2: 0, 3: 0,        # Compact high/mid/low-rise
    4: 0, 5: 0, 6: 0,        # Open high/mid/low-rise
    7: 0,                     # Lightweight low-rise
    8: 0,                     # Large low-rise
    9: 2,                     # Sparsely built -> Transition
    10: 0,                    # Heavy industry
    11: 1, 12: 1, 13: 1,     # Dense trees, Scattered trees, Bush
    14: 1,                    # Low plants
    15: 1,                    # Bare rock/paved (natural context)
    16: 1,                    # Bare soil
    17: 1,                    # Water
}

EUROSAT_DOWNLOAD_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"

# =========================================================================
# 1. EuroSAT Dataset
# =========================================================================

def download_eurosat(data_dir: str = None) -> str:
    """
    Download and extract EuroSAT RGB dataset from Zenodo.

    Returns:
        Path to the extracted EuroSAT_RGB directory.
    """
    import urllib.request

    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, "eurosat")
    os.makedirs(data_dir, exist_ok=True)

    extracted_dir = os.path.join(data_dir, "EuroSAT_RGB")
    if os.path.isdir(extracted_dir) and len(os.listdir(extracted_dir)) > 0:
        logger.info("EuroSAT already present at %s", extracted_dir)
        return extracted_dir

    zip_path = os.path.join(data_dir, "EuroSAT_RGB.zip")
    if not os.path.isfile(zip_path):
        print(f"[EuroSAT] Downloading from {EUROSAT_DOWNLOAD_URL} ...")
        print(f"[EuroSAT] Saving to {zip_path}")
        urllib.request.urlretrieve(EUROSAT_DOWNLOAD_URL, zip_path)
        print("[EuroSAT] Download complete.")
    else:
        print(f"[EuroSAT] Zip already exists: {zip_path}")

    print("[EuroSAT] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print(f"[EuroSAT] Extracted to {extracted_dir}")

    return extracted_dir


class EuroSATDataset(Dataset):
    """
    EuroSAT RGB dataset mapped to 3-class urban expansion schema.

    Each 64x64 Sentinel-2 RGB patch is:
      - Resized to PATCH_SIZE x PATCH_SIZE (256x256) via bilinear interpolation
      - Expanded from 3 to 6 channels:
          ch0-2: R, G, B  (normalised to [0,1])
          ch3: pseudo-NDVI = (G - R) / (G + R + eps)
          ch4: brightness  = mean(R, G, B)
          ch5: texture     = local std-dev of brightness
    """

    def __init__(
        self,
        root_dir: str = None,
        transform: Optional[Callable] = None,
        transition_fraction: float = 0.30,
        download: bool = False,
        seed: int = SEED,
    ):
        super().__init__()
        if root_dir is None:
            root_dir = os.path.join(DATA_DIR, "eurosat", "EuroSAT_RGB")

        self.root_dir = root_dir
        self.transform = transform
        self.transition_fraction = transition_fraction
        self.seed = seed
        self.samples: List[Tuple[str, int]] = []  # (path, mapped_label)

        if download and not os.path.isdir(root_dir):
            download_eurosat(os.path.dirname(root_dir))

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f"EuroSAT data not found at {root_dir}. "
                "Pass download=True or run download_eurosat() first."
            )

        self._load_file_list()

    # ------------------------------------------------------------------
    def _load_file_list(self):
        rng = random.Random(self.seed)
        border_classes = {"AnnualCrop", "PermanentCrop", "Pasture"}

        for class_name, mapped_label in EUROSAT_CLASS_MAP.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                logger.warning("EuroSAT class folder missing: %s", class_dir)
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith(".jpg"):
                    continue
                fpath = os.path.join(class_dir, fname)
                label = mapped_label
                # Randomly assign ~transition_fraction of border cases to Transition
                if class_name in border_classes and rng.random() < self.transition_fraction:
                    label = 2  # Transition
                self.samples.append((fpath, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {self.root_dir}. "
                "Check that the directory contains class sub-folders with .jpg images."
            )
        logger.info("EuroSAT: loaded %d samples", len(self.samples))

    # ------------------------------------------------------------------
    @staticmethod
    def _rgb_to_6ch(img_rgb: np.ndarray) -> np.ndarray:
        """
        Convert (H, W, 3) uint8 RGB image to (6, PATCH_SIZE, PATCH_SIZE) float32.
        """
        # Resize: use 64x64 (native EuroSAT resolution) to avoid
        # upsampling artifacts and reduce compute cost.
        target_size = 64
        pil_img = Image.fromarray(img_rgb)
        if pil_img.size != (target_size, target_size):
            pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        eps = 1e-7
        ndvi_like = (g - r) / (g + r + eps)              # ch3
        brightness = np.mean(arr, axis=2)                  # ch4

        # Texture: local std of brightness (5x5 window via uniform_filter)
        try:
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(brightness, size=5)
            local_sq_mean = uniform_filter(brightness ** 2, size=5)
            texture = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0.0))
        except ImportError:
            # Fallback: simple gradient magnitude
            dy = np.diff(brightness, axis=0, prepend=brightness[:1, :])
            dx = np.diff(brightness, axis=1, prepend=brightness[:, :1])
            texture = np.sqrt(dx ** 2 + dy ** 2)

        # Stack to (6, H, W)
        patch = np.stack([r, g, b, ndvi_like, brightness, texture], axis=0)
        return patch.astype(np.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label = self.samples[idx]
        img = Image.open(fpath).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)

        patch = self._rgb_to_6ch(img_np)  # (6, 256, 256)
        patch_tensor = torch.from_numpy(patch)

        if self.transform is not None:
            patch_tensor = self.transform(patch_tensor)

        return patch_tensor, label


# =========================================================================
# 2. So2Sat LCZ42 Dataset
# =========================================================================

def print_so2sat_download_instructions():
    """Print instructions to download So2Sat LCZ42 manually."""
    msg = """
    ================================================================
    So2Sat LCZ42 Dataset — Download Instructions
    ================================================================
    The So2Sat LCZ42 dataset requires registration:

    1. Go to  https://mediatum.ub.tum.de/1454690
    2. Register / log in with an institutional account.
    3. Download the HDF5 files:
         - training.h5  (~27 GB)
         - validation.h5 (~3 GB)
         - testing.h5    (if available)
    4. Place them in:
         {data_dir}

    Expected structure:
         data/so2sat/training.h5
         data/so2sat/validation.h5
    ================================================================
    """.format(data_dir=os.path.join(DATA_DIR, "so2sat"))
    print(msg)
    return msg


class So2SatDataset(Dataset):
    """
    So2Sat LCZ42 dataset: paired Sentinel-1 (SAR) + Sentinel-2 (optical)
    patches for Local Climate Zone classification.

    Returns:
        (optical_patch, sar_patch, label)
        - optical_patch : (10, 32, 32)  — 10 Sentinel-2 bands
        - sar_patch     : (8, 32, 32)   — 8 Sentinel-1 bands
        - label         : int in {0, 1, 2}
    """

    def __init__(
        self,
        h5_path: str = None,
        split: str = "training",
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for So2Sat LCZ42. Install with: pip install h5py"
            )

        if h5_path is None:
            h5_path = os.path.join(DATA_DIR, "so2sat", f"{split}.h5")

        self.h5_path = h5_path
        self.transform = transform

        if not os.path.isfile(h5_path):
            print_so2sat_download_instructions()
            raise FileNotFoundError(
                f"So2Sat HDF5 file not found: {h5_path}"
            )

        # Read labels and map them
        with h5py.File(h5_path, "r") as f:
            raw_labels = np.array(f["label"]).flatten().astype(int)
            self.n_samples = len(raw_labels)
            # Keys for Sentinel-2 and Sentinel-1
            self._sen2_key = "sen2" if "sen2" in f else "sentinel2"
            self._sen1_key = "sen1" if "sen1" in f else "sentinel1"

        self.labels = np.array(
            [SO2SAT_LCZ_MAP.get(int(l), 1) for l in raw_labels], dtype=np.int64
        )

        # We will read patches lazily from the h5 file
        self._h5_file = None
        logger.info("So2Sat (%s): %d samples loaded", split, self.n_samples)

    # Lazy HDF5 handle (safe for DataLoader num_workers > 0)
    def _get_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        f = self._get_h5()
        optical = np.array(f[self._sen2_key][idx], dtype=np.float32)  # (H, W, C)
        sar = np.array(f[self._sen1_key][idx], dtype=np.float32)

        # Transpose to (C, H, W) if needed
        if optical.ndim == 3 and optical.shape[2] < optical.shape[0]:
            pass  # already (C, H, W)
        elif optical.ndim == 3:
            optical = np.transpose(optical, (2, 0, 1))
        if sar.ndim == 3 and sar.shape[2] < sar.shape[0]:
            pass
        elif sar.ndim == 3:
            sar = np.transpose(sar, (2, 0, 1))

        optical_t = torch.from_numpy(optical)
        sar_t = torch.from_numpy(sar)
        label = int(self.labels[idx])

        if self.transform is not None:
            optical_t = self.transform(optical_t)
            sar_t = self.transform(sar_t)

        return optical_t, sar_t, label

    def __del__(self):
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass


# =========================================================================
# 3. SpaceNet Dataset
# =========================================================================

def print_spacenet_download_instructions():
    """Print instructions to download SpaceNet data."""
    msg = """
    ================================================================
    SpaceNet Dataset — Download Instructions
    ================================================================
    SpaceNet datasets are hosted on AWS S3 (requester-pays):

    1. Install AWS CLI:  pip install awscli
    2. Configure credentials:  aws configure
    3. Download SpaceNet Building Detection:

       aws s3 cp s3://spacenet-dataset/spacenet/SN2_buildings/tarballs/ \\
           {data_dir} --request-payer requester --recursive

    4. Extract the tar.gz files into:
           {data_dir}/AOI_*/

    Expected structure:
        data/spacenet/AOI_2_Vegas/
            RGB-PanSharpen/  (GeoTIFF images)
            geojson/buildings/  (building polygon labels)
    ================================================================
    """.format(data_dir=os.path.join(DATA_DIR, "spacenet"))
    print(msg)
    return msg


class SpaceNetDataset(Dataset):
    """
    SpaceNet building footprint segmentation dataset.

    Loads GeoTIFF image tiles and rasterises building polygon labels
    into binary segmentation masks.

    Returns:
        (image_patch, building_mask)
        - image_patch  : (C, H, W) float32  (3 or 4 channels)
        - building_mask: (1, H, W) float32  (1=building, 0=background)
    """

    def __init__(
        self,
        root_dir: str = None,
        aoi: str = "AOI_2_Vegas",
        patch_size: int = PATCH_SIZE,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        if not HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for SpaceNet. "
                "Install with: pip install rasterio"
            )

        if root_dir is None:
            root_dir = os.path.join(DATA_DIR, "spacenet")

        self.root_dir = root_dir
        self.aoi = aoi
        self.patch_size = patch_size
        self.transform = transform

        self.image_dir = os.path.join(root_dir, aoi, "RGB-PanSharpen")
        self.geojson_dir = os.path.join(root_dir, aoi, "geojson", "buildings")

        if not os.path.isdir(self.image_dir):
            print_spacenet_download_instructions()
            raise FileNotFoundError(
                f"SpaceNet image directory not found: {self.image_dir}"
            )

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".tif", ".tiff"))
        ])

        if len(self.image_files) == 0:
            raise RuntimeError(f"No GeoTIFF images found in {self.image_dir}")

        logger.info("SpaceNet (%s): %d tiles found", aoi, len(self.image_files))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        import rasterio
        from rasterio.features import rasterize
        import json

        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Read image
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)  # (C, H, W)
            img_transform = src.transform
            h, w = src.height, src.width

        # Normalise to [0, 1]
        for c in range(img.shape[0]):
            cmax = img[c].max()
            if cmax > 0:
                img[c] /= cmax

        # Build mask from GeoJSON
        mask = np.zeros((h, w), dtype=np.float32)
        # Derive geojson filename from image filename
        geojson_name = img_name.replace("RGB-PanSharpen_", "buildings_")
        geojson_name = os.path.splitext(geojson_name)[0] + ".geojson"
        geojson_path = os.path.join(self.geojson_dir, geojson_name)

        if os.path.isfile(geojson_path):
            try:
                with open(geojson_path, "r") as gf:
                    geo = json.load(gf)
                shapes = []
                for feature in geo.get("features", []):
                    geom = feature.get("geometry")
                    if geom is not None:
                        shapes.append((geom, 1))
                if shapes:
                    mask = rasterize(
                        shapes,
                        out_shape=(h, w),
                        transform=img_transform,
                        dtype=np.float32,
                    )
            except Exception as e:
                logger.warning("Failed to rasterize %s: %s", geojson_path, e)

        # Resize to patch_size
        if h != self.patch_size or w != self.patch_size:
            # Resize image channels
            resized_channels = []
            for c in range(img.shape[0]):
                pil_c = Image.fromarray(img[c])
                pil_c = pil_c.resize(
                    (self.patch_size, self.patch_size), Image.BILINEAR
                )
                resized_channels.append(np.array(pil_c, dtype=np.float32))
            img = np.stack(resized_channels, axis=0)

            # Resize mask (nearest to preserve binary)
            pil_m = Image.fromarray(mask)
            pil_m = pil_m.resize(
                (self.patch_size, self.patch_size), Image.NEAREST
            )
            mask = np.array(pil_m, dtype=np.float32)

        mask = mask[np.newaxis, :, :]  # (1, H, W)

        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)

        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, mask_tensor


# =========================================================================
# 3b. Dataset Adapters (make So2Sat/SpaceNet work with classification pipeline)
# =========================================================================

class So2SatClassificationAdapter(Dataset):
    """
    Wraps So2SatDataset to return (6ch_patch, label) for the classification pipeline.
    Selects 6 optical bands from the 10 Sentinel-2 channels to match project schema:
    Blue(B2), Green(B3), Red(B4), NIR(B8), SWIR1(B11), SWIR2(B12).
    So2Sat S2 bands order: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 → indices 0,1,2,6,8,9
    """
    S2_BAND_INDICES = [0, 1, 2, 6, 8, 9]  # Blue,Green,Red,NIR,SWIR1,SWIR2

    def __init__(self, so2sat_dataset):
        self.ds = so2sat_dataset
        self.labels = so2sat_dataset.labels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        optical, sar, label = self.ds[idx]
        # Select 6 bands matching our schema, resize from 32x32 to 64x64
        patch_6ch = optical[self.S2_BAND_INDICES]  # (6, 32, 32)
        # Normalize to [0,1]
        pmax = patch_6ch.max()
        if pmax > 1.0:
            patch_6ch = patch_6ch / pmax
        return patch_6ch, label

    @property
    def samples(self):
        return [(None, int(l)) for l in self.labels]


class SpaceNetClassificationAdapter(Dataset):
    """
    Wraps SpaceNetDataset to return (patch, label) for classification.
    Label is derived from building mask density:
      - >50% buildings → Urban (0)
      - <10% buildings → Non-Urban (1)
      - otherwise → Transition (2)
    """
    def __init__(self, spacenet_dataset):
        self.ds = spacenet_dataset
        self._labels = None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        building_pct = mask.mean().item()
        if building_pct > 0.5:
            label = 0  # Urban
        elif building_pct < 0.1:
            label = 1  # Non-Urban
        else:
            label = 2  # Transition

        # Use first 3 channels (RGB) + derive 3 pseudo channels
        if img.shape[0] >= 3:
            r, g, b = img[0], img[1], img[2]
            ndvi = (g - r) / (g + r + 1e-7)
            brightness = (r + g + b) / 3
            texture = brightness  # placeholder
            patch = torch.stack([r, g, b, ndvi, brightness, texture], dim=0)
        else:
            patch = img
        return patch, label


class IndianCityDataset(Dataset):
    """
    Dataset loader for GEE-downloaded Indian city patches.
    Reads .npy patches extracted by download_data.extract_patches_from_geotiff().

    Searches recursively for _img.npy / _lbl.npy pairs in:
        data/indian_cities/{city}/**/{city}_000001_img.npy
        data/processed/{city}/**/{city}_000001_img.npy
    """
    def __init__(self, cities=None, patch_dir=None, transform=None):
        super().__init__()
        self.transform = transform
        self.samples = []  # (img_path, label)
        seen_paths = set()

        cities = _env_city_filter(cities or CITIES)
        max_patches_per_city = _env_max_patches_per_city()

        # Search multiple possible locations
        env_patch_root = _env_indian_patch_root()
        if patch_dir:
            search_dirs = [patch_dir]
        elif env_patch_root:
            # When a locked subset is explicitly selected, do not mix it with
            # exploratory/legacy patch directories.
            search_dirs = [env_patch_root]
        else:
            search_dirs = [
                os.path.join(DATA_DIR, "indian_cities"),
                PROCESSED_DATA_DIR,
            ]

        for base_dir in search_dirs:
            for city in cities:
                city_samples = []
                city_dir = os.path.join(base_dir, city)
                if not os.path.isdir(city_dir):
                    continue
                # Recursively find all _img.npy files
                for root, dirs, files in os.walk(city_dir):
                    img_files = sorted([f for f in files if f.endswith("_img.npy")])
                    for img_f in img_files:
                        lbl_f = img_f.replace("_img.npy", "_lbl.npy")
                        lbl_path = os.path.join(root, lbl_f)
                        if not os.path.isfile(lbl_path):
                            continue
                        img_path = os.path.join(root, img_f)
                        if img_path in seen_paths:
                            continue
                        lbl_data = np.load(lbl_path)
                        majority_label = int(np.bincount(lbl_data.flatten().astype(int)).argmax())
                        city_samples.append((img_path, majority_label))
                        seen_paths.add(img_path)

                if max_patches_per_city is not None and len(city_samples) > max_patches_per_city:
                    rng = random.Random(SEED)
                    city_samples = rng.sample(city_samples, max_patches_per_city)

                self.samples.extend(city_samples)

        if len(self.samples) == 0:
            logger.warning("IndianCityDataset: no patches found in %s", patch_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = np.load(img_path).astype(np.float32)  # (C, H, W)
        patch = torch.from_numpy(img)
        if patch.ndim == 3 and tuple(patch.shape[-2:]) != (PATCH_SIZE, PATCH_SIZE):
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(PATCH_SIZE, PATCH_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.transform:
            patch = self.transform(patch)
        return patch, label


# =========================================================================
# 4. RealDataManager
# =========================================================================

class RealDataManager:
    """
    Unified manager that checks dataset availability and provides
    DataLoaders for EuroSAT, So2Sat, and SpaceNet, with fallback
    to synthetic data if a real dataset is not found.
    """

    def __init__(self, batch_size: int = BATCH_SIZE, num_workers: int = 0,
                 seed: int = SEED):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self._availability: Dict[str, bool] = {}
        self._check_availability()

    # ------------------------------------------------------------------
    def _check_availability(self):
        eurosat_dir = os.path.join(DATA_DIR, "eurosat", "EuroSAT_RGB")
        self._availability["eurosat"] = (
            os.path.isdir(eurosat_dir)
            and len(os.listdir(eurosat_dir)) > 0
        )

        so2sat_train = os.path.join(DATA_DIR, "so2sat", "training.h5")
        self._availability["so2sat"] = os.path.isfile(so2sat_train) and HAS_H5PY

        spacenet_dir = os.path.join(DATA_DIR, "spacenet")
        self._availability["spacenet"] = (
            os.path.isdir(spacenet_dir)
            and HAS_RASTERIO
            and any(
                os.path.isdir(os.path.join(spacenet_dir, d))
                for d in os.listdir(spacenet_dir)
                if d.startswith("AOI_")
            )
            if os.path.isdir(spacenet_dir) else False
        )

        # Check for Indian city patches from either supported extraction layout:
        #   data/indian_cities/{city}/**/*_img.npy
        #   data/processed/{city}/**/*_img.npy
        indian_count = 0
        env_patch_root = _env_indian_patch_root()
        if env_patch_root:
            search_roots = [env_patch_root]
        else:
            search_roots = [
                os.path.join(DATA_DIR, "indian_cities"),
                PROCESSED_DATA_DIR,
            ]
        for base_dir in search_roots:
            if not os.path.isdir(base_dir):
                continue
            for city in CITIES:
                city_dir = os.path.join(base_dir, city)
                if not os.path.isdir(city_dir):
                    continue
                for root, _, files in os.walk(city_dir):
                    indian_count += sum(1 for f in files if f.endswith("_img.npy"))
        self._availability["indian_cities"] = indian_count > 0
        self._indian_patch_count = indian_count

        print("=" * 60)
        print("Real Dataset Availability")
        print("=" * 60)
        for name, avail in self._availability.items():
            status = "AVAILABLE" if avail else "NOT FOUND"
            print(f"  {name:12s} : {status}")
        print("=" * 60)

    def is_available(self, name: str) -> bool:
        return self._availability.get(name, False)

    # ------------------------------------------------------------------
    @staticmethod
    def _stratified_split(dataset, labels, train_ratio=TRAIN_RATIO,
                          val_ratio=VAL_RATIO, seed=SEED):
        """Split dataset indices with stratification."""
        n = len(dataset)
        indices = list(range(n))
        labels_arr = np.array(labels)

        # First split: train vs. (val+test)
        train_idx, valtest_idx = train_test_split(
            indices,
            test_size=1.0 - train_ratio,
            stratify=labels_arr[indices],
            random_state=seed,
        )
        # Second split: val vs. test
        relative_val = val_ratio / (val_ratio + (1.0 - train_ratio - val_ratio))
        val_idx, test_idx = train_test_split(
            valtest_idx,
            test_size=1.0 - relative_val,
            stratify=labels_arr[valtest_idx],
            random_state=seed,
        )
        return (
            Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx),
        )

    # ------------------------------------------------------------------
    def _make_loaders(self, train_ds, val_ds, test_ds):
        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Weighted sampling for training to handle class imbalance (especially Transition)
        train_labels = self._extract_labels(train_ds)
        if train_labels is not None:
            from collections import Counter
            counts = Counter(train_labels)
            num_samples = len(train_labels)
            class_weights = {c: num_samples / (len(counts) * cnt) for c, cnt in counts.items()}
            sample_weights = [class_weights[lbl] for lbl in train_labels]
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=num_samples, replacement=True
            )
            train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
        else:
            train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
        return train_loader, val_loader, test_loader

    @staticmethod
    def _extract_labels(dataset):
        """Extract labels from a dataset (handles Subset wrapping)."""
        try:
            if hasattr(dataset, 'samples'):
                return [lbl for _, lbl in dataset.samples]
            if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
                # torch Subset
                base = dataset.dataset
                if hasattr(base, 'samples'):
                    return [base.samples[i][1] for i in dataset.indices]
                if hasattr(base, 'labels'):
                    labels = base.labels
                    return [labels[i] for i in dataset.indices]
            if hasattr(dataset, 'labels'):
                return list(dataset.labels)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def get_eurosat_loaders(self, download: bool = False):
        """
        Returns (train_loader, val_loader, test_loader) for EuroSAT.
        Falls back to synthetic data if EuroSAT is unavailable.
        """
        if not self.is_available("eurosat") and not download:
            warnings.warn(
                "EuroSAT not found — falling back to synthetic data.",
                UserWarning,
            )
            return self._synthetic_fallback()

        dataset = EuroSATDataset(download=download, seed=self.seed)
        labels = [lbl for _, lbl in dataset.samples]
        train_ds, val_ds, test_ds = self._stratified_split(
            dataset, labels, seed=self.seed
        )
        print(f"[EuroSAT] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        return self._make_loaders(train_ds, val_ds, test_ds)

    def get_so2sat_loaders(self):
        """
        Returns (train_loader, val_loader, test_loader) for So2Sat.
        Falls back to synthetic data if So2Sat is unavailable.
        """
        if not self.is_available("so2sat"):
            warnings.warn(
                "So2Sat LCZ42 not found — falling back to synthetic data.",
                UserWarning,
            )
            print_so2sat_download_instructions()
            return self._synthetic_fallback()

        train_h5 = os.path.join(DATA_DIR, "so2sat", "training.h5")
        val_h5 = os.path.join(DATA_DIR, "so2sat", "validation.h5")

        train_ds = So2SatDataset(h5_path=train_h5, split="training")

        if os.path.isfile(val_h5):
            # Use validation.h5 directly; split it into val + test
            full_val_ds = So2SatDataset(h5_path=val_h5, split="validation")
            n_val = len(full_val_ds)
            half = n_val // 2
            val_ds = Subset(full_val_ds, list(range(half)))
            test_ds = Subset(full_val_ds, list(range(half, n_val)))
        else:
            # Split training into train / val / test
            labels = list(train_ds.labels)
            train_ds, val_ds, test_ds = self._stratified_split(
                train_ds, labels, seed=self.seed
            )

        print(f"[So2Sat] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        return self._make_loaders(train_ds, val_ds, test_ds)

    def get_spacenet_loaders(self, aoi: str = "AOI_2_Vegas"):
        """
        Returns (train_loader, val_loader, test_loader) for SpaceNet.
        Falls back to synthetic data if SpaceNet is unavailable.
        """
        if not self.is_available("spacenet"):
            warnings.warn(
                "SpaceNet not found — falling back to synthetic data.",
                UserWarning,
            )
            print_spacenet_download_instructions()
            return self._synthetic_fallback()

        dataset = SpaceNetDataset(aoi=aoi)
        n = len(dataset)
        indices = list(range(n))
        random.Random(self.seed).shuffle(indices)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train_ds = Subset(dataset, indices[:n_train])
        val_ds = Subset(dataset, indices[n_train:n_train + n_val])
        test_ds = Subset(dataset, indices[n_train + n_val:])

        print(f"[SpaceNet] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        return self._make_loaders(train_ds, val_ds, test_ds)

    # ------------------------------------------------------------------
    def get_indian_city_loaders(self, cities=None):
        """
        Returns (train_loader, val_loader, test_loader) for Indian city patches.
        Requires GEE data to be downloaded and extracted first.
        Falls back to EuroSAT if no Indian data available.
        """
        if not self.is_available("indian_cities"):
            warnings.warn(
                f"Indian city patches not found in {PROCESSED_DATA_DIR}. "
                "Run: python src/download_data.py --download  then  --extract. "
                "Falling back to EuroSAT.",
                UserWarning,
            )
            if self.is_available("eurosat"):
                return self.get_eurosat_loaders()
            return self._synthetic_fallback()

        cities = _env_city_filter(cities or CITIES)
        dataset = IndianCityDataset(cities=cities)
        labels = [lbl for _, lbl in dataset.samples]
        train_ds, val_ds, test_ds = self._stratified_split(
            dataset, labels, seed=self.seed
        )
        print(f"[IndianCities] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        print(f"[IndianCities] cities={cities}")
        print(f"[IndianCities] env_limited_total={len(dataset)} | discovered_total={self._indian_patch_count}")
        return self._make_loaders(train_ds, val_ds, test_ds)

    # ------------------------------------------------------------------
    def get_so2sat_classification_loaders(self):
        """
        So2Sat wrapped for classification pipeline (returns 6ch optical + label).
        For SAR fusion (Pillar I), use get_so2sat_loaders() directly.
        """
        if not self.is_available("so2sat"):
            warnings.warn("So2Sat not found — falling back.", UserWarning)
            print_so2sat_download_instructions()
            return self._synthetic_fallback()

        train_h5 = os.path.join(DATA_DIR, "so2sat", "training.h5")
        val_h5 = os.path.join(DATA_DIR, "so2sat", "validation.h5")
        base_ds = So2SatDataset(h5_path=train_h5, split="training")
        adapted = So2SatClassificationAdapter(base_ds)
        labels = list(adapted.labels)
        train_ds, val_ds, test_ds = self._stratified_split(adapted, labels, seed=self.seed)
        print(f"[So2Sat-Classification] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        return self._make_loaders(train_ds, val_ds, test_ds)

    # ------------------------------------------------------------------
    def _synthetic_fallback(self):
        """
        Create synthetic DataLoaders using the project's existing generator.
        """
        from src.dataset import get_data_loaders
        print("[Fallback] Using synthetic data loaders.")
        return get_data_loaders(
            batch_size=self.batch_size,
            data_source="synthetic",
        )


# =========================================================================
# 5. Dataset Statistics
# =========================================================================

def compute_class_distribution(dataset: Dataset, label_index: int = -1) -> Dict[int, int]:
    """
    Count per-class samples.  ``label_index`` indicates which position
    in the tuple returned by __getitem__ holds the label (-1 = last).
    """
    counts: Dict[int, int] = Counter()
    for i in range(len(dataset)):
        item = dataset[i]
        label = int(item[label_index]) if isinstance(item, (tuple, list)) else int(item)
        counts[label] += 1
    return dict(sorted(counts.items()))


def compute_channel_stats(
    dataset: Dataset,
    max_samples: int = 500,
    data_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute channel-wise mean and std over a random subset of the dataset.

    Args:
        dataset: a PyTorch Dataset whose __getitem__ returns (tensor, ...).
        max_samples: number of samples to use (for speed).
        data_index: which element of the returned tuple contains the tensor.

    Returns:
        (mean_per_channel, std_per_channel) as numpy arrays.
    """
    rng = random.Random(SEED)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:max_samples]

    running_sum = None
    running_sq_sum = None
    n_pixels = 0

    for idx in indices:
        item = dataset[idx]
        tensor = item[data_index] if isinstance(item, (tuple, list)) else item
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        # tensor shape: (C, H, W)
        c = tensor.shape[0]
        if running_sum is None:
            running_sum = np.zeros(c, dtype=np.float64)
            running_sq_sum = np.zeros(c, dtype=np.float64)
        pixels = tensor.reshape(c, -1)
        running_sum += pixels.sum(axis=1)
        running_sq_sum += (pixels ** 2).sum(axis=1)
        n_pixels += pixels.shape[1]

    mean = running_sum / n_pixels
    std = np.sqrt(running_sq_sum / n_pixels - mean ** 2)
    return mean.astype(np.float32), std.astype(np.float32)


def print_dataset_statistics(
    dataset: Dataset,
    name: str,
    label_index: int = -1,
    data_index: int = 0,
    max_samples: int = 500,
):
    """Compute and print statistics for a dataset."""
    print(f"\n{'=' * 60}")
    print(f"Statistics for: {name}")
    print(f"{'=' * 60}")
    print(f"  Total samples : {len(dataset)}")

    # Class distribution
    print("\n  Per-class counts:")
    counts = compute_class_distribution(dataset, label_index=label_index)
    total = sum(counts.values())
    for cls_id, cnt in counts.items():
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class-{cls_id}"
        pct = 100.0 * cnt / total if total else 0
        print(f"    {cls_name:12s} (id={cls_id}): {cnt:6d}  ({pct:5.1f}%)")

    # Channel stats
    print(f"\n  Channel-wise mean/std (computed on up to {max_samples} samples):")
    try:
        mean, std = compute_channel_stats(
            dataset, max_samples=max_samples, data_index=data_index
        )
        for ch_i, (m, s) in enumerate(zip(mean, std)):
            print(f"    Channel {ch_i}: mean={m:.4f}, std={s:.4f}")
    except Exception as e:
        print(f"    [Could not compute channel stats: {e}]")

    print(f"{'=' * 60}\n")


def compare_with_synthetic(real_counts: Dict[int, int]):
    """
    Print a side-by-side comparison of real vs. synthetic class distributions.
    """
    synthetic_dist = dict(enumerate(CLASS_DISTRIBUTION))
    real_total = sum(real_counts.values())

    print(f"\n{'=' * 60}")
    print("Class Distribution Comparison: Real vs Synthetic Target")
    print(f"{'=' * 60}")
    print(f"  {'Class':12s}  {'Real':>8s}  {'Real %':>8s}  {'Synthetic %':>12s}")
    print(f"  {'-' * 44}")
    for cls_id in range(NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_id]
        r_cnt = real_counts.get(cls_id, 0)
        r_pct = 100.0 * r_cnt / real_total if real_total else 0
        s_pct = 100.0 * synthetic_dist.get(cls_id, 0)
        print(f"  {cls_name:12s}  {r_cnt:8d}  {r_pct:7.1f}%  {s_pct:11.1f}%")
    print(f"{'=' * 60}\n")


# =========================================================================
# Main — test all available datasets
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 60)
    print("  Real Data Loaders — Diagnostics")
    print("=" * 60)

    manager = RealDataManager(batch_size=BATCH_SIZE)

    # ── EuroSAT ──────────────────────────────────────────
    if manager.is_available("eurosat"):
        print("\n>>> Testing EuroSAT ...")
        try:
            ds = EuroSATDataset()
            print_dataset_statistics(ds, "EuroSAT", label_index=-1, data_index=0)

            # Compare with synthetic
            counts = compute_class_distribution(ds, label_index=-1)
            compare_with_synthetic(counts)

            # Test a batch
            train_ld, val_ld, test_ld = manager.get_eurosat_loaders()
            batch = next(iter(train_ld))
            imgs, labels = batch
            print(f"  Batch shape: images={tuple(imgs.shape)}, labels={tuple(labels.shape)}")
        except Exception as e:
            print(f"  EuroSAT test failed: {e}")
    else:
        print("\n[SKIP] EuroSAT not available locally.")
        print("       To download: download_eurosat() or pass download=True")

    # ── So2Sat ───────────────────────────────────────────
    if manager.is_available("so2sat"):
        print("\n>>> Testing So2Sat LCZ42 ...")
        try:
            ds = So2SatDataset()
            print(f"  So2Sat samples: {len(ds)}")
            optical, sar, label = ds[0]
            print(f"  Sample 0: optical={tuple(optical.shape)}, sar={tuple(sar.shape)}, label={label}")

            train_ld, val_ld, test_ld = manager.get_so2sat_loaders()
            batch = next(iter(train_ld))
            opt_b, sar_b, lab_b = batch
            print(f"  Batch: optical={tuple(opt_b.shape)}, sar={tuple(sar_b.shape)}, labels={tuple(lab_b.shape)}")
        except Exception as e:
            print(f"  So2Sat test failed: {e}")
    else:
        print("\n[SKIP] So2Sat LCZ42 not available locally.")
        if not HAS_H5PY:
            print("       h5py is not installed (pip install h5py)")
        print_so2sat_download_instructions()

    # ── SpaceNet ─────────────────────────────────────────
    if manager.is_available("spacenet"):
        print("\n>>> Testing SpaceNet ...")
        try:
            ds = SpaceNetDataset()
            print(f"  SpaceNet tiles: {len(ds)}")
            img, mask = ds[0]
            print(f"  Sample 0: image={tuple(img.shape)}, mask={tuple(mask.shape)}")

            train_ld, val_ld, test_ld = manager.get_spacenet_loaders()
            batch = next(iter(train_ld))
            img_b, mask_b = batch
            print(f"  Batch: images={tuple(img_b.shape)}, masks={tuple(mask_b.shape)}")
        except Exception as e:
            print(f"  SpaceNet test failed: {e}")
    else:
        print("\n[SKIP] SpaceNet not available locally.")
        if not HAS_RASTERIO:
            print("       rasterio is not installed (pip install rasterio)")
        print_spacenet_download_instructions()

    # ── Fallback test ────────────────────────────────────
    print("\n>>> Testing synthetic fallback ...")
    try:
        fb_train, fb_val, fb_test = manager._synthetic_fallback()
        batch = next(iter(fb_train))
        if isinstance(batch, (tuple, list)):
            print(f"  Synthetic fallback batch: {tuple(batch[0].shape)}")
        else:
            print(f"  Synthetic fallback batch: {tuple(batch.shape)}")
    except Exception as e:
        print(f"  Synthetic fallback failed: {e}")

    print("\n" + "=" * 60)
    print("  Diagnostics complete.")
    print("=" * 60)
