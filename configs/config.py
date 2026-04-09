"""
Configuration for Urban Expansion Monitoring project.
Focused on Indian metropolitan areas.
"""
import os

# ── Paths ──────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ── Study Areas (Indian Metropolitan Regions) ─────────
CITIES = ["Mumbai", "Delhi_NCR", "Bangalore", "Hyderabad", "Chennai", "Pune", "Ahmedabad"]

# Bounding boxes for satellite data download [lon_min, lat_min, lon_max, lat_max]
CITY_BOUNDS = {
    "Mumbai":     [72.75, 18.85, 73.05, 19.30],
    "Delhi_NCR":  [76.85, 28.40, 77.45, 28.85],
    "Bangalore":  [77.45, 12.85, 77.75, 13.15],
    "Hyderabad":  [78.30, 17.30, 78.60, 17.55],
    "Chennai":    [80.15, 12.95, 80.35, 13.20],
    "Pune":       [73.75, 18.45, 73.95, 18.65],
    "Ahmedabad":  [72.50, 22.95, 72.70, 23.15],
}

# Reported urban area expansion 1990-2023 (growth multiplier)
CITY_GROWTH = {
    "Mumbai":     2.87,   # +287% - compact but fast, intense peri-urban
    "Delhi_NCR":  3.45,   # +345% - massive radial sprawl, satellite towns
    "Bangalore":  3.78,   # +378% - IT-driven rapid expansion
    "Hyderabad":  2.95,   # +295% - infrastructure-led (ORR, IT corridor)
    "Chennai":    2.42,   # +242% - coastal, post-flood urban reshaping
    "Pune":       3.15,   # +315% - IT + manufacturing corridor to Mumbai
    "Ahmedabad":  2.68,   # +268% - GIFT city, smart city expansion
}

CITY_DESCRIPTIONS = {
    "Mumbai":     "Compact but fast-growing fabric; intense peri-urban development",
    "Delhi_NCR":  "Massive radial sprawl; satellite towns (Noida, Gurgaon, Faridabad)",
    "Bangalore":  "IT-driven rapid expansion; lake encroachment and tech corridors",
    "Hyderabad":  "Infrastructure-led growth; Outer Ring Road and IT corridor",
    "Chennai":    "Coastal city; post-flood urban reshaping and IT growth",
    "Pune":       "IT + manufacturing corridor expansion toward Mumbai",
    "Ahmedabad":  "GIFT city, Smart City Mission; planned expansion",
}

PATCHES_PER_CITY = 3000
TOTAL_PATCHES = PATCHES_PER_CITY * len(CITIES)  # 21,000
PATCH_SIZE = 256
NUM_CHANNELS = 6  # Blue, Green, Red, NIR, SWIR1/NDVI, SWIR2/NDBI
NUM_CLASSES = 3
CLASS_NAMES = ["Urban", "Non-Urban", "Transition"]
CLASS_DISTRIBUTION = [0.45, 0.40, 0.15]  # Urban, Non-Urban, Transition
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Satellite Data Sources ────────────────────────────
# Landsat: historical backbone (1990-2023), 30m resolution
LANDSAT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
LANDSAT_RESOLUTION = 30  # metres

# Sentinel-2: contemporary detail (2015-present), 10m resolution
SENTINEL2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
SENTINEL2_RESOLUTION = 10  # metres

# Sentinel-1 SAR: all-weather monitoring
SAR_BANDS = ["VV", "VH"]  # Polarization channels
SAR_RESOLUTION = 10  # metres

# Time periods for analysis
TIME_PERIODS = {
    "historical": [(1990, 2000), (2000, 2010)],  # Landsat only
    "recent":     [(2015, 2020), (2020, 2023)],   # Sentinel-2 + Landsat
}

# ── Model ──────────────────────────────────────────────
BACKBONES = [
    "vgg16", "resnet50", "efficientnet_b0",       # Original 3
    "mobilenet_v3_small", "swin_tiny",             # New: edge + transformer
    "convnext_tiny", "prithvi",                    # New: modern CNN + foundation model
]
DEFAULT_BACKBONE = "resnet50"

# ── Progressive fine-tuning stages ─────────────────────
STAGES = [
    {"name": "stage1_frozen", "lr": 1e-3, "epochs": 20, "unfreeze": "head"},
    {"name": "stage2_partial", "lr": 1e-4, "epochs": 20, "unfreeze": "last_blocks"},
    {"name": "stage3_full", "lr": 1e-5, "epochs": 20, "unfreeze": "all"},
]

# ── Training ───────────────────────────────────────────
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
LOSS_WEIGHTS = {"ce": 0.6, "focal": 0.3, "dice": 0.1}
FOCAL_GAMMA = 2.0
EARLY_STOP_PATIENCE = 5
NUM_WORKERS = 2

# ── FPN ────────────────────────────────────────────────
FPN_CHANNELS = 256

# ── Augmentation ───────────────────────────────────────
MIXUP_ALPHA = 0.2
AUGMENT = True

SEED = 42

# Data source selection
DATA_SOURCE = "real"        # "synthetic" or "real"
REAL_DATASET = "eurosat"    # "eurosat", "so2sat", "spacenet"
ALLOW_REAL_DATA_DOWNLOAD = False

# ── Socio-Economic Data Sources (Pillar IV) ────────────
# Census of India, RBI, World Bank
SOCIO_ECONOMIC_FEATURES = [
    "population_density",       # Census of India
    "population_growth_rate",   # Census decadal growth
    "gdp_district",             # RBI district-level GDP
    "infrastructure_investment",# Smart City / AMRUT allocation
    "distance_to_nh",           # Distance to National Highway
    "distance_to_metro",        # Distance to metro/rail station
    "land_price_index",         # Property registration data
    "employment_rate",          # NSSO/PLFS data
    "construction_permits",     # Municipal corporation data
    "green_cover_ratio",        # Forest Survey of India
    "public_transport_score",   # UMTA data
    "elevation",                # SRTM DEM
    "slope",                    # Derived from DEM
    "water_proximity",          # Distance to river/coast
    "sez_distance",             # Distance to SEZ/IT park
    "school_hospital_density",  # Census amenities data
]
