"""Central path resolution for the demo backend.

Everything is resolved relative to this file so the app works regardless of the
current working directory (important for `uvicorn` launched from anywhere).
"""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = BACKEND_DIR.parent
PROJECT_ROOT = WEBAPP_DIR.parent

OUTPUTS = PROJECT_ROOT / "outputs"
RESULTS = OUTPUTS / "research_results"
MODELS = OUTPUTS / "models"
FIGURES = OUTPUTS / "figures"
PAPER = OUTPUTS / "paper"
INTEGRATION = OUTPUTS / "integration"
ALERTS = OUTPUTS / "alerts"

DATA = PROJECT_ROOT / "data" / "indian_cities_locked"

# Self-contained, offline demo assets bundled into the backend.
SAMPLES = BACKEND_DIR / "sample_patches"
