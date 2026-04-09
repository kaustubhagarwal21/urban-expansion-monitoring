# Urban Expansion Monitoring Using Transfer Learning on Historical Satellite Imagery

Conference-grade research project targeting **IGARSS / IEEE JSTARS / ISPRS / ACM SIGSPATIAL / CVPR EarthVision**.

**Focus:** Indian metropolitan areas — Mumbai, Delhi NCR, Bangalore, Hyderabad, Chennai, Pune, Ahmedabad.

---

## Key Results

| Model | OA | F1 | mIoU |
|---|---|---|---|
| SVM | 89.2 ± 0.4% | 0.890 ± 0.005 | 0.763 ± 0.009 |
| Random Forest | 88.2 ± 1.7% | 0.875 ± 0.017 | 0.733 ± 0.028 |
| MobileNetV3-Small | 91.5 ± 1.6% | 0.920 ± 0.014 | 0.823 ± 0.028 |
| EfficientNet-B0 | 93.4 ± 2.3% | 0.936 ± 0.022 | 0.857 ± 0.044 |
| Swin-Tiny | 93.6 ± 2.6% | 0.939 ± 0.024 | 0.862 ± 0.046 |
| **ResNet50** | **97.5 ± 0.2%** | **0.976 ± 0.002** | **0.939 ± 0.005** |

All results on real Indian satellite data (Mumbai, Delhi NCR, Bangalore), 3-seed mean ± std (seeds 42, 123, 7).

---

## Architecture: Five-Pillar Framework

- **Base Pipeline** — ResNet50 / EfficientNet-B0 / Swin-Tiny / MobileNetV3 + FPN + progressive fine-tuning
- **Pillar I** — SAR-optical fusion (Sentinel-1 + Sentinel-2)
- **Pillar II** — Self-supervised pretraining (SimCLR on unlabelled EO data)
- **Pillar III** — High-resolution imagery analysis
- **Pillar IV** — Bi-LSTM + Multi-Head Attention urban sprawl forecasting (R² = 0.9564)
- **Pillar V** — Real-time encroachment alert engine (99.33% accuracy, 18.75ms latency)

---

## Data Download

Training data (~7.5 GB) is hosted on Google Drive (excluded from this repo due to size):

**[Download Data from Google Drive](https://drive.google.com/drive/folders/1mmgGRtjHyVpQLMTSQE8xUOFv1gslrpWv?usp=sharing)**

| Folder | Contents | Size |
|---|---|---|
| `models/` | Trained model checkpoints (.pth) | ~439 MB |
| `data/indian_cities_locked/` | Extracted Sentinel-2 patches (.npy) — Mumbai, Delhi NCR, Bangalore | ~4.7 GB |
| `data/levir_cd/` | LEVIR-CD change detection dataset | ~2.35 GB |

After downloading, place the folders under the project root to match this structure:
```
AISD PROJECT/
├── data/
│   ├── indian_cities_locked/
│   └── levir_cd/
└── outputs/
    └── models/
```

Raw GeoTIFF satellite files (Sentinel-2, Sentinel-1 SAR, Landsat) were exported from Google Earth Engine and are available in the same Drive folder under `urban_expansion_india/`.

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.11, PyTorch >= 2.0, NVIDIA GPU recommended (tested on RTX 4070).

---

## Usage

```bash
python main.py                          # Full pipeline
python main.py --base-only              # Base pipeline only
python main.py --pillars-only           # Extended pillars only
python main.py --pillar 4 5             # Specific pillars
python main.py --epochs-override 2      # Quick test
python main.py --data-source real --real-dataset indian_cities
```

---

## Paper Outputs

All conference-ready materials are in `outputs/`:

- `outputs/figures/` — Architecture diagrams, model comparison charts, GradCAM visualizations, LOCO heatmaps
- `outputs/tables/` — LaTeX tables (main benchmark, LOCO, ablation, SOTA comparison, efficiency)
- `outputs/paper/` — Full paper sections, novelty statement, reviewer defense, reproducibility protocol
- `outputs/research_results/` — Authoritative JSON results for all experiments (3-seed)

---

## Hardware

- GPU: NVIDIA GeForce RTX 4070 Laptop
- OS: Windows 11
- Python: 3.11
