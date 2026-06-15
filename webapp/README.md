# The Living Map — Demo Web App

Interactive demo for the IEEE CHANDICON 2026 paper
**"Urban Expansion Monitoring via Transfer Learning on Historical Satellite Imagery."**

A two-tier app that lets you *show* the full research pipeline live during the talk:

```
React + Vite frontend  ──HTTP──►  FastAPI backend  ──►  trained models (.pth) + result JSONs
   (charts, map, UI)                (live inference)        in ../outputs/
```

## What it shows

| Page | Backed by |
|------|-----------|
| **Overview** | The 5-stage pipeline, headline numbers (`/api/overview`) |
| **Live Classification** | Real CPU inference on bundled Sentinel-2 patches + Grad-CAM (`resnet50_best.pth` …) |
| **Urban Growth** | Per-city built-up time series 1990→2023 (`outputs/integration/urban_timeseries.json`) |
| **Sprawl Forecast** | Pillar IV LSTM forecasts with 95% CI bands (`outputs/pillar4_forecasts.json`) |
| **Encroachment Alerts** | Pillar V alerts on an offline geo-map (`outputs/alerts/alerts.json`) |
| **Globe** | The alerts on an interactive 3D Earth (react-globe.gl, offline — country outlines bundled in `frontend/public/countries-110m.geojson`) |
| **Benchmarks** | 3-seed leaderboard, LOCO, ablation, efficiency (`outputs/research_results/*.json`) |
| **Figures** | Gallery of the ~15 real paper figures with captions + click-to-zoom (`backend/figures_content.py`) |
| **Novelty** | The contribution story: pitch + 8 novelty pillars (with "no prior work" + evidence) + SOTA positioning (`backend/novelty_content.py`) |
| **Paper, Explained** | Every paper line paired with a plain-English translation + searchable glossary (`backend/paper_content.py`) — your live teleprompter |
| **Reviewer Q&A** | 17 anticipated questions with layman answers, defense points, "if-pressed" one-liners + venue tips (`backend/reviewer_content.py`) |

## Built for a conference room (offline-first)

- **Live inference runs on CPU** — no GPU needed at the venue.
- **No internet needed at runtime**: fonts are self-hosted (`@fontsource`), the alerts map is a custom
  SVG (no tile server), and a curated set of sample patches is bundled into the backend.
- The app reads **pre-computed results + checkpoints already in `../outputs/`** — it never touches the
  6.9 GB of raw GeoTIFFs on `G:\`.

---

## First-time setup (only needed on a fresh clone)

Run these once. The backend must use the Python that has the project's deps
(torch, timm) — on this machine that's the Windows Store Python 3.11, **not**
Anaconda base, so `conda deactivate` first.

```powershell
# backend deps + demo patches
cd "C:\Users\KAUSTUBH\Desktop\AISD PROJECT\webapp\backend"
conda deactivate
python -m pip install -r requirements.txt
python bundle_samples.py

# frontend deps
cd "..\frontend"
npm install
```

## Run the demo

```powershell
cd "C:\Users\KAUSTUBH\Desktop\AISD PROJECT\webapp"
./run.ps1
```

`run.ps1` opens the backend and frontend each in their own window (using the
correct torch-Python, not Anaconda), waits for the frontend, then opens the
browser. The backend takes ~15-20 s to start (importing torch) — wait for its
window to say `Uvicorn running on http://127.0.0.1:8000`.

To stop both servers / free the ports before re-running:

```powershell
./stop.ps1     # kills whatever is on ports 8000 and 5173
```

- Backend → http://127.0.0.1:8000  (API docs at `/docs`)
- Frontend → http://127.0.0.1:5173  ← **open this in your browser**

Or run the two tiers manually in separate terminals:

```powershell
# terminal 1 — backend
cd webapp\backend
python -m uvicorn main:app --port 8000

# terminal 2 — frontend
cd webapp\frontend
npm run dev
```

## Re-bundle demo patches (optional)

The 18 bundled patches live in `backend/sample_patches/`. To regenerate from the real dataset:

```powershell
cd webapp\backend
python bundle_samples.py
```

## Notes

- The backend serves the project's figures at `/figures/...` and demo patches at `/samples/...`.
- CORS is open (`*`) — demo only, fine for localhost.
- If port 8000 is taken, change it in `run.ps1` and set `VITE_API_BASE` for the frontend.
