"""FastAPI backend for the Urban Expansion Monitoring demo.

Serves the project's pre-computed research results as clean JSON APIs, exposes
live patch classification + Grad-CAM (CPU, offline), and statically serves the
paper figures and bundled demo patches.

Run:  uvicorn main:app --reload --port 8000   (from webapp/backend/)
"""
import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app_paths as P
import data_access as DA
import inference
import paper_content
import reviewer_content
import novelty_content
import limitations_content
import presenter_content
import figures_content
import tour_content

app = FastAPI(title="Urban Expansion Monitoring API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- static assets ---------------------------------------------------------
if P.FIGURES.exists():
    app.mount("/figures", StaticFiles(directory=str(P.FIGURES)), name="figures")
if P.SAMPLES.exists():
    app.mount("/samples", StaticFiles(directory=str(P.SAMPLES)), name="samples")


# ---- meta ------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "models_available": inference.available_models()}


@app.get("/api/overview")
def overview():
    return DA.get_overview()


# ---- results / tables ------------------------------------------------------
@app.get("/api/models")
def models():
    return DA.get_models()


@app.get("/api/loco")
def loco():
    return DA.get_loco()


@app.get("/api/ablation")
def ablation():
    return DA.get_ablation()


@app.get("/api/pillars")
def pillars():
    return DA.get_pillars()


@app.get("/api/paper")
def paper():
    return DA.get_paper()


@app.get("/api/explain")
def explain():
    """Paper broken into verbatim-quote + plain-English blocks, plus glossary."""
    return {
        "title": paper_content.PAPER_TITLE,
        "venue": paper_content.VENUE,
        "authors": paper_content.AUTHORS,
        "sections": paper_content.SECTIONS,
        "glossary": paper_content.GLOSSARY,
    }


@app.get("/api/reviewer")
def reviewer():
    """Anticipated reviewer questions with layman answers + defense points."""
    return {"qa": reviewer_content.QA, "venue_tips": reviewer_content.VENUE_TIPS}


@app.get("/api/novelty")
def novelty():
    """The contribution / novelty story, cleaned for presentation."""
    return {
        "pitch": novelty_content.PITCH,
        "pillars": novelty_content.PILLARS,
        "positioning": novelty_content.POSITIONING,
    }


@app.get("/api/limitations")
def limitations():
    """Limitations paired with mitigations and future work."""
    return {"limitations": limitations_content.LIMITATIONS}


@app.get("/api/presenter")
def presenter():
    """In-app talk script for the demo."""
    return {
        "total_minutes": presenter_content.TOTAL_MINUTES,
        "beats": presenter_content.BEATS,
        "tips": presenter_content.TIPS,
    }


@app.get("/api/figures")
def figures():
    """Curated, captioned list of real paper figures (served from /figures)."""
    return {"figures": figures_content.list_figures()}


@app.get("/api/tour")
def tour():
    """Plain-language 'Start Here' walkthrough for first-time / non-expert viewers."""
    return {"slides": tour_content.TOUR}


# ---- pipeline data ---------------------------------------------------------
@app.get("/api/timeseries")
def timeseries():
    return DA.get_timeseries()


@app.get("/api/forecasts")
def forecasts():
    return DA.get_forecasts()


@app.get("/api/alerts")
def alerts():
    return DA.get_alerts()


# ---- live inference --------------------------------------------------------
def _list_samples():
    fp = P.SAMPLES / "samples.json"
    if not fp.exists():
        return []
    return json.loads(fp.read_text(encoding="utf-8"))


@app.get("/api/samples")
def samples():
    """Curated demo patches available for live classification."""
    return _list_samples()


def _load_sample_patch(sample_id):
    fp = P.SAMPLES / f"{sample_id}.npy"
    if not fp.exists():
        raise HTTPException(404, f"sample '{sample_id}' not found")
    return np.load(fp).astype(np.float32)


@app.get("/api/classify/{sample_id}")
def classify_sample(sample_id: str, backbone: str = Query("resnet50")):
    patch = _load_sample_patch(sample_id)
    result = inference.classify(patch, backbone=backbone)
    meta = next((s for s in _list_samples() if s["id"] == sample_id), {})
    result["true_label"] = meta.get("true_label")
    result["city"] = meta.get("city")
    return result


@app.get("/api/gradcam/{sample_id}")
def gradcam_sample(sample_id: str, backbone: str = Query("resnet50")):
    patch = _load_sample_patch(sample_id)
    png, info = inference.gradcam_png(patch, backbone=backbone)
    if png is None:
        raise HTTPException(400, info.get("reason", "Grad-CAM unavailable"))
    return Response(content=png, media_type="image/png")


@app.post("/api/classify-upload")
async def classify_upload(file: UploadFile = File(...), backbone: str = Query("resnet50")):
    """Classify an uploaded 6-channel patch saved as .npy (shape (6,256,256))."""
    raw = await file.read()
    import io
    try:
        patch = np.load(io.BytesIO(raw)).astype(np.float32)
    except Exception:
        raise HTTPException(400, "Could not read .npy file (expected array of shape (6,256,256))")
    if patch.ndim != 3 or patch.shape[0] < 3:
        raise HTTPException(400, f"Unexpected patch shape {patch.shape}; expected (C>=3, H, W)")
    result = inference.classify(patch, backbone=backbone)
    return result
