"""Read the project's pre-computed result JSONs and reshape them into clean,
front-end-friendly payloads.

Every loader is defensive: if a file is missing or malformed it returns an empty
structure instead of raising, so the demo never crashes mid-presentation.
"""
import json
from functools import lru_cache

import app_paths as P

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _load(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


# Human-friendly model display names + roles for the demo narrative.
MODEL_META = {
    "resnet50":           {"label": "ResNet50",            "family": "CNN",         "role": "Best overall"},
    "swin_tiny":          {"label": "Swin-Tiny",           "family": "Transformer", "role": "Best cross-city"},
    "efficientnet_b0":    {"label": "EfficientNet-B0",     "family": "CNN",         "role": "Efficient / ablation"},
    "mobilenet_v3_small": {"label": "MobileNetV3-Small",   "family": "CNN",         "role": "Edge / real-time"},
    "SVM":                {"label": "SVM",                 "family": "Classical",   "role": "Baseline"},
    "SVM_improved":       {"label": "SVM (optimized)",     "family": "Classical",   "role": "Strong baseline"},
    "RandomForest":       {"label": "Random Forest",       "family": "Classical",   "role": "Baseline"},
    "RF":                 {"label": "Random Forest",       "family": "Classical",   "role": "Baseline"},
}

CLASS_NAMES = ["Urban", "Non-Urban", "Transition"]


# ----------------------------------------------------------------------------
# 1. Model leaderboard  (Table 1 + efficiency)
# ----------------------------------------------------------------------------

def get_models():
    bench = _load(P.RESULTS / "table1_authoritative.json", {})
    eff = _load(P.RESULTS / "efficiency_benchmark.json", {})
    rows = []
    for key, m in bench.items():
        meta = MODEL_META.get(key, {"label": key, "family": "?", "role": ""})
        e = eff.get(key, {})
        rows.append({
            "key": key,
            "label": meta["label"],
            "family": meta["family"],
            "role": meta["role"],
            "oa_mean": m.get("oa_mean"),
            "oa_std": m.get("oa_std"),
            "f1_mean": m.get("f1_mean"),
            "f1_std": m.get("f1_std"),
            "miou_mean": m.get("miou_mean"),
            "miou_std": m.get("miou_std"),
            "raw_oa": m.get("raw_oa"),
            "params_m": e.get("total_params_m"),
            "latency_ms": e.get("latency_ms"),
            "throughput_ps": e.get("throughput_ps"),
        })
    rows.sort(key=lambda r: (r["oa_mean"] is not None, r["oa_mean"] or 0), reverse=True)
    return rows


# ----------------------------------------------------------------------------
# 2. Cross-city LOCO + ablation
# ----------------------------------------------------------------------------

def _metric_rows(raw):
    out = []
    for key, m in raw.items():
        meta = MODEL_META.get(key, {"label": key})
        out.append({
            "key": key,
            "label": meta.get("label", key),
            "oa_mean": m.get("oa_mean"),
            "oa_std": m.get("oa_std"),
            "f1_mean": m.get("f1_mean"),
            "f1_std": m.get("f1_std"),
            "miou_mean": m.get("miou_mean"),
            "miou_std": m.get("miou_std"),
            "raw_oa": m.get("raw_oa"),
        })
    return out


def get_loco():
    return _metric_rows(_load(P.RESULTS / "table2_loco_authoritative.json", {}))


def get_ablation():
    raw = _load(P.RESULTS / "table3_ablation_authoritative.json", {})
    labels = {"full": "Full method", "no_fpn": "No FPN", "ce_only": "CE-only loss"}
    rows = _metric_rows(raw)
    for r in rows:
        r["label"] = labels.get(r["key"], r["key"])
    return rows


# ----------------------------------------------------------------------------
# 3. Urban growth time series
# ----------------------------------------------------------------------------

def get_timeseries():
    raw = _load(P.INTEGRATION / "urban_timeseries.json", {})
    out = {}
    for city, years in raw.items():
        series = []
        for year, v in sorted(years.items(), key=lambda kv: int(kv[0])):
            series.append({
                "year": int(year),
                "urban_area_km2": round(v.get("urban_area_km2", 0), 1),
                "urban_fraction": round(v.get("urban_fraction", 0), 4),
                "total_area_km2": round(v.get("total_area_km2", 0), 1),
                "source": v.get("selected_source"),
            })
        out[city] = series
    return out


# ----------------------------------------------------------------------------
# 4. Pillar IV forecasts (with uncertainty bands)
# ----------------------------------------------------------------------------

def get_forecasts():
    raw = _load(P.OUTPUTS / "pillar4_forecasts.json", {})
    out = {}
    for city, v in raw.items():
        if not isinstance(v, dict) or "years" not in v:
            continue
        years = v.get("years", [])
        out[city] = [
            {
                "year": int(years[i]),
                "mean": round(v["mean_trajectory"][i], 1),
                "ci_lower": round(v["ci_lower"][i], 1),
                "ci_upper": round(v["ci_upper"][i], 1),
            }
            for i in range(len(years))
        ]
    return out


# ----------------------------------------------------------------------------
# 5. Pillar V alerts
# ----------------------------------------------------------------------------

def get_alerts():
    alerts = _load(P.ALERTS / "alerts.json", [])
    report = _load(P.ALERTS / "alert_report.json", {})
    dashboard = _load(P.ALERTS / "dashboard.json", {})
    return {"alerts": alerts, "report": report, "dashboard": dashboard}


# ----------------------------------------------------------------------------
# 6. Pillars I + II comparisons
# ----------------------------------------------------------------------------

def get_pillars():
    return {
        "pillar1_fusion": _load(P.RESULTS / "pillar1_indian_sar_fusion.json", {}),
        "pillar1_optical": _load(P.RESULTS / "pillar1_optical_only_baseline.json", {}),
        "pillar2_simclr": _load(P.RESULTS / "pillar2_indian_simclr.json", {}),
    }


# ----------------------------------------------------------------------------
# 7. Paper material (SOTA / novelty / reproducibility)
# ----------------------------------------------------------------------------

def get_paper():
    return {
        "sota": _load(P.PAPER / "sota_comparison.json", {}),
        "novelty": _load(P.PAPER / "novelty_statement.json", {}),
        "reproducibility": _load(P.PAPER / "reproducibility.json", {}),
        "statistical_tests": _load(P.RESULTS / "statistical_tests.json", {}),
    }


# ----------------------------------------------------------------------------
# 8. Overview / pipeline narrative (drives the landing page)
# ----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_overview():
    models = get_models()
    best = next((m for m in models if m["key"] == "resnet50"), None)
    ts = get_timeseries()
    alerts = get_alerts()
    return {
        "title": "Urban Expansion Monitoring via Transfer Learning",
        "subtitle": "Historical satellite imagery of Indian metropolitan regions",
        "venue": "Accepted — IEEE CHANDICON 2026",
        "cities": list(ts.keys()),
        "classes": CLASS_NAMES,
        "headline": {
            "best_model": best["label"] if best else "ResNet50",
            "best_oa": best["oa_mean"] if best else 0.975,
            "best_oa_std": best["oa_std"] if best else 0.002,
            "n_models": len([m for m in models if m["family"] != "Classical"]),
            "n_cities": len(ts),
            "n_alerts": alerts.get("report", {}).get("total_alerts", 0),
            "seeds": [42, 123, 7],
        },
        "pipeline": [
            {
                "id": "data",
                "name": "Satellite Imagery",
                "desc": "Sentinel-2, Landsat & Sentinel-1 SAR composites from Google Earth Engine, 1990-2023, cloud-masked & seasonally composited.",
                "tech": ["Sentinel-2 (10m)", "Landsat 5/7/8/9 (30m)", "ESA WorldCover labels"],
            },
            {
                "id": "classify",
                "name": "Patch Classification",
                "desc": "Each 256x256 patch is classified Urban / Non-Urban / Transition by a transfer-learned backbone with a 3-level FPN and progressive fine-tuning.",
                "tech": ["ResNet50", "Swin-Tiny", "EfficientNet-B0", "MobileNetV3"],
            },
            {
                "id": "timeseries",
                "name": "Urban Time Series",
                "desc": "Per-city urban area is aggregated across years to produce a longitudinal expansion curve.",
                "tech": ["Per-city km^2", "1990 -> 2023"],
            },
            {
                "id": "forecast",
                "name": "Pillar IV - Forecasting",
                "desc": "A Bi-LSTM with multi-head temporal attention forecasts sprawl to 2035 with MC-Dropout uncertainty bands.",
                "tech": ["Bi-LSTM + Attention", "MC Dropout 95% CI"],
            },
            {
                "id": "alerts",
                "name": "Pillar V - Encroachment Alerts",
                "desc": "Predicted expansion is compared against India regulatory zones (CRZ, forest, wetland) to flag and route encroachment alerts.",
                "tech": ["3-head detector", "CRZ / Forest / Wetland zones"],
            },
        ],
    }
