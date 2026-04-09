"""
Locked Phase 6 integration runner for the active 3-city pipeline.

Uses the current best benchmark checkpoint and the locked real-data selection:
  - Cities: Mumbai, Delhi_NCR, Bangalore
  - Sentinel-2 years: 2019, 2021, 2023
  - Landsat anchors: 1990, 2000/2005, 2010, 2023
  - Season: pre_monsoon
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.integration_pipeline import (
    build_urban_timeseries,
    classify_all_geotiffs,
    run_alerts_with_predictions,
    run_full_pipeline,
    run_prediction_with_real_data,
)


LOCKED_CITIES = ["Mumbai", "Delhi_NCR", "Bangalore"]


def _configure_env():
    os.environ.setdefault("INTEGRATION_USE_LOCKED_SELECTION", "1")
    os.environ.setdefault("INTEGRATION_CITY_FILTER", ",".join(LOCKED_CITIES))
    os.environ.setdefault("INTEGRATION_YEAR_FILTER", "2019,2021,2023")
    os.environ.setdefault("INTEGRATION_SEASON_FILTER", "pre_monsoon")
    os.environ.setdefault("INTEGRATION_INCLUDE_S2", "1")
    os.environ.setdefault("INTEGRATION_INCLUDE_LANDSAT", "1")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 locked integration runner")
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--timeseries", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--alerts", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--backbone", type=str, default=None)
    args = parser.parse_args()

    _configure_env()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.full:
        run_full_pipeline(device=device)
    elif args.classify:
        classify_all_geotiffs(backbone_name=args.backbone, device=device)
    elif args.timeseries:
        build_urban_timeseries()
    elif args.predict:
        run_prediction_with_real_data(device=device)
    elif args.alerts:
        run_alerts_with_predictions(device=device)
    else:
        parser.error("Specify one of --classify, --timeseries, --predict, --alerts, or --full.")


if __name__ == "__main__":
    main()
