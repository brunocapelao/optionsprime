"""
Quantile Bands Training Module
============================

Production-ready module for training quantile regression models with conformal prediction.
Implements anti-leakage CPCV, Mondrian calibration, and deterministic artifacts.

Usage:
    python -m quant_bands.train --config configs/02a.yaml
"""

__version__ = "1.0.1"
__author__ = "Algo Trading Team"

from .conformal import fit_conformal, nonconformity_errors
from .cv import build_xy_datasets, evaluate_cpcv
from .predict import daily_inference_pipeline, load_and_predict, save_predictions_parquet
from .predict_incremental import check_predictions_status, update_predictions_incremental
from .utils import load_cpcv, load_features, set_seeds

__all__ = [
    "set_seeds",
    "load_features",
    "load_cpcv",
    "build_xy_datasets",
    "evaluate_cpcv",
    "fit_conformal",
    "nonconformity_errors",
    "daily_inference_pipeline",
    "load_and_predict",
    "save_predictions_parquet",
    "update_predictions_incremental",
    "check_predictions_status",
]
