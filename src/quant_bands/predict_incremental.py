"""
Incremental Prediction Update Module
====================================

Automatically detects new bars in features and generates missing predictions.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from .cv import rearrange_quantiles


def update_predictions_incremental(
    features_path: Path,
    models_dir: Path,
    preds_dir: Path,
    targets_T: List[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Update predictions incrementally: generate only for ts0 that don't exist yet.

    This function:
    1. Loads existing predictions from preds_T=*.parquet files
    2. Identifies new ts0 timestamps in features that don't have predictions
    3. Generates predictions for missing ts0 only
    4. Appends new predictions to existing files

    Args:
        features_path: Path to features_4H.parquet
        models_dir: Directory with trained models (models_T*.joblib, calib_T*.json)
        preds_dir: Directory for predictions (preds_T=*.parquet)
        targets_T: List of prediction horizons (default: [42, 48, 54, 60])
        verbose: Show progress logs

    Returns:
        Dict with update statistics:
        - total_ts0_available: Total timestamps in features
        - new_predictions_by_T: Dict {T: count of new predictions}
        - errors: List of error messages
        - duration_seconds: Time taken

    Example:
        >>> from quant_bands.predict_incremental import update_predictions_incremental
        >>> stats = update_predictions_incremental(
        ...     features_path=Path("data/processed/features/features_4H.parquet"),
        ...     models_dir=Path("data/processed/preds"),
        ...     preds_dir=Path("data/processed/preds"),
        ...     targets_T=[42, 48, 54, 60]
        ... )
        >>> print(f"Added {sum(stats['new_predictions_by_T'].values())} new predictions")
    """
    start_time = time.time()

    if targets_T is None:
        targets_T = [42, 48, 54, 60]

    if verbose:
        print("🔄 Starting incremental prediction update...\n")

    # Load features
    if verbose:
        print(f"📂 Loading features from: {features_path}")

    df_features = pd.read_parquet(features_path)
    df_features["ts"] = pd.to_datetime(df_features["ts"], utc=True)
    df_features = df_features.set_index("ts").sort_index()

    # Get all available ts0 in features
    available_ts0 = set(df_features.index)

    if verbose:
        print(f"   Found {len(available_ts0)} timestamps")
        print(f"   Range: {df_features.index.min()} to {df_features.index.max()}\n")

    stats = {
        "total_ts0_available": len(available_ts0),
        "new_predictions_by_T": {},
        "errors": [],
        "files_updated": [],
    }

    # Process each horizon T
    for T in targets_T:
        pred_file = preds_dir / f"preds_T={T}.parquet"

        if verbose:
            print("=" * 60)
            print(f"Processing T={T} ({T*4/24:.1f} days)")
            print("=" * 60)

        # Load existing predictions (if any)
        if pred_file.exists():
            df_existing = pd.read_parquet(pred_file)
            df_existing["ts0"] = pd.to_datetime(df_existing["ts0"], utc=True)
            existing_ts0 = set(df_existing["ts0"])

            if verbose:
                print(f"📊 Existing predictions: {len(existing_ts0)}")
        else:
            df_existing = None
            existing_ts0 = set()
            if verbose:
                print("📊 No existing file (will be created)")

        # Identify missing ts0
        missing_ts0 = sorted(available_ts0 - existing_ts0)

        if not missing_ts0:
            if verbose:
                print("✅ No new predictions needed\n")
            stats["new_predictions_by_T"][T] = 0
            continue

        if verbose:
            print(f"🆕 New predictions needed: {len(missing_ts0)}")
            print(f"   First: {missing_ts0[0]}")
            print(f"   Last:  {missing_ts0[-1]}")

        # Load models
        models_T_path = models_dir / f"models_T{T}.joblib"
        calib_T_path = models_dir / f"calib_T{T}.json"

        if not models_T_path.exists():
            error_msg = f"T={T}: Models not found at {models_T_path}"
            if verbose:
                print(f"⚠️  {error_msg}\n")
            stats["errors"].append(error_msg)
            stats["new_predictions_by_T"][T] = 0
            continue

        if verbose:
            print(f"📦 Loading models from: {models_T_path.name}")

        models_T = joblib.load(models_T_path)

        # Load calibrators
        calibrators_T = {}
        if calib_T_path.exists():
            with open(calib_T_path, "r") as f:
                calibrators_T = json.load(f)
            if verbose and "q_hat_global" in calibrators_T:
                print(f"📐 Conformal calibration: q_hat={calibrators_T['q_hat_global']:.6f}")

        # Get feature names from first model
        first_model = models_T[list(models_T.keys())[0]]
        model_features = first_model.feature_name()

        # Generate predictions for each missing ts0
        new_records = []
        errors_count = 0

        if verbose:
            print("🔮 Generating predictions...")

        for idx, ts0 in enumerate(missing_ts0):
            try:
                # Get features at ts0
                x_t_full = df_features.loc[ts0:ts0]
                S0 = x_t_full["close"].iloc[0]

                # Filter features (exclude non-numeric columns)
                exclude_cols = {"close", "asset", "dt"}
                feature_cols = [c for c in x_t_full.columns if c not in exclude_cols]
                x_t = x_t_full[feature_cols]

                # Select only features used in training
                x_t_model = x_t[model_features]

                # Generate predictions for each quantile
                predictions = {}
                taus = [0.05, 0.25, 0.50, 0.75, 0.95]

                for tau in taus:
                    if tau in models_T:
                        pred = models_T[tau].predict(x_t_model)
                        predictions[tau] = pred

                # Apply conformal calibration
                if calibrators_T and "q_hat_global" in calibrators_T:
                    q_hat = calibrators_T["q_hat_global"]
                    if 0.05 in predictions and 0.95 in predictions:
                        predictions[0.05] = predictions[0.05] - q_hat
                        predictions[0.95] = predictions[0.95] + q_hat

                # Ensure quantile monotonicity
                predictions = rearrange_quantiles(predictions)

                # Calculate RV from bands width
                if 0.05 in predictions and 0.95 in predictions:
                    width = predictions[0.95][0] - predictions[0.05][0]
                    z_95 = 1.645  # 95th percentile of standard normal
                    rvhat_ann = (width / (2 * z_95)) * np.sqrt(252 / (T * 4 / 24))
                else:
                    rvhat_ann = np.nan

                # Calculate forecast target timestamp
                ts_forecast = ts0 + pd.Timedelta(hours=T * 4)
                h_days = T * 4 / 24

                # Create record
                record = {
                    "ts0": ts0,
                    "ts_forecast": ts_forecast,
                    "T": T,
                    "h_days": h_days,
                    "S0": S0,
                    "rvhat_ann": rvhat_ann,
                }

                # Add quantiles (log-return space and absolute prices)
                q_keys = ["q05", "q25", "q50", "q75", "q95"]
                for q_key, tau in zip(q_keys, taus):
                    if tau in predictions:
                        log_return = predictions[tau][0]
                        record[q_key] = log_return

                        # Convert to absolute price
                        absolute_price = S0 * np.exp(log_return)
                        record[f"p_{q_key[1:]}"] = absolute_price

                # Add convenience price levels
                if "q05" in record and "q95" in record:
                    record["p_low"] = record["p_05"]
                    record["p_high"] = record["p_95"]
                if "q50" in record:
                    record["p_med"] = record["p_50"]

                new_records.append(record)

                # Progress indicator (every 10%)
                if verbose and len(missing_ts0) > 10:
                    progress = (idx + 1) / len(missing_ts0) * 100
                    if progress % 10 < (100 / len(missing_ts0)):
                        print(f"   Progress: {progress:.0f}% ({idx+1}/{len(missing_ts0)})")

            except Exception as e:
                errors_count += 1
                error_msg = f"T={T}, ts0={ts0}: {str(e)}"
                stats["errors"].append(error_msg)

                if verbose and errors_count <= 3:  # Show first 3 errors
                    print(f"   ⚠️  Error at ts0={ts0}: {e}")

        if errors_count > 3 and verbose:
            print(f"   ⚠️  ... and {errors_count - 3} more errors")

        if not new_records:
            if verbose:
                print("❌ No predictions generated successfully\n")
            stats["new_predictions_by_T"][T] = 0
            continue

        # Create DataFrame with new predictions
        df_new = pd.DataFrame(new_records)

        # Ensure timezone
        df_new["ts0"] = pd.to_datetime(df_new["ts0"], utc=True)
        df_new["ts_forecast"] = pd.to_datetime(df_new["ts_forecast"], utc=True)

        # Combine with existing predictions
        if df_existing is not None:
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.sort_values("ts0").reset_index(drop=True)
        else:
            df_combined = df_new

        # Save to parquet
        preds_dir.mkdir(parents=True, exist_ok=True)
        df_combined.to_parquet(pred_file, index=False, engine="pyarrow")

        stats["new_predictions_by_T"][T] = len(new_records)
        stats["files_updated"].append(str(pred_file))

        if verbose:
            print(f"✅ Added {len(new_records)} new predictions")
            print(f"   Total now: {len(df_combined)} predictions")
            print(f"   Saved to: {pred_file.name}\n")

    # Calculate duration
    duration = time.time() - start_time
    stats["duration_seconds"] = duration

    # Print summary
    if verbose:
        print("=" * 60)
        print("📊 UPDATE SUMMARY")
        print("=" * 60)
        print(f"Duration: {duration:.1f}s")
        print(f"Total ts0 available: {stats['total_ts0_available']}")
        print("\nNew predictions generated:")
        total_new = 0
        for T, count in stats["new_predictions_by_T"].items():
            print(f"  • T={T}: {count} new predictions")
            total_new += count

        print(f"\nTotal new predictions: {total_new}")

        if stats["errors"]:
            print(f"\n⚠️  {len(stats['errors'])} errors encountered")
            if verbose and len(stats["errors"]) <= 5:
                for error in stats["errors"][:5]:
                    print(f"   - {error}")
        else:
            print("\n✅ Update completed successfully!")
        print("=" * 60)

    return stats


def check_predictions_status(
    features_path: Path, preds_dir: Path, targets_T: List[int] = None
) -> Dict[str, Any]:
    """
    Check status of predictions: how many ts0 are missing predictions.

    Args:
        features_path: Path to features_4H.parquet
        preds_dir: Directory with predictions
        targets_T: List of horizons to check

    Returns:
        Dict with status information
    """
    if targets_T is None:
        targets_T = [42, 48, 54, 60]

    # Load features
    df_features = pd.read_parquet(features_path)
    df_features["ts"] = pd.to_datetime(df_features["ts"], utc=True)
    available_ts0 = set(df_features["ts"])

    status = {
        "total_ts0_available": len(available_ts0),
        "ts0_range": (df_features["ts"].min(), df_features["ts"].max()),
        "status_by_T": {},
    }

    for T in targets_T:
        pred_file = preds_dir / f"preds_T={T}.parquet"

        if pred_file.exists():
            df_pred = pd.read_parquet(pred_file)
            df_pred["ts0"] = pd.to_datetime(df_pred["ts0"], utc=True)
            existing_ts0 = set(df_pred["ts0"])
            missing_ts0 = available_ts0 - existing_ts0

            status["status_by_T"][T] = {
                "file_exists": True,
                "total_predictions": len(existing_ts0),
                "missing_predictions": len(missing_ts0),
                "coverage_pct": len(existing_ts0) / len(available_ts0) * 100,
                "up_to_date": len(missing_ts0) == 0,
            }
        else:
            status["status_by_T"][T] = {
                "file_exists": False,
                "total_predictions": 0,
                "missing_predictions": len(available_ts0),
                "coverage_pct": 0.0,
                "up_to_date": False,
            }

    return status
