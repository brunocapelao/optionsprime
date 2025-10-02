"""
Conformal Prediction Module
==========================

Implements Conformal Quantile Regression (CQR) with Mondrian conditioning
and time-decay weighting for robust prediction intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.neighbors import NearestNeighbors


def compute_nonconformity_scores(
    y_true: np.ndarray,
    q_low: np.ndarray, 
    q_high: np.ndarray
) -> np.ndarray:
    """
    Compute two-sided nonconformity scores for CQR.
    
    e_i = max{q_low(x_i) - y_i, y_i - q_high(x_i), 0}
    """
    lower_miss = q_low - y_true
    upper_miss = y_true - q_high
    zero_baseline = np.zeros_like(y_true)
    
    nonconformity = np.maximum.reduce([lower_miss, upper_miss, zero_baseline])
    return nonconformity


def get_mondrian_buckets(
    features_df: pd.DataFrame,
    bucket_config: Dict[str, Any]
) -> pd.Series:
    """
    Create Mondrian conditioning buckets based on market regime/volatility.
    
    Args:
        features_df: DataFrame with features
        bucket_config: Configuration for bucketing strategy
        
    Returns:
        Series with bucket labels for each sample
    """
    n_samples = len(features_df)
    
    # Initialize with single bucket
    buckets = pd.Series(['global'] * n_samples, index=features_df.index)
    
    # Volatility buckets with robust fallback
    vol_col = bucket_config.get('vol_bucket_from', 'rv_cc_1d_rank_l1')
    
    # Try multiple fallback options for volatility features
    vol_candidates = [vol_col, 'rv_cc_1d_l1', 'vol_l1', 'r_4h_l1']
    vol_feature = None
    
    for candidate in vol_candidates:
        if candidate in features_df.columns:
            vol_feature = candidate
            break
    
    if vol_feature:
        vol_values = features_df[vol_feature]
        # Handle potential NaN values
        vol_values = vol_values.fillna(vol_values.median())
        vol_terciles = vol_values.quantile([1/3, 2/3])
        
        vol_buckets = pd.cut(
            vol_values, 
            bins=[-np.inf, vol_terciles.iloc[0], vol_terciles.iloc[1], np.inf],
            labels=['vol_low', 'vol_mid', 'vol_high']
        ).astype(str)
        print(f"Using {vol_feature} for volatility buckets: {vol_buckets.value_counts().to_dict()}")
    else:
        vol_buckets = pd.Series(['vol_unknown'] * n_samples, index=features_df.index)
        print("Warning: No suitable volatility feature found, using single bucket")
    
    # Regime buckets (if available)
    if bucket_config.get('use_regime', False) and 'regime_label' in features_df.columns:
        regime_buckets = features_df['regime_label'].astype(str)
        # Combine regime and vol
        buckets = regime_buckets + '|' + vol_buckets
    else:
        # Squeeze buckets as regime proxy with robust fallback
        squeeze_col = bucket_config.get('squeeze_bucket_from', 'bb_bw_1m_rank_l1')
        
        # Try multiple fallback options for squeeze features  
        squeeze_candidates = [squeeze_col, 'bb_bw_1m_l1', 'bb_z_1m_l1', 'pband_1m_l1']
        squeeze_feature = None
        
        for candidate in squeeze_candidates:
            if candidate in features_df.columns:
                squeeze_feature = candidate
                break
        
        if squeeze_feature:
            squeeze_values = features_df[squeeze_feature]
            squeeze_values = squeeze_values.fillna(squeeze_values.median())
            squeeze_terciles = squeeze_values.quantile([1/3, 2/3])
            
            squeeze_buckets = pd.cut(
                squeeze_values,
                bins=[-np.inf, squeeze_terciles.iloc[0], squeeze_terciles.iloc[1], np.inf],
                labels=['squeeze_low', 'squeeze_mid', 'squeeze_high']
            ).astype(str)
            
            buckets = squeeze_buckets + '|' + vol_buckets
            print(f"Using {squeeze_feature} for squeeze buckets")
        else:
            buckets = vol_buckets
            print("Warning: No suitable squeeze feature found, using only vol buckets")
    
    # Clean bucket names
    buckets = buckets.fillna('unknown')
    
    return buckets


def compute_weighted_quantile(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    quantile: float = 0.9
) -> float:
    """Compute weighted quantile with midpoint-CDF interpolation.

    - If weights are uniform (or None), matches np.quantile behavior.
    - Uses CDF defined at bin midpoints: F_i = (cum_w - 0.5*w) / sum_w.
    """
    values = np.asarray(values)
    if values.size == 0:
        return float('nan')

    if weights is None:
        return float(np.quantile(values, quantile))

    weights = np.asarray(weights)
    if values.shape[0] != weights.shape[0]:
        raise ValueError("Values and weights must have same length")

    # Treat uniform weights as unweighted for consistency with np.quantile
    if np.allclose(weights, weights[0]):
        return float(np.quantile(values, quantile))

    # Sort by values
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    w = np.clip(w, 0.0, np.inf)
    total = w.sum()
    if total <= 0:
        return float(np.quantile(values, quantile))

    cum = np.cumsum(w)
    # Midpoint CDF at each support point
    cdf = (cum - 0.5 * w) / total

    # Guard monotonicity
    cdf = np.clip(cdf, 0.0, 1.0)

    # Interpolate
    q = float(np.interp(quantile, cdf, v))
    return q


def get_time_decay_weights(
    timestamps: pd.DatetimeIndex,
    lambda_decay: float = 0.01
) -> np.ndarray:
    """Compute time decay weights for conformal calibration."""
    if len(timestamps) == 0:
        return np.array([])
    
    # Days from most recent
    max_time = timestamps.max()
    delta_days = (max_time - timestamps).total_seconds() / (24 * 3600)
    
    weights = np.exp(-lambda_decay * delta_days)
    return weights / np.sum(weights)  # normalize


def fit_conformal_calibrator(
    nonconformity_scores: np.ndarray,
    bucket_labels: pd.Series,
    timestamps: pd.DatetimeIndex,
    alpha: float = 0.1,
    weights_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fit conformal calibrator with Mondrian conditioning and optional weighting.
    
    Args:
        nonconformity_scores: Array of nonconformity scores
        bucket_labels: Mondrian bucket for each score
        timestamps: Timestamps for time weighting
        alpha: Miscoverage level (1-coverage)
        weights_config: Time decay configuration
        
    Returns:
        Calibrator configuration dict
    """
    if len(nonconformity_scores) != len(bucket_labels):
        raise ValueError("Scores and buckets must have same length")
    
    # Prepare weights
    weights = None
    if weights_config and weights_config.get('time_decay_lambda'):
        weights = get_time_decay_weights(
            timestamps, 
            weights_config['time_decay_lambda']
        )
    
    # Global quantile
    global_quantile = compute_weighted_quantile(
        nonconformity_scores, weights, 1 - alpha
    )
    
    # Per-bucket quantiles
    bucket_quantiles = {}
    bucket_stats = {}
    
    unique_buckets = bucket_labels.unique()
    
    for bucket in unique_buckets:
        bucket_mask = bucket_labels == bucket
        bucket_scores = nonconformity_scores[bucket_mask]
        
        if len(bucket_scores) == 0:
            continue
        
        bucket_weights = weights[bucket_mask] if weights is not None else None
        
        bucket_quantile = compute_weighted_quantile(
            bucket_scores, bucket_weights, 1 - alpha
        )
        
        bucket_quantiles[bucket] = float(bucket_quantile)
        bucket_stats[bucket] = {
            'n_samples': int(len(bucket_scores)),
            'mean_score': float(np.mean(bucket_scores)),
            'std_score': float(np.std(bucket_scores)),
            'effective_weight': float(np.sum(bucket_weights)) if bucket_weights is not None else int(len(bucket_scores))
        }
    
    calibrator = {
        'method': 'CQR',
        'alpha': alpha,
        'coverage_target': 1 - alpha,
        'quantile_global': float(global_quantile),
        'quantile_by_bucket': bucket_quantiles,
        'bucket_stats': bucket_stats,
        'n_calibration_samples': int(len(nonconformity_scores)),
        'weights_config': weights_config
    }
    
    return calibrator


def nonconformity_errors(
    models: Dict[float, Any],
    dataset: Dict[str, Any],
    window_days: int = 90,
    q_low: float = 0.05,
    q_high: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Compute nonconformity errors on calibration window.
    
    Args:
        models: Dict of trained quantile models {tau: model}
        dataset: Dataset dict with X, y, timestamps, valid_mask
        window_days: Size of calibration window in days
        q_low, q_high: Quantile levels for interval
        
    Returns:
        Tuple of (nonconformity_scores, valid_indices, timestamps)
    """
    timestamps = dataset['timestamps']
    valid_mask = dataset['valid_mask']
    
    # Get calibration window (recent data)
    end_time = timestamps.max()
    start_time = end_time - pd.Timedelta(days=window_days)
    
    # Find samples in calibration window
    calib_mask = (timestamps >= start_time) & (timestamps <= end_time) & valid_mask
    calib_indices = np.where(calib_mask)[0]
    
    if len(calib_indices) == 0:
        raise ValueError(f"No samples found in calibration window ({window_days} days)")
    
    # Get calibration data
    X_calib = dataset['X'].iloc[calib_indices]
    y_calib = dataset['y'].iloc[calib_indices].values
    calib_timestamps = timestamps[calib_indices]
    
    # Predict quantiles
    if q_low not in models or q_high not in models:
        raise ValueError(f"Models must contain quantiles {q_low} and {q_high}")
    
    pred_low = models[q_low].predict(X_calib)
    pred_high = models[q_high].predict(X_calib)
    
    # Compute nonconformity scores
    nonconformity_scores = compute_nonconformity_scores(y_calib, pred_low, pred_high)
    
    print(f"Calibration: {len(calib_indices)} samples, "
          f"mean nonconformity: {np.mean(nonconformity_scores):.4f}")
    
    return nonconformity_scores, calib_indices, calib_timestamps


def fit_conformal_mondrian(
    models_dict: Dict[str, Any],
    X_calib: pd.DataFrame, 
    y_calib: pd.Series,
    alpha: float = 0.1,
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
    bucket_config: Optional[Dict[str, Any]] = None,
    time_decay_lambda: float = 0.01,
    min_bucket_size: int = 300
) -> Dict[str, Any]:
    """
    Fit conformal prediction using Mondrian CQR with time-decay weighting.
    
    Args:
        models_dict: Dictionary of trained quantile models
        X_calib: Calibration features  
        y_calib: Calibration targets
        alpha: Miscoverage level (0.1 for 90% intervals)
        quantiles: List of quantile levels
        bucket_config: Configuration for Mondrian buckets
        time_decay_lambda: Time decay parameter for weighting
        min_bucket_size: Minimum samples for bucket-specific calibration
        
    Returns:
        Dictionary with conformal correction factors (global + by bucket)
    """
    n_calib = len(X_calib)
    
    # Get predictions for low and high quantiles (typically 0.05, 0.95)
    q_low_key = f"q_{quantiles[0]:.2f}"
    q_high_key = f"q_{quantiles[-1]:.2f}"
    
    if q_low_key not in models_dict or q_high_key not in models_dict:
        raise ValueError(f"Models for {q_low_key} and {q_high_key} not found")
        
    # Predict intervals
    q_low_pred = models_dict[q_low_key].predict(X_calib)
    q_high_pred = models_dict[q_high_key].predict(X_calib)
    
    # Compute nonconformity scores
    nonconf_scores = compute_nonconformity_scores(y_calib.values, q_low_pred, q_high_pred)
    
    # Time-decay weights (more recent = higher weight)
    if hasattr(X_calib, 'index') and hasattr(X_calib.index, 'to_series'):
        time_idx = np.arange(len(X_calib))
        time_weights = np.exp(time_decay_lambda * (time_idx - time_idx[-1]))
    else:
        time_weights = np.ones(n_calib)
    
    # Global conformal quantile
    corrected_alpha = (1.0 - alpha) * (n_calib + 1) / n_calib
    q_hat_global = compute_weighted_quantile(nonconf_scores, time_weights, corrected_alpha)
    
    # Mondrian buckets
    bucket_corrections = {}
    bucket_keys = ['global']
    
    if bucket_config:
        try:
            buckets = get_mondrian_buckets(X_calib, bucket_config)
            bucket_keys = buckets.unique().tolist()
            
            for bucket in bucket_keys:
                if bucket == 'global':
                    continue
                    
                bucket_mask = buckets == bucket
                bucket_scores = nonconf_scores[bucket_mask]
                bucket_weights = time_weights[bucket_mask]
                
                if len(bucket_scores) >= min_bucket_size:
                    # Bucket-specific quantile
                    q_hat_bucket = compute_weighted_quantile(
                        bucket_scores, bucket_weights, corrected_alpha
                    )
                    
                    # Shrinkage towards global
                    n_bucket = len(bucket_scores)
                    shrink_weight = n_bucket / (n_bucket + min_bucket_size)
                    q_hat_shrunk = (shrink_weight * q_hat_bucket + 
                                  (1 - shrink_weight) * q_hat_global)
                    
                    bucket_corrections[bucket] = {
                        'q_hat': q_hat_shrunk,
                        'q_hat_raw': q_hat_bucket,
                        'n_samples': n_bucket,
                        'shrink_weight': shrink_weight
                    }
                else:
                    # Too few samples, use global with inflation
                    inflation_factor = 1.1  # 10% inflation for small buckets
                    bucket_corrections[bucket] = {
                        'q_hat': q_hat_global * inflation_factor,
                        'q_hat_raw': q_hat_global,
                        'n_samples': len(bucket_scores),
                        'shrink_weight': 0.0,
                        'inflation_applied': True
                    }
        except Exception as e:
            print(f"Warning: Mondrian bucketing failed: {e}, using global calibration")
            bucket_keys = ['global']
    
    corrections = {
        'q_hat_global': q_hat_global,
        'bucket_corrections': bucket_corrections,
        'bucket_keys': bucket_keys,
        'alpha': alpha,
        'n_calib': n_calib,
        'corrected_alpha': corrected_alpha,
        'low_quantile': quantiles[0],
        'high_quantile': quantiles[-1],
        'time_decay_lambda': time_decay_lambda,
        'min_bucket_size': min_bucket_size
    }
    
    return corrections


def fit_conformal(
    models_dict: Dict[str, Any],
    X_calib: pd.DataFrame, 
    y_calib: pd.Series,
    alpha: float = 0.1,
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
) -> Dict[str, Any]:
    """
    Legacy interface for simple conformal prediction (backwards compatibility).
    """
    return fit_conformal_mondrian(
        models_dict, X_calib, y_calib, alpha, quantiles,
        bucket_config=None, time_decay_lambda=0.0, min_bucket_size=300
    )


def apply_conformal_adjustment(
    pred_low: np.ndarray,
    pred_high: np.ndarray,
    features_df: pd.DataFrame,
    calibrator: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply conformal adjustment to base quantile predictions.
    
    Args:
        pred_low, pred_high: Base quantile predictions
        features_df: Features for bucket assignment
        calibrator: Fitted calibrator
        
    Returns:
        Tuple of adjusted (q_low, q_high) predictions
    """
    # Get bucket assignments
    mondrian_config = calibrator.get('mondrian_config', {})
    bucket_labels = get_mondrian_buckets(features_df, mondrian_config)
    
    # Initialize adjustments
    adjustments = np.full(len(pred_low), calibrator['quantile_global'])
    
    # Apply bucket-specific adjustments
    bucket_quantiles = calibrator.get('quantile_by_bucket', {})
    
    for bucket, quantile_val in bucket_quantiles.items():
        bucket_mask = bucket_labels == bucket
        if bucket_mask.any():
            adjustments[bucket_mask] = quantile_val
    
    # Adjust predictions
    adjusted_low = pred_low - adjustments
    adjusted_high = pred_high + adjustments
    
    return adjusted_low, adjusted_high


def apply_conformal_correction(
    y_pred_dict: Dict[str, np.ndarray],
    corrections: Dict[str, Any],
    X_test: Optional[pd.DataFrame] = None,
    bucket_config: Optional[Dict[str, Any]] = None
) -> Dict[str, np.ndarray]:
    """
    Apply conformal corrections to predictions.
    
    Args:
        y_pred_dict: Dictionary of quantile predictions
        corrections: Conformal correction factors from fit_conformal_mondrian
        X_test: Test features (needed for Mondrian buckets)
        bucket_config: Bucket configuration (for Mondrian)
        
    Returns:
        Dictionary with corrected predictions
    """
    corrected_preds = y_pred_dict.copy()
    
    low_q = corrections['low_quantile']
    high_q = corrections['high_quantile']
    q_low_key = f"q_{low_q:.2f}"
    q_high_key = f"q_{high_q:.2f}"
    
    q_low_orig = y_pred_dict[q_low_key]
    q_high_orig = y_pred_dict[q_high_key]
    
    # Determine which correction to use
    if X_test is not None and bucket_config and corrections['bucket_corrections']:
        # Mondrian correction
        try:
            buckets = get_mondrian_buckets(X_test, bucket_config)
            q_low_corrected = np.zeros_like(q_low_orig)
            q_high_corrected = np.zeros_like(q_high_orig)
            
            for bucket in buckets.unique():
                bucket_mask = buckets == bucket
                
                if bucket in corrections['bucket_corrections']:
                    q_hat = corrections['bucket_corrections'][bucket]['q_hat']
                else:
                    q_hat = corrections['q_hat_global']
                
                q_low_corrected[bucket_mask] = q_low_orig[bucket_mask] - q_hat
                q_high_corrected[bucket_mask] = q_high_orig[bucket_mask] + q_hat
            
            corrected_preds[q_low_key] = q_low_corrected
            corrected_preds[q_high_key] = q_high_corrected
            
        except Exception as e:
            print(f"Warning: Mondrian correction failed: {e}, using global")
            # Fall back to global correction
            q_hat = corrections['q_hat_global']
            corrected_preds[q_low_key] = q_low_orig - q_hat
            corrected_preds[q_high_key] = q_high_orig + q_hat
    else:
        # Global correction
        q_hat = corrections['q_hat_global']
        corrected_preds[q_low_key] = q_low_orig - q_hat
        corrected_preds[q_high_key] = q_high_orig + q_hat
    
    return corrected_preds


def compute_post_conformal_metrics(
    y_true: np.ndarray,
    y_pred_corrected: Dict[str, np.ndarray],
    X_test: Optional[pd.DataFrame] = None,
    bucket_config: Optional[Dict[str, Any]] = None,
    quantiles: List[float] = [0.05, 0.95]
) -> Dict[str, Any]:
    """
    Compute post-conformal metrics including coverage by bucket.
    
    Returns:
        Dictionary with coverage metrics (global + by bucket)
    """
    q_low_key = f"q_{quantiles[0]:.2f}"
    q_high_key = f"q_{quantiles[1]:.2f}"
    
    q_low = y_pred_corrected[q_low_key]
    q_high = y_pred_corrected[q_high_key]
    
    # Global coverage
    in_interval = (y_true >= q_low) & (y_true <= q_high)
    coverage_global = in_interval.mean()
    
    metrics = {
        'coverage_post_global': coverage_global,
        'width_post_global': (q_high - q_low).mean(),
        'crossing_rate_post': ((q_low > q_high).sum() / len(q_low)).item(),
        'target_coverage': 0.9  # Fixed: Always 90% coverage target regardless of quantiles used
    }
    
    # Coverage by bucket (Mondrian)
    if X_test is not None and bucket_config:
        try:
            buckets = get_mondrian_buckets(X_test, bucket_config)
            coverage_by_bucket = {}
            width_by_bucket = {}
            
            for bucket in buckets.unique():
                bucket_mask = buckets == bucket
                if bucket_mask.sum() > 0:  # Check bucket has samples
                    bucket_coverage = in_interval[bucket_mask].mean()
                    bucket_width = (q_high[bucket_mask] - q_low[bucket_mask]).mean()
                    
                    coverage_by_bucket[bucket] = bucket_coverage
                    width_by_bucket[bucket] = bucket_width
            
            metrics['coverage_post_by_bucket'] = coverage_by_bucket
            metrics['width_post_by_bucket'] = width_by_bucket
            metrics['buckets'] = buckets.unique().tolist()
            
        except Exception as e:
            print(f"Warning: Bucket metrics computation failed: {e}")
    
    return metrics
