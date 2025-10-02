"""
Daily Prediction Module
======================

Implements daily inference pipeline with predictions output in parquet format
as per 02a specification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import joblib
import json

from .utils import write_json_atomic, get_env_versions, compute_content_hash


def save_predictions_parquet(
    predictions_dict: Dict[str, np.ndarray],
    ts0: pd.Timestamp,
    S0: float,
    T: int,
    h_days: float,
    rvhat_ann: float,
    har_rv_ann_baseline: Optional[float] = None,
    output_path: Path = None
) -> Path:
    """
    Save daily predictions in parquet format according to 02a specification.
    
    The output parquet includes:
    - ts0: Reference timestamp (when prediction was made)
    - ts_forecast: Target forecast timestamp (ts0 + T × 4H)
    - T: Horizon in 4H bars
    - h_days: Horizon in days
    - S0: Reference price
    - Quantiles in log-return space: q05, q25, q50, q75, q95
    - Quantiles in absolute prices: p_05, p_25, p_50, p_75, p_95
    - Convenience fields: p_low, p_high, p_med
    - rvhat_ann: Annualized RV from bands width
    
    Args:
        predictions_dict: Dict with quantile predictions {tau: array}
        ts0: Reference timestamp (bar close)
        S0: Reference price
        T: Horizon in 4H bars
        h_days: Horizon in days
        rvhat_ann: Annualized RV from bands width
        har_rv_ann_baseline: Optional HAR-RV baseline
        output_path: Output file path
        
    Returns:
        Path to saved file
    """
    # Convert log-return quantiles to absolute prices
    q_keys = ['q05', 'q25', 'q50', 'q75', 'q95']
    tau_keys = [0.05, 0.25, 0.50, 0.75, 0.95]
    
    # Calculate forecast target date (ts0 + T barras × 4H)
    ts_forecast = ts0 + pd.Timedelta(hours=T * 4)
    
    # Initialize record
    record = {
        'ts0': ts0,
        'ts_forecast': ts_forecast,
        'T': T,
        'h_days': h_days,
        'S0': S0,
        'rvhat_ann': rvhat_ann
    }
    
    # Add quantiles (log-return space and absolute prices)
    for q_key, tau in zip(q_keys, tau_keys):
        if tau in predictions_dict:
            log_return = predictions_dict[tau][0] if len(predictions_dict[tau]) > 0 else 0.0
            record[q_key] = log_return
            
            # Convert to absolute price: K = S0 * exp(log_return)
            absolute_price = S0 * np.exp(log_return)
            record[f'p_{q_key[1:]}'] = absolute_price  # p_05, p_25, etc.
    
    # Add convenience price levels
    if 'q05' in record and 'q95' in record:
        record['p_low'] = record['p_05']
        record['p_high'] = record['p_95']
    if 'q50' in record:
        record['p_med'] = record['p_50']
    
    # Add HAR-RV baseline if available
    if har_rv_ann_baseline is not None:
        record['har_rv_ann_T'] = har_rv_ann_baseline
        # Optional blend
        record['rvhat_blend_ann'] = 0.7 * rvhat_ann + 0.3 * har_rv_ann_baseline
    
    # Create DataFrame
    df = pd.DataFrame([record])
    
    # Ensure timezone for both timestamps (use tz_convert if already timezone-aware)
    df['ts0'] = pd.to_datetime(df['ts0'])
    if df['ts0'].dt.tz is None:
        df['ts0'] = df['ts0'].dt.tz_localize('UTC')
    else:
        df['ts0'] = df['ts0'].dt.tz_convert('UTC')
    
    df['ts_forecast'] = pd.to_datetime(df['ts_forecast'])
    if df['ts_forecast'].dt.tz is None:
        df['ts_forecast'] = df['ts_forecast'].dt.tz_localize('UTC')
    else:
        df['ts_forecast'] = df['ts_forecast'].dt.tz_convert('UTC')
    
    # Define output path if not provided
    if output_path is None:
        output_path = Path(f"data/processed/preds/preds_T={T}.parquet")
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with proper dtypes
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    return output_path


def daily_inference_pipeline(
    ts0: pd.Timestamp,
    features_df: pd.DataFrame,
    models_dir: Path,
    targets_T: List[int] = [42, 48, 54, 60],
    output_dir: Path = None
) -> Dict[str, Any]:
    """
    Run daily inference pipeline generating predictions for all T horizons.
    
    Args:
        ts0: Reference timestamp for inference
        features_df: DataFrame with features up to ts0
        models_dir: Directory containing trained models
        targets_T: List of horizons to predict
        output_dir: Output directory for predictions
        
    Returns:
        Dict with inference results and saved file paths
    """
    if output_dir is None:
        output_dir = Path("data/processed/preds")
    
    # Load models and calibrators
    models_path = models_dir / "quantiles_model.joblib"
    if not models_path.exists():
        raise FileNotFoundError(f"Models not found at {models_path}")
    
    models_data = joblib.load(models_path)
    
    # Extract features at ts0 (only up to bar close)
    ts0_mask = features_df.index <= ts0
    if not ts0_mask.any():
        raise ValueError(f"No features found up to ts0={ts0}")
    
    latest_idx = features_df.index[ts0_mask][-1]
    x_t_full = features_df.loc[latest_idx:latest_idx]  # Single row
    
    # Get reference price
    S0 = x_t_full['close'].iloc[0]
    
    # Filter features using same logic as training, plus exclude non-numeric columns
    exclude_cols = {'close', 'asset', 'ts', 'dt'}  # Exclude non-numeric
    feature_cols = [c for c in x_t_full.columns if c not in exclude_cols]
    x_t = x_t_full[feature_cols]
    
    results = {
        'ts0': ts0,
        'S0': S0,
        'predictions_by_T': {},
        'saved_files': []
    }
    
    for T in targets_T:
        print(f"🔮 Generating predictions for T={T} ({T*4/24:.1f} days)")
        
        # Load T-specific models and calibrators
        models_T_path = models_dir / f"models_T{T}.joblib"
        calib_T_path = models_dir / f"calib_T{T}.json"
        
        if not models_T_path.exists():
            print(f"⚠️  No models found for T={T}, skipping")
            continue
        
        models_T = joblib.load(models_T_path)
        
        # Load calibration parameters
        calibrators_T = {}
        if calib_T_path.exists():
            with open(calib_T_path, 'r') as f:
                calibrators_T = json.load(f)
        
        # Generate base quantile predictions
        predictions = {}
        taus = [0.05, 0.25, 0.50, 0.75, 0.95]
        
        # Get the feature names from first model to ensure consistency
        first_model = models_T[list(models_T.keys())[0]]
        model_features = first_model.feature_name()
        
        # Select only the features that were used in training
        x_t_model = x_t[model_features]
        
        for tau in taus:
            if tau in models_T:
                pred = models_T[tau].predict(x_t_model)
                predictions[tau] = pred
        
        # Apply conformal calibration if available
        if calibrators_T and 'q_hat_global' in calibrators_T:
            q_hat = calibrators_T['q_hat_global']
            
            # Apply adjustment to prediction intervals
            if 0.05 in predictions and 0.95 in predictions:
                predictions[0.05] = predictions[0.05] - q_hat
                predictions[0.95] = predictions[0.95] + q_hat
        
        # Ensure quantile monotonicity
        from .cv import rearrange_quantiles
        predictions = rearrange_quantiles(predictions)
        
        # Calculate RV from bands width
        if 0.05 in predictions and 0.95 in predictions:
            width = predictions[0.95][0] - predictions[0.05][0]
            z_95 = 1.645  # 95th percentile of standard normal
            rvhat_ann = (width / (2 * z_95)) * np.sqrt(252 / (T * 4 / 24))  # Annualized
        else:
            rvhat_ann = np.nan
        
        # Save predictions to parquet
        h_days = T * 4 / 24  # Convert 4H bars to days
        
        pred_file = save_predictions_parquet(
            predictions, ts0, S0, T, h_days, rvhat_ann,
            output_path=output_dir / f"preds_T={T}.parquet"
        )
        
        results['predictions_by_T'][T] = {
            'predictions': predictions,
            'rvhat_ann': rvhat_ann,
            'h_days': h_days,
            'file_path': str(pred_file)
        }
        
        results['saved_files'].append(str(pred_file))
        
        print(f"✅ T={T}: RV_hat={rvhat_ann:.4f}, saved to {pred_file.name}")
    
    # Save meta_pred.json
    meta_pred = {
        'run_id': f"inference_{ts0.strftime('%Y%m%d_%H%M')}",
        'ts0': ts0.isoformat(),
        'tz': 'UTC',
        'S0': S0,
        'horizons_T': targets_T,
        'n_predictions': len(results['predictions_by_T']),
        'environment_versions': get_env_versions(),
        'data_hash': compute_content_hash(str(features_df.shape)),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    meta_path = output_dir / "meta_pred.json"
    write_json_atomic(meta_pred, meta_path)
    results['saved_files'].append(str(meta_path))
    
    print(f"📋 Meta predictions saved to: {meta_path}")
    
    return results


def load_and_predict(
    config_path: str,
    ts0: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    """
    Convenience function to load config and run daily inference.
    
    Args:
        config_path: Path to YAML configuration
        ts0: Reference timestamp (defaults to latest available)
        
    Returns:
        Inference results
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load features - resolve relative paths from config directory
    features_path = config['data']['features_path']
    if not Path(features_path).is_absolute():
        # Resolve relative to config file directory
        config_dir = Path(config_path).parent
        features_path = config_dir / '..' / features_path
        features_path = features_path.resolve()
    
    features_df = pd.read_parquet(str(features_path))
    
    # Convert ts column to datetime index if needed (same as utils.py pattern)
    if 'ts' in features_df.columns:
        features_df['ts'] = pd.to_datetime(features_df['ts'], utc=True)
        features_df = features_df.set_index('ts').sort_index()
    
    if ts0 is None:
        ts0 = features_df.index.max()
        # Ensure ts0 is a proper Timestamp
        if not isinstance(ts0, pd.Timestamp):
            ts0 = pd.Timestamp(ts0, tz='UTC')
    
    # Run inference - resolve relative paths from config directory
    models_dir = Path(config['output']['dir'])
    if not models_dir.is_absolute():
        # Resolve relative to config file directory
        config_dir = Path(config_path).parent
        models_dir = config_dir / '..' / models_dir
        models_dir = models_dir.resolve()
    
    return daily_inference_pipeline(
        ts0=ts0,
        features_df=features_df,
        models_dir=models_dir,
        targets_T=config['targets_T']
    )