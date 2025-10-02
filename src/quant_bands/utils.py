"""
Utilities Module
===============

Core utilities for seeds, I/O, atomic writes, and data loading.
"""

import json
import os
import random
import tempfile
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone


def set_seeds(seed: int = 17) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # LightGBM seed is set per model


def load_features(path: str) -> pd.DataFrame:
    """
    Load features from parquet, keeping only modelable columns.
    Returns: DataFrame with *_l1 features + calendar + close + targets.
    """
    df = pd.read_parquet(path)
    
    # Keep only modelable columns: *_l1 + calendar + close + targets
    modelable_cols = [c for c in df.columns if c.endswith('_l1')]
    calendar_cols = ['dow', 'hod', 'is_weekend']  # no _l1 suffix
    essential_cols = ['close', 'ts', 'asset'] if 'ts' in df.columns else ['close', 'asset']
    target_cols = [c for c in df.columns if c.startswith('target_') and c.endswith('h')]
    
    keep_cols = modelable_cols + calendar_cols + essential_cols + target_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    df_clean = df[keep_cols].copy()
    
    # Ensure ts is datetime index if available
    if 'ts' in df_clean.columns:
        df_clean['ts'] = pd.to_datetime(df_clean['ts'], utc=True)
        df_clean = df_clean.set_index('ts').sort_index()
    
    return df_clean


def load_cpcv(path: str) -> Dict[str, Any]:
    """Load CV splits configuration."""
    with open(path, 'r') as f:
        return json.load(f)


def write_json_atomic(data: Dict[str, Any], path: str) -> None:
    """Atomic write of JSON data."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        mode='w', 
        dir=path_obj.parent, 
        delete=False
    ) as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp_path = tmp.name
    
    shutil.move(tmp_path, path)


def save_joblib_atomic(obj: Any, path: str) -> None:
    """Atomic save of joblib object."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        dir=path_obj.parent, 
        delete=False,
        suffix='.joblib'
    ) as tmp:
        joblib.dump(obj, tmp.name)
        tmp_path = tmp.name
    
    shutil.move(tmp_path, path)


def compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of string content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def get_env_versions() -> Dict[str, str]:
    """Get environment versions for reproducibility."""
    import sys
    import platform
    import subprocess
    
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        git_sha = None
    
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "git_sha": git_sha,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    try:
        import lightgbm
        versions["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass
        
    return versions


def remove_constant_columns(df: pd.DataFrame, threshold: float = 0.0) -> List[str]:
    """Remove columns with constant values. Returns list of removed columns."""
    # Protect essential columns
    protected_cols = {'close', 'open', 'high', 'low', 'volume', 'ts', 'asset'}
    
    removed = []
    for col in df.columns:
        if col not in protected_cols and df[col].nunique() <= 1:
            removed.append(col)
    return removed


def remove_high_nan_columns(df: pd.DataFrame, threshold: float = 0.3) -> List[str]:
    """Remove columns with >threshold NaN share. Returns list of removed columns."""
    removed = []
    for col in df.columns:
        nan_rate = df[col].isna().mean()
        if nan_rate > threshold:
            removed.append(col)
    return removed


def remove_collinear_features(df: pd.DataFrame, threshold: float = 0.95) -> Dict[str, Any]:
    """
    Remove highly correlated features, keeping the one with higher mutual info.
    Returns dict with removal decisions.
    """
    from sklearn.feature_selection import mutual_info_regression
    
    # Protect essential columns
    protected_cols = {'close', 'open', 'high', 'low', 'volume', 'ts', 'asset'}
    
    # Only consider numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return {"removed": [], "decisions": []}
    
    # Choose a target for MI calculation
    target_col = 'target' if 'target' in df.columns else (numeric_cols[0])
    features_df = df[numeric_cols].drop(columns=[target_col])
    
    if len(features_df.columns) < 2:
        return {"removed": [], "decisions": []}
    
    # Fill NaN for correlation computation
    corr_df = features_df.fillna(features_df.median())
    corr_matrix = corr_df.corr().abs()
    
    removed = []
    decisions = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
            
            if col1 in removed or col2 in removed:
                continue
                
            correlation = corr_matrix.iloc[i, j]
            
            if correlation > threshold:
                # Compute mutual info to decide which to keep
                try:
                    y = df[target_col].fillna(df[target_col].median())
                    mi1 = mutual_info_regression(
                        corr_df[[col1]].fillna(0), y, random_state=17
                    )[0]
                    mi2 = mutual_info_regression(
                        corr_df[[col2]].fillna(0), y, random_state=17
                    )[0]
                    
                    # Prefer higher MI; if nearly equal, keep the first column deterministically
                    if abs(mi1 - mi2) <= 1e-3:
                        kept, dropped = col1, col2
                    elif mi1 >= mi2:
                        kept, dropped = col1, col2
                    else:
                        kept, dropped = col2, col1
                        
                except:
                    # Fallback: keep first column
                    kept, dropped = col1, col2
                
                # Don't remove protected columns
                if dropped not in protected_cols:
                    removed.append(dropped)
                elif kept not in protected_cols:
                    # If dropped is protected but kept is not, swap them
                    kept, dropped = dropped, kept
                    removed.append(dropped)
                # If both are protected, don't remove either
                else:
                    continue
                decisions.append({
                    "kept": kept,
                    "dropped": dropped,
                    "correlation": float(correlation),
                    "mi_kept": float(mi1 if kept == col1 else mi2),
                    "mi_dropped": float(mi2 if dropped == col2 else mi1)
                })
    
    return {"removed": removed, "decisions": decisions}


def apply_time_decay_weights(
    timestamps, 
    lambda_decay: float = 0.01
) -> np.ndarray:
    """Apply exponential time decay weights: w_t = exp(-λ * Δt)."""
    if len(timestamps) == 0:
        return np.array([])
    
    # Convert to pandas Series or DatetimeIndex if needed
    if not isinstance(timestamps, (pd.DatetimeIndex, pd.Series)):
        timestamps = pd.to_datetime(timestamps)
    
    # Convert to days from most recent
    max_time = timestamps.max()
    time_diff = max_time - timestamps
    
    # Handle both Series and Index cases
    if hasattr(time_diff, 'dt'):
        delta_days = time_diff.dt.total_seconds() / (24 * 3600)
    else:
        delta_days = time_diff.total_seconds() / (24 * 3600)
    
    # Convert to numpy array to ensure .sum() works
    delta_days = np.array(delta_days)
    weights = np.exp(-lambda_decay * delta_days)
    return weights / weights.sum()  # normalize
