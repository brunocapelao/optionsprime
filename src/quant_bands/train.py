"""
Training Pipeline Module
=======================

Main training pipeline for quantile bands with conformal prediction.
Implements the complete 02a specification with CLI interface.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid
import joblib
from tqdm import tqdm

from .utils import (
    set_seeds, load_features, load_cpcv, write_json_atomic, 
    save_joblib_atomic, get_env_versions, compute_file_hash, compute_content_hash,
    remove_constant_columns, remove_high_nan_columns, 
    remove_collinear_features, apply_time_decay_weights
)
from .cv import (
    build_xy_datasets, get_cpcv_folds, evaluate_cpcv, 
    rearrange_quantiles, check_quantile_crossing_rate
)
from .conformal import fit_conformal, nonconformity_errors
from .qc import run_comprehensive_qc, apply_isotonic_width_correction


def save_checkpoint(checkpoint_data: Dict[str, Any], output_dir: Path) -> None:
    """Save training checkpoint to resume interrupted training."""
    checkpoint_path = output_dir / "training_checkpoint.json"
    checkpoint_data['timestamp'] = time.time()
    write_json_atomic(checkpoint_data, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load training checkpoint if exists."""
    checkpoint_path = output_dir / "training_checkpoint.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                print(f"🔄 Checkpoint found: {len(checkpoint.get('completed_horizons', []))} horizons completed")
                return checkpoint
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
    return None


def clear_checkpoint(output_dir: Path) -> None:
    """Clear checkpoint after successful completion."""
    checkpoint_path = output_dir / "training_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("✅ Checkpoint cleared")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"✅ Loaded config: {list(config.keys()) if config else 'None'}")
            return config
    except ImportError as e:
        print(f"⚠️  YAML import failed: {e}, falling back to JSON")
        # Fallback to JSON if YAML not available
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        raise


def prepare_datasets(
    features_df: pd.DataFrame, 
    targets_T: List[int], 
    data_config: Dict[str, Any]
) -> Dict[int, Dict[str, Any]]:
    """
    Prepare datasets for different T horizons with feature engineering.
    
    Returns:
        Dict mapping T -> {'X': features, 'y': targets, 'timestamps': index, 'T_horizon': T}
    """
    print("🔧 Preparing datasets...")
    
    # Use build_xy_datasets to create all datasets at once
    raw_datasets = build_xy_datasets(features_df, targets_T)
    
    datasets = {}
    
    for T in targets_T:
        if T not in raw_datasets:
            print(f"⚠️  No data for T={T}, skipping")
            continue
        
        data = raw_datasets[T]
        
        if data['X'].empty:
            print(f"⚠️  No data for T={T}, skipping")
            continue
        
        # Feature engineering
        print(f"📊 T={T}: {data['X'].shape[0]} samples, {data['X'].shape[1]} features")
        
        # Remove problematic features
        X_clean = data['X'].copy()
        
        # Remove constant columns
        constant_cols = remove_constant_columns(X_clean)
        if constant_cols:
            X_clean = X_clean.drop(columns=constant_cols)
            print(f"🗑️  Removed {len(constant_cols)} constant columns")
        
        # Remove high NaN columns
        high_nan_cols = remove_high_nan_columns(X_clean, threshold=data_config.get('max_nan_ratio', 0.5))
        if high_nan_cols:
            X_clean = X_clean.drop(columns=high_nan_cols)
            print(f"🗑️  Removed {len(high_nan_cols)} high-NaN columns")
        
        # Remove collinear features
        collinear_result = remove_collinear_features(X_clean, threshold=data_config.get('collinearity_threshold', 0.95))
        if collinear_result.get('removed'):
            X_clean = X_clean.drop(columns=collinear_result['removed'])
            print(f"🗑️  Removed {len(collinear_result['removed'])} collinear columns")
        
        print(f"🧹 T={T}: After cleaning: {X_clean.shape[1]} features")
        
        datasets[T] = {
            'X': X_clean,
            'y': data['y'],
            'timestamps': data['timestamps'],
            'valid_mask': data['valid_mask'],
            'T_horizon': T
        }
    
    print(f"✅ Prepared {len(datasets)} datasets")
    return datasets


def hyperparameter_search(
    dataset: Dict[str, Any],
    folds: List[Dict[str, Any]],
    model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Grid search hyperparameter optimization with progress tracking.
    
    Args:
        dataset: Dataset dict with X, y, timestamps
        folds: CPCV fold definitions
        model_config: Model configuration with param_grid
        
    Returns:
        Dict with best_params, best_score, search_results
    """
    T = dataset['T_horizon']
    param_grid = model_config.get('param_grid', {})
    
    if not param_grid:
        print(f"⚠️  No param_grid found for T={T}, using defaults")
        return {'best_params': {}, 'best_score': float('inf'), 'search_results': []}
    
    # Generate parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    print(f"🔍 T={T}: Testing {len(param_combinations)} parameter combinations")
    
    search_results = []
    best_score = float('inf')
    best_params = {}
    
    # Progress bar for hyperparameter search
    param_pbar = tqdm(param_combinations, desc=f"T={T} HyperSearch", unit="combo")
    
    for param_idx, params in enumerate(param_pbar):
        # Evaluate params with CPCV
        fold_scores = []
        
        for fold in folds:
            if len(fold['train_indices']) == 0 or len(fold['val_indices']) == 0:
                continue
            
            # Prepare fold data
            X_train = dataset['X'].iloc[fold['train_indices']]
            y_train = dataset['y'].iloc[fold['train_indices']]
            X_val = dataset['X'].iloc[fold['val_indices']]
            y_val = dataset['y'].iloc[fold['val_indices']].values
            
            # Quick single-quantile evaluation (τ=0.5)
            lgb_params = {
                'objective': 'quantile',
                'alpha': 0.5,
                'metric': 'quantile',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'seed': 17,
                **params
            }
            
            # Create datasets with proper validation
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model with early stopping
            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            
            pred = model.predict(X_val)
            fold_score = np.mean(np.abs(pred - y_val))  # MAE for τ=0.5
            fold_scores.append(fold_score)
        
        # Average score across folds
        avg_score = np.mean(fold_scores) if fold_scores else float('inf')
        
        search_results.append({
            'params': params,
            'score': avg_score,
            'fold_scores': fold_scores
        })
        
        # Update best
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
        
        # Update progress bar
        param_pbar.set_postfix({
            "best_score": f"{best_score:.6f}",
            "current": f"{avg_score:.6f}"
        })
    
    param_pbar.close()
    
    print(f"✅ T={T}: Best score = {best_score:.6f}")
    print(f"📋 T={T}: Best params = {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'search_results': search_results
    }


def train_quantile_models(
    dataset: Dict[str, Any],
    folds: List[Dict[str, Any]],
    best_params: Dict[str, Any],
    taus: List[float],
    weights_config: Optional[Dict[str, Any]] = None
) -> Dict[float, Any]:
    """
    Train quantile regression models for all tau values.
    
    Args:
        dataset: Dataset dict with X, y, timestamps
        folds: CPCV fold definitions  
        best_params: Optimal hyperparameters from search
        taus: List of quantile levels
        weights_config: Optional time-decay weighting config
        
    Returns:
        Dict mapping tau -> trained LightGBM model
    """
    T = dataset['T_horizon']
    print(f"🚀 Training quantile models for T={T}")
    
    models = {}
    
    # Combine all training data
    all_train_indices = []
    for fold in folds:
        all_train_indices.extend(fold['train_indices'])
    all_train_indices = sorted(list(set(all_train_indices)))
    
    if len(all_train_indices) == 0:
        raise ValueError(f"No training indices found for T={T}")
    
    X_train = dataset['X'].iloc[all_train_indices]
    y_train = dataset['y'].iloc[all_train_indices]
    
    # Apply time-decay weights if configured
    sample_weights = None
    if weights_config and weights_config.get('time_decay_lambda'):
        train_timestamps = dataset['timestamps'][all_train_indices]
        
        # Ensure timestamps is a proper DatetimeIndex or Series
        if hasattr(train_timestamps, 'index'):
            train_timestamps = train_timestamps.index
        elif not isinstance(train_timestamps, (pd.DatetimeIndex, pd.Series)):
            # Convert to DatetimeIndex if needed
            train_timestamps = pd.to_datetime(train_timestamps)
        
        sample_weights = apply_time_decay_weights(
            train_timestamps, 
            weights_config['time_decay_lambda']
        )
        print(f"⚖️  Applied time-decay weights (λ={weights_config['time_decay_lambda']})")
    
    # Train models for each quantile with progress bar
    tau_pbar = tqdm(taus, desc=f"T={T} Models", unit="τ")
    
    for tau in tau_pbar:
        tau_pbar.set_description(f"T={T} τ={tau}")
        
        lgb_params = {
            'objective': 'quantile',
            'alpha': tau,
            'metric': 'quantile',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'seed': 17,
            **best_params
        }
        
        # Prepare training data
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        
        # Train final model without validation (fixed rounds)
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        models[tau] = model
    
    tau_pbar.close()
    print(f"✅ Trained {len(models)} quantile models for T={T}")
    return models


def fit_conformal_predictors(
    models: Dict[float, Any],
    dataset: Dict[str, Any],
    folds: List[Dict[str, Any]],
    taus: List[float],
    conformal_config: Dict[str, Any]
) -> Dict[float, Any]:
    """
    Fit conformal predictors for uncertainty quantification.
    
    Args:
        models: Trained quantile models
        dataset: Dataset dict
        folds: CPCV fold definitions
        taus: Quantile levels
        conformal_config: Conformal prediction configuration
        
    Returns:
        Dict mapping tau -> conformal predictor
    """
    T = dataset['T_horizon']
    print(f"🎯 Fitting conformal predictors for T={T}")
    
    calibrators = {}
    
    for tau in taus:
        if tau not in models:
            continue
        
        model = models[tau]
        nonconformity_scores = []
        
        # Collect nonconformity scores from each fold
        for fold in folds:
            if len(fold['val_indices']) == 0:
                continue
            
            X_cal = dataset['X'].iloc[fold['val_indices']]
            y_cal = dataset['y'].iloc[fold['val_indices']].values
            
            predictions = model.predict(X_cal)
            scores = nonconformity_errors(y_cal, predictions, tau)
            nonconformity_scores.extend(scores)
        
        if nonconformity_scores:
            calibrator = fit_conformal(
                nonconformity_scores, 
                alpha=conformal_config.get('alpha', 0.1)
            )
            calibrators[tau] = calibrator
        else:
            print(f"⚠️  No calibration data for τ={tau}")
    
    print(f"✅ Fitted {len(calibrators)} conformal predictors")
    return calibrators


def save_training_artifacts(
    models: Dict[float, Any],
    cv_metrics: Dict[str, Any],
    calibrators: Dict[str, Any],
    T: int,
    output_dir: Path,
    config: Dict[str, Any],
    dataset: Dict[str, Any] = None
) -> Dict[str, str]:
    """
    Save all training artifacts for a given T horizon per 02a spec.
    
    Implements atomic writes and saves required artifacts:
    - quantiles_model.joblib: Combined models for all quantiles
    - calib_T{T}.json: Conformal calibration parameters  
    - cv_metrics.json: Cross-validation metrics
    - meta_train.json: Training metadata
    - feature_importance_T{T}.csv: Feature importance (optional)
    
    Returns:
        Dict with file paths for saved artifacts
    """
    print(f"💾 Saving artifacts for T={T} (02a spec compliant)")
    
    saved_files = {}
    
    # 1. Save quantiles_model.joblib (02a spec requirement)
    quantiles_model_path = output_dir / "quantiles_model.joblib"
    combined_models = {T: models}  # Wrap in T-indexed dict
    save_joblib_atomic(combined_models, quantiles_model_path)
    saved_files['quantiles_model'] = str(quantiles_model_path)
    
    # 2. Save individual models for this T
    models_T_path = output_dir / f"models_T{T}.joblib"
    save_joblib_atomic(models, models_T_path)
    saved_files[f'models_T{T}'] = str(models_T_path)
    
    # 3. Save cv_metrics.json (02a spec requirement)
    cv_metrics_path = output_dir / "cv_metrics.json"
    # Aggregate metrics across all T (will be overwritten by each T, final one wins)
    write_json_atomic(cv_metrics, cv_metrics_path)
    saved_files['cv_metrics'] = str(cv_metrics_path)
    
    # 4. Save T-specific CV metrics
    cv_metrics_T_path = output_dir / f"cv_metrics_T{T}.json"
    write_json_atomic(cv_metrics, cv_metrics_T_path)
    saved_files[f'cv_metrics_T{T}'] = str(cv_metrics_T_path)
    
    # 5. Save conformal calibrators (calib_T{T}.json already saved in pipeline)
    if calibrators:
        calibrators_path = output_dir / f"calibrators_T{T}.joblib"
        save_joblib_atomic(calibrators, calibrators_path)
        saved_files[f'calibrators_T{T}'] = str(calibrators_path)
    
    # 6. Save feature importance (02a spec optional)
    if models:
        try:
            # Extract feature importance from first model (tau=0.5 if available)
            reference_tau = 0.5 if 0.5 in models else list(models.keys())[0]
            reference_model = models[reference_tau]
            
            feature_names = getattr(reference_model, 'feature_name_', None)
            feature_importance = getattr(reference_model, 'feature_importance_', None)
            
            if feature_names and feature_importance is not None:
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': feature_importance,
                    'tau_reference': reference_tau
                }).sort_values('importance', ascending=False)
                
                feature_importance_path = output_dir / f"feature_importance_T{T}.csv"
                importance_df.to_csv(feature_importance_path, index=False)
                saved_files[f'feature_importance_T{T}'] = str(feature_importance_path)
                
        except Exception as e:
            print(f"⚠️  Could not save feature importance for T={T}: {e}")
    
    # 7. Save meta_train.json (02a spec requirement)
    embargo_bars = config.get('cpcv', {}).get('embargo_bars_4h', 42)
    
    metadata = {
        'T_horizon': T,
        'embargo': embargo_bars,  # 42 bars as per 02a spec
        'n_samples_total': int(dataset['valid_mask'].sum()) if 'valid_mask' in dataset else 0,
        'n_features': (list(models.values())[0].num_feature() if (models and hasattr(list(models.values())[0], 'num_feature')) else 0),
        'n_models': len(models),
        'n_calibrators': len(calibrators) if calibrators else 0,
        'quantiles': list(models.keys()) if models else [],
        'targets_T': config.get('targets_T', []),
        'folds': len(cv_metrics.get('fold_results', [])) if 'fold_results' in cv_metrics else 0,
        'timestamp': time.time(),
        'config_hash': str(hash(str(sorted(config.items())))),
        'data_hash': compute_file_hash(config['data']['features_path']),
        'git_sha': get_env_versions().get('git_sha', 'unknown'),
        'seed': 17,
        'cv_type': 'CPCV_purged',  # 02a spec compliance
        'environment_versions': get_env_versions()
    }
    
    meta_train_path = output_dir / "meta_train.json"
    write_json_atomic(metadata, meta_train_path)
    saved_files['meta_train'] = str(meta_train_path)
    
    print(f"✅ Saved {len(saved_files)} artifact files for T={T} (02a compliant)")
    return saved_files


def run_training_pipeline(config: Dict[str, Any]) -> None:
    """
    Run complete training pipeline with progress tracking.
    
    Main pipeline implementing 02a specification:
    1. Data loading and preparation
    2. Feature engineering and cleaning
    3. Hyperparameter search with CPCV
    4. Quantile model training
    5. Conformal predictor fitting
    6. Cross-validation evaluation
    7. Quality control and validation
    8. Artifact saving
    """
    print("🚀 Starting 02a Training Pipeline")
    print("=" * 50)
    
    # Set seeds for reproducibility
    set_seeds(17)
    
    # Load configuration
    targets_T = config['targets_T']
    taus = config['taus']
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint (if enabled)
    checkpoint = None
    completed_horizons = set()
    checkpoint_config = config.get('checkpoint', {})
    
    if checkpoint_config.get('enabled', True) and checkpoint_config.get('auto_resume', True):
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            completed_horizons = set(checkpoint.get('completed_horizons', []))
            if completed_horizons:
                remaining = [t for t in targets_T if t not in completed_horizons]
                print(f"🔄 Resuming training: {len(completed_horizons)} completed, {len(remaining)} remaining")
                if not remaining:
                    print("✅ All horizons already completed!")
                    return
    
    # Load data
    print("💾 Loading data...")
    features_df = load_features(config['data']['features_path'])
    cv_splits = load_cpcv(config['data']['cv_splits_path'])
    
    print(f"📊 Features shape: {features_df.shape}")
    print(f"🎯 Targets: {targets_T}")
    print(f"📏 Quantiles: {taus}")
    
    # Prepare datasets for all T horizons
    datasets = prepare_datasets(features_df, targets_T, config['data'])
    
    if not datasets:
        raise ValueError("No valid datasets prepared")
    
    # Training results storage
    all_models = {}
    all_cv_metrics = {}
    all_calibrators = {}
    all_saved_files = {}
    
    # Load existing results if resuming
    if checkpoint:
        all_models = checkpoint.get('all_models', {})
        all_cv_metrics = checkpoint.get('all_cv_metrics', {})
        all_calibrators = checkpoint.get('all_calibrators', {})
        all_saved_files = checkpoint.get('all_saved_files', {})
    
    start_time = time.time()
    
    # Main training loop with progress bar
    target_pbar = tqdm(targets_T, desc="🎯 Training", unit="T")
    
    for T in target_pbar:
        target_pbar.set_description(f"🎯 Training T={T}")
        
        # Skip if already completed (checkpoint resume)
        if T in completed_horizons:
            print(f"✅ T={T} already completed, skipping")
            continue
            
        print(f"\n{'='*20} Training T={T} {'='*20}")
        
        if T not in datasets:
            print(f"⚠️  No dataset for T={T}, skipping")
            continue
        
        dataset = datasets[T]
        
        # Get CPCV folds
        folds = get_cpcv_folds(
            cv_splits, 
            dataset, 
            embargo_bars=config['cpcv']['embargo_bars_4h']
        )
        
        print(f"📂 Generated {len(folds)} CPCV folds")
        
        if len(folds) == 0:
            print(f"⚠️  No valid folds for T={T}, skipping")
            continue
        
        # Hyperparameter search
        search_result = hyperparameter_search(dataset, folds, config['model'])
        
        # Train quantile models
        models_T = train_quantile_models(
            dataset, folds, search_result['best_params'], 
            taus, config.get('weights')
        )
        
        # Conformal prediction with Mondrian conditioning (02a spec compliant)
        print(f"🎯 Fitting conformal predictors for T={T}")
        
        # Prepare calibration data from validation folds
        calib_X_list = []
        calib_y_list = []
        
        for fold in folds:
            if len(fold['val_indices']) > 0:
                calib_X_list.append(dataset['X'].iloc[fold['val_indices']])
                calib_y_list.append(dataset['y'].iloc[fold['val_indices']])
        
        calibrators_T = {}
        post_conf_metrics = {}
        
        if calib_X_list:
            X_calib = pd.concat(calib_X_list, axis=0)
            y_calib = pd.concat(calib_y_list, axis=0)
            
            # Import conformal functions
            from .conformal import fit_conformal_mondrian, apply_conformal_correction, compute_post_conformal_metrics
            
            # Convert models dict to expected format
            models_dict = {f"q_{tau:.2f}": model for tau, model in models_T.items()}
            
            # Fit conformal predictor with Mondrian conditioning
            conformal_config = config['conformal']
            conformal_corrections = fit_conformal_mondrian(
                models_dict,
                X_calib,
                y_calib,
                alpha=conformal_config['alpha'],
                quantiles=taus,
                bucket_config=conformal_config.get('mondrian'),
                time_decay_lambda=conformal_config.get('lambda', 0.01),
                min_bucket_size=conformal_config.get('shrinkage', {}).get('min_bucket_size', 300)
            )
            
            # Save conformal calibration results (02a spec compliance)
            calib_path = output_dir / f"calib_T{T}.json"
            calib_data = {
                'T_horizon': T,
                'window_days': conformal_config.get('window_days', 90),
                'lambda': conformal_config.get('lambda', 0.01),
                'kappa': conformal_corrections.get('corrected_alpha'),
                'bucket_keys': conformal_corrections['bucket_keys'],
                'q_hat_global': float(conformal_corrections['q_hat_global']),
                'bucket_corrections': {
                    k: {k2: float(v2) if isinstance(v2, (int, float, np.number)) else v2 
                        for k2, v2 in v.items()} 
                    for k, v in conformal_corrections['bucket_corrections'].items()
                },
                'alpha': conformal_config['alpha'],
                'n_calib': len(X_calib),
                'low_quantile': taus[0],
                'high_quantile': taus[-1]
            }
            write_json_atomic(calib_data, calib_path)
            calibrators_T = conformal_corrections
            
            # Generate corrected predictions for evaluation
            pred_dict = {}
            for tau, model in models_T.items():
                pred_dict[f"q_{tau:.2f}"] = model.predict(X_calib)
            
            # Apply conformal correction
            corrected_pred_dict = apply_conformal_correction(
                pred_dict,
                conformal_corrections, 
                X_calib,
                conformal_config.get('mondrian')
            )
            
            # Compute post-conformal metrics (critical for 02a compliance)
            post_conf_metrics = compute_post_conformal_metrics(
                y_calib.values,
                corrected_pred_dict,
                X_calib,
                conformal_config.get('mondrian'),
                [taus[0], taus[-1]]  # Use first and last quantiles for coverage
            )
            
            print(f"📊 T={T} Post-Conformal Coverage: {post_conf_metrics['coverage_post_global']:.3f} (target: 0.90 ±0.03)")
            print(f"📏 T={T} Post-Conformal Width: {post_conf_metrics['width_post_global']:.4f}")
            
        else:
            print(f"⚠️  No calibration data available for T={T}")
        
        # Cross-validation evaluation (pre-conformal)
        print("📈 Evaluating with CPCV...")
        cv_metrics = evaluate_cpcv(models_T, dataset, folds, taus)
        cv_metrics['hyperparameter_search'] = search_result
        
        # Add post-conformal metrics to CV results
        if post_conf_metrics:
            cv_metrics['post_conformal'] = post_conf_metrics
        
        # Quality control with 02a spec compliance
        print("🔍 Running quality control...")
        try:
            valid_mask = dataset['valid_mask']
            X_all = dataset['X'].loc[valid_mask]
            y_all = dataset['y'].loc[valid_mask].values

            preds_T = {tau: model.predict(X_all) for tau, model in models_T.items()}
            qc_results = run_comprehensive_qc({T: preds_T}, {T: y_all})
            cv_metrics['quality_control'] = qc_results
        except Exception as e:
            print(f"⚠️  QC step failed: {e}")
        
        # Save artifacts
        saved_files = save_training_artifacts(
            models_T, cv_metrics, calibrators_T, T, output_dir, config, dataset
        )
        
        # Store results
        all_models[T] = models_T
        all_cv_metrics[T] = cv_metrics
        all_calibrators[T] = calibrators_T
        all_saved_files[T] = saved_files
        
        # Save checkpoint after each horizon completion (if enabled)
        completed_horizons.add(T)
        if checkpoint_config.get('enabled', True):
            checkpoint_data = {
                'completed_horizons': list(completed_horizons),
                'all_models': {str(k): v for k, v in all_models.items()},  # JSON serializable
                'all_cv_metrics': {str(k): v for k, v in all_cv_metrics.items()},
                'all_calibrators': {str(k): v for k, v in all_calibrators.items()},
                'all_saved_files': {str(k): v for k, v in all_saved_files.items()},
                'config': config,
                'progress': f"{len(completed_horizons)}/{len(targets_T)}"
            }
            save_checkpoint(checkpoint_data, output_dir)
        
        # Update progress
        target_pbar.set_postfix({
            "completed": f"{len(all_models)}/{len(targets_T)}",
            "score": f"{search_result['best_score']:.3f}"
        })
    
    target_pbar.close()
    
    # Final summary
    execution_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("🎉 Training Pipeline Complete!")
    print(f"⏱️  Total execution time: {execution_time:.1f} seconds")
    print(f"🏆 Successfully trained models for {len(all_models)} horizons")
    print(f"💾 Output directory: {output_dir}")
    
    # Save final summary
    summary = {
        'pipeline_version': '02a',
        'execution_time': execution_time,
        'trained_horizons': list(all_models.keys()),
        'total_models': sum(len(models) for models in all_models.values()),
        'timestamp': time.time(),
        'config': config,
        'saved_files': all_saved_files
    }
    
    summary_path = output_dir / "training_summary.json"
    write_json_atomic(summary, summary_path)
    
    print(f"📋 Training summary saved to: {summary_path}")
    
    # Clear checkpoint after successful completion (if enabled)
    if checkpoint_config.get('enabled', True):
        clear_checkpoint(output_dir)
    print("🎉 Training completed successfully!")


def main():
    """CLI entry point for training pipeline."""
    parser = argparse.ArgumentParser(description="02a Training Pipeline")
    parser.add_argument("--config", required=True, help="Configuration YAML file")
    parser.add_argument("--features", help="Override features path")
    parser.add_argument("--cv-splits", help="Override CV splits path")
    parser.add_argument("--out-dir", help="Override output directory")
    parser.add_argument("--clear-checkpoint", action="store_true", help="Clear existing checkpoint before training")
    parser.add_argument("--show-checkpoint", action="store_true", help="Show checkpoint status and exit")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.features:
        config['data']['features_path'] = args.features
    if args.cv_splits:
        config['data']['cv_splits_path'] = args.cv_splits
    if args.out_dir:
        config['output']['dir'] = args.out_dir
    
    output_dir = Path(config['output']['dir'])
    
    # Handle checkpoint commands
    if args.show_checkpoint:
        checkpoint = load_checkpoint(output_dir)
        if checkpoint:
            print(f"📄 Checkpoint found:")
            print(f"   Progress: {checkpoint.get('progress', 'Unknown')}")
            print(f"   Completed: {checkpoint.get('completed_horizons', [])}")
            print(f"   Timestamp: {time.ctime(checkpoint.get('timestamp', 0))}")
        else:
            print("📄 No checkpoint found")
        exit(0)
    
    if args.clear_checkpoint:
        clear_checkpoint(output_dir)
        print("🗑️  Checkpoint cleared")
    
    # Run pipeline
    try:
        run_training_pipeline(config)
        print("🎉 Training completed successfully!")
        exit(0)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
