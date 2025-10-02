#!/usr/bin/env python3
"""
Generate Historical Predictions
================================

Generates out-of-sample predictions for the last N days using trained models.
This creates the time series data needed for Module 03 (momentum analysis).

Usage:
    python scripts/generate_historical_predictions.py --days 180
    python scripts/generate_historical_predictions.py --days 180 --horizons 42 48
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from quant_bands.predict import daily_inference_pipeline


def generate_historical_predictions(
    features_path: Path,
    models_dir: Path,
    output_dir: Path,
    n_days: int = 180,
    horizons: list = [42, 48, 54, 60],
    frequency: str = '1D'
):
    """
    Generate historical predictions for the last N days.
    
    Args:
        features_path: Path to features parquet file
        models_dir: Directory with trained models
        output_dir: Output directory for predictions
        n_days: Number of days to generate (default: 180)
        horizons: List of horizons to predict (default: [42, 48, 54, 60])
        frequency: Frequency of predictions (default: '1D' = daily)
    """
    print("=" * 80)
    print("🚀 GENERATE HISTORICAL PREDICTIONS")
    print("=" * 80)
    print()
    
    # Load features
    print(f"📂 Loading features from: {features_path}")
    features_df = pd.read_parquet(features_path)
    
    # Prepare index
    if 'ts' in features_df.columns:
        features_df['ts'] = pd.to_datetime(features_df['ts'], utc=True)
        features_df = features_df.set_index('ts').sort_index()
    
    print(f"   ✅ Features loaded: shape {features_df.shape}")
    print(f"   📅 Date range: {features_df.index.min()} to {features_df.index.max()}")
    print()
    
    # Determine timestamps to predict
    end_timestamp = features_df.index.max()
    start_timestamp = end_timestamp - pd.Timedelta(days=n_days)
    
    # Generate daily timestamps (00:00 UTC each day)
    timestamps = pd.date_range(
        start=start_timestamp, 
        end=end_timestamp, 
        freq=frequency,
        tz='UTC'
    )
    
    # Filter to timestamps that exist in features
    valid_timestamps = []
    for ts in timestamps:
        # Find nearest timestamp in features (within 24h)
        mask = (features_df.index >= ts - pd.Timedelta(hours=12)) & \
               (features_df.index <= ts + pd.Timedelta(hours=12))
        
        if mask.any():
            nearest_ts = features_df.index[mask][0]
            valid_timestamps.append(nearest_ts)
    
    print(f"🎯 Generating predictions for {len(valid_timestamps)} timestamps")
    print(f"   📅 Start: {valid_timestamps[0]}")
    print(f"   📅 End:   {valid_timestamps[-1]}")
    print(f"   🎲 Horizons: {horizons}")
    print()
    
    # Initialize storage for each horizon
    predictions_by_horizon = {T: [] for T in horizons}
    
    # Generate predictions for each timestamp
    print("⏳ Generating predictions...")
    print()
    
    failed_timestamps = []
    
    for ts0 in tqdm(valid_timestamps, desc="Processing timestamps", unit="timestamp"):
        try:
            # Run inference
            results = daily_inference_pipeline(
                ts0=ts0,
                features_df=features_df,
                models_dir=models_dir,
                targets_T=horizons,
                output_dir=output_dir
            )
            
            # Collect predictions for each horizon
            for T in horizons:
                if T in results['predictions_by_T']:
                    # Load the file that was just created
                    pred_file = output_dir / f"preds_T={T}.parquet"
                    if pred_file.exists():
                        df_pred = pd.read_parquet(pred_file)
                        predictions_by_horizon[T].append(df_pred)
            
        except Exception as e:
            print(f"\n⚠️  Error at {ts0}: {e}")
            failed_timestamps.append(ts0)
            continue
    
    print()
    print("📊 Combining predictions into time series...")
    print()
    
    # Combine predictions into single files per horizon
    for T in horizons:
        if predictions_by_horizon[T]:
            # Concatenate all predictions
            df_combined = pd.concat(predictions_by_horizon[T], ignore_index=True)
            
            # Sort by timestamp
            df_combined = df_combined.sort_values('ts0').reset_index(drop=True)
            
            # Remove duplicates (keep last)
            df_combined = df_combined.drop_duplicates(subset=['ts0'], keep='last')
            
            # Save combined file
            output_file = output_dir / f"preds_T={T}.parquet"
            df_combined.to_parquet(output_file, index=False, engine='pyarrow')
            
            print(f"✅ T={T:2d}: {len(df_combined):4d} timestamps saved to {output_file.name}")
            print(f"         Date range: {df_combined['ts0'].min()} to {df_combined['ts0'].max()}")
        else:
            print(f"❌ T={T:2d}: No predictions generated")
    
    print()
    print("=" * 80)
    print("✅ HISTORICAL PREDICTIONS GENERATED SUCCESSFULLY!")
    print("=" * 80)
    print()
    
    if failed_timestamps:
        print(f"⚠️  {len(failed_timestamps)} timestamps failed:")
        for ts in failed_timestamps[:5]:
            print(f"   • {ts}")
        if len(failed_timestamps) > 5:
            print(f"   ... and {len(failed_timestamps) - 5} more")
        print()
    
    # Summary statistics
    print("📊 SUMMARY:")
    print(f"   • Total timestamps processed: {len(valid_timestamps)}")
    print(f"   • Successful: {len(valid_timestamps) - len(failed_timestamps)}")
    print(f"   • Failed: {len(failed_timestamps)}")
    print(f"   • Horizons: {len(horizons)}")
    print()
    
    for T in horizons:
        if predictions_by_horizon[T]:
            df = pd.concat(predictions_by_horizon[T], ignore_index=True)
            print(f"   T={T:2d}: {len(df):4d} predictions, shape {df.shape}")
    
    print()
    print("🎯 Ready for Module 03 (momentum analysis)!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate historical predictions for momentum analysis"
    )
    parser.add_argument(
        '--days', 
        type=int, 
        default=180,
        help='Number of days to generate (default: 180)'
    )
    parser.add_argument(
        '--horizons',
        type=int,
        nargs='+',
        default=[42, 48, 54, 60],
        help='Horizons to predict (default: 42 48 54 60)'
    )
    parser.add_argument(
        '--features',
        type=str,
        default='data/processed/features/features_4H.parquet',
        help='Path to features file'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='data/processed/preds',
        help='Directory with trained models'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/preds',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--frequency',
        type=str,
        default='1D',
        choices=['1D', '12H', '6H', '4H'],
        help='Frequency of predictions (default: 1D = daily)'
    )
    
    args = parser.parse_args()
    
    # Convert paths
    project_root = Path(__file__).parent.parent
    features_path = project_root / args.features
    models_dir = project_root / args.models_dir
    output_dir = project_root / args.output_dir
    
    # Validate paths
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        sys.exit(1)
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        sys.exit(1)
    
    # Check for trained models
    required_models = [models_dir / f"models_T{T}.joblib" for T in args.horizons]
    missing_models = [m for m in required_models if not m.exists()]
    
    if missing_models:
        print("❌ Missing trained models:")
        for m in missing_models:
            print(f"   • {m}")
        print()
        print("Please train models first using: python run_train.py --config config/base.yaml")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run generation
    generate_historical_predictions(
        features_path=features_path,
        models_dir=models_dir,
        output_dir=output_dir,
        n_days=args.days,
        horizons=args.horizons,
        frequency=args.frequency
    )


if __name__ == '__main__':
    main()
