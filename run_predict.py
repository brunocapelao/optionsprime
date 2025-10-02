#!/usr/bin/env python3
"""
Daily Prediction Runner
======================

Runs daily inference pipeline to generate preds_T=*.parquet files
with quantile predictions for next T bars.

Usage:
    python run_predict.py --config config/base.yaml --ts0 "2024-01-15 08:00:00"
    python run_predict.py --config config/base.yaml  # uses latest timestamp
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, 'src')

from quant_bands.predict import load_and_predict


def main():
    parser = argparse.ArgumentParser(description='Run daily quantile predictions')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to YAML configuration file')
    parser.add_argument('--ts0', type=str, default=None,
                      help='Reference timestamp (ISO format), defaults to latest')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for predictions')
    
    args = parser.parse_args()
    
    # Parse timestamp if provided
    ts0 = None
    if args.ts0:
        ts0 = pd.Timestamp(args.ts0, tz='UTC')
        print(f"📅 Using reference timestamp: {ts0}")
    else:
        print("📅 Using latest available timestamp")
    
    print(f"⚙️  Loading configuration: {args.config}")
    
    try:
        # Run inference
        results = load_and_predict(args.config, ts0=ts0)
        
        print(f"\n✅ Daily inference completed successfully!")
        print(f"📊 Reference: ts0={results['ts0']}, S0=${results['S0']:.2f}")
        print(f"🎯 Generated predictions for {len(results['predictions_by_T'])} horizons:")
        
        for T, pred_info in results['predictions_by_T'].items():
            h_days = pred_info['h_days']
            rv_hat = pred_info['rvhat_ann']
            print(f"   T={T} ({h_days:.1f}d): RV_hat={rv_hat:.4f}")
        
        print(f"\n📁 Saved files:")
        for file_path in results['saved_files']:
            print(f"   {file_path}")
            
    except Exception as e:
        print(f"❌ Error running daily inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())