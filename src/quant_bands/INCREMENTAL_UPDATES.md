# Incremental Prediction Updates

## Overview

The `predict_incremental.py` module provides automatic detection and generation of missing predictions when new data bars are added to `features_4H.parquet`.

## Problem Solved

When you download new BTC price data:
1. ✅ `features_4H.parquet` gets updated with new bars
2. ❌ `preds_T=*.parquet` files are **NOT** automatically updated
3. ❌ Visualizations show incomplete data

**Solution**: Incremental update detects missing predictions and generates only what's needed.

## Functions

### `update_predictions_incremental()`

Automatically updates prediction files with missing ts0 timestamps.

```python
from quant_bands.predict_incremental import update_predictions_incremental

stats = update_predictions_incremental(
    features_path=Path("data/processed/features/features_4H.parquet"),
    models_dir=Path("data/processed/preds"),
    preds_dir=Path("data/processed/preds"),
    targets_T=[42, 48, 54, 60],
    verbose=True
)

print(f"Added {sum(stats['new_predictions_by_T'].values())} new predictions")
```

**How it works:**
1. Loads existing `preds_T=*.parquet` files
2. Identifies ts0 timestamps in features that don't have predictions
3. Generates predictions for missing ts0 only
4. Appends new predictions to existing files (preserves history)

**Parameters:**
- `features_path`: Path to features parquet file
- `models_dir`: Directory containing trained models (`models_T*.joblib`, `calib_T*.json`)
- `preds_dir`: Directory for prediction files
- `targets_T`: List of horizons (default: [42, 48, 54, 60])
- `verbose`: Show detailed progress logs

**Returns:**
```python
{
    'total_ts0_available': 1000,  # Total timestamps in features
    'new_predictions_by_T': {
        42: 10,   # Number of new predictions for T=42
        48: 10,
        54: 10,
        60: 10
    },
    'errors': [],  # List of error messages
    'duration_seconds': 5.2,
    'files_updated': ['preds_T=42.parquet', ...]
}
```

### `check_predictions_status()`

Check coverage status without generating predictions.

```python
from quant_bands.predict_incremental import check_predictions_status

status = check_predictions_status(
    features_path=Path("data/processed/features/features_4H.parquet"),
    preds_dir=Path("data/processed/preds"),
    targets_T=[42, 48, 54, 60]
)

for T, info in status['status_by_T'].items():
    print(f"T={T}: {info['coverage_pct']:.1f}% coverage")
    if info['missing_predictions'] > 0:
        print(f"  Missing {info['missing_predictions']} predictions")
```

**Returns:**
```python
{
    'total_ts0_available': 1000,
    'ts0_range': (Timestamp('2020-01-01'), Timestamp('2025-10-02')),
    'status_by_T': {
        42: {
            'file_exists': True,
            'total_predictions': 990,
            'missing_predictions': 10,
            'coverage_pct': 99.0,
            'up_to_date': False
        },
        # ... for each T
    }
}
```

## Usage in Notebooks

### Before Visualization (Recommended)

Add this cell at the beginning of visualization notebooks:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('../src').resolve()))

from quant_bands.predict_incremental import update_predictions_incremental

# Auto-update predictions
update_predictions_incremental(
    features_path=Path('../data/processed/features/features_4H.parquet'),
    models_dir=Path('../data/processed/preds'),
    preds_dir=Path('../data/processed/preds'),
    targets_T=[42, 48, 54, 60],
    verbose=True
)
```

### In Production Scripts

```python
from quant_bands.predict_incremental import check_predictions_status, update_predictions_incremental

# Check if update is needed
status = check_predictions_status(...)
needs_update = any(
    not info['up_to_date']
    for info in status['status_by_T'].values()
)

if needs_update:
    update_predictions_incremental(...)
```

## Workflow Integration

### Daily Workflow

```bash
# 1. Download new data
jupyter execute notebooks/00_baixar_dados.ipynb

# 2. Update features
jupyter execute notebooks/01_data_features.ipynb

# 3. Auto-update predictions (built into notebook 04)
jupyter execute notebooks/04_forecast_visualization.ipynb
```

The predictions update automatically in step 3! 🎉

### Manual Update

```python
# In Python script or notebook
from pathlib import Path
from quant_bands.predict_incremental import update_predictions_incremental

update_predictions_incremental(
    features_path=Path("data/processed/features/features_4H.parquet"),
    models_dir=Path("data/processed/preds"),
    preds_dir=Path("data/processed/preds")
)
```

## Performance

- **Fast**: Only generates missing predictions (~0.5s per prediction)
- **Safe**: Appends to existing files, never overwrites
- **Idempotent**: Can run multiple times safely
- **Memory efficient**: Processes one ts0 at a time

## Error Handling

The function handles errors gracefully:
- Missing model files → Skip that T horizon
- Feature errors → Log and continue with next ts0
- All errors returned in `stats['errors']`

Example:
```python
stats = update_predictions_incremental(...)

if stats['errors']:
    print(f"⚠️  {len(stats['errors'])} errors occurred:")
    for error in stats['errors'][:5]:  # Show first 5
        print(f"  - {error}")
```

## Best Practices

### ✅ DO:
- Run before visualizations to ensure up-to-date data
- Check status first with `check_predictions_status()`
- Use `verbose=True` during development
- Review errors in production

### ❌ DON'T:
- Delete prediction files (use incremental updates instead)
- Run in parallel (models are loaded sequentially)
- Ignore errors (may indicate data/model issues)

## Future Enhancements

Planned improvements:
- [ ] Parallel prediction generation (multiprocessing)
- [ ] Automatic model reloading on changes
- [ ] Integration with MLflow for tracking
- [ ] Support for partial updates (specific T only)
- [ ] Caching of loaded models

## Troubleshooting

### "Models not found for T=X"
→ Train models first using `notebooks/02a_train_report_gold.ipynb`

### "No features found up to ts0"
→ Check timezone consistency between features and ts0

### Slow performance
→ Consider updating less frequently (e.g., once daily)
→ Or implement parallel processing (future enhancement)

### Predictions don't match manual generation
→ Ensure model files haven't changed
→ Check random seeds and feature engineering consistency

## Example Output

```
🔄 Starting incremental prediction update...

📂 Loading features from: features_4H.parquet
   Found 1000 timestamps
   Range: 2020-01-01 to 2025-10-02

============================================================
Processing T=42 (7.0 days)
============================================================
📊 Existing predictions: 990
🆕 New predictions needed: 10
   First: 2025-09-23 00:00:00+00:00
   Last:  2025-10-02 00:00:00+00:00
📦 Loading models from: models_T42.joblib
📐 Conformal calibration: q_hat=0.002345
🔮 Generating predictions...
✅ Added 10 new predictions
   Total now: 1000 predictions
   Saved to: preds_T=42.parquet

[... similar for T=48, 54, 60 ...]

============================================================
📊 UPDATE SUMMARY
============================================================
Duration: 5.2s
Total ts0 available: 1000

New predictions generated:
  • T=42: 10 new predictions
  • T=48: 10 new predictions
  • T=54: 10 new predictions
  • T=60: 10 new predictions

Total new predictions: 40

✅ Update completed successfully!
============================================================
```

## See Also

- `predict.py` - Core prediction pipeline
- `train.py` - Model training
- `cv.py` - Cross-validation and conformal calibration
- Notebook `04_forecast_visualization.ipynb` - Uses incremental updates
