# Quantile Bands Training Pipeline (02a)

Production-ready quantile regression with conformal prediction for multi-horizon price forecasting.

## 🎯 Overview

This module implements the complete **02a specification** for training quantile regression models with conformal prediction bands. It follows best practices for production ML with:

- **Anti-leakage CPCV**: Purged cross-validation with embargo for overlapping prediction windows
- **Mondrian Conformal**: Stratified conformal prediction with time-decay weighting  
- **Quality Control**: Automated monotonicity, coverage, and coherence checks
- **Deterministic Artifacts**: Atomic writes, checksums, and reproducible results
- **Modular Architecture**: Code as source of truth, notebooks for reporting only

## 🏗️ Architecture

```
src/quant_bands/
├── __init__.py          # Package interface
├── train.py             # Main training pipeline (CLI)
├── cv.py                # Cross-validation with purging
├── conformal.py         # Conformal prediction (CQR/EnbPI)
├── qc.py                # Quality control and monotonicity
└── utils.py             # I/O, seeds, atomic operations

configs/
└── 02a.yaml             # Training configuration

tests/
└── test_02a_train.py    # Comprehensive unit tests

notebooks/
└── 02a_train_report.ipynb  # Reporting notebook (thin)
```

## 🚀 Quick Start

### Prerequisites

1. **Features Dataset**: Run `01_data_features.ipynb` to generate `features_4H.parquet`
2. **Dependencies**: Install requirements (see below)
3. **Configuration**: Review `configs/02a.yaml`

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
make dev-setup
```

### Training Pipeline

```bash
# Basic training
make train

# With custom config
make train CONFIG=configs/02a_production.yaml

# Full pipeline with reporting
make all

# CLI interface
python -m quant_bands.train --config configs/02a.yaml
```

### Testing

```bash
# Run unit tests
make test

# Smoke test environment
make smoke-test

# Check pipeline status
make status
```

## 📊 Outputs

The pipeline generates deterministic artifacts in `data/processed/02a_train/`:

### Core Models
- **`quantiles_model.joblib`**: LightGBM quantile models for all T×τ combinations
- **`calib_T{T}.json`**: Conformal calibrators with Mondrian conditioning

### Metrics & QC
- **`cv_metrics.json`**: CPCV performance metrics (pinball loss, interval score, coverage)
- **`qc_train.json`**: Quality control report (monotonicity, width coherence)
- **`feature_importance_T{T}.csv`**: Feature importance by horizon

### Metadata
- **`meta_train.json`**: Environment, configuration, execution metadata
- **`report_train.html`**: Comprehensive HTML report with visualizations

## ⚙️ Configuration

Key configuration sections in `configs/02a.yaml`:

```yaml
# Prediction horizons and quantiles
targets_T: [42, 48, 54, 60]  # 4H bars (7-10 days)
taus: [0.05, 0.25, 0.5, 0.75, 0.95]

# Cross-validation
cpcv:
  embargo_bars_4h: 42  # Anti-leakage embargo
  selection_metric: "interval_score"

# Model hyperparameters
model:
  kind: "lightgbm_quantile"
  grid:
    num_leaves: [31, 63, 127]
    learning_rate: [0.02, 0.03, 0.05]

# Conformal prediction
conformal:
  method: "cqr"
  alpha: 0.10  # 90% coverage target
  mondrian:
    use_regime: true
    vol_bucket_from: "rv_cc_1d_rank_l1"

# Sample weighting
weights:
  time_decay_lambda: 0.01
```

## 🔬 Quality Control

The pipeline implements comprehensive QC checks:

### SLO Thresholds
- **Coverage**: 90% ± 3% (pre-conformal diagnostic)
- **Quantile Crossing**: < 0.5% before rearrangement
- **Width Coherence**: Median width non-decreasing with T
- **Execution Time**: ≤ 30 minutes for 2 years of 4H data

### Automated Fixes
- **Quantile Rearrangement**: Chernozhukov monotonicity correction
- **Isotonic Width Correction**: Pool-adjacent-violators for width coherence
- **Collinearity Removal**: Mutual information-based feature selection

### Fail-Fast Conditions
- Missing features file or CV splits
- Coverage < 75% or > 98% in multiple horizons
- High quantile crossing rates (> 0.5%)
- Insufficient calibration data

## 📈 Performance Metrics

### Model Evaluation
- **Pinball Loss**: τ-quantile regression loss
- **Interval Score**: Width + penalty for miscoverage  
- **Coverage Rate**: Empirical interval coverage
- **CRPS**: Continuous ranked probability score (optional)

### Conformal Diagnostics
- **Nonconformity Distribution**: By Mondrian bucket
- **Calibration Sample Size**: Per bucket and global
- **Time Decay Effectiveness**: Weight distribution analysis

### Feature Analysis
- **Importance by Gain**: LightGBM native importance
- **Permutation Importance**: Hold-out validation importance
- **Ablation Studies**: Block-wise feature removal impact

## 🧪 Testing

Comprehensive test suite covering:

```bash
# Anti-leakage tests
test_no_leakage_in_labels()      # y_t construction
test_cpcv_respects_embargo()     # Training/validation separation
test_purged_indices_no_overlap() # Window overlap detection

# Monotonicity tests  
test_quantile_crossing_detection()  # Violation detection
test_rearrangement_fixes_crossings() # Chernozhukov rearrangement
test_width_coherence_check()        # Width-by-T validation

# Conformal tests
test_nonconformity_scores()      # CQR score computation
test_weighted_quantile()         # Time-decay quantiles
test_mondrian_buckets()          # Stratification logic

# Utility tests
test_collinearity_removal()      # Feature selection
test_time_decay_weights()        # Temporal weighting
test_seed_reproducibility()      # Deterministic results

# Artifact tests
test_model_artifact_structure()     # Model serialization
test_calibrator_artifact_structure() # Calibrator format
```

## 🔧 Development

### Adding New Models

1. **Extend `train.py`**: Add new model type in `train_quantile_models()`
2. **Update Config**: Add model parameters to YAML schema
3. **Add Tests**: Include model-specific test cases
4. **Update QC**: Extend quality checks if needed

### Custom Conformal Methods

1. **Extend `conformal.py`**: Implement new conformalization method
2. **Update `fit_conformal()`**: Add method selection logic
3. **Add Configuration**: Extend YAML schema
4. **Test Coverage**: Add method-specific tests

### Monitoring Integration

The pipeline generates structured metadata suitable for ML monitoring:

```python
# Model metrics
cv_metrics = {
    "T=42": {
        "coverage_90_mean": 0.891,
        "interval_score_mean": 0.0234,
        "pinball_0.05_mean": 0.0121
    }
}

# QC status
qc_report = {
    "qc_passed": True,
    "summary": {"status": "PASS", "total_warnings": 2},
    "monotonicity_by_T": {...},
    "width_coherence": {...}
}
```

## 📝 Best Practices

### Production Deployment
1. **Use CLI Interface**: `python -m quant_bands.train --config prod.yaml`
2. **Version Control**: Pin configuration files and track git SHA
3. **Atomic Operations**: All file writes are atomic (no partial writes)
4. **Monitoring**: Parse `meta_train.json` for execution metrics
5. **Rollback**: Keep model artifacts with timestamps for rollback

### Performance Optimization
1. **Parallel Training**: Models per τ can be trained in parallel
2. **Feature Selection**: Use collinearity removal and ablation studies
3. **Memory Management**: Stream large datasets, don't load all in memory
4. **Early Stopping**: LightGBM with validation-based early stopping

### Debugging
1. **Verbose Logging**: Set logging level in configuration
2. **Debug Mode**: Use `make train-debug` for detailed output
3. **Partial Runs**: Comment out horizons in config for faster iteration
4. **QC Analysis**: Check `qc_train.json` for specific failures

## 📚 References

- **Conformal Prediction**: Shafer & Vovk (2008), "A Tutorial on Conformal Prediction"
- **Quantile Crossing**: Chernozhukov et al. (2010), "Quantile and Probability Curves"
- **Purged CV**: López de Prado (2018), "Advances in Financial Machine Learning"
- **LightGBM**: Ke et al. (2017), "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"

---

**Pipeline Status**: ✅ Production Ready | **Last Updated**: 2024-10-01 | **Version**: 1.0.0