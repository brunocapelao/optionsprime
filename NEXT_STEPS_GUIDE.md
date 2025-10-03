# 🚀 Next Steps Guide - Production Deployment

**Current Status:** HPO Complete ✅ | Validation Phase 🔄
**Target:** Production by Oct 5, 2025
**Progress:** 75% Complete

---

## 📋 Immediate Tasks (Priority Order)

### 🔴 CRITICAL - Must Complete Today (Oct 3)

#### 1. Walk-Forward Validation (4-6 hours)

**Purpose:** Validate model performance on unseen recent data

**Steps:**
```bash
cd /Users/brunocapelao/Projects/algo/project
source ../.venv/bin/activate

# Create walk-forward validation script
cat > scripts/walk_forward_validation.py << 'EOF'
import pandas as pd
import numpy as np
from pathlib import Path
import json

def walk_forward_validation(horizons, window_days=365, step_days=30):
    """
    Perform walk-forward validation for coverage empirical testing.

    Args:
        horizons: List of T values [42, 48, 54, 60]
        window_days: Training window size
        step_days: Step size for moving window
    """
    results = {}

    for T in horizons:
        # Load predictions
        preds = pd.read_parquet(f'data/processed/preds/preds_T={T}.parquet')

        # Implement walk-forward logic
        # Calculate coverage on each out-of-sample window
        # Aggregate results

        results[f'T{T}'] = {
            'coverage_90': None,  # Calculate
            'coverage_50': None,  # Calculate
            'crossing_rate': None,  # Calculate
            'calibration_error': None  # Calculate
        }

    # Save results
    with open('data/processed/validation_wf/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == '__main__':
    results = walk_forward_validation([42, 48, 54, 60])
    print("Walk-Forward Validation Complete!")
    print(json.dumps(results, indent=2))
EOF

# Run validation
mkdir -p data/processed/validation_wf
python scripts/walk_forward_validation.py
```

**Success Criteria:**
- Coverage 90% CI: 87-93% for all horizons
- Coverage 50% CI: 47-53% for all horizons
- Results documented in `validation_wf/results.json`

---

#### 2. Conformal Calibration Refinement (2-3 hours)

**Purpose:** Adjust q_hat to improve interval calibration

**Steps:**
```bash
cd /Users/brunocapelao/Projects/algo/project
source ../.venv/bin/activate

# Test different calibration windows
python << 'EOF'
import pandas as pd
import numpy as np
import json

results = {}

for T in [42, 48, 54, 60]:
    preds = pd.read_parquet(f'data/processed/preds/preds_T={T}.parquet')

    best_config = None
    best_coverage = None

    for window_days in [30, 60, 90, 120]:
        # Calculate q_hat with this window
        # Measure resulting coverage
        # Track best configuration

        coverage = None  # Calculate empirical coverage

        if best_coverage is None or abs(coverage - 0.90) < abs(best_coverage - 0.90):
            best_coverage = coverage
            best_config = window_days

    results[f'T{T}'] = {
        'best_window': best_config,
        'coverage': best_coverage,
        'q_hat': None  # Calculate final q_hat
    }

# Save optimal calibration config
with open('data/processed/calibration_optimal.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Optimal Calibration:")
print(json.dumps(results, indent=2))
EOF
```

**Success Criteria:**
- q_hat > 0 for all horizons
- Coverage within [87%, 93%]
- Configuration documented

---

#### 3. MLflow Model Registry (1-2 hours)

**Purpose:** Formally register HPO-optimized models

**Steps:**
```bash
cd /Users/brunocapelao/Projects/algo/project
source ../.venv/bin/activate

# Register models from HPO
python << 'EOF'
import mlflow
import json
from pathlib import Path

mlflow.set_tracking_uri("sqlite:///mlruns.db")

for T in [42, 48, 54, 60]:
    # Load best params
    with open(f'data/processed/hpo/best_params_T={T}.json') as f:
        hpo_results = json.load(f)

    # Create experiment run
    with mlflow.start_run(experiment_id="2", run_name=f"HPO_Final_T{T}"):
        # Log parameters
        mlflow.log_params(hpo_results['best_params'])

        # Log metrics
        mlflow.log_metric("best_pinball_loss", hpo_results['best_pinball'])
        mlflow.log_metric("n_trials", hpo_results['n_trials'])
        mlflow.log_metric("n_pruned", hpo_results['n_pruned'])

        # Log model artifacts
        mlflow.log_artifact(f'data/processed/preds/models_T{T}.joblib')
        mlflow.log_artifact(f'data/processed/preds/calibrators_T{T}.joblib')

        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/models"
        mlflow.register_model(
            model_uri=model_uri,
            name=f"CQR_LightGBM_HPO_T{T}",
            tags={
                "version": "1.0.0-hpo",
                "stage": "staging",
                "hpo": "true",
                "date": "2025-10-03"
            }
        )

print("✅ All models registered in MLflow!")
EOF
```

**Success Criteria:**
- 4 models registered (T=42, 48, 54, 60)
- Tagged with version v1.0.0-hpo
- Visible in MLflow UI

---

### 🟡 HIGH PRIORITY - Complete Tomorrow (Oct 4)

#### 4. Quality Gates Validation (3-4 hours)

**Purpose:** Validate all 4 quality gates for production approval

**Checklist:**

**Gate 1: CV Quality** ✅ Already Approved
- [x] Pinball loss < 0.05
- [x] Coverage 90%: 87-93%
- [x] Variance between folds < 10%
- [x] Quantile crossing < 1%

**Gate 2: Out-of-Sample** ⏳ To Validate
- [ ] Run walk-forward validation
- [ ] Verify coverage 87-93%
- [ ] Check calibration error < 0.03
- [ ] Document results

**Gate 3: Technical** ✅ Already Approved
- [x] Models well-formed
- [x] Features coherent
- [x] Infrastructure stable
- [x] MLflow tracking active

**Gate 4: Production Readiness** ⏳ To Validate
- [ ] Measure latency < 1s
- [ ] Test memory < 2GB
- [ ] Version artifacts
- [ ] Configure monitoring

**Command:**
```bash
# Run full quality gate validation
python scripts/validate_quality_gates.py \
  --cv-results data/processed/preds/cv_metrics_T*.json \
  --wf-results data/processed/validation_wf/results.json \
  --output data/processed/quality_gates_report.json
```

---

#### 5. Production Artifacts Preparation (2-3 hours)

**Purpose:** Package everything needed for deployment

**Steps:**
```bash
# Create production package
mkdir -p production_v1.0.0-hpo/{models,calibrators,configs,scripts,docs}

# Copy models
for T in 42 48 54 60; do
  cp data/processed/preds/models_T${T}.joblib production_v1.0.0-hpo/models/
  cp data/processed/preds/calibrators_T${T}.joblib production_v1.0.0-hpo/calibrators/
done

# Copy configurations
cp data/processed/hpo/best_params_T*.json production_v1.0.0-hpo/configs/
cp data/processed/calibration_optimal.json production_v1.0.0-hpo/configs/

# Generate requirements
pip freeze > production_v1.0.0-hpo/requirements.txt

# Copy prediction script
cp src/quant_bands/predict.py production_v1.0.0-hpo/scripts/

# Generate documentation
cat > production_v1.0.0-hpo/README.md << 'EOF'
# CQR_LightGBM Production Package v1.0.0-hpo

## Contents
- models/: 20 trained models (4 horizons × 5 quantiles)
- calibrators/: 4 conformal calibrators
- configs/: Optimal hyperparameters and calibration settings
- scripts/: Prediction pipeline
- requirements.txt: Python dependencies

## Usage
See DEPLOYMENT_GUIDE.md for deployment instructions.
EOF

# Create archive
tar -czf production_v1.0.0-hpo.tar.gz production_v1.0.0-hpo/

echo "✅ Production artifacts ready: production_v1.0.0-hpo.tar.gz"
```

---

#### 6. Performance Benchmarking (1-2 hours)

**Purpose:** Measure production performance metrics

**Script:**
```bash
python << 'EOF'
import time
import numpy as np
import pandas as pd
import joblib
import psutil
import os

def benchmark_prediction(T, n_samples=1000):
    """Benchmark prediction latency and memory."""

    # Load model
    models = joblib.load(f'data/processed/preds/models_T{T}.joblib')
    calibrators = joblib.load(f'data/processed/preds/calibrators_T{T}.joblib')

    # Load sample features
    features = pd.read_parquet('data/processed/features/features_4H.parquet')
    X_sample = features.sample(n_samples)

    # Measure memory before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024**2  # MB

    # Benchmark prediction
    latencies = []
    for _ in range(100):
        start = time.time()
        # Predict quantiles
        for tau, model in models.items():
            pred = model.predict(X_sample)
        elapsed = time.time() - start
        latencies.append(elapsed)

    # Measure memory after
    mem_after = process.memory_info().rss / 1024**2  # MB

    return {
        'T': T,
        'mean_latency_ms': np.mean(latencies) * 1000,
        'p95_latency_ms': np.percentile(latencies, 95) * 1000,
        'memory_usage_mb': mem_after - mem_before,
        'throughput_pred_per_sec': 1 / np.mean(latencies)
    }

# Benchmark all horizons
results = []
for T in [42, 48, 54, 60]:
    result = benchmark_prediction(T)
    results.append(result)
    print(f"T={T}: {result['mean_latency_ms']:.1f}ms, {result['memory_usage_mb']:.1f}MB")

# Check against targets
for r in results:
    latency_ok = r['mean_latency_ms'] < 1000  # < 1s
    memory_ok = r['memory_usage_mb'] < 2048   # < 2GB
    status = "✅" if (latency_ok and memory_ok) else "❌"
    print(f"{status} T={r['T']}: {r['mean_latency_ms']:.0f}ms, {r['memory_usage_mb']:.0f}MB")
EOF
```

**Success Criteria:**
- Latency < 1s per prediction
- Memory < 2GB total
- Documented in benchmark report

---

### 🟢 MEDIUM PRIORITY - Deploy Day (Oct 5)

#### 7. Final Approval & Go-Live

**Pre-Deployment Checklist:**
- [ ] 3/4 quality gates passed
- [ ] Walk-forward validation complete
- [ ] Production artifacts prepared
- [ ] Benchmarking results documented
- [ ] Stakeholder approval obtained
- [ ] Rollback plan tested

**Deployment Commands:**
```bash
# Final check
python scripts/pre_deployment_check.py

# Deploy
./scripts/deploy_production.sh v1.0.0-hpo

# Smoke test
python tests/test_production_smoke.py

# Enable monitoring
python scripts/start_monitoring.py
```

---

## 📊 Success Criteria Summary

| Task | Duration | Success Metric |
|------|----------|----------------|
| Walk-Forward Validation | 4-6h | Coverage 87-93% |
| Conformal Calibration | 2-3h | q_hat > 0, Coverage optimal |
| MLflow Registry | 1-2h | 4 models registered |
| Quality Gates | 3-4h | 3/4 gates passed |
| Production Artifacts | 2-3h | Package ready |
| Benchmarking | 1-2h | Latency < 1s |
| **TOTAL** | **13-20h** | **Production Ready** |

---

## 🚨 Escalation Contacts

**Technical Issues:**
- Architect/Lead: Critical decisions
- Data Scientist: Validation questions

**Business Decisions:**
- PO: Timeline, priorities
- Stakeholder: Final approval

**Emergency:**
- If coverage fails: Recalibrate immediately
- If latency too high: Optimize or prune models
- If quality gates fail: Postpone to Oct 6

---

## 📚 Reference Documents

- **PRODUCTION_ROADMAP_v2.md**: Detailed timeline
- **PROJECT_STATUS_REPORT.md**: Full technical status
- **EXECUTIVE_SUMMARY.md**: Business summary
- **EVALUATION_REPORT.md**: Model evaluation

---

**Start immediately with Task 1: Walk-Forward Validation**

This is the critical blocker for production approval. All other tasks depend on this completing successfully.

Good luck! 🚀
