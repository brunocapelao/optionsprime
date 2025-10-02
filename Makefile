# Quantile Bands Training Pipeline
# ================================
# Professional ML pipeline with make automation

.PHONY: help install test train report clean all

# Default target
help:
	@echo "Quantile Bands Training Pipeline"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  test       - Run unit tests"
	@echo "  train      - Run 02a training pipeline"
	@echo "  report     - Generate training report"
	@echo "  clean      - Clean generated artifacts"
	@echo "  all        - Run complete pipeline (train + report)"
	@echo ""
	@echo "Configuration:"
	@echo "  CONFIG     - Config file (default: configs/02a.yaml)"
	@echo "  FEATURES   - Features file (default: data/processed/features/features_4H.parquet)"
	@echo "  OUT_DIR    - Output directory (default: data/processed/02a_train)"

# Configuration variables
CONFIG ?= configs/02a.yaml
FEATURES ?= data/processed/features/features_4H.parquet
CV_SPLITS ?= data/processed/features/cv_splits.json
OUT_DIR ?= data/processed/02a_train

# Python environment
PYTHON ?= python
PIP ?= pip

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

# Run unit tests
test:
	@echo "🧪 Running unit tests..."
	$(PYTHON) -m pytest tests/test_02a_train.py -v
	@echo "✅ Tests completed"

# Validate prerequisites
check-prereqs:
	@echo "🔍 Checking prerequisites..."
	@if [ ! -f "$(FEATURES)" ]; then \
		echo "❌ Features file not found: $(FEATURES)"; \
		echo "Please run the 01_data_features pipeline first"; \
		exit 1; \
	fi
	@if [ ! -f "$(CV_SPLITS)" ]; then \
		echo "❌ CV splits file not found: $(CV_SPLITS)"; \
		echo "Please run the 01_data_features pipeline first"; \
		exit 1; \
	fi
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "❌ Config file not found: $(CONFIG)"; \
		exit 1; \
	fi
	@echo "✅ Prerequisites validated"

# Run training pipeline
train: check-prereqs
	@echo "🚀 Starting 02a training pipeline..."
	@echo "Config: $(CONFIG)"
	@echo "Features: $(FEATURES)"
	@echo "Output: $(OUT_DIR)"
	@echo ""
	$(PYTHON) -m quant_bands.train \
		--config $(CONFIG) \
		--features $(FEATURES) \
		--cv-splits $(CV_SPLITS) \
		--out-dir $(OUT_DIR)
	@echo ""
	@echo "✅ Training pipeline completed"
	@echo "📁 Artifacts saved to: $(OUT_DIR)"

# Generate training report
report:
	@echo "📊 Generating training report..."
	@if [ ! -d "$(OUT_DIR)" ]; then \
		echo "❌ Training artifacts not found in $(OUT_DIR)"; \
		echo "Please run 'make train' first"; \
		exit 1; \
	fi
	cd notebooks && $(PYTHON) 02a_train_report.ipynb
	@echo "✅ Report generated: $(OUT_DIR)/report_train.html"

# Clean generated artifacts
clean:
	@echo "🧹 Cleaning artifacts..."
	@if [ -d "$(OUT_DIR)" ]; then \
		rm -rf $(OUT_DIR); \
		echo "🗑️  Removed $(OUT_DIR)"; \
	fi
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup completed"

# Run complete pipeline
all: train report

# Development targets
dev-setup: install
	@echo "🛠️  Setting up development environment..."
	$(PIP) install pytest jupyter matplotlib seaborn scikit-learn
	@echo "✅ Development environment ready"

# Lint code
lint:
	@echo "🔍 Linting code..."
	$(PYTHON) -m flake8 src/quant_bands/ --max-line-length=100
	@echo "✅ Linting completed"

# Run training with debug output
train-debug: check-prereqs
	@echo "🐛 Running training with debug output..."
	$(PYTHON) -m quant_bands.train \
		--config $(CONFIG) \
		--features $(FEATURES) \
		--cv-splits $(CV_SPLITS) \
		--out-dir $(OUT_DIR) \
		2>&1 | tee $(OUT_DIR)/train_debug.log

# Quick smoke test
smoke-test:
	@echo "💨 Running smoke test..."
	@echo "import sys; print('Python:', sys.version)" | $(PYTHON)
	@$(PYTHON) -c "import pandas as pd; print('Pandas:', pd.__version__)"
	@$(PYTHON) -c "import numpy as np; print('NumPy:', np.__version__)"
	@$(PYTHON) -c "import lightgbm as lgb; print('LightGBM:', lgb.__version__)" 2>/dev/null || echo "⚠️  LightGBM not installed"
	@echo "✅ Environment check completed"

# Show pipeline status
status:
	@echo "📊 Pipeline Status"
	@echo "=================="
	@echo ""
	@echo "Configuration:"
	@echo "  Config file: $(CONFIG)"
	@echo "  Features:    $(FEATURES)"
	@echo "  CV splits:   $(CV_SPLITS)"
	@echo "  Output dir:  $(OUT_DIR)"
	@echo ""
	@echo "Prerequisites:"
	@if [ -f "$(FEATURES)" ]; then echo "  ✅ Features available"; else echo "  ❌ Features missing"; fi
	@if [ -f "$(CV_SPLITS)" ]; then echo "  ✅ CV splits available"; else echo "  ❌ CV splits missing"; fi
	@if [ -f "$(CONFIG)" ]; then echo "  ✅ Config available"; else echo "  ❌ Config missing"; fi
	@echo ""
	@echo "Training artifacts:"
	@if [ -d "$(OUT_DIR)" ]; then \
		echo "  📁 Output directory exists"; \
		ls -la $(OUT_DIR) | grep -E '\.(json|joblib|csv|html)$$' | wc -l | xargs echo "  📄 Artifact files:"; \
	else \
		echo "  📁 No training artifacts found"; \
	fi

# Extended help
help-extended:
	@echo "Quantile Bands Training Pipeline - Extended Help"
	@echo "==============================================="
	@echo ""
	@echo "This Makefile orchestrates the 02a training pipeline following"
	@echo "the specification for production-ready quantile regression with"
	@echo "conformal prediction."
	@echo ""
	@echo "Pipeline Flow:"
	@echo "  1. Prerequisites: 01_data_features must generate features_4H.parquet"
	@echo "  2. Training: LightGBM quantile models with CPCV validation"
	@echo "  3. Calibration: Conformal prediction with Mondrian conditioning"
	@echo "  4. QC: Monotonicity, coverage, and width coherence checks"
	@echo "  5. Artifacts: Models, calibrators, metrics, and metadata"
	@echo "  6. Reporting: HTML report with visualizations"
	@echo ""
	@echo "Key Features:"
	@echo "  - Anti-leakage CPCV with embargo for multi-horizon targets"
	@echo "  - Mondrian conformal prediction with time-decay weighting"
	@echo "  - Automated QC with quantile rearrangement"
	@echo "  - Deterministic artifacts with atomic writes"
	@echo "  - Comprehensive test suite"
	@echo ""
	@echo "Production Usage:"
	@echo "  make train CONFIG=configs/02a_production.yaml"
	@echo "  make all OUT_DIR=data/processed/models/$(shell date +%Y%m%d)"
	@echo ""
	@echo "Development Workflow:"
	@echo "  make dev-setup  # One-time setup"
	@echo "  make test       # Run unit tests"
	@echo "  make lint       # Code quality checks"
	@echo "  make train      # Full training pipeline"
	@echo "  make report     # Generate HTML report"