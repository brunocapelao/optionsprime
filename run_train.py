#!/usr/bin/env python3
"""
Entry point for training pipeline
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run
from src.quant_bands.train import main

if __name__ == "__main__":
    main()