#!/usr/bin/env python3
"""
Entry point for the quantile bands training module.
Allows execution as: python -m quant_bands.train
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import and run main
from quant_bands.train import main

if __name__ == '__main__':
    main()