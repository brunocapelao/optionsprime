"""
Setup script for quantile bands training pipeline.
"""
from setuptools import setup, find_packages

setup(
    name="quant-bands",
    version="0.1.0",
    description="Quantile Bands Training Pipeline with Anti-Leakage CPCV and Conformal Prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.20.0",
        "lightgbm>=4.0.0",
        "scikit-learn>=1.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.60.0",
        "optuna>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "quant-bands-train=quant_bands.train:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)