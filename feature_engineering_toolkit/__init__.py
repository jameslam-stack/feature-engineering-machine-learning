"""Feature Engineering Toolkit for Machine Learning.

Create, select, and transform features to improve ML model performance.
Handles scaling, encoding, interaction terms, and importance analysis.
"""

from feature_engineering_toolkit.pipeline import FeatureEngineeringPipeline
from feature_engineering_toolkit.ml_model import MLDeveloperPipeline
from feature_engineering_toolkit.config import load_config

__version__ = "1.0.0"

__all__ = [
    "FeatureEngineeringPipeline",
    "MLDeveloperPipeline",
    "load_config",
]