"""
Auto-Sklearn2 - A Python 3.11+ compatible version of auto-sklearn
"""

from .classifier import AutoSklearnClassifier
from .regressor import AutoSklearnRegressor

__version__ = "0.1.2"
__all__ = ["AutoSklearnClassifier", "AutoSklearnRegressor"]
