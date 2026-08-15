"""A NumPy-only Kalman filter, and structural time-series models built on it."""

from .filter import FilterResult, KalmanFilter
from .models import local_linear_trend

__all__ = ["KalmanFilter", "FilterResult", "local_linear_trend"]
__version__ = "0.1.0"
