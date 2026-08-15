"""A NumPy-only Kalman filter, and structural time-series models built on it."""

from .filter import FilterResult, KalmanFilter
from .models import local_linear_trend
from .adaptive import (
    AdaptiveResult,
    AdaptiveRobustKalmanFilter,
    adaptive_local_linear_trend,
)

__all__ = [
    "KalmanFilter",
    "FilterResult",
    "local_linear_trend",
    "AdaptiveRobustKalmanFilter",
    "AdaptiveResult",
    "adaptive_local_linear_trend",
]
__version__ = "0.2.0"
