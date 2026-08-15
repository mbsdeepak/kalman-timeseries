"""Ready-made state-space models built on top of `KalmanFilter`.

Right now there is one: the *local linear trend* model, a classic structural
time-series model that decomposes a noisy 1-D signal into a slowly varying
`level` and a `trend` (slope). It is a natural fit for denoising daily
temperature, sensor readings, or any smooth-ish series corrupted by noise, and
its two-component state makes short-horizon forecasting trivial.
"""

from __future__ import annotations

import numpy as np

from .filter import KalmanFilter


def local_linear_trend(
    observation_noise: float,
    level_process_noise: float,
    trend_process_noise: float,
    initial_level: float = 0.0,
    initial_trend: float = 0.0,
    initial_uncertainty: float = 1e3,
) -> KalmanFilter:
    """Build a local-linear-trend Kalman filter.

    State vector is [level, trend]:

        level_k = level_{k-1} + trend_{k-1} + noise
        trend_k = trend_{k-1}               + noise
        z_k     = level_k                   + measurement noise

    Args:
        observation_noise:    std-dev of the measurement noise (R = this**2).
            Larger -> trust the model more, smooth harder.
        level_process_noise:  std-dev of level innovations. Larger -> track
            fast changes; smaller -> smoother output.
        trend_process_noise:  std-dev of trend innovations. Controls how quickly
            the slope is allowed to change.
        initial_level:        starting guess for the level.
        initial_trend:        starting guess for the slope (per step).
        initial_uncertainty:  large value = "I don't trust the initial guess",
            letting the first few measurements dominate.
    """
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[level_process_noise ** 2, 0.0],
                  [0.0, trend_process_noise ** 2]])
    R = np.array([[observation_noise ** 2]])
    x0 = np.array([initial_level, initial_trend])
    P0 = np.eye(2) * initial_uncertainty

    return KalmanFilter(
        transition_matrix=F,
        observation_matrix=H,
        process_covariance=Q,
        measurement_covariance=R,
        initial_state=x0,
        initial_covariance=P0,
    )
