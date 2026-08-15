"""Generate a synthetic 'noisy daily temperature' series.

We build a ground-truth signal with a seasonal swing plus a slow warming
drift, then corrupt it with Gaussian measurement noise. Keeping the data
synthetic makes the demo fully reproducible with no downloads, and lets us
compare the filter's output against a known truth.
"""

from __future__ import annotations

import numpy as np


def noisy_temperature(
    n_days: int = 365,
    base: float = 12.0,
    seasonal_amplitude: float = 10.0,
    warming_per_year: float = 1.5,
    noise_std: float = 2.5,
    seed: int = 42,
):
    """Return (days, truth, measurements).

    Args:
        n_days:             number of daily samples.
        base:               mean temperature (deg C).
        seasonal_amplitude: peak-to-mean seasonal swing (deg C).
        warming_per_year:   linear drift added over 365 days (deg C).
        noise_std:          std-dev of sensor noise (deg C).
        seed:               RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)

    seasonal = seasonal_amplitude * np.sin(2 * np.pi * (days - 80) / 365.0)
    drift = warming_per_year * days / 365.0
    truth = base + seasonal + drift

    measurements = truth + rng.normal(0.0, noise_std, size=n_days)
    return days, truth, measurements


def noisy_temperature_hard(
    n_days: int = 365,
    noise_std: float = 2.5,
    n_outliers: int = 12,
    outlier_scale: float = 12.0,
    shift_day: int = 220,
    shift_size: float = 8.0,
    seed: int = 7,
):
    """A deliberately nasty version of the temperature series.

    On top of the smooth seasonal signal we add the two things that break a
    naive Kalman filter:

      * **outliers** — `n_outliers` random days get a large spike (a stuck /
        glitching sensor), size ~ `outlier_scale` * noise_std.
      * **a regime shift** — at `shift_day` the true temperature jumps by
        `shift_size` degrees (e.g. the sensor was relocated), and stays there.

    Returns (days, truth, measurements). `truth` already includes the shift, so
    RMSE is measured against the real post-shift signal.
    """
    days, truth, measurements = noisy_temperature(
        n_days=n_days, noise_std=noise_std, seed=seed
    )
    truth = truth.copy()
    measurements = measurements.copy()

    # Regime shift: a permanent step change in the underlying signal.
    if 0 <= shift_day < n_days:
        truth[shift_day:] += shift_size
        measurements[shift_day:] += shift_size

    # Outliers: large spikes on random days.
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(n_days, size=min(n_outliers, n_days), replace=False)
    signs = rng.choice([-1.0, 1.0], size=idx.size)
    measurements[idx] += signs * outlier_scale * noise_std

    return days, truth, measurements
