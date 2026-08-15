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
