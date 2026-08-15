"""Tests for the Kalman filter and the local-linear-trend model."""

import numpy as np
import pytest

from kalman import KalmanFilter, local_linear_trend
from examples.synthetic import noisy_temperature


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_shape_validation():
    with pytest.raises(ValueError):
        KalmanFilter(
            transition_matrix=np.eye(2),
            observation_matrix=np.array([[1.0, 0.0]]),
            process_covariance=np.eye(3),   # wrong size
            measurement_covariance=np.eye(1),
            initial_state=np.zeros(2),
            initial_covariance=np.eye(2),
        )


def test_constant_signal_converges_to_true_mean():
    """Filtering noisy samples of a constant should recover the constant."""
    true_value = 7.0
    rng = np.random.default_rng(0)
    z = true_value + rng.normal(0, 1.0, size=500)

    kf = local_linear_trend(
        observation_noise=1.0,
        level_process_noise=1e-4,   # signal is (almost) constant
        trend_process_noise=1e-6,
        initial_level=0.0,
    )
    result = kf.filter(z)
    final_level = (kf.H @ result.states[-1]).item()
    assert abs(final_level - true_value) < 0.2


def test_covariance_stays_symmetric_and_psd():
    _, _, z = noisy_temperature(n_days=120)
    kf = local_linear_trend(observation_noise=2.5, level_process_noise=0.5,
                            trend_process_noise=0.01, initial_level=z[0])
    for P in kf.filter(z).covariances:
        assert np.allclose(P, P.T, atol=1e-8)          # symmetric
        assert np.min(np.linalg.eigvalsh(P)) > -1e-8   # positive semidefinite


def test_filter_reduces_noise():
    _, truth, z = noisy_temperature(noise_std=2.5)
    kf = local_linear_trend(observation_noise=2.5, level_process_noise=0.5,
                            trend_process_noise=0.01, initial_level=z[0])
    filtered = kf.filter(z)
    level = filtered.levels(kf.H)
    assert rmse(level, truth) < rmse(z, truth)


def test_smoother_beats_filter():
    """Using the whole series (offline) should be at least as good as causal."""
    _, truth, z = noisy_temperature(noise_std=2.5)
    kf = local_linear_trend(observation_noise=2.5, level_process_noise=0.5,
                            trend_process_noise=0.01, initial_level=z[0])
    filtered = kf.filter(z)
    smoothed = kf.smooth(filtered)
    assert rmse(smoothed.levels(kf.H), truth) <= rmse(filtered.levels(kf.H), truth)


def test_handles_missing_measurements():
    """None / NaN gaps should be predicted through without crashing."""
    _, truth, z = noisy_temperature(n_days=200)
    z = list(z)
    for i in range(50, 80):        # a 30-day sensor outage
        z[i] = None
    kf = local_linear_trend(observation_noise=2.5, level_process_noise=0.5,
                            trend_process_noise=0.01, initial_level=z[0])
    filtered = kf.filter(z)
    assert len(filtered.states) == len(z)
    assert np.isfinite((kf.H @ filtered.states[-1]).item())


def test_forecast_shape_and_growing_uncertainty():
    _, _, z = noisy_temperature(n_days=200)
    kf = local_linear_trend(observation_noise=2.5, level_process_noise=0.5,
                            trend_process_noise=0.01, initial_level=z[0])
    filtered = kf.filter(z)
    preds, variances = kf.forecast(filtered.states[-1], filtered.covariances[-1], steps=15)
    assert preds.shape == (15,)
    # Uncertainty should not shrink as we predict further out.
    assert variances[-1] >= variances[0]
