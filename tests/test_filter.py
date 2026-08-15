"""Tests for the Kalman filter and the local-linear-trend model."""

import numpy as np
import pytest

from kalman import (
    KalmanFilter,
    adaptive_local_linear_trend,
    local_linear_trend,
)
from examples.synthetic import noisy_temperature, noisy_temperature_hard


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


# --------------------------------------------------------------------------- #
# Adaptive / robust filter (v0.2.0)                                           #
# --------------------------------------------------------------------------- #

def test_adaptive_requires_scalar_observation():
    from kalman import AdaptiveRobustKalmanFilter
    with pytest.raises(ValueError):
        AdaptiveRobustKalmanFilter(
            transition_matrix=np.eye(2),
            observation_matrix=np.eye(2),   # (2,2) -> not scalar measurements
            process_covariance=np.eye(2),
            initial_state=np.zeros(2),
            initial_covariance=np.eye(2),
        )


def test_adaptive_self_tunes_R_near_truth():
    """With no R supplied, the online estimate should land near true variance."""
    noise_std = 2.5
    _, _, z = noisy_temperature(noise_std=noise_std)
    kf = adaptive_local_linear_trend(initial_level=z[0])
    res = kf.filter(z)
    true_var = noise_std ** 2
    assert 0.5 * true_var < res.r_estimates[-1] < 1.6 * true_var


def test_adaptive_beats_baseline_on_hard_series():
    """On noise + outliers + a regime shift, adaptive should clearly win."""
    _, truth, z = noisy_temperature_hard()

    base = local_linear_trend(observation_noise=2.5, level_process_noise=0.5,
                              trend_process_noise=0.01, initial_level=z[0])
    base_rmse = rmse(base.filter(z).levels(base.H), truth)

    adapt = adaptive_local_linear_trend(initial_level=z[0])
    adapt_rmse = rmse(adapt.filter(z).levels(adapt.H), truth)

    assert adapt_rmse < 0.8 * base_rmse   # at least 20% better


def test_adaptive_flags_and_downweights_outliers():
    _, truth, z = noisy_temperature_hard(n_outliers=10)
    kf = adaptive_local_linear_trend(initial_level=z[0])
    res = kf.filter(z)
    # Some samples must be flagged as outliers, and every flagged sample must
    # have received a weight strictly below 1.
    flagged = np.where(res.outlier_flags)[0]
    assert flagged.size >= 5
    assert all(res.weights[i] < 1.0 for i in flagged)


def test_adaptive_catches_regime_shift():
    """After a level shift the estimate should re-converge quickly."""
    _, truth, z = noisy_temperature_hard(n_outliers=0, shift_day=180, shift_size=10.0)
    kf = adaptive_local_linear_trend(initial_level=z[0])
    res = kf.filter(z)
    level = res.levels(kf.H)
    # 15 days after the shift, the estimate should be within 2 deg of truth,
    # and the regime catch-up should have engaged at least once.
    k = 180 + 15
    assert abs(level[k] - truth[k]) < 2.0
    assert any(res.regime_flags)


def test_adaptive_handles_missing_measurements():
    _, _, z = noisy_temperature_hard()
    z = list(z)
    for i in range(60, 90):
        z[i] = None
    kf = adaptive_local_linear_trend(initial_level=z[0])
    res = kf.filter(z)
    assert len(res.states) == len(z)
    assert np.isfinite((kf.H @ res.states[-1]).item())
