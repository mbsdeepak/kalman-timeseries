"""Adaptive, self-tuning, robust Kalman filter (the v0.2.0 improvement).

The baseline `KalmanFilter` in `filter.py` is the textbook algorithm: you hand
it fixed noise matrices `Q` and `R` and it trusts every measurement equally.
That has two well-known failure modes:

  1. **You have to hand-tune the noise.** Set `R` wrong and the output is
     either sluggish or noisy. In practice the true noise is unknown and can
     drift over time.
  2. **One bad sample wrecks the estimate.** A single sensor glitch enters the
     update with full weight and yanks the state off course, and the filter
     lags badly when the underlying signal genuinely jumps.

`AdaptiveRobustKalmanFilter` addresses both by folding three established ideas
from the adaptive-filtering literature into one online filter, plus a small new
routing rule that ties them together:

  (A) Self-tuning measurement noise via *innovation-based adaptive estimation*
      (covariance matching). `R` is re-estimated from a sliding window of
      innovations instead of being supplied by hand.
      Ref: Zhang et al., "On the Identification of Noise Covariances and
      Adaptive Kalman Filtering: A New Look at a 50-Year-Old Problem" (2020).

  (B) Robust measurement update via a Huber-style weight gated by a chi-square
      test on the normalized innovation squared (Mahalanobis distance). An
      outlier is down-weighted, not trusted.
      Ref: Wang, Li & Fang, "Robust Gaussian Kalman Filter With Outlier
      Detection" (IEEE SP Letters, 2018).

  (C) Adaptive process-noise inflation. When innovations are large, `Q` is
      temporarily inflated so the filter can react to real change quickly.

  (*) The new bit — an **outlier-vs-regime-shift router**. A large innovation is
      ambiguous: it could be a one-off glitch (down-weight it, path B) or the
      first sample of a genuine level shift (trust it and inflate Q, path C).
      We disambiguate by *sign persistence*: an EMA/CUSUM of the robustly-
      weighted innovations. Isolated outliers have random signs and mostly
      cancel in the EMA (and are pre-shrunk by their Huber weight), but a real
      level shift drives a run of same-sign innovations whose EMA grows past
      regime_k * sqrt(R). When it does, catch-up engages: Q is inflated so the
      filter jumps to the new level, then decays back. This is what lets the
      filter tell "sensor glitch" (suppress) from "the world changed" (follow).

Everything here is still a linear-Gaussian filter; only the *weighting* and the
*noise matrices* adapt online.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .filter import FilterResult


@dataclass
class AdaptiveResult(FilterResult):
    """FilterResult plus per-step diagnostics from the adaptive machinery."""

    weights: List[float] = field(default_factory=list)       # robust weight in (0, 1]
    r_estimates: List[float] = field(default_factory=list)   # online estimate of R
    q_scales: List[float] = field(default_factory=list)      # Q inflation factor
    regime_flags: List[bool] = field(default_factory=list)   # regime-shift catch-up active?
    outlier_flags: List[bool] = field(default_factory=list)  # sample down-weighted as outlier?


class AdaptiveRobustKalmanFilter:
    """Self-tuning robust Kalman filter for a scalar-measurement model.

    Assumes a single measurement per step (m = 1), which covers the
    local-linear-trend denoising use case and keeps the chi-square gate and the
    R-estimator simple. The state dimension n is arbitrary.

    Args:
        transition_matrix:  F, shape (n, n)
        observation_matrix: H, shape (1, n)
        process_covariance: Q, shape (n, n) — the *base* process noise; it gets
            inflated adaptively, never shrunk below this.
        initial_state:      x_0, shape (n,)
        initial_covariance: P_0, shape (n, n)
        measurement_noise:  initial guess for R (scalar variance). If None, it
            is bootstrapped from the data inside `filter()` (see `_bootstrap_R`).
        adapt_window:       sliding-window length for the R estimate.
        gate:               chi-square threshold on normalized innovation². The
            default 6.635 is the 99% point of chi-square with 1 dof.
        bias_lambda:        EMA weight for the running innovation-bias (CUSUM)
            statistic used to spot a regime shift. Higher ⇒ reacts faster.
        regime_k:           regime catch-up engages when |EMA innovation bias|
            exceeds regime_k * sqrt(R). Symmetric outliers cancel in the EMA; a
            genuine level shift builds a persistent same-sign bias that trips it.
        q_boost:            factor Q is multiplied by when catch-up engages.
        q_decay:            per-step geometric decay of the Q inflation back to 1.
        r_smoothing:        EMA factor for the online R estimate (0..1).
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        observation_matrix: np.ndarray,
        process_covariance: np.ndarray,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        measurement_noise: Optional[float] = None,
        adapt_window: int = 30,
        gate: float = 6.635,
        bias_lambda: float = 0.35,
        regime_k: float = 1.5,
        q_boost: float = 100.0,
        q_decay: float = 0.6,
        r_smoothing: float = 0.05,
    ) -> None:
        self.F = np.atleast_2d(np.asarray(transition_matrix, dtype=float))
        self.H = np.atleast_2d(np.asarray(observation_matrix, dtype=float))
        self.Q_base = np.atleast_2d(np.asarray(process_covariance, dtype=float))
        self.x0 = np.asarray(initial_state, dtype=float).reshape(-1)
        self.P0 = np.atleast_2d(np.asarray(initial_covariance, dtype=float))

        n = self.F.shape[0]
        if self.H.shape != (1, n):
            raise ValueError("observation_matrix H must be (1, n) — scalar measurements only")
        if self.Q_base.shape != (n, n):
            raise ValueError("process_covariance Q must be (n, n)")
        self.n = n

        self.R_init = None if measurement_noise is None else float(measurement_noise)
        self.adapt_window = int(adapt_window)
        self.gate = float(gate)
        self.bias_lambda = float(bias_lambda)
        self.regime_k = float(regime_k)
        self.q_boost = float(q_boost)
        self.q_decay = float(q_decay)
        self.r_smoothing = float(r_smoothing)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _bootstrap_R(measurements: np.ndarray) -> float:
        """Robust initial R from the measurements themselves.

        Uses the median absolute deviation of first differences. For a smooth
        signal the differences are dominated by noise, and MAD ignores the
        occasional outlier, so 0.5 * (1.4826 * MAD(diff))**2 is a decent,
        tuning-free starting guess for the measurement variance.
        """
        z = np.asarray([v for v in measurements if v is not None and np.isfinite(v)], dtype=float)
        if z.size < 3:
            return 1.0
        diffs = np.diff(z)
        mad = np.median(np.abs(diffs - np.median(diffs)))
        sigma = 1.4826 * mad
        return max(1e-6, 0.5 * sigma ** 2)

    # -- main pass -------------------------------------------------------------

    def filter(self, measurements) -> AdaptiveResult:
        measurements = list(measurements)
        R = self.R_init if self.R_init is not None else self._bootstrap_R(measurements)

        x, P = self.x0.copy(), self.P0.copy()
        q_scale = 1.0
        ewma_bias = 0.0                     # running innovation bias (CUSUM/EMA)
        innov_sq_window: List[float] = []   # innovation² of *inlier* samples
        prior_var_window: List[float] = []  # matching H P^- H^T of those samples

        out = AdaptiveResult()

        for z in measurements:
            # --- predict (with the current Q inflation) ---
            Q_eff = self.Q_base * q_scale
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + Q_eff
            out.pred_states.append(x_pred)
            out.pred_covs.append(P_pred)

            missing = z is None or (np.ndim(z) == 0 and not np.isfinite(z))
            prior_var = (self.H @ P_pred @ self.H.T).item()  # H P^- H^T (scalar)

            if missing:
                x, P = x_pred, P_pred
                weight, is_outlier, is_regime = 1.0, False, False
                out.innovations.append(np.array([np.nan]))
            else:
                innovation = float(z) - (self.H @ x_pred).item()
                S = prior_var + R
                d2 = innovation * innovation / S          # normalized innovation² (Mahalanobis)

                # --- path B: chi-square gate -> Huber weight ---
                if d2 <= self.gate:
                    weight, is_outlier = 1.0, False
                else:
                    weight, is_outlier = self.gate / d2, True   # down-weight

                # --- the router: is this a glitch or a real shift? ---
                # EMA of robustly-weighted innovations. Outliers are pre-shrunk
                # by `weight` and have random signs, so they wash out; a genuine
                # level shift produces a run of same-sign innovations that builds
                # a persistent bias.
                ewma_bias = (1 - self.bias_lambda) * ewma_bias + self.bias_lambda * (weight * innovation)

                is_regime = abs(ewma_bias) > self.regime_k * np.sqrt(R)
                if is_regime:
                    # path C: the world changed. Trust the measurement and
                    # inflate Q so the state jumps to the new level.
                    weight, is_outlier = 1.0, False
                    q_scale = self.q_boost
                    ewma_bias = 0.0                          # consume the evidence

                # Robust update: down-weighting == inflating R by 1/weight.
                R_eff = R / weight
                S_eff = prior_var + R_eff
                K = (P_pred @ self.H.T) / S_eff          # (n, 1)
                x = x_pred + (K * innovation).reshape(-1)
                A = np.eye(self.n) - K @ self.H
                P = A @ P_pred @ A.T + (K @ K.T) * R_eff  # Joseph form
                out.innovations.append(np.array([innovation]))

                # --- online R estimate (robust covariance matching) ---
                # Feed every non-regime innovation into the window and estimate
                # the innovation scale with the MAD (median absolute deviation),
                # which is immune to the outliers — so we neither truncate the
                # tail (biasing R low, as gated mean-of-squares does) nor let a
                # glitch inflate R. Then E[innov²] = var(innov) = R + HP⁻Hᵀ, so
                # R ≈ (1.4826·MAD)² − mean(HP⁻Hᵀ).
                if not is_regime:
                    innov_sq_window.append(innovation)      # raw innovations
                    prior_var_window.append(prior_var)
                    if len(innov_sq_window) > self.adapt_window:
                        innov_sq_window.pop(0)
                        prior_var_window.pop(0)
                if len(innov_sq_window) >= 8:
                    arr = np.asarray(innov_sq_window)
                    mad = np.median(np.abs(arr - np.median(arr)))
                    innov_var = (1.4826 * mad) ** 2
                    r_hat = max(1e-6, innov_var - np.mean(prior_var_window))
                    R = (1 - self.r_smoothing) * R + self.r_smoothing * r_hat

            # decay the Q inflation back toward the base level
            if q_scale > 1.0:
                q_scale = max(1.0, q_scale * self.q_decay)

            out.states.append(x)
            out.covariances.append(P)
            out.weights.append(weight)
            out.r_estimates.append(R)
            out.q_scales.append(q_scale)
            out.regime_flags.append(is_regime)
            out.outlier_flags.append(is_outlier)

        return out

    def forecast(self, x: np.ndarray, P: np.ndarray, steps: int, R: Optional[float] = None):
        """Forecast `steps` ahead (uses the base Q; no measurements)."""
        if R is None:
            R = self.R_init if self.R_init is not None else 1.0
        preds = np.empty(steps)
        variances = np.empty(steps)
        for i in range(steps):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q_base
            preds[i] = (self.H @ x).item()
            variances[i] = (self.H @ P @ self.H.T).item() + R
        return preds, variances


def adaptive_local_linear_trend(
    initial_level: float = 0.0,
    initial_trend: float = 0.0,
    level_process_noise: float = 0.5,
    trend_process_noise: float = 0.01,
    measurement_noise: Optional[float] = None,
    initial_uncertainty: float = 1e3,
    **kwargs,
) -> AdaptiveRobustKalmanFilter:
    """Build the adaptive/robust filter for the local-linear-trend model.

    Unlike `local_linear_trend`, `measurement_noise` is optional — leave it None
    and the filter bootstraps and then continuously re-estimates it from the
    data. Extra keyword args (gate, q_boost, regime_patience, ...) pass straight
    through to `AdaptiveRobustKalmanFilter`.
    """
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[level_process_noise ** 2, 0.0],
                  [0.0, trend_process_noise ** 2]])
    x0 = np.array([initial_level, initial_trend])
    P0 = np.eye(2) * initial_uncertainty
    return AdaptiveRobustKalmanFilter(
        transition_matrix=F,
        observation_matrix=H,
        process_covariance=Q,
        initial_state=x0,
        initial_covariance=P0,
        measurement_noise=measurement_noise,
        **kwargs,
    )
