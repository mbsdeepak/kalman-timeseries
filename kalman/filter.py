"""A small, dependency-light (NumPy-only) linear Kalman filter.

The implementation is intentionally general: it works for any linear-Gaussian
state-space model of the form

    x_k = F x_{k-1} + w,   w ~ N(0, Q)      (state transition)
    z_k = H x_k     + v,   v ~ N(0, R)      (measurement)

You supply the matrices; the filter does the bookkeeping. A Rauch-Tung-Striebel
(RTS) smoother is included because, for offline denoising, using future
observations to refine past estimates gives a noticeably cleaner signal than
the forward filter alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class FilterResult:
    """Container for the forward-pass output.

    Each attribute is a list with one entry per time step.

    Attributes:
        states:       posterior state means  x_{k|k}
        covariances:  posterior state covariances  P_{k|k}
        pred_states:  prior state means  x_{k|k-1}   (needed by the smoother)
        pred_covs:    prior state covariances  P_{k|k-1}
        innovations:  measurement residuals  z_k - H x_{k|k-1}
    """

    states: List[np.ndarray] = field(default_factory=list)
    covariances: List[np.ndarray] = field(default_factory=list)
    pred_states: List[np.ndarray] = field(default_factory=list)
    pred_covs: List[np.ndarray] = field(default_factory=list)
    innovations: List[np.ndarray] = field(default_factory=list)

    def levels(self, observation_matrix: np.ndarray) -> np.ndarray:
        """Project each posterior state back into measurement space (H x)."""
        return np.array([(observation_matrix @ x).item() for x in self.states])


class KalmanFilter:
    """A time-invariant linear Kalman filter.

    Args:
        transition_matrix:  F, shape (n, n)
        observation_matrix: H, shape (m, n)
        process_covariance: Q, shape (n, n)
        measurement_covariance: R, shape (m, m)
        initial_state:      x_0, shape (n,)
        initial_covariance: P_0, shape (n, n)
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        observation_matrix: np.ndarray,
        process_covariance: np.ndarray,
        measurement_covariance: np.ndarray,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> None:
        self.F = np.atleast_2d(np.asarray(transition_matrix, dtype=float))
        self.H = np.atleast_2d(np.asarray(observation_matrix, dtype=float))
        self.Q = np.atleast_2d(np.asarray(process_covariance, dtype=float))
        self.R = np.atleast_2d(np.asarray(measurement_covariance, dtype=float))
        self.x0 = np.asarray(initial_state, dtype=float).reshape(-1)
        self.P0 = np.atleast_2d(np.asarray(initial_covariance, dtype=float))

        n = self.F.shape[0]
        if self.F.shape != (n, n):
            raise ValueError("transition_matrix F must be square (n, n)")
        if self.H.shape[1] != n:
            raise ValueError("observation_matrix H must have n columns")
        if self.Q.shape != (n, n):
            raise ValueError("process_covariance Q must be (n, n)")
        m = self.H.shape[0]
        if self.R.shape != (m, m):
            raise ValueError("measurement_covariance R must be (m, m)")
        if self.x0.shape != (n,):
            raise ValueError("initial_state x0 must have length n")
        if self.P0.shape != (n, n):
            raise ValueError("initial_covariance P0 must be (n, n)")

        self.n = n
        self.m = m

    # -- single-step primitives ------------------------------------------------

    def predict(self, x: np.ndarray, P: np.ndarray):
        """Time update: propagate the state one step forward."""
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray):
        """Measurement update: correct the prediction using observation z."""
        z = np.asarray(z, dtype=float).reshape(-1)
        innovation = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        x_new = x_pred + K @ innovation
        # Joseph form keeps P symmetric and positive-definite under rounding.
        I = np.eye(self.n)
        A = I - K @ self.H
        P_new = A @ P_pred @ A.T + K @ self.R @ K.T
        return x_new, P_new, innovation

    # -- full passes -----------------------------------------------------------

    def filter(self, measurements) -> FilterResult:
        """Run the forward filter over a sequence of measurements.

        `measurements` may contain None to represent a gap (no observation at
        that step); the filter then predicts through the gap without updating.
        """
        x, P = self.x0.copy(), self.P0.copy()
        result = FilterResult()

        for z in measurements:
            x_pred, P_pred = self.predict(x, P)
            result.pred_states.append(x_pred)
            result.pred_covs.append(P_pred)

            if z is None or (np.ndim(z) == 0 and np.isnan(z)):
                x, P = x_pred, P_pred
                result.innovations.append(np.full(self.m, np.nan))
            else:
                x, P, innovation = self.update(x_pred, P_pred, z)
                result.innovations.append(innovation)

            result.states.append(x)
            result.covariances.append(P)

        return result

    def smooth(self, result: Optional[FilterResult] = None, measurements=None) -> FilterResult:
        """RTS smoother: refine the filtered estimates using the whole series.

        Pass either a `FilterResult` from `filter()`, or raw `measurements`
        (in which case `filter()` is run first).
        """
        if result is None:
            if measurements is None:
                raise ValueError("provide either a FilterResult or measurements")
            result = self.filter(measurements)

        n_steps = len(result.states)
        smoothed = FilterResult()
        smoothed.states = [None] * n_steps
        smoothed.covariances = [None] * n_steps

        # Last smoothed estimate equals the last filtered estimate.
        smoothed.states[-1] = result.states[-1]
        smoothed.covariances[-1] = result.covariances[-1]

        for k in range(n_steps - 2, -1, -1):
            P_filt = result.covariances[k]
            P_pred_next = result.pred_covs[k + 1]
            # Smoother gain C = P_k F^T (P_{k+1|k})^-1
            C = P_filt @ self.F.T @ np.linalg.inv(P_pred_next)
            x_s = result.states[k] + C @ (smoothed.states[k + 1] - result.pred_states[k + 1])
            P_s = P_filt + C @ (smoothed.covariances[k + 1] - P_pred_next) @ C.T
            smoothed.states[k] = x_s
            smoothed.covariances[k] = P_s

        return smoothed

    def forecast(self, x: np.ndarray, P: np.ndarray, steps: int):
        """Forecast `steps` ahead from state (x, P) with no new measurements.

        Returns (predicted measurements, measurement variances) as arrays of
        length `steps`.
        """
        preds = np.empty(steps)
        variances = np.empty(steps)
        for i in range(steps):
            x, P = self.predict(x, P)
            preds[i] = (self.H @ x).item()
            variances[i] = (self.H @ P @ self.H.T + self.R).item()
        return preds, variances
