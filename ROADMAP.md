# Roadmap

This is a **living project**. Each release ships one genuinely new capability
*and* a benchmark showing it moved the needle, so the repo tells a story of
measured progress rather than sitting frozen. Nothing here is a commitment to a
date — we pick features off this shelf one at a time.

## How we work

- **One feature per iteration.** Understand → implement → benchmark → document →
  bump version → signed tag. No half-features on `main`.
- **Every release carries a number.** A new capability that changes behaviour
  gets a benchmark row and a `CHANGELOG.md` entry. Nothing ships without a
  before/after number against a baseline.
- **Honesty about novelty.** We separate "established technique, well
  implemented" from "genuinely new here," and we never inflate the latter.
- **Reproducible.** Fixed seeds, released data/generators, tests that assert the
  claimed improvement so regressions are caught.

## Shipped

- **v0.1.0 — baseline.** From-scratch NumPy local-linear-trend Kalman filter:
  predict/update (Joseph form), RTS smoother, missing-data handling, short-horizon
  forecasting. ~80% noise removed on a clean synthetic temperature series.
- **v0.2.0 — adaptive / robust.** `AdaptiveRobustKalmanFilter`: self-tuning `R`
  (robust MAD covariance-matching), Huber + χ² robust update, adaptive `Q`
  inflation, and a **novel outlier-vs-regime-shift router** (CUSUM of the
  robustly-weighted innovations). ~51% lower RMSE than v0.1.0 on a series with
  outliers + a regime shift, with no hand-set measurement noise. Includes the
  head-to-head demo and the academic paper in `paper/`.

## Planned

Ordered by rough value; we can reorder freely.

### v0.2.x — "running project" foundation (infra, no algorithm change)
- [ ] `CHANGELOG.md` (Keep-a-Changelog style), backfilled for v0.1.0 / v0.2.0.
- [ ] **GitHub Actions CI** — run `pytest` on every push/PR across Python 3.9–3.12.
- [ ] README status badges (CI, license, Python versions).
- [ ] Turn the two tags into proper **GitHub Releases** with notes + the figures.
- [ ] Optional: publish to TestPyPI so `pip install kalman-timeseries` works.

### v0.3.0 — EM auto-tuning of Q *and* R
- **What.** Learn the *full* noise model (both process noise `Q` and measurement
  noise `R`) by Expectation-Maximization over the RTS-smoothed states, instead of
  hand-setting `Q`. Truly zero-knob.
- **Why.** v0.2.0 self-tunes `R` but still takes `σ_level`, `σ_trend` by hand.
  This closes the "no tuning" story completely.
- **How.** E-step = RTS smoother (already implemented); M-step = closed-form
  covariance re-estimates from smoothed second moments; iterate to convergence.
- **Benchmark.** Show EM-learned `Q,R` matches or beats hand-tuned on the clean
  series, and recovers known parameters on data generated with a known `Q`.
- **Novelty.** Established (Shumway & Stoffer 1982); value is completeness +
  integration with the robust machinery. Honest label: "textbook, well done."

### v0.4.0 — heavy-tailed robustness (Student-t via variational Bayes)
- **What.** Replace the Gaussian observation model with a Student-t one, whose
  degrees-of-freedom are inferred, giving principled fat-tail robustness that
  subsumes the Huber gate.
- **Why.** Our χ²+Huber assumes Gaussian inliers; real sensor noise is
  heavy-tailed. The paper lists this as future work.
- **How.** Variational-Bayes update with a per-sample precision scaling
  (Gamma prior) — the standard VB robust KF.
- **Benchmark.** Contaminate inlier noise with a t-distribution; show lower RMSE
  than v0.2.0's hard gate, especially near the outlier/inlier boundary.
- **Novelty.** Established family (e.g. Agamennoni et al. 2012); our contribution
  is composing it with the self-tuning + router stack.

### v0.5.0 — BOCPD changepoint router (the flagship research step)
- **What.** Replace the *heuristic* CUSUM router with **Bayesian Online
  Changepoint Detection** (Adams & MacKay 2007) coupled to the filter: maintain a
  posterior over the "run length since last change" and let it drive when to
  reset/inflate.
- **Why.** This upgrades our signature piece from a threshold heuristic to a
  principled probabilistic detector with a tunable hazard rate and uncertainty.
- **How.** Run-length posterior over the filter's predictive likelihood; on a
  high changepoint probability, branch/reset the state — a proper
  Bayesian analogue of the `q_boost`.
- **Benchmark.** Multi-changepoint series with varied shift sizes; measure
  detection delay and false-alarm rate vs the CUSUM router; ROC curve.
- **Novelty.** **Highest of the set.** BOCPD and robust adaptive KFs both exist,
  but a BOCPD-driven router arbitrating between outlier-suppression and
  process-noise adaptation, on top of self-tuned noise, is a genuinely new
  combination worth writing up properly.

### v0.6.0 — real data + seasonal model + library benchmark
- **What.** (a) A seasonal structural component added to the state (local level +
  trend + seasonal), (b) validation on *real* datasets (e.g. NOAA daily
  temperature, an IoT/air-quality sensor feed, an FX/price series), (c) a
  reproducible benchmark table vs `filterpy`, `statsmodels` (UnobservedComponents),
  and `pykalman`.
- **Why.** Synthetic wins are suggestive; real-data wins + library baselines are
  credible.
- **Novelty.** None claimed — this is the credibility/validation milestone.

## Backlog / ideas (unscheduled)
- KalmanNet-style learned Kalman gain (small NN) — "deep Kalman filter" branch.
- Interacting Multiple Model (IMM) — blend constant-velocity / -acceleration models.
- Square-root / UD-factorized implementation for numerical robustness at scale.
- Multivariate (m > 1) measurements: full-covariance MAD analogue + Mahalanobis gate.
- A `streamlit`/notebook interactive demo (tune knobs, watch the filter react).
- Ablation study: contribution of each of the four v0.2.0 components in isolation.

## Evaluation standards (apply to every release)
- Multiple random seeds; report mean ± std, not a single lucky run.
- An `assert`-backed test that encodes the claimed improvement.
- A figure and a table added to `paper/` where the change is substantive.
- CHANGELOG entry + signed tag + (from v0.2.x) green CI.
