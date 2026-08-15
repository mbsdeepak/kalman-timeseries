# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
See [ROADMAP.md](ROADMAP.md) for planned work.

## [Unreleased]

### Added
- `CHANGELOG.md` (this file), backfilled for the shipped releases.
- **GitHub Actions CI** (`.github/workflows/ci.yml`): runs the test suite on
  every push and pull request across Python 3.9–3.13.
- README status badges (CI, license, Python versions, changelog).
- `ROADMAP.md` describing the planned v0.3.0–v0.6.0 improvements and the
  project's working cadence.

## [0.2.0] - 2026-08-15

Adaptive / robust filter. **~51% lower RMSE than v0.1.0** on a series
contaminated with outliers and a regime shift, with no hand-set measurement
noise (learned R = 5.96 vs. true variance 6.25).

### Added
- `AdaptiveRobustKalmanFilter` (`kalman/adaptive.py`) and the
  `adaptive_local_linear_trend(...)` builder, combining:
  - **Self-tuning measurement noise** `R` via robust (MAD) covariance matching —
    no manual noise knob.
  - **Robust measurement update**: Huber down-weighting gated by a χ² test on the
    normalized innovation.
  - **Adaptive process-noise inflation** on detected change.
  - **Outlier-vs-regime-shift router** (new): a CUSUM/EMA of the
    robustly-weighted innovations that decides whether a large innovation is a
    glitch to suppress or a real level shift to follow.
- `examples/compare_versions.py` — v0.1.0 vs v0.2.0 head-to-head on a hard series.
- `noisy_temperature_hard(...)` generator (noise + outliers + regime shift).
- Academic paper: `paper/kalman_improvements.tex` and compiled `.pdf`.
- 6 new tests (13 total): self-tuning R, outlier down-weighting, regime catch-up,
  beats-baseline, gaps, scalar-observation validation.

### Changed
- `__version__` bumped to `0.2.0`; package now exports the adaptive filter.

## [0.1.0] - 2026-08-15

Baseline local-linear-trend Kalman filter. ~80% noise removed on a clean
synthetic temperature series (RMSE 2.35 → 0.47 with the smoother).

### Added
- `KalmanFilter` (`kalman/filter.py`): general linear filter with predict/update
  (Joseph-form covariance), RTS smoother, multi-step forecasting, and
  missing-data handling.
- `local_linear_trend(...)` structural time-series model (`kalman/models.py`).
- `examples/synthetic.py` reproducible noisy-temperature generator and
  `examples/denoise_temperature.py` end-to-end demo.
- 7 tests covering correctness, noise reduction, smoother, gaps, and forecasting.
- Packaging (`pyproject.toml`), MIT license, and README.

[Unreleased]: https://github.com/mbsdeepak/kalman-timeseries/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mbsdeepak/kalman-timeseries/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mbsdeepak/kalman-timeseries/releases/tag/v0.1.0
