# Kalman Time Series

Denoise and forecast a noisy 1-D time series with a **Kalman filter**, written
from scratch in NumPy — no `filterpy`, no `statsmodels`, no black boxes. The
worked example takes a year of noisy daily temperature readings and recovers
the underlying signal, then forecasts the next 30 days.

![Kalman filtering of a noisy temperature series](assets/denoise.png)

On the synthetic benchmark (sensor noise σ = 2.5 °C), the filter cuts the error
against ground truth by ~64% online and ~80% offline:

| Signal                          | RMSE vs truth | Notes                    |
|---------------------------------|---------------|--------------------------|
| Raw measurements                | 2.35 °C       | what the sensor gives you |
| Kalman filtered (causal/online) | 0.85 °C       | uses only past + present  |
| RTS smoothed (offline)          | 0.47 °C       | uses the whole series     |

## The problem

A cheap sensor reports a daily value that is the true quantity plus noise. You
want (a) the best estimate of the *true* value at each day, and (b) a short
forecast with honest uncertainty. This is exactly what a Kalman filter is for:
it maintains a probabilistic estimate of a hidden state, predicting it forward
and correcting it each time a noisy measurement arrives.

## The model — local linear trend

We model the hidden state as a slowly-varying **level** plus a **trend**
(slope). This is a classic *structural time-series* model:

```
state x = [level, trend]

level_k = level_{k-1} + trend_{k-1} + noise      (level drifts by its slope)
trend_k =               trend_{k-1} + noise      (slope drifts slowly)
z_k     = level_k                   + noise       (we only observe the level)
```

In Kalman-filter matrix form:

```
F = [[1, 1],      H = [1, 0]      Q = diag(σ_level², σ_trend²)      R = [σ_obs²]
     [0, 1]]
```

The two knobs that matter:

- **`observation_noise` (σ_obs)** — how much you trust each measurement. Bigger
  ⇒ smooth harder.
- **`level_process_noise` / `trend_process_noise`** — how fast you allow the
  underlying signal to move. Smaller ⇒ smoother, but slower to react to real
  change.

## Quick start

```bash
git clone https://github.com/mbsdeepak/kalman-timeseries.git
cd kalman-timeseries
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the end-to-end demo (prints metrics, writes assets/denoise.png)
python -m examples.denoise_temperature

# metrics only, no matplotlib
python -m examples.denoise_temperature --no-plot
```

## Using it on your own data

```python
import numpy as np
from kalman import local_linear_trend

measurements = np.loadtxt("my_series.csv")   # a 1-D array of noisy readings

kf = local_linear_trend(
    observation_noise=2.5,      # ≈ std-dev of your sensor noise
    level_process_noise=0.5,    # how fast the true level can move
    trend_process_noise=0.01,   # how fast the slope can change
    initial_level=measurements[0],
)

filtered = kf.filter(measurements)          # online estimate
smoothed = kf.smooth(filtered)              # offline, cleaner estimate

clean = smoothed.levels(kf.H)               # denoised signal

# forecast 30 steps beyond the last observation, with 95% band
preds, var = kf.forecast(filtered.states[-1], filtered.covariances[-1], steps=30)
lo, hi = preds - 1.96*np.sqrt(var), preds + 1.96*np.sqrt(var)
```

Gaps are supported: pass `None` (or `np.nan`) for missing days and the filter
predicts through them without a correction step.

## What's inside

```
kalman/
  filter.py     # general linear Kalman filter: predict / update / RTS smoother / forecast
  models.py     # local_linear_trend(...) — builds the state-space model above
examples/
  synthetic.py            # reproducible noisy-temperature generator
  denoise_temperature.py  # the end-to-end demo behind the plot & table above
tests/
  test_filter.py          # correctness, noise-reduction, smoother, gaps, forecast
```

The core `KalmanFilter` is model-agnostic — give it `F, H, Q, R` and an initial
`(x, P)` and it works for any linear-Gaussian model (GPS tracking, sensor
fusion, etc.), not just this one. Notable implementation details:

- **Joseph-form covariance update** for numerical stability (keeps `P`
  symmetric positive-definite).
- **RTS smoother** for offline denoising — a backward pass that lets future
  observations refine past estimates.
- **Missing-data handling** — predict-only steps when no measurement is present.

## Tests

```bash
pip install pytest
pytest -q
```

The suite checks that the filter recovers a known constant, reduces RMSE vs the
raw signal, that the smoother is at least as good as the filter, that
covariances stay symmetric PSD, that measurement gaps are handled, and that
forecast uncertainty grows with the horizon.

## License

MIT — see [LICENSE](LICENSE).
