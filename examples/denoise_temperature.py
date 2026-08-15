"""End-to-end demo: denoise a noisy temperature series and forecast ahead.

Run it:

    python -m examples.denoise_temperature            # prints metrics, saves plot
    python -m examples.denoise_temperature --no-plot  # metrics only

The script:
  1. Generates a noisy synthetic daily-temperature series.
  2. Runs the local-linear-trend Kalman filter (causal, online estimate).
  3. Runs the RTS smoother (offline, uses the whole series).
  4. Forecasts the next 30 days with an uncertainty band.
  5. Reports how much noise each stage removed vs the known ground truth.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from examples.synthetic import noisy_temperature
from kalman import local_linear_trend


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="skip the matplotlib figure")
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--noise-std", type=float, default=2.5)
    args = parser.parse_args()

    days, truth, measurements = noisy_temperature(noise_std=args.noise_std)

    # Tuning: trust the measurements moderately, allow the level to wander a bit
    # day-to-day, and let the trend drift only very slowly.
    kf = local_linear_trend(
        observation_noise=args.noise_std,
        level_process_noise=0.5,
        trend_process_noise=0.01,
        initial_level=measurements[0],
    )

    filtered = kf.filter(measurements)
    smoothed = kf.smooth(filtered)

    filtered_level = filtered.levels(kf.H)
    smoothed_level = smoothed.levels(kf.H)

    # Forecast from the final filtered state.
    preds, variances = kf.forecast(filtered.states[-1], filtered.covariances[-1], args.forecast_days)
    forecast_days = np.arange(len(days), len(days) + args.forecast_days)
    band = 1.96 * np.sqrt(variances)  # 95% interval

    print("=" * 56)
    print("Denoising a noisy temperature series with a Kalman filter")
    print("=" * 56)
    print(f"Samples:                 {len(days)} days")
    print(f"Sensor noise (std):      {args.noise_std:.2f} degC")
    print("-" * 56)
    print(f"RMSE raw measurements:   {rmse(measurements, truth):.3f} degC")
    print(f"RMSE Kalman filtered:    {rmse(filtered_level, truth):.3f} degC  (causal/online)")
    print(f"RMSE RTS smoothed:       {rmse(smoothed_level, truth):.3f} degC  (offline)")
    reduction = 100 * (1 - rmse(smoothed_level, truth) / rmse(measurements, truth))
    print(f"Noise removed (smoother):{reduction:6.1f}%")
    print("-" * 56)
    print(f"{args.forecast_days}-day forecast, first 3 days:")
    for i in range(min(3, args.forecast_days)):
        print(f"  day +{i + 1:<2d}  {preds[i]:6.2f} degC  +/- {band[i]:.2f}")
    print("=" * 56)

    if args.no_plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -> skipping plot)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(days, measurements, s=8, alpha=0.35, color="#9aa0a6", label="Noisy measurements")
    ax.plot(days, truth, color="#34a853", lw=1.5, label="Ground truth", alpha=0.8)
    ax.plot(days, filtered_level, color="#4285f4", lw=1.2, label="Kalman filtered (online)")
    ax.plot(days, smoothed_level, color="#ea4335", lw=1.8, label="RTS smoothed (offline)")

    ax.plot(forecast_days, preds, color="#fbbc04", lw=1.8, ls="--", label="30-day forecast")
    ax.fill_between(forecast_days, preds - band, preds + band, color="#fbbc04", alpha=0.2, label="95% interval")

    ax.set_xlabel("Day of year")
    ax.set_ylabel("Temperature (degC)")
    ax.set_title("Kalman filtering of a noisy temperature series")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    out = os.path.join(os.path.dirname(__file__), "..", "assets", "denoise.png")
    out = os.path.normpath(out)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
