"""Head-to-head: baseline Kalman filter (v0.1.0) vs adaptive/robust (v0.2.0).

Both filters are run on the *same* deliberately nasty series — smooth seasonal
signal + measurement noise + sensor outliers + a permanent regime shift. The
baseline uses fixed, hand-tuned noise and trusts every sample; the adaptive
filter self-tunes its measurement noise, down-weights outliers, and inflates its
process noise to catch the shift.

Run:

    python -m examples.compare_versions            # metrics + comparison plot
    python -m examples.compare_versions --no-plot  # metrics only
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from examples.synthetic import noisy_temperature_hard
from kalman import adaptive_local_linear_trend, local_linear_trend


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    days, truth, z = noisy_temperature_hard()

    # --- v0.1.0: baseline, fixed hand-tuned noise, no robustness ---
    baseline = local_linear_trend(
        observation_noise=2.5, level_process_noise=0.5,
        trend_process_noise=0.01, initial_level=z[0],
    )
    base_res = baseline.filter(z)
    base_level = base_res.levels(baseline.H)

    # --- v0.2.0: adaptive, self-tuning R, robust update, regime catch-up ---
    #     Note: measurement_noise is NOT supplied — it is learned from the data.
    adaptive = adaptive_local_linear_trend(
        initial_level=z[0], level_process_noise=0.5, trend_process_noise=0.01,
    )
    adapt_res = adaptive.filter(z)
    adapt_level = adapt_res.levels(adaptive.H)

    print("=" * 64)
    print("Baseline (v0.1.0)  vs  Adaptive/Robust (v0.2.0)")
    print("Series: seasonal signal + noise + outliers + regime shift")
    print("=" * 64)
    print(f"{'':24}{'RMSE':>10}{'MAE':>10}")
    print(f"{'Raw measurements':24}{rmse(z, truth):>10.3f}{mae(z, truth):>10.3f}")
    print(f"{'Baseline KF (v0.1.0)':24}{rmse(base_level, truth):>10.3f}{mae(base_level, truth):>10.3f}")
    print(f"{'Adaptive KF (v0.2.0)':24}{rmse(adapt_level, truth):>10.3f}{mae(adapt_level, truth):>10.3f}")
    print("-" * 64)
    improvement = 100 * (1 - rmse(adapt_level, truth) / rmse(base_level, truth))
    print(f"RMSE improvement of v0.2.0 over v0.1.0: {improvement:.1f}%")
    print(f"Outliers auto-detected & down-weighted: {int(np.sum(adapt_res.outlier_flags))}")
    print(f"Regime catch-up steps engaged:          {int(np.sum(adapt_res.regime_flags))}")
    print(f"Final self-tuned R (est.):              {adapt_res.r_estimates[-1]:.2f}"
          f"   (true noise var = {2.5**2:.2f})")
    print("=" * 64)

    if args.no_plot:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -> skipping plot)")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.scatter(days, z, s=10, alpha=0.35, color="#9aa0a6", label="Noisy measurements (+ outliers)")
    ax1.plot(days, truth, color="#34a853", lw=1.6, label="Ground truth", alpha=0.85)
    ax1.plot(days, base_level, color="#4285f4", lw=1.3, label="Baseline KF (v0.1.0)")
    ax1.plot(days, adapt_level, color="#ea4335", lw=1.9, label="Adaptive/robust KF (v0.2.0)")

    # mark detected outliers and the regime-shift catch-up window
    out_idx = np.where(np.array(adapt_res.outlier_flags))[0]
    ax1.scatter(days[out_idx], z[out_idx], s=70, facecolors="none",
                edgecolors="#ea4335", lw=1.5, label="Detected outliers")
    reg_idx = np.where(np.array(adapt_res.regime_flags))[0]
    if reg_idx.size:
        ax1.axvspan(days[reg_idx.min()], days[reg_idx.max()] + 1, color="#fbbc04",
                    alpha=0.25, label="Regime catch-up")

    ax1.set_ylabel("Temperature (degC)")
    ax1.set_title("Baseline vs adaptive/robust Kalman filter on a nasty series")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.2)

    # bottom panel: the self-tuned measurement noise over time
    ax2.plot(days, adapt_res.r_estimates, color="#ea4335", lw=1.5, label="self-tuned R estimate")
    ax2.axhline(2.5 ** 2, color="#34a853", ls="--", lw=1.2, label="true noise variance")
    ax2.set_ylabel("R estimate")
    ax2.set_xlabel("Day of year")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.2)

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "assets", "comparison.png"))
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
