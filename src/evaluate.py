"""Held-out evaluation figures and metrics for the report.

Run:  python -m src.evaluate   (after python -m src.train)

Reads the test-period results that train_one stores before its final refit,
writes four PNGs to reports/, and prints a per-series metrics table. Every
number comes from the held-out test months; nothing is refitted here.
Needs matplotlib (requirements-train.txt), which production does not.
"""

from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")  # files only, no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import config  # noqa: E402

REPORT_DIR = config.ROOT / "reports"

SHIPPABLE = "#C2410C"
NOT_SHIPPABLE = "#A8A29E"
INK = "#1C1917"
MUTED_INK = "#57534E"
NAIVE = "#D6D3D1"
GRID = "#E7E5E4"

TEST_KEYS = ("test_index", "test_actual", "test_predicted", "test_naive", "test_seasonal")


def _style(ax, grid: bool = True) -> None:
    """No top/right spines, light horizontal gridlines only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    ax.grid(False)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    path = REPORT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(config.ROOT)}")


def _name(entry: dict) -> str:
    return f"{entry['commodity']} ({entry['unit']}, {entry['pricetype'].lower()})"


def load() -> dict[str, dict]:
    if not config.ARTIFACT.exists():
        raise SystemExit(f"No artifact at {config.ARTIFACT}. Run: python -m src.train")
    series = joblib.load(config.ARTIFACT)["series"]
    stale = [k for k, e in series.items() if any(t not in e for t in TEST_KEYS)]
    if stale:
        raise SystemExit(
            f"{len(stale)} series in the artifact have no stored test results "
            "(trained before they were recorded). Run: python -m src.train"
        )
    return series


def test_frame(entry: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "actual": entry["test_actual"],
            "predicted": entry["test_predicted"],
            "naive": entry["test_naive"],
            "seasonal": entry["test_seasonal"],
        },
        index=pd.to_datetime(entry["test_index"]),
    )


def actual_vs_predicted(series: dict[str, dict]) -> None:
    shippable = [e for e in series.values() if e["shippable"]]
    if not shippable:
        print("skipped actual_vs_predicted.png: no shippable series")
        return

    frames = [test_frame(e) for e in shippable]
    pooled = pd.concat(frames)
    model_mape = (pooled["predicted"] / pooled["actual"] - 1).abs().mean() * 100
    naive_mape = (pooled["naive"] / pooled["actual"] - 1).abs().mean() * 100

    cols = 3 if len(shippable) > 4 else 2 if len(shippable) > 1 else 1
    rows = int(np.ceil(len(shippable) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.9 * rows), squeeze=False)

    for ax, entry, df in zip(axes.flat, shippable, frames):
        # Test months are not contiguous (rows with missing lags are dropped),
        # so put them on a full monthly calendar: lines break at gaps instead
        # of drawing a straight line through months that have no data.
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="MS"))
        dot = dict(marker="o", markersize=3.5, markeredgewidth=0)
        ax.plot(df.index, df["naive"], color=NAIVE, linewidth=2, zorder=1, **dot)
        ax.plot(df.index, df["actual"], color=INK, linewidth=2, zorder=3, **dot)
        ax.plot(df.index, df["predicted"], color=SHIPPABLE, linewidth=2,
                linestyle=(0, (4, 2)), zorder=2, **dot)
        ax.set_title(_name(entry), fontsize=10, color=INK, loc="left")
        ax.set_ylabel(f"KES per {entry['unit']}", fontsize=9, color=MUTED_INK)
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
        _style(ax)
    for ax in axes.flat[len(shippable):]:
        ax.set_visible(False)

    fig.legend(
        handles=[
            Line2D([], [], color=INK, linewidth=2, label="Actual price"),
            Line2D([], [], color=SHIPPABLE, linewidth=2, linestyle=(0, (4, 2)),
                   label="Shipped model forecast"),
            Line2D([], [], color=NAIVE, linewidth=2, label="Naive (last month's price)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9,
    )
    verdict = "closer to" if model_mape < naive_mape else "no closer to"
    fig.suptitle(
        f"One-month forecasts land {verdict} actual prices than the naive baseline: "
        f"mean error {model_mape:.1f}% vs {naive_mape:.1f}% over held-out months",
        x=0.01, y=1.06, ha="left", fontsize=11, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "actual_vs_predicted.png")


def mase_by_series(series: dict[str, dict]) -> None:
    rows = sorted(
        (
            (_name(e), e["scores"][e["model_name"]]["mase"], e["shippable"])
            for e in series.values()
            if e["scores"][e["model_name"]]["mase"] is not None
        ),
        key=lambda r: r[1],
        reverse=True,  # barh draws bottom-up, so best ends up on top
    )
    names, mase, ship = zip(*rows)
    beat = sum(m < 1.0 for m in mase)

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.4))
    ax.barh(names, mase, color=[SHIPPABLE if s else NOT_SHIPPABLE for s in ship],
            height=0.7, edgecolor="white", linewidth=2)
    for y, m in enumerate(mase):
        # Two decimals, unless that would print a value under 1.0 as "1.00".
        label = f"{m:.3f}" if round(m, 2) == 1.0 else f"{m:.2f}"
        ax.text(m + 0.02, y, label, va="center", fontsize=8, color=MUTED_INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ax.axvline(1.0, color=INK, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(1.0, len(rows) - 0.4, " no better than naive", fontsize=8,
            color=INK, va="bottom", ha="left")
    ax.set_xlabel("MASE against naive baseline (ratio; below 1.0 beats naive)",
                  fontsize=9, color=MUTED_INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(max(mase) * 1.12, 1.15))
    # Horizontal gridlines only, and on a horizontal bar chart they would
    # just repeat the bars, so this chart carries none.
    _style(ax, grid=False)
    ax.legend(
        handles=[Patch(color=SHIPPABLE, label="Shippable (beats both baselines)"),
                 Patch(color=NOT_SHIPPABLE, label="Not shippable")],
        loc="upper right", frameon=False, fontsize=8,
    )
    ax.set_title(
        f"{beat} of {len(rows)} series beat the naive baseline; "
        f"{sum(ship)} also beat the seasonal one and ship",
        fontsize=11, color=INK, loc="left",
    )
    _save(fig, "mase_by_series.png")


def _pooled_shippable(series: dict[str, dict]) -> pd.DataFrame:
    frames = [
        test_frame(e).assign(series=_name(e))
        for e in series.values() if e["shippable"]
    ]
    df = pd.concat(frames)
    df["residual"] = df["actual"] - df["predicted"]
    df["pct_error"] = (df["predicted"] / df["actual"] - 1) * 100
    return df


def residuals(series: dict[str, dict]) -> None:
    df = _pooled_shippable(series)
    median_pct = (df["residual"] / df["predicted"]).median() * 100
    # Does the size of the error grow with the price level?
    spread_corr = df["predicted"].rank().corr(df["residual"].abs().rank())
    bias = (
        "centre on zero" if abs(median_pct) < 2
        else f"lean {'positive (forecasts too low)' if median_pct > 0 else 'negative (forecasts too high)'}"
    )
    spread = (
        "and widen at higher prices" if spread_corr > 0.3
        else "with no clear pattern against price level"
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0, color=INK, linewidth=1, zorder=1)
    ax.scatter(df["predicted"], df["residual"], s=28, color=SHIPPABLE, alpha=0.75,
               edgecolor="white", linewidth=1, zorder=2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("Predicted price (KES per series unit, log scale)", fontsize=9,
                  color=MUTED_INK)
    ax.set_ylabel("Residual: actual − predicted (KES)", fontsize=9, color=MUTED_INK)
    _style(ax)
    ax.set_title(
        f"Residuals {bias} {spread}\n"
        f"(median {median_pct:+.1f}% of predicted; {len(df)} held-out months, "
        f"{df['series'].nunique()} shippable series)",
        fontsize=11, color=INK, loc="left",
    )
    _save(fig, "residuals.png")


def error_distribution(series: dict[str, dict]) -> None:
    df = _pooled_shippable(series)
    abs_err = df["pct_error"].abs()
    p50, p90 = abs_err.quantile(0.5), abs_err.quantile(0.9)

    # 5-point bins over the main body; months beyond CLIP are piled into the
    # edge bins and labelled, so one extreme month cannot flatten the chart.
    clip = 60.0
    width = 5.0
    shown = df["pct_error"].clip(-clip - width / 2, clip + width / 2)
    bins = np.arange(-clip - width, clip + width * 1.01, width)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(shown, bins=bins, color=SHIPPABLE, edgecolor="white", linewidth=1.5)
    ax.axvline(0, color=INK, linewidth=1)
    for side, mask in (("high", df["pct_error"] > clip), ("low", df["pct_error"] < -clip)):
        if mask.any():
            x = clip + width / 2 if side == "high" else -clip - width / 2
            worst = df.loc[mask, "pct_error"]
            ax.annotate(
                f"{int(mask.sum())} beyond {'+' if side == 'high' else '−'}{clip:.0f}%\n"
                f"(up to {worst.abs().max():+.0f}%)".replace("+-", "−"),
                xy=(x, mask.sum()), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8, color=MUTED_INK,
            )
    ax.set_xlim(-clip - width * 1.5, clip + width * 1.5)
    ax.set_xlabel("Forecast error (% of actual price; positive = forecast too high)",
                  fontsize=9, color=MUTED_INK)
    ax.set_ylabel("Held-out months (count)", fontsize=9, color=MUTED_INK)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _style(ax)
    ax.set_title(
        f"Half of forecasts fall within ±{p50:.1f}% of the actual price, "
        f"90% within ±{p90:.1f}%",
        fontsize=11, color=INK, loc="left",
    )
    _save(fig, "error_distribution.png")


def metrics_table(series: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for e in series.values():
        s = e["scores"][e["model_name"]]
        rows.append(
            {
                "series": _name(e),
                "shipped": e["model_name"],
                "mase": s["mase"],
                "mase_seasonal": s["mase_seasonal"],
                "rmse_kes": s["rmse"],
                "mae_kes": s["mae"],
                "test_months": e["n_test"],
                "shippable": e["shippable"],
            }
        )
    return pd.DataFrame(rows).sort_values("mase").reset_index(drop=True)


def main() -> None:
    series = load()
    REPORT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlepad": 10})

    actual_vs_predicted(series)
    mase_by_series(series)
    residuals(series)
    error_distribution(series)

    table = metrics_table(series)
    print("\n(mase = vs naive lag_1, mase_seasonal = vs same month last year; "
          "below 1.0 beats the baseline. RMSE/MAE in KES per series unit.)")
    print(table.to_string(index=False))
    print(f"\nshippable: {int(table['shippable'].sum())} of {len(table)}")


if __name__ == "__main__":
    main()
