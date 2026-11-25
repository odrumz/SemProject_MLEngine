"""Work out why series are being dropped, before changing any settings.

Run:  python -m src.diagnose

Resampling to monthly turns every missing month into NaN, and long lags
propagate those gaps: one missing month can wipe out rows up to twelve
months later. This prints how many rows survive feature building, and what
you would gain by shortening the lags.
"""

from __future__ import annotations

import pandas as pd

from . import config, data


def usable_rows(series: pd.Series, lags, windows) -> int:
    """Rows left after building features with a given lag/window set."""
    frame = series.to_frame(name="price")
    past = frame["price"].shift(1)
    cols = []

    for n in lags:
        frame[f"lag_{n}"] = frame["price"].shift(n)
        cols.append(f"lag_{n}")
    for w in windows:
        frame[f"rm_{w}"] = past.rolling(w, min_periods=w).mean()
        cols.append(f"rm_{w}")

    return int(len(frame.dropna(subset=cols + ["price"])))


def main() -> None:
    raw = data.clean(data.load_raw())
    monthly = data.to_monthly(raw)

    dated = {k: s for k, s in monthly.items() if s.last_valid_index() is not None}
    newest = max(s.last_valid_index() for s in dated.values())
    cutoff = newest - pd.DateOffset(months=config.MAX_STALENESS_MONTHS)
    fresh = {k: s for k, s in dated.items() if s.last_valid_index() >= cutoff}

    print(f"\nnewest month in file : {newest:%Y-%m}")
    print(f"active series        : {len(fresh)} of {len(monthly)}")

    rows = []
    for key, s in fresh.items():
        observed = int(s.notna().sum())
        span = len(s)  # calendar months from first to last observation
        rows.append(
            {
                "series": key,
                "observed": observed,
                "span": span,
                "gaps": span - observed,
                "current": usable_rows(s, config.LAGS, config.ROLL_WINDOWS),
                "short_lags": usable_rows(s, [1, 2, 3], [3]),
            }
        )

    df = pd.DataFrame(rows)

    print("\n--- observed months per active series ---")
    print(df["observed"].describe().round(1).to_string())

    print("\n--- how many series survive MIN_OBSERVATIONS ---")
    print(f"{'threshold':>10} {'current features':>18} {'short lags only':>18}")
    for t in (24, 36, 48, 60, 72):
        print(f"{t:>10} {int((df['current'] >= t).sum()):>18} "
              f"{int((df['short_lags'] >= t).sum()):>18}")

    print("\n--- best 15 series by usable rows (current features) ---")
    top = df.sort_values("current", ascending=False).head(15)
    print(top.to_string(index=False))

    lost = df["observed"].sum() - df["current"].sum()
    print(f"\nrows lost to gaps + lag warm-up: {lost:,} "
          f"of {df['observed'].sum():,} observed")
    print("If 'short lags only' is far higher, gaps are the problem, not data volume.")


if __name__ == "__main__":
    main()
