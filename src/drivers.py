"""External price drivers. Currently: the USD/KES exchange rate.

Run:  python -m src.drivers --inspect [--path FILE]

Two sources, chosen by config.FX_SOURCE:
  cbk - the CBK export at config.FX_CSV (daily, averaged per month)
  wfp - the rate implied by the WFP file itself, median of price / usdprice
"auto" prefers cbk when the file exists and falls back to wfp.

CBK exports are not consistent about column names ("Mean", "EXCHANGE RATE",
"Mean Rate", ...) so they are detected, not hardcoded. If the export stacks
several currencies, only US dollar rows are kept.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data

# Gaps in the FX series up to this many consecutive months are forward-filled
# with the previous month's rate. A one-month hole is better bridged than
# allowed to delete every row that looks back over it. Forward-fill, not
# interpolation: an interpolated month would carry the following month's rate,
# so row T's fx_lag1 could contain fx[T].
FX_MAX_GAP_FILL = 1

# Implied KES per USD outside this range is a rounding artefact or bad row.
WFP_RATE_BOUNDS = (50.0, 250.0)
SOURCES = ("auto", "cbk", "wfp")

# Checked in order; the first column whose normalised name matches wins.
_DATE_PATTERNS = [r"^date$", r"date"]
_RATE_PATTERNS = [r"^mean$", r"mean", r"exchange ?rate", r"rate"]
_CURRENCY_PATTERNS = [r"^currency$", r"currency"]
_USD = "US DOLLAR"


def _clean(values: pd.Series) -> pd.Series:
    """Strip whitespace and any stray quote characters CBK leaves in cells."""
    return values.fillna("").str.strip().str.strip("\"'").str.strip()


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).replace("﻿", "")).strip().lower()


def _find(columns, patterns) -> str | None:
    for pattern in patterns:
        for col in columns:
            if re.search(pattern, _norm(col)):
                return col
    return None


def load_daily(path: Path | str | None = None) -> pd.DataFrame:
    """Parsed daily rows: columns `date` and `rate`, sorted, USD only."""
    path = Path(path or config.FX_CSV)
    if not path.exists():
        raise FileNotFoundError(f"FX file not found: {path}")

    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    date_col = _find(raw.columns, _DATE_PATTERNS)
    rate_col = _find([c for c in raw.columns if c != date_col], _RATE_PATTERNS)
    if date_col is None or rate_col is None:
        raise ValueError(
            f"Could not detect date/rate columns in {path.name}. "
            f"Columns found: {list(raw.columns)}. "
            f"Detected date={date_col!r}, rate={rate_col!r}."
        )

    currency_col = _find(raw.columns, _CURRENCY_PATTERNS)
    if currency_col is not None:
        is_usd = _clean(raw[currency_col]).str.upper() == _USD
        if not is_usd.any():
            found = sorted(raw[currency_col].dropna().unique())[:10]
            raise ValueError(
                f"No US dollar rows in column {currency_col!r}. Values found: {found}"
            )
        raw = raw[is_usd]

    raw_date = _clean(raw[date_col])
    out = pd.DataFrame(
        {
            "raw_date": raw_date,
            # CBK writes dates as DD/MM/YYYY.
            "date": pd.to_datetime(raw_date, dayfirst=True, errors="coerce"),
            "rate": pd.to_numeric(
                _clean(raw[rate_col]).str.replace(",", ""), errors="coerce"
            ),
        }
    )
    return out.dropna(subset=["date", "rate"]).sort_values("date").reset_index(drop=True)


def load_cbk(path: Path | str | None = None) -> pd.Series:
    """Mean CBK USD/KES rate per calendar month, indexed by month start.

    Months with no quotes are absent from the index, not NaN-filled. The
    feature join reindexes onto the price calendar, so a missing month turns
    into NaN there and the row is dropped.
    """
    daily = load_daily(path)
    monthly = daily.set_index("date")["rate"].resample("MS").mean().dropna()
    monthly.name = "fx"
    return monthly


def fx_from_wfp(df: pd.DataFrame) -> pd.Series:
    """Implied USD/KES per calendar month from a cleaned WFP frame.

    Each row gives price / usdprice. usdprice is rounded to two decimals, so
    a 30 KES item (usdprice 0.23) can be off by several percent; the median
    across the month's rows is used instead of the mean so those rows do not
    drag the estimate.
    """
    usd = pd.to_numeric(df["usdprice"], errors="coerce")
    kes = pd.to_numeric(df["price"], errors="coerce")
    ok = usd.notna() & (usd > 0) & kes.notna()
    rate = (kes[ok] / usd[ok]).replace([np.inf, -np.inf], np.nan)
    lo, hi = WFP_RATE_BOUNDS
    rate = rate[(rate >= lo) & (rate <= hi)]

    month = df.loc[rate.index, "date"].dt.to_period("M").dt.to_timestamp()
    monthly = rate.groupby(month).median().sort_index()
    monthly.index = pd.DatetimeIndex(monthly.index, freq=None, name="date")
    monthly.name = "fx"
    return monthly


def _clean_wfp(**load_kwargs) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):  # silence clean()'s row count
        return data.clean(data.load_raw(**load_kwargs))


def fill_single_gaps(fx: pd.Series) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Forward-fill runs of up to FX_MAX_GAP_FILL missing months.

    Each filled month takes the last known rate before it, so it never holds
    information from a later month. Longer runs are left as gaps entirely
    (not partly filled). Returns the series and the months filled.
    """
    calendar = fx.reindex(pd.date_range(fx.index.min(), fx.index.max(), freq="MS"))
    missing = calendar.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    fillable = missing & (run_len <= FX_MAX_GAP_FILL)
    filled = calendar.where(~fillable, calendar.ffill())
    new = filled.index[fillable]
    filled = filled.dropna()
    filled.index.name, filled.name = fx.index.name, fx.name
    return filled, new


def resolve_source(source: str | None = None, path: Path | str | None = None) -> str:
    """Turn "auto" into "cbk" or "wfp"; validate anything else."""
    source = (source or config.FX_SOURCE).lower()
    if source not in SOURCES:
        raise ValueError(f"FX_SOURCE must be one of {SOURCES}, got {source!r}")
    if source == "auto":
        return "cbk" if Path(path or config.FX_CSV).exists() else "wfp"
    return source


def load_fx(path: Path | str | None = None, source: str | None = None) -> pd.Series:
    """Monthly USD/KES series from the configured source. Prints which one."""
    chosen = resolve_source(source, path)
    if chosen == "cbk":
        fx = load_cbk(path)
        where = f"CBK export {Path(path or config.FX_CSV).name}"
    else:
        fx = fx_from_wfp(_clean_wfp())
        where = "implied by WFP price / usdprice (median per month)"
    fx, filled = fill_single_gaps(fx)
    print(
        f"fx source: {where} "
        f"[{fx.index.min():%Y-%m} -> {fx.index.max():%Y-%m}, {len(fx)} months]"
    )
    print(
        f"fx gaps forward-filled (runs of <= {FX_MAX_GAP_FILL} month): {len(filled)}"
        + (f" -> {', '.join(m.strftime('%Y-%m') for m in filled)}" if len(filled) else "")
    )
    return fx


def _missing_months(monthly: pd.Series) -> pd.DatetimeIndex:
    full = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    return full.difference(monthly.index)


def _comparison(cbk: pd.Series, wfp: pd.Series) -> pd.DataFrame:
    both = pd.concat({"cbk": cbk, "wfp_implied": wfp}, axis=1, join="inner")
    both["diff_pct"] = (both["wfp_implied"] / both["cbk"] - 1) * 100
    both.index = both.index.strftime("%Y-%m")
    return both.round({"cbk": 4, "wfp_implied": 4, "diff_pct": 2})


def _inspect(path: Path | str | None) -> None:
    path = Path(path or config.FX_CSV)
    print(f"FX_SOURCE = {config.FX_SOURCE!r} -> using {resolve_source(None, path)!r}")

    cbk = None
    print("\n== CBK ==")
    if not path.exists():
        print(f"not found: {path}")
    else:
        daily = load_daily(path)
        cbk = load_cbk(path)
        missing = _missing_months(cbk)
        # Raw text beside the parsed date, so a day/month swap is visible by eye.
        first, last = daily.iloc[0], daily.iloc[-1]
        print(f"file:          {path}")
        print(f"daily rows:    {len(daily)}")
        print(f"first date:    {first['raw_date']!r} -> {first['date']:%d %b %Y}")
        print(f"last date:     {last['raw_date']!r} -> {last['date']:%d %b %Y}")
        print(f"rate range:    {daily['rate'].min():.4f} -> {daily['rate'].max():.4f}")
        print(f"months:        {len(cbk)} ({cbk.index.min():%Y-%m} -> {cbk.index.max():%Y-%m})")
        print(f"missing months: {len(missing)}"
              + (f" -> {', '.join(m.strftime('%Y-%m') for m in missing)}" if len(missing) else ""))

    print("\n== WFP-implied (cleaned rows) ==")
    if not config.RAW_CSV.exists():
        print(f"not found: {config.RAW_CSV}")
        return
    wfp = fx_from_wfp(_clean_wfp())
    missing = _missing_months(wfp)
    print(f"months:        {len(wfp)} ({wfp.index.min():%Y-%m} -> {wfp.index.max():%Y-%m})")
    print(f"rate range:    {wfp.min():.2f} -> {wfp.max():.2f}")
    print(f"missing months: {len(missing)}"
          + (f" -> {', '.join(m.strftime('%Y-%m') for m in missing[:24])}"
             + (" ..." if len(missing) > 24 else "") if len(missing) else ""))
    _, filled = fill_single_gaps(wfp)
    print(f"would forward-fill: {len(filled)}"
          + (f" -> {', '.join(m.strftime('%Y-%m') for m in filled)}" if len(filled) else ""))
    print("last 6 months:")
    print(wfp.tail(6).round(2).to_string())

    if cbk is None:
        return
    print("\n== CBK vs WFP-implied, overlapping months ==")
    table = _comparison(cbk, wfp)
    if len(table):
        print(table.to_string())
        return
    print(
        f"none: cleaned WFP rows end {wfp.index.max():%Y-%m}, "
        f"CBK starts {cbk.index.min():%Y-%m}."
    )
    # Diagnostic only: the same method on every WFP row, including the
    # aggregate-flagged ones clean() drops, to check the method against CBK.
    every = data.load_raw()
    every = every.dropna(subset=["date"])
    table = _comparison(cbk, fx_from_wfp(every))
    print("\ndiagnostic, all WFP price flags (not used for training):")
    print(table.to_string() if len(table) else "none either: WFP has no rows in those months.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inspect", action="store_true", help="print a parse check")
    parser.add_argument("--path", help=f"FX CSV (default {config.FX_CSV})")
    args = parser.parse_args()
    if not args.inspect:
        parser.error("nothing to do; pass --inspect")
    _inspect(args.path)


if __name__ == "__main__":
    main()
