"""Load the WFP file and turn it into clean monthly series.

One series = one (commodity, market, unit, pricetype) combination.
Keeping `unit` in the key matters: "Maize" appears both as a 90 KG wholesale
sack and a 1 KG retail packet, and mixing them produces a nonsense series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Read the CSV, skipping the HXL tag row that sits under the header."""
    path = path or config.RAW_CSV
    df = pd.read_csv(path, skiprows=[1])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("price", "usdprice"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be modelled, and rows WFP flagged as estimates."""
    before = len(df)

    df = df[df["priceflag"].isin(config.PRICE_FLAGS)]
    df = df.dropna(subset=["date", "price", "market", "commodity", "unit"])
    df = df[df["price"] > 0]
    df = df.drop_duplicates()

    print(f"clean: {before} -> {len(df)} rows")
    return df


def series_key(row) -> str:
    """Stable id used in the API and as the model dictionary key."""
    return "|".join(
        str(row[c]).strip().replace(" ", "_")
        for c in ("commodity", "market", "unit", "pricetype")
    )


def to_monthly(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Collapse to one price per calendar month, per series.

    With config.AGGREGATION == "market" this is the mean per market; with
    "national" it is the median across markets (see _to_monthly_national).

    Gaps of up to config.MAX_GAP_FILL months are linearly interpolated.
    Longer gaps stay as NaN: filling six months of missing prices would be
    inventing data, while filling one month between two observations is a
    reasonable estimate and saves every later row whose lag window spans it.
    """
    if config.AGGREGATION == "national":
        return _to_monthly_national(df)
    if config.AGGREGATION != "market":
        raise ValueError(
            f"config.AGGREGATION must be 'market' or 'national', got {config.AGGREGATION!r}"
        )

    out: dict[str, pd.Series] = {}

    grouped = df.groupby(["commodity", "market", "unit", "pricetype"], sort=False)
    for (commodity, market, unit, pricetype), part in grouped:
        key = series_key(
            {"commodity": commodity, "market": market,
             "unit": unit, "pricetype": pricetype}
        )
        monthly = (
            part.set_index("date")["price"]
            .resample("MS")
            .mean()
            .sort_index()
        )
        out[key] = fill_short_gaps(monthly)

    return out


def _to_monthly_national(df: pd.DataFrame) -> dict[str, pd.Series]:
    """One series per (commodity, unit, pricetype), using the median across markets.

    Each market is first collapsed to its own monthly mean, so a market with
    many reports in a month counts once. The median across those markets
    is then used rather than the mean, because the set of reporting markets
    changes over time and the median is less affected by which markets happen
    to be in the sample in a given month.
    """
    out: dict[str, pd.Series] = {}

    grouped = df.groupby(["commodity", "unit", "pricetype"], sort=False)
    for (commodity, unit, pricetype), part in grouped:
        key = series_key(
            {"commodity": commodity, "market": "NATIONAL",
             "unit": unit, "pricetype": pricetype}
        )
        per_market = (
            part.assign(month=part["date"].dt.to_period("M").dt.to_timestamp())
            .groupby(["month", "market"])["price"]
            .mean()
        )
        monthly = (
            per_market.groupby(level="month").median()
            .resample("MS")
            .asfreq()
            .sort_index()
        )
        monthly.index.name = "date"
        monthly.name = "price"
        out[key] = fill_short_gaps(monthly)

    return out


def fill_short_gaps(series: pd.Series, limit: int | None = None) -> pd.Series:
    """Linearly interpolate runs of NaN no longer than `limit` months."""
    limit = config.MAX_GAP_FILL if limit is None else limit
    if limit <= 0 or series.notna().sum() < 2:
        return series

    filled = series.interpolate(method="linear", limit_area="inside")

    # interpolate() has no "only runs shorter than N" option, so undo the
    # fills that belong to runs longer than the limit.
    missing = series.isna()
    run_id = (missing != missing.shift()).cumsum()
    too_long = (
        missing.groupby(run_id).transform("sum").where(missing, 0) > limit
    )
    filled[too_long] = np.nan
    return filled


def describe_series(monthly: dict[str, pd.Series]) -> pd.DataFrame:
    """A table of what is available, for deciding what is worth modelling."""
    rows = []
    for key, s in monthly.items():
        observed = s.notna().sum()
        rows.append(
            {
                "series_key": key,
                "observed_months": int(observed),
                "missing_months": int(s.isna().sum()),
                "start": s.first_valid_index(),
                "end": s.last_valid_index(),
                "mean_price": round(float(s.mean()), 2) if observed else None,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("observed_months", ascending=False)
        .reset_index(drop=True)
    )
