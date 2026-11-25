"""Build one stacked training table across all market-level series.

A pooled model learns from every series at once, so a series with only ten
months of data can still borrow strength from the rest. To make series with
very different price levels comparable, every feature and the target are
log returns rather than prices.

The feature set is deliberately small: most market series have only about
ten observations, and every extra lag costs one more row from each of them.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from . import config, data

NUMERIC = ["r1", "r2", "month_sin", "month_cos"]
CATEGORICAL = ["commodity", "unit", "pricetype"]
TARGET = "y"


def _rows_for(key: str, monthly: pd.Series) -> pd.DataFrame:
    commodity, _market, unit, pricetype = key.split("|")
    price = monthly.astype(float)
    month = price.index.month
    return pd.DataFrame(
        {
            "series_key": key,
            "month": price.index,
            "last_price": price,
            "r1": np.log(price / price.shift(1)),
            "r2": np.log(price.shift(1) / price.shift(2)),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "commodity": commodity,
            "unit": unit,
            "pricetype": pricetype,
            TARGET: np.log(price.shift(-1) / price),
        },
        index=price.index,
    )


def build(path=None) -> pd.DataFrame:
    """Return the stacked frame, one row per (series, month) with a full row."""
    # The pooled model always works on market-level series, whatever
    # config.AGGREGATION is set to: pooling national medians would leave
    # only a handful of series and defeat the point. The setting is
    # restored afterwards so the per-series path is unaffected.
    #
    # Gap filling is also switched off here. An interpolated month is a
    # straight line toward the next observed price, so its log return
    # leaks the target: on this data almost every test row touched a filled
    # month and the model looked far better than it is. Only observed
    # prices are used; rows that span a gap are dropped by dropna below.
    saved = config.AGGREGATION, config.MAX_GAP_FILL
    config.AGGREGATION, config.MAX_GAP_FILL = "market", 0
    try:
        # Silence data.clean()'s progress line so the row count below is the
        # first thing this path prints.
        with contextlib.redirect_stdout(io.StringIO()):
            monthly = data.to_monthly(data.clean(data.load_raw(path)))
    finally:
        config.AGGREGATION, config.MAX_GAP_FILL = saved

    parts = [
        _rows_for(key, s)
        for key, s in monthly.items()
        if s.notna().sum() >= config.POOLED_MIN_OBS
    ]
    stacked = (
        pd.concat(parts, ignore_index=True)
        .dropna(subset=NUMERIC + [TARGET, "last_price"])
        .reset_index(drop=True)
    )
    for col in CATEGORICAL:
        stacked[col] = stacked[col].astype("category")

    print(
        f"pooled: {len(stacked):,} rows from "
        f"{stacked['series_key'].nunique():,} contributing series"
    )
    return stacked
