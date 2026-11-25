"""Turn one monthly price series into a supervised learning table.

Every rolling statistic is shifted by one month. Without that shift the
rolling mean would include the month being predicted, which leaks the target
into its own feature and produces scores that collapse in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from functools import lru_cache

from . import config, drivers


@lru_cache(maxsize=1)
def _fx() -> pd.Series:
    return drivers.load_fx()


def uses_fx() -> bool:
    """Whether the live feature set includes any FX column."""
    return any(f in config.FEATURES for f in config.FX_FEATURES)


def _fx_features(fx: pd.Series, months: pd.DatetimeIndex) -> pd.DataFrame:
    """FX columns for each target month, using only FX from T-1 and earlier.

    FX is put on a gap-free monthly calendar *before* shifting, so shift(1)
    is always the previous calendar month even where the FX file has holes
    (a hole becomes NaN, not a silently older month).
    """
    span = pd.date_range(
        min(fx.index.min(), months.min()), max(fx.index.max(), months.max()), freq="MS"
    )
    fx = fx.reindex(span)
    out = pd.DataFrame(
        {
            "fx_lag1": fx.shift(1),
            "fx_change_3m": np.log(fx.shift(1) / fx.shift(4)),
        }
    )
    return out.reindex(months)


def build_features(monthly: pd.Series, fx: pd.Series | None = None) -> pd.DataFrame:
    """Return a frame indexed by month with FEATURES + TARGET columns.

    FX columns are added when `fx` is passed (src.compare_fx does this) or
    when config.FEATURES includes them, in which case drivers.load_fx() is
    used. Otherwise FX is never loaded.
    """
    frame = monthly.to_frame(name=config.TARGET).copy()
    past = frame[config.TARGET].shift(1)  # everything is built from the past only

    for n in config.LAGS:
        frame[f"lag_{n}"] = frame[config.TARGET].shift(n)

    for w in config.ROLL_WINDOWS:
        frame[f"roll_mean_{w}"] = past.rolling(w, min_periods=w).mean()
        frame[f"roll_std_{w}"] = past.rolling(w, min_periods=w).std()

    frame["mom_change"] = frame["lag_1"] / frame["lag_2"] - 1.0

    # Cyclical month: December sits next to January instead of 11 units away.
    month = frame.index.month
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Row T predicts price[T]; FX features come from T-1 and earlier, never fx[T].
    if fx is not None or uses_fx():
        frame = frame.join(_fx_features(_fx() if fx is None else fx, frame.index))

    return frame


def model_frame(monthly: pd.Series, fx: pd.Series | None = None) -> pd.DataFrame:
    """Feature frame with rows that cannot be used dropped."""
    frame = build_features(monthly, fx)
    return frame.dropna(subset=config.FEATURES + [config.TARGET])


def row_from_history(
    history: pd.Series, target_month: pd.Timestamp, fx: pd.Series | None = None
) -> pd.DataFrame:
    """Build a single feature row for a month we do not have a price for.

    `history` is a month-indexed series of known or already-predicted prices,
    ending the month before `target_month`. Used by the recursive forecaster.
    FX columns follow the same rule as build_features; the forecaster passes
    an extended copy of `fx` when FX is in use.
    """
    history = history.sort_index()
    values = {}

    for n in config.LAGS:
        stamp = target_month - pd.DateOffset(months=n)
        values[f"lag_{n}"] = history.get(stamp, np.nan)

    prior = history[history.index < target_month]
    for w in config.ROLL_WINDOWS:
        window = prior.tail(w)
        values[f"roll_mean_{w}"] = window.mean() if len(window) == w else np.nan
        values[f"roll_std_{w}"] = window.std() if len(window) == w else np.nan

    values["mom_change"] = (
        values["lag_1"] / values["lag_2"] - 1.0
        if values.get("lag_2") not in (None, 0) and not pd.isna(values.get("lag_2"))
        else np.nan
    )
    values["month_sin"] = np.sin(2 * np.pi * target_month.month / 12)
    values["month_cos"] = np.cos(2 * np.pi * target_month.month / 12)

    # Same T-1 rule as training. Beyond the last month in `fx` these are NaN;
    # the forecaster decides whether to extend `fx` (see predict.forecast).
    if fx is not None or uses_fx():
        fx_row = _fx_features(
            _fx() if fx is None else fx, pd.DatetimeIndex([target_month])
        ).iloc[0]
        values["fx_lag1"] = fx_row["fx_lag1"]
        values["fx_change_3m"] = fx_row["fx_change_3m"]

    return pd.DataFrame([values], index=[target_month])[config.FEATURES]
