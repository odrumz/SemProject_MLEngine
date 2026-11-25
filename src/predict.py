"""Load the saved models and produce forecasts.

Multi-month forecasting is recursive: the prediction for month t+1 is fed
back in as the lag_1 feature for t+2. Errors compound, so the uncertainty
band widens with the square root of the horizon. Treat anything past about
three months as indicative.
"""

from __future__ import annotations

import functools

import pandas as pd

from . import config, explain, features, narrative

# 80% interval under a normal assumption.
Z_80 = 1.2816


@functools.lru_cache(maxsize=1)
def _load() -> dict:
    import joblib

    if not config.ARTIFACT.exists():
        raise FileNotFoundError(
            f"No model artifact at {config.ARTIFACT}. Run: python -m src.train"
        )
    return joblib.load(config.ARTIFACT)


def reload_models() -> None:
    """Call after retraining so a running API picks up the new file."""
    _load.cache_clear()


def trained_at() -> str:
    return _load()["trained_at"]


def is_shippable(entry: dict) -> bool:
    """Whether the chosen model beat both baselines on held-out months.

    Artifacts trained before the seasonal baseline existed have no flag;
    they are treated as shippable so older deployments keep working.
    """
    return bool(entry.get("shippable", True))


def list_series() -> list[dict]:
    """Everything the API can forecast, for the app's pickers."""
    out = []
    for key, entry in _load()["series"].items():
        out.append(
            {
                "series_key": key,
                "commodity": entry["commodity"],
                "market": entry["market"],
                "unit": entry["unit"],
                "pricetype": entry["pricetype"],
                "last_observed_month": entry["last_observed_month"].strftime("%Y-%m"),
                "last_observed_price": round(entry["last_observed_price"], 2),
                "observations": entry["n_observations"],
                "shippable": is_shippable(entry),
            }
        )
    return sorted(out, key=lambda r: (r["commodity"], r["market"]))


def get_series(series_key: str) -> dict:
    series = _load()["series"]
    if series_key not in series:
        raise KeyError(series_key)
    return series[series_key]


def forecast(series_key: str, horizon: int = 1) -> dict:
    """Recursive multi-step forecast for one series."""
    horizon = max(1, min(int(horizon), config.MAX_HORIZON))
    entry = get_series(series_key)

    model = entry["model"]
    history = entry["history"].copy()
    sigma = entry["residual_std"]
    last_price = float(history.iloc[-1])

    # FX handling only when the model actually uses FX features; otherwise
    # no rate is loaded and no assumption is reported.
    use_fx = features.uses_fx()
    fx = features._fx().copy() if use_fx else None
    fx_last_known = fx.index.max() if use_fx else None
    fx_assumption = None

    points = []
    first_row = None
    blocked_reason = None
    for step in range(1, horizon + 1):
        target_month = history.index.max() + pd.DateOffset(months=1)
        needed = target_month - pd.DateOffset(months=1)  # features use FX at T-1
        if use_fx and needed > fx.index.max():
            # FX for these months is not observed yet: hold the last known
            # rate for every month up to T-1, so fx_change_3m's T-4 lookup
            # never lands in a gap.
            gap = pd.date_range(fx.index.max(), needed, freq="MS")[1:]
            fx = pd.concat([fx, pd.Series(fx.loc[fx_last_known], index=gap)])
        fx_held = use_fx and needed > fx_last_known
        row = features.row_from_history(history, target_month, fx)

        if row.isna().any(axis=None):
            # Not enough history to build a complete feature row.
            if step == 1:
                blocked_reason = _blocked_reason(row, entry["history"], target_month)
            break

        if step == 1:
            first_row = row  # the explanation describes the next month

        predicted = float(model.predict(row)[0])
        # Compounding error: band widens with sqrt(step).
        band = Z_80 * sigma * (step ** 0.5)

        points.append(
            {
                "month": target_month.strftime("%Y-%m"),
                "horizon": step,
                "predicted_price": round(predicted, 2),
                "lower_80": round(max(predicted - band, 0.0), 2),
                "upper_80": round(predicted + band, 2),
                "change_pct": round((predicted / last_price - 1) * 100, 2),
            }
        )
        if fx_held:
            # Only recorded once a forecast point actually used the held rate.
            fx_assumption = f"held flat from {fx_last_known:%Y-%m}"
        history.loc[target_month] = predicted

    scores = entry["scores"][entry["model_name"]]
    result = {
        "series_key": series_key,
        "commodity": entry["commodity"],
        "market": entry["market"],
        "unit": entry["unit"],
        "pricetype": entry["pricetype"],
        "currency": "KES",
        "last_observed_month": entry["last_observed_month"].strftime("%Y-%m"),
        "last_observed_price": round(last_price, 2),
        "model": entry["model_name"],
        "model_quality": {
            "rmse": scores["rmse"],
            "mae": scores["mae"],
            "mase": scores["mase"],
            "beats_naive_baseline": bool(scores["mase"] is not None and scores["mase"] < 1.0),
            # None when the artifact predates the seasonal baseline.
            "mase_seasonal": scores.get("mase_seasonal"),
            "seasonal_fallbacks": entry.get("seasonal_fallbacks"),
        },
        "shippable": is_shippable(entry),
        "forecast": points,
        "explanation": (
            explain.explain(entry, first_row) if first_row is not None else None
        ),
        "data_quality": explain.data_quality(entry),
        "disclaimer": (
            "Forecasts are statistical estimates from historical WFP price data, "
            "not guarantees. Accuracy falls as the horizon increases."
        ),
    }
    if fx_assumption is not None:
        result["fx_assumption"] = fx_assumption
    if blocked_reason is not None:
        result["blocked_reason"] = blocked_reason
    result["summary"] = narrative.build(result, entry["history"])
    return result


def _blocked_reason(row: pd.DataFrame, history: pd.Series, target_month: pd.Timestamp) -> str:
    """Explain why no feature row could be built for `target_month`."""
    nan_cols = [col for col in row.columns if row[col].isna().any()]
    observed = history.dropna().index
    missing = [
        (target_month - pd.DateOffset(months=n)).strftime("%Y-%m")
        for n in range(4, 0, -1)
        if (target_month - pd.DateOffset(months=n)) not in observed
    ]
    reason = (
        f"Could not build features for {target_month.strftime('%Y-%m')}: "
        f"NaN in {', '.join(nan_cols)}."
    )
    if missing:
        reason += f" Missing months in the 4 months before it: {', '.join(missing)}."
    else:
        reason += " None of the 4 months before it are missing."
    return reason


def history(series_key: str, months: int = 24) -> dict:
    """Recent observed prices, for drawing the chart behind the forecast."""
    entry = get_series(series_key)
    tail = entry["history"].tail(max(1, int(months)))
    return {
        "series_key": series_key,
        "commodity": entry["commodity"],
        "market": entry["market"],
        "unit": entry["unit"],
        "currency": "KES",
        "points": [
            {"month": stamp.strftime("%Y-%m"), "price": round(float(value), 2)}
            for stamp, value in tail.items()
        ],
    }
