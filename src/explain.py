"""Explain a single forecast: which features moved it, and by how much.

For Ridge this is exact. A linear model's prediction is the intercept plus
the sum of coefficient x scaled-feature, so each term is that feature's
contribution in KES, measured against the training average.

Random Forest has no such decomposition without extra machinery, so it
reports global feature importance instead, clearly labelled as such.

When fuel, exchange rate and rainfall features are added later, they appear
here automatically. Nothing in this file needs to change.
"""

from __future__ import annotations

import pandas as pd

from . import config

# Human wording for the Drivers screen. Unknown features fall back to the
# raw column name, so new features still show up rather than disappearing.
LABELS = {
    "lag_1": "Last month's price",
    "lag_2": "Price two months ago",
    "lag_3": "Price three months ago",
    "lag_6": "Price six months ago",
    "lag_12": "Price a year ago",
    "roll_mean_3": "Three-month average",
    "roll_mean_6": "Six-month average",
    "roll_std_3": "Recent volatility (3mo)",
    "roll_std_6": "Recent volatility (6mo)",
    "mom_change": "Month-on-month momentum",
    "season": "Time of year",
    "fx_lag1": "Exchange rate (USD/KES)",
    "fx_change_3m": "Exchange rate trend, 3mo",
}

# month_sin and month_cos are one idea split across two columns; they are
# summed into a single "season" entry so the output reads sensibly.
SEASON_PARTS = ("month_sin", "month_cos")


def _label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").capitalize())


def _merge_season(pairs: dict[str, float]) -> dict[str, float]:
    season = sum(pairs.pop(p, 0.0) for p in SEASON_PARTS)
    if season:
        pairs["season"] = season
    return pairs


def ridge_contributions(pipeline, row: pd.DataFrame) -> dict[str, float]:
    """Exact per-forecast contribution of each feature, in KES."""
    scaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]
    scaled = scaler.transform(row[config.FEATURES])[0]
    return dict(zip(config.FEATURES, model.coef_ * scaled))


def forest_importances(model) -> dict[str, float]:
    """Global importance. Not specific to this forecast."""
    return dict(zip(config.FEATURES, model.feature_importances_))


def narrate(ranked: list[dict], exact: bool) -> str:
    """One sentence a non-technical reader can follow."""
    if not ranked:
        return "No feature breakdown is available for this forecast."

    top = ranked[0]
    opening = (
        f"Mostly carried by {_label(top['feature']).lower()}"
        if exact
        else f"This model relies most on {_label(top['feature']).lower()}"
    )

    if len(ranked) > 1:
        second = ranked[1]
        direction = "pushing it up" if second["direction"] == "up" else "pulling it down"
        return (
            f"{opening} ({top['share_pct']}% of the movement), "
            f"with {_label(second['feature']).lower()} {direction}."
        )
    return f"{opening} ({top['share_pct']}% of the movement)."


def explain(entry: dict, row: pd.DataFrame) -> dict:
    """Build the explanation block for one forecast."""
    model = entry["model"]
    exact = entry["model_name"] == "ridge"

    if exact:
        raw = ridge_contributions(model, row)
        unit = "kes"
    else:
        raw = forest_importances(model)
        unit = "importance"

    merged = _merge_season(dict(raw))
    total = sum(abs(v) for v in merged.values())
    if total == 0:
        return {"summary": "No feature breakdown is available.", "contributions": []}

    ranked = sorted(
        (
            {
                "feature": name,
                "label": _label(name),
                "effect": round(float(value), 3),
                "share_pct": round(abs(value) / total * 100, 1),
                "direction": "up" if value >= 0 else "down",
            }
            for name, value in merged.items()
        ),
        key=lambda d: d["share_pct"],
        reverse=True,
    )

    return {
        "summary": narrate(ranked, exact),
        "method": "ridge_coefficients" if exact else "random_forest_global_importance",
        "is_per_forecast": exact,
        "effect_unit": unit,
        "caveat": (
            "Contributions show what the model used, not proven cause. "
            "The model currently sees price history only — fuel, exchange "
            "rate and rainfall are not yet inputs."
        ),
        "contributions": ranked,
    }


def data_quality(entry: dict, today: pd.Timestamp | None = None) -> dict:
    """Flag series whose market stopped being monitored."""
    today = today or pd.Timestamp.today().normalize()
    last = entry["last_observed_month"]
    months = (today.year - last.year) * 12 + (today.month - last.month)
    stale = months > config.MAX_STALENESS_MONTHS

    return {
        "last_observed_month": last.strftime("%Y-%m"),
        "months_since_last_observation": int(months),
        "is_stale": bool(stale),
        "note": (
            f"This market has no reported prices since {last:%B %Y}. "
            "The forecast continues from that date, not from today."
            if stale
            else "Data is current."
        ),
    }