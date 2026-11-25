"""Turn a forecast into plain English.

This is template-based, not a language model. Every sentence is derived
from a number in the forecast, so it cannot invent a figure, costs nothing
to run, works offline, and gives the same wording for the same inputs.
A language model would be able to phrase things more naturally, but it
would also occasionally state a price that is not in the data, and that is
not a trade worth making for an app about prices.
"""

from __future__ import annotations

import pandas as pd

from . import config


def _where(market: str) -> str:
    """'nationally' for the national median series, else 'in <market>'."""
    return "nationally" if market.strip().upper() == "NATIONAL" else f"in {market}"


def _inputs_sentence() -> str:
    """What the model can and cannot see, read from config.FEATURES.

    Kept in step with the live feature set so it stays true if the FX
    features (config.FX_FEATURES) are ever added back.
    """
    if any(f in config.FEATURES for f in config.FX_FEATURES):
        return (
            "The model sees price history and the USD/KES exchange rate only, "
            "so it cannot attribute this to fuel costs or rainfall."
        )
    return (
        "The model sees price history only, so it cannot attribute this to "
        "the exchange rate, fuel costs or rainfall."
    )


def _month_name(stamp: str) -> str:
    return pd.Timestamp(stamp + "-01").strftime("%B %Y")


def _volatility(history: pd.Series, months: int = 24) -> float | None:
    """Typical month-to-month swing, as a percentage."""
    recent = history.dropna().tail(months + 1)
    if len(recent) < 6:
        return None
    changes = recent.pct_change().dropna().abs()
    return float(changes.mean() * 100) if len(changes) else None


def _volatility_sentence(vol: float | None) -> str | None:
    if vol is None:
        return None
    if vol > 8:
        return (
            f"This is a volatile series: prices have moved an average of "
            f"{vol:.0f}% month to month over the past two years, which is "
            f"why the range is wide."
        )
    if vol >= 5:
        return (
            f"Prices here move moderately, an average of {vol:.0f}% month "
            f"to month over the past two years."
        )
    return (
        f"Prices here are relatively steady, moving an average of "
        f"{vol:.0f}% month to month over the past two years."
    )


def _not_shippable(result: dict, history: pd.Series) -> dict:
    """Summary for a series whose model did not beat both baselines.

    The model's own number is not presented as the forecast; the fallback is
    that the price stays near its last observed value.
    """
    currency = result["currency"]
    last_price = result["last_observed_price"]
    headline = (
        f"For {result['commodity']} {_where(result['market'])}, the best available "
        f"estimate is that the price stays near its last observed value of "
        f"{currency} {last_price:,.0f} per {result['unit']} "
        f"({_month_name(result['last_observed_month'])})."
    )

    quality = result.get("model_quality") or {}
    note = (
        "The trained model did not beat both baselines on held-out months "
        "(last month's price, and the price in the same month a year "
        "earlier), so its forecast is not used here"
    )
    figures = [
        f"{label} {quality[key]:.2f}"
        for label, key in (("MASE", "mase"), ("seasonal MASE", "mase_seasonal"))
        if quality.get(key) is not None
    ]
    note += f" ({', '.join(figures)}; below 1.0 is needed on both)." if figures else "."
    sentences = [note]

    vol = _volatility(history)
    vol_sentence = _volatility_sentence(vol)
    if vol_sentence:
        sentences.append(vol_sentence)

    dq = result.get("data_quality") or {}
    if dq.get("is_stale"):
        sentences.append(dq["note"])

    return {
        "headline": headline,
        "confidence": "none",
        "actionable": False,
        "uncertainty_pct": None,
        "historical_volatility_pct": round(vol, 1) if vol is not None else None,
        "sentences": sentences,
        "text": headline + " " + " ".join(sentences),
    }


def build(result: dict, history: pd.Series) -> dict:
    """Return a summary block: a headline, a verdict, and the reasoning."""
    points = result.get("forecast") or []
    if not points:
        sentences = [
            "There is not enough recent history to build the features the "
            "model needs."
        ]
        if result.get("blocked_reason"):
            sentences.append(result["blocked_reason"])
        return {
            "headline": "No forecast could be produced for this series.",
            "confidence": "unknown",
            "sentences": sentences,
            "actionable": False,
        }

    if not result.get("shippable", True):
        return _not_shippable(result, history)

    first = points[0]
    currency = result["currency"]
    predicted = first["predicted_price"]
    change = first["change_pct"]
    band = first["upper_80"] - first["lower_80"]
    half_band_pct = (band / 2) / predicted * 100 if predicted else 0.0

    direction = "rise" if change > 0 else "fall" if change < 0 else "hold steady"
    verb = "to rise" if change > 0 else "to fall" if change < 0 else "to hold steady"

    sentences: list[str] = []

    # 1. What the model expects.
    headline = (
        f"{result['commodity']} {_where(result['market'])} is forecast {verb}"
        f"{'' if change == 0 else f' {abs(change):.1f}%'} to about "
        f"{currency} {predicted:,.0f} per {result['unit']} "
        f"by {_month_name(first['month'])}."
    )

    # 2. Whether that expectation is worth acting on. The decisive test is
    #    the predicted move against the width of the uncertainty band.
    actionable = abs(change) > half_band_pct
    if actionable:
        confidence = "moderate"
        sentences.append(
            f"The predicted move is larger than the model's uncertainty "
            f"(±{half_band_pct:.0f}%), so the direction is meaningful."
        )
    else:
        confidence = "low"
        sentences.append(
            f"Treat the direction with caution: the uncertainty is "
            f"±{half_band_pct:.0f}%, wider than the {abs(change):.1f}% move "
            f"itself, so a {direction} is plausible but not established. "
            f"The 80% range runs {currency} {first['lower_80']:,.0f} to "
            f"{currency} {first['upper_80']:,.0f}."
        )

    # 3. How volatile this series is historically, for context.
    vol = _volatility(history)
    vol_sentence = _volatility_sentence(vol)
    if vol_sentence:
        sentences.append(vol_sentence)

    # 4. What drove the number.
    explanation = result.get("explanation") or {}
    contributions = explanation.get("contributions") or []
    if contributions:
        top = contributions[0]
        sentences.append(
            f"The forecast leans mainly on {top['label'].lower()} "
            f"({top['share_pct']}% of the movement). {_inputs_sentence()}"
        )

    # 5. Honest limits.
    quality = result.get("model_quality") or {}
    if quality.get("mase") is not None:
        if quality.get("beats_naive_baseline"):
            gain = (1 - quality["mase"]) * 100
            sentences.append(
                f"On held-out months this model was {gain:.0f}% more accurate "
                f"than simply assuming next month equals this month."
            )
        else:
            sentences.append(
                "On held-out months this model was no better than assuming "
                "next month equals this month, so the forecast should not be "
                "relied on."
            )

    dq = result.get("data_quality") or {}
    if dq.get("is_stale"):
        sentences.append(dq["note"])

    if len(points) > 1:
        last = points[-1]
        sentences.append(
            f"Beyond one month the forecast is recursive and compounds its "
            f"own error: by {_month_name(last['month'])} the range widens to "
            f"{currency} {last['lower_80']:,.0f}–{last['upper_80']:,.0f}."
        )
        if result.get("fx_assumption"):
            held_from = result["fx_assumption"].rsplit(" ", 1)[-1]
            sentences.append(
                f"The exchange rate was assumed to hold at its last known "
                f"level ({_month_name(held_from)}), since USD/KES for future "
                f"months is not yet observable."
            )

    return {
        "headline": headline,
        "confidence": confidence,
        "actionable": bool(actionable),
        "uncertainty_pct": round(half_band_pct, 1),
        "historical_volatility_pct": round(vol, 1) if vol is not None else None,
        "sentences": sentences,
        "text": headline + " " + " ".join(sentences),
    }
