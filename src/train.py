"""Train one model per series and save everything the API needs.

Run:  python -m src.train

The split is strictly chronological and asserted, so no future month can
reach the model during training. Every series is scored against two baselines:
naive ("next month equals this month") and seasonal naive ("next month equals
the same month last year"). A series is shippable only if the chosen model
beats both, i.e. both of its MASE scores are below 1.0.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data, features


def _ratio(mae: float, baseline_mae: float) -> float | None:
    return round(float(mae / baseline_mae), 3) if baseline_mae > 0 else None


def metrics(actual, predicted, naive, seasonal) -> dict:
    mae = mean_absolute_error(actual, predicted)
    return {
        "mae": round(float(mae), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 3),
        "r2": round(float(r2_score(actual, predicted)), 3),
        # MASE < 1 means we beat the naive (lag_1) baseline on this series.
        "mase": _ratio(mae, mean_absolute_error(actual, naive)),
        # Same, against the seasonal naive (same month last year) baseline.
        "mase_seasonal": _ratio(mae, mean_absolute_error(actual, seasonal)),
    }


def seasonal_naive(monthly: pd.Series, test: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Price twelve months before each test month, from the series history.

    Where that month has no price, fall back to lag_1. Returns the
    predictions and how many test months needed the fallback.
    """
    year_ago = monthly.shift(12).reindex(test.index)
    missing = year_ago.isna()
    filled = year_ago.where(~missing, test["lag_1"])
    return filled.to_numpy(), int(missing.sum())


def pick_alpha(train: pd.DataFrame, feature_cols: list[str] | None = None) -> float:
    """Chronological inner CV over alpha. Falls back if the series is short."""
    feature_cols = feature_cols or config.FEATURES
    if len(train) < 40:
        return 1.0

    splitter = TimeSeriesSplit(n_splits=3)
    scores = {}

    for alpha in config.RIDGE_ALPHAS:
        errors = []
        for tr_idx, val_idx in splitter.split(train):
            fold_tr, fold_val = train.iloc[tr_idx], train.iloc[val_idx]
            model = Pipeline(
                [("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]
            )
            model.fit(fold_tr[feature_cols], fold_tr[config.TARGET])
            pred = model.predict(fold_val[feature_cols])
            errors.append(
                np.sqrt(mean_squared_error(fold_val[config.TARGET], pred))
            )
        scores[alpha] = float(np.mean(errors))

    return min(scores, key=scores.get)


def evaluate(
    key: str, frame: pd.DataFrame, monthly: pd.Series, feature_cols: list[str]
) -> dict:
    """Chronological split, fit ridge + forest, score, pick the winner.

    Shared by train_one and src.compare_fx so both use identical selection.
    `frame` must already be the final row set; nothing is dropped here.
    """
    cut = int(len(frame) * config.TRAIN_FRACTION)
    train, test = frame.iloc[:cut], frame.iloc[cut:]
    assert train.index.max() < test.index.min(), f"{key}: split leaks time"

    X_train, y_train = train[feature_cols], train[config.TARGET]
    X_test, y_test = test[feature_cols], test[config.TARGET]

    naive = X_test["lag_1"].to_numpy()  # last month's price
    seasonal, seasonal_fallbacks = seasonal_naive(monthly, test)

    alpha = pick_alpha(train, feature_cols)
    ridge = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    forest = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    forest.fit(X_train, y_train)
    forest_pred = forest.predict(X_test)

    scores = {
        "naive": metrics(y_test, naive, naive, seasonal),
        "seasonal_naive": metrics(y_test, seasonal, naive, seasonal),
        "ridge": metrics(y_test, ridge_pred, naive, seasonal),
        "random_forest": metrics(y_test, forest_pred, naive, seasonal),
    }

    # Ship whichever actually wins on held-out RMSE.
    best = min(("ridge", "random_forest"), key=lambda m: scores[m]["rmse"])
    chosen, predictions = (ridge, ridge_pred) if best == "ridge" else (forest, forest_pred)

    chosen_scores = scores[best]
    shippable = all(
        chosen_scores[k] is not None and chosen_scores[k] < 1.0
        for k in ("mase", "mase_seasonal")
    )

    return {
        "test": test,
        "naive": naive,
        "seasonal": seasonal,
        "scores": scores,
        "best": best,
        "chosen": chosen,
        "predictions": predictions,
        "alpha": alpha,
        "shippable": shippable,
        "seasonal_fallbacks": seasonal_fallbacks,
    }


def active_series(
    monthly: dict[str, pd.Series],
) -> tuple[dict[str, pd.Series], pd.Timestamp, pd.Timestamp]:
    """Series still being monitored, relative to the newest date in the file."""
    dated = {k: s for k, s in monthly.items() if s.last_valid_index() is not None}
    newest = max(s.last_valid_index() for s in dated.values())
    cutoff = newest - pd.DateOffset(months=config.MAX_STALENESS_MONTHS)
    fresh = {k: s for k, s in dated.items() if s.last_valid_index() >= cutoff}
    return fresh, newest, cutoff


def train_one(key: str, monthly: pd.Series) -> dict | None:
    frame = features.model_frame(monthly)
    if len(frame) < config.MIN_OBSERVATIONS:
        return None

    ev = evaluate(key, frame, monthly, config.FEATURES)
    test, scores, best = ev["test"], ev["scores"], ev["best"]
    chosen, predictions, alpha = ev["chosen"], ev["predictions"], ev["alpha"]
    y_test = test[config.TARGET]

    # Held-out results, captured before the refit below replaces the model
    # that produced them. src.evaluate draws its figures from these.
    test_results = {
        "test_index": [m.date().isoformat() for m in test.index],
        "test_actual": [float(v) for v in y_test],
        "test_predicted": [float(v) for v in predictions],
        "test_naive": [float(v) for v in ev["naive"]],
        "test_seasonal": [float(v) for v in ev["seasonal"]],
    }

    # Refit on everything so the live forecast uses all known history.
    chosen.fit(frame[config.FEATURES], frame[config.TARGET])

    commodity, market, unit, pricetype = key.split("|")
    return {
        "series_key": key,
        "commodity": commodity.replace("_", " "),
        "market": market.replace("_", " "),
        "unit": unit.replace("_", " "),
        "pricetype": pricetype.replace("_", " "),
        "model": chosen,
        "model_name": best,
        "alpha": alpha if best == "ridge" else None,
        "residual_std": float(np.std(y_test.to_numpy() - predictions)),
        "history": monthly.dropna(),
        "last_observed_month": monthly.last_valid_index(),
        "last_observed_price": float(monthly.dropna().iloc[-1]),
        "n_observations": int(len(frame)),
        "scores": scores,
        "seasonal_fallbacks": ev["seasonal_fallbacks"],
        "n_test": int(len(test)),
        "shippable": ev["shippable"],
        **test_results,
    }


def main() -> None:
    raw = data.clean(data.load_raw())
    monthly = data.to_monthly(raw)
    print(f"series found: {len(monthly)}")

    # Drop series that stopped being monitored, relative to the newest date
    # in the file. Forecasting a market last seen in 2019 is not useful.
    fresh, newest, cutoff = active_series(monthly)
    print(
        f"newest data: {newest:%Y-%m} | active since {cutoff:%Y-%m}: "
        f"{len(fresh)} series, {len(monthly) - len(fresh)} stale and dropped"
    )

    trained, skipped = {}, 0
    for key, series in fresh.items():
        result = train_one(key, series)
        if result is None:
            skipped += 1
            continue
        trained[key] = result

    if not trained:
        raise SystemExit(
            "No series had enough history. Lower MIN_OBSERVATIONS in config.py "
            "or check the filters in data.clean()."
        )

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"trained_at": datetime.now(timezone.utc).isoformat(), "series": trained},
        config.ARTIFACT,
    )

    report = pd.DataFrame(
        [
            {
                "series": k,
                "n": v["n_observations"],
                "shipped": v["model_name"],
                "ridge_mase": v["scores"]["ridge"]["mase"],
                "ridge_smase": v["scores"]["ridge"]["mase_seasonal"],
                "forest_mase": v["scores"]["random_forest"]["mase"],
                "forest_smase": v["scores"]["random_forest"]["mase_seasonal"],
                "rmse": v["scores"][v["model_name"]]["rmse"],
                "s_fallback": f'{v["seasonal_fallbacks"]}/{v["n_test"]}',
                "shippable": v["shippable"],
            }
            for k, v in trained.items()
        ]
    ).sort_values("ridge_mase")

    print(f"\ntrained {len(trained)} series, skipped {skipped} (too short)")
    print("(mase = vs naive lag_1, smase = vs seasonal naive; "
          "s_fallback = test months where the year-ago price was missing "
          "and lag_1 was used)")
    print(report.to_string(index=False))

    beat = (report[["ridge_mase", "forest_mase"]].min(axis=1) < 1.0).sum()
    fallbacks = sum(v["seasonal_fallbacks"] for v in trained.values())
    test_months = sum(v["n_test"] for v in trained.values())
    print(f"\nseries beating the naive baseline: {beat} of {len(report)}")
    print(f"shippable (chosen model beats both baselines): "
          f"{int(report['shippable'].sum())} of {len(report)}")
    print(f"seasonal baseline fell back to lag_1 on {fallbacks} of "
          f"{test_months} test months")
    print(f"saved -> {config.ARTIFACT}")

    (config.MODEL_DIR / "training_report.json").write_text(
        json.dumps(report.to_dict(orient="records"), indent=2, default=str)
    )


if __name__ == "__main__":
    main()
