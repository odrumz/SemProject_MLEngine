"""Train and evaluate the pooled global model.

    python -m src.train_pooled

Splits the stacked table from src.pooled at config.POOLED_TEST_FROM, fits
Ridge and a tree model (XGBoost if installed, otherwise random forest),
and scores both in price space against the naive "next month = this month"
baseline. Nothing is saved; this path exists to compare against src.train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, pooled
from .train import metrics

def _tree():
    """XGBoost if installed, otherwise random forest.

    Imported here rather than at module level: xgboost lives in
    requirements-train.txt, not the deploy requirements, so this module must
    import cleanly without it.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        from sklearn.ensemble import RandomForestRegressor

        return "random_forest", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1,
        )
    return "xgboost", XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1,
    )


def _pipeline(model) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), pooled.NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), pooled.CATEGORICAL),
        ]
    )
    return Pipeline([("prep", prep), ("model", model)])


def main() -> None:
    frame = pooled.build()

    # Split on the month the target belongs to, not the row's own month.
    # A row dated m predicts the price in m+1, so splitting on m would put a
    # training row dated just before the cutoff with its target inside the
    # test period.
    cutoff = pd.Timestamp(config.POOLED_TEST_FROM + "-01")
    target_month = frame["month"] + pd.DateOffset(months=1)
    train = frame[target_month < cutoff]
    test = frame[target_month >= cutoff].copy()
    print(f"split at {config.POOLED_TEST_FROM}: {len(train):,} train rows, "
          f"{len(test):,} test rows\n")

    X_cols = pooled.NUMERIC + pooled.CATEGORICAL
    tree_name, tree_model = _tree()
    models = {"ridge": Ridge(alpha=1.0), tree_name: tree_model}

    # Price space: predicted log return -> price, compared with the actual
    # next price. The naive forecast is last_price itself.
    actual = test["last_price"] * np.exp(test[pooled.TARGET])
    naive = test["last_price"]

    rows = []
    for name, model in models.items():
        pipe = _pipeline(model).fit(train[X_cols], train[pooled.TARGET])
        test[f"pred_{name}"] = test["last_price"] * np.exp(pipe.predict(test[X_cols]))
        rows.append({"model": name, **metrics(actual, test[f"pred_{name}"], naive)})
    rows.append({"model": "naive", **metrics(actual, naive, naive)})

    table = pd.DataFrame(rows)[["model", "mae", "rmse", "mase"]]
    print(table.to_string(index=False))

    best = min((r for r in rows if r["model"] != "naive"), key=lambda r: r["mase"])["model"]
    print(f"\nbetter pooled model: {best}\n")

    test["actual"] = actual
    per_series = []
    for key, part in test.groupby("series_key", sort=False):
        model_mae = (part["actual"] - part[f"pred_{best}"]).abs().mean()
        naive_mae = (part["actual"] - part["last_price"]).abs().mean()
        per_series.append(
            {
                "series_key": key,
                "n_test": len(part),
                "mase": round(model_mae / naive_mae, 3) if naive_mae > 0 else np.nan,
            }
        )
    per = pd.DataFrame(per_series)
    # A MASE over one or two rows says nothing, so short series are left out.
    too_few = (per["n_test"] < config.POOLED_MIN_TEST_ROWS).sum()
    per = per[per["n_test"] >= config.POOLED_MIN_TEST_ROWS]
    per = per.sort_values("mase", na_position="last")
    print(f"ranking series with at least {config.POOLED_MIN_TEST_ROWS} test rows; "
          f"{too_few} series excluded for having fewer\n")

    print(per.head(20).to_string(index=False))
    undefined = per["mase"].isna().sum()
    print(
        f"\nseries beating naive (MASE < 1.0): {(per['mase'] < 1.0).sum()} of "
        f"{len(per)} in the test period"
        + (f" ({undefined} with undefined MASE: naive error was zero)" if undefined else "")
    )


if __name__ == "__main__":
    main()
