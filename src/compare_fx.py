"""Controlled A/B: does adding the FX features help?

Run:  python -m src.compare_fx

For each active series the feature frame is built once, with FX, and rows
missing any feature are dropped. Both arms then train on exactly those rows
with the same chronological split and the same selection logic
(train.evaluate). The only difference is the feature list, so any change in
MASE is down to the FX columns, not to different rows or test months.

MASE here is the chosen model's (the one train would ship), against the
naive lag_1 baseline. Shippable means the chosen model beats both baselines.
Nothing is saved.
"""

from __future__ import annotations

import contextlib
import io

import pandas as pd

from . import config, data, drivers, features, train

# Independent of whether FX is currently in config.FEATURES, so the
# comparison is reproducible either way.
FX_FEATURES = list(config.FX_FEATURES)
BASE_FEATURES = [f for f in config.FEATURES if f not in FX_FEATURES]
ALL_FEATURES = BASE_FEATURES + FX_FEATURES

# Differences smaller than this (scores are rounded to 3 dp) count as "same".
TOLERANCE = 0.001


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        monthly = data.to_monthly(data.clean(data.load_raw()))
    fresh, _newest, _cutoff = train.active_series(monthly)
    fx = drivers.load_fx()
    print(f"without FX: {BASE_FEATURES}")
    print(f"with FX:    adds {FX_FEATURES}")

    rows = []
    for key, series in fresh.items():
        # All features, FX included; rows missing any of them are dropped once.
        frame = features.build_features(series, fx).dropna(
            subset=ALL_FEATURES + [config.TARGET]
        )
        if len(frame) < config.MIN_OBSERVATIONS:
            continue

        a = train.evaluate(key, frame, series, BASE_FEATURES)
        b = train.evaluate(key, frame, series, ALL_FEATURES)
        assert a["test"].index.equals(b["test"].index), f"{key}: test months differ"

        mase_a = a["scores"][a["best"]]["mase"]
        mase_b = b["scores"][b["best"]]["mase"]
        diff = mase_b - mase_a
        rows.append(
            {
                "series": key.split("|")[0],
                "n": len(frame),
                "test": f"{a['test'].index.min():%Y-%m}..{a['test'].index.max():%Y-%m} ({len(a['test'])})",
                "model_noFX": a["best"],
                "mase_noFX": mase_a,
                "model_FX": b["best"],
                "mase_FX": mase_b,
                "diff": round(diff, 3),
                "smase_noFX": a["scores"][a["best"]]["mase_seasonal"],
                "smase_FX": b["scores"][b["best"]]["mase_seasonal"],
                "ship_noFX": a["shippable"],
                "ship_FX": b["shippable"],
                "ship_change": (
                    "gained" if b["shippable"] and not a["shippable"]
                    else "lost" if a["shippable"] and not b["shippable"]
                    else ""
                ),
            }
        )

    if not rows:
        raise SystemExit("No series had enough rows to compare.")

    table = pd.DataFrame(rows).sort_values("diff")
    print("\n(mase = chosen model vs naive lag_1; diff = with FX minus without, "
          "negative is better)")
    print(table.to_string(index=False))

    improved = int((table["diff"] <= -TOLERANCE).sum())
    worsened = int((table["diff"] >= TOLERANCE).sum())
    same = len(table) - improved - worsened
    print(f"\nseries compared: {len(table)}")
    print(f"improved: {improved} | worsened: {worsened} | same: {same}")
    print(f"median MASE change: {table['diff'].median():+.3f}")
    print(f"shippable: {int(table['ship_noFX'].sum())} without FX -> "
          f"{int(table['ship_FX'].sum())} with FX "
          f"(gained {int((table['ship_change'] == 'gained').sum())}, "
          f"lost {int((table['ship_change'] == 'lost').sum())})")


if __name__ == "__main__":
    main()
