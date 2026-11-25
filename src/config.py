"""Central settings. Change things here, not scattered through the code."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

RAW_CSV = DATA_DIR / "wfp_food_prices_ken.csv"
# CBK USD/KES export (date + mean rate). Loaded by src.drivers.
FX_CSV = DATA_DIR / "cbk_usd_kes.csv"
# Where the USD/KES series comes from:
#   "auto" - FX_CSV if it exists, otherwise the rate implied by the WFP file
#   "cbk"  - FX_CSV only (error if missing)
#   "wfp"  - implied rate from WFP price / usdprice only
FX_SOURCE = "auto"
ARTIFACT = MODEL_DIR / "beialert_models.joblib"

# --- which rows of the WFP file we model -------------------------------------
# priceflag "actual" excludes WFP's own aggregated estimates.
PRICE_FLAGS = ["actual"]

# How prices are grouped into series:
#   "market"   - one series per (commodity, market, unit, pricetype).
#   "national" - one series per (commodity, unit, pricetype), using the median
#                across all markets reporting that month. The key carries
#                NATIONAL in the market slot so it still splits into four parts.
AGGREGATION = "national"

# A series must have at least this many usable monthly observations after
# lagging, or it is skipped. Below ~36 the train/test split gets thin.
MIN_OBSERVATIONS = 36

# Interpolate gaps no longer than this many consecutive months. WFP series
# are patchy; one missing month otherwise invalidates every later row whose
# lag window reaches back over it. Longer gaps are left as gaps, because
# inventing half a year of prices would be dishonest.
MAX_GAP_FILL = 2

# Fraction of each series used for training; the rest is the held-out test set.
TRAIN_FRACTION = 0.80

# Skip series that stopped being monitored. Measured against the newest date
# anywhere in the file, so it stays correct as you drop in newer downloads.
# A market last seen in 2019 cannot forecast anything useful today.
MAX_STALENESS_MONTHS = 12

# --- features ----------------------------------------------------------------
# Long lags are expensive on gappy data: lag_12 invalidates a row whenever
# the month a year earlier is missing. Seasonality is carried by month_sin
# and month_cos instead, which cost nothing. Add 6 and 12 back only if
# python -m src.diagnose shows your series are complete enough to afford it.
LAGS = [1, 2, 3]
ROLL_WINDOWS = [3]

FEATURES = (
    [f"lag_{n}" for n in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + ["mom_change", "month_sin", "month_cos"]
)

# USD/KES exchange-rate features, built from T-1 and earlier only (see
# features.build_features). Deliberately NOT in FEATURES: the controlled A/B
# in src/compare_fx.py (same rows, same split, same test months) found no
# improvement - 6 series better, 4 worse, median MASE change -0.004, and
# shippable stayed at 6 of 11. Re-run `python -m src.compare_fx` with fresh
# data before adding them back with FEATURES = FEATURES + FX_FEATURES.
FX_FEATURES = ["fx_lag1", "fx_change_3m"]
TARGET = "price"

# --- forecasting -------------------------------------------------------------
MAX_HORIZON = 6          # months ahead the API will return
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# --- pooled global model (src.pooled, src.train_pooled) -----------------------
# A series needs at least this many observed months to contribute rows.
POOLED_MIN_OBS = 4
# Global chronological cutoff, applied to the target month (row month + 1):
# rows whose target falls before this train, the rest test.
POOLED_TEST_FROM = "2024-10"
# Per-series ranking only includes series with at least this many test rows.
POOLED_MIN_TEST_ROWS = 6
