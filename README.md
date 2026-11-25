# BeiAlert — price forecasting engine

Monthly commodity price forecasts for Kenyan markets, trained on WFP food
price data and served over HTTP for the BeiAlert Android app.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-train.txt   # optional: xgboost, for src.train_pooled
```

Download **Kenya - Food Prices** from
<https://data.humdata.org/dataset/wfp-food-prices-for-kenya>
and save it as `data/wfp_food_prices_ken.csv`.

## Train

```bash
python -m src.train
```

Prints a per-series table and saves `models/beialert_models.joblib`.
Watch the **MASE** column: below 1.0 means the model beats a naive
"next month equals this month" forecast. Above 1.0 means it does not, and
that series should not ship.

## Run

```bash
uvicorn src.api:app --reload
```

- Test console: <http://127.0.0.1:8000/>
- Interactive API docs: <http://127.0.0.1:8000/docs>

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness, and whether models are loaded |
| GET | `/markets` | List of market names |
| GET | `/series` | Every forecastable series; filter with `?market=` or `?commodity=` |
| GET | `/forecast` | `?series_key=…&horizon=1..6` |
| GET | `/history` | `?series_key=…&months=24` observed prices for charting |

A `series_key` looks like `Maize_flour|Nakuru|KG|Retail` — commodity, market,
unit and price type joined by `|`. Unit is part of the key on purpose: the
same commodity appears as both a 90 KG wholesale sack and a 1 KG retail
packet, and those are different series.

### Example response

```json
{
  "series_key": "Maize_flour|Nakuru|KG|Retail",
  "commodity": "Maize flour",
  "market": "Nakuru",
  "unit": "KG",
  "currency": "KES",
  "last_observed_month": "2026-09",
  "last_observed_price": 62.4,
  "model": "ridge",
  "model_quality": {
    "rmse": 4.07, "mae": 3.1, "mase": 0.96, "beats_naive_baseline": true
  },
  "forecast": [
    { "month": "2026-10", "horizon": 1, "predicted_price": 64.2,
      "lower_80": 59.1, "upper_80": 69.3, "change_pct": 2.88 }
  ]
}
```

## Updating with new data

WFP publishes monthly. Replace `data/wfp_food_prices_ken.csv` with the new
download, re-run `python -m src.train`, and push. Render rebuilds and
retrains automatically.

## Deployment

Deployed on Render as a Python web service, configured by `render.yaml`:

- **Build:** `pip install -r requirements.txt && python -m src.train`. The
  model is trained at build time, so `models/*.joblib` is not committed;
  `data/wfp_food_prices_ken.csv` must be.
- **Start:** `uvicorn src.api:app --host 0.0.0.0 --port $PORT`.
- **Python:** pinned to 3.10.12 via `PYTHON_VERSION`.

Steps:

1. Push the repo to GitHub, including `data/wfp_food_prices_ken.csv`.
2. On Render, choose **New → Blueprint** and pick the repo. Render reads
   `render.yaml` and creates the `beialert-api` service.
3. Wait for the build (training runs here), then open `/health`. It should
   report `models_loaded: true` and the series count.

xgboost is deliberately not in `requirements.txt`. Only the offline pooled
experiment (`src.train_pooled`) uses it, and on Linux it pulls a ~200 MB
NVIDIA dependency. Install `requirements-train.txt` locally if you need it.

GitHub Pages cannot host this. It serves static files only and cannot run
Python.

## Layout

```
src/config.py        settings: filters, features, horizons
src/data.py          load, clean, collapse to monthly series
src/features.py      lags, rolling stats, cyclical month (+ optional FX columns)
src/drivers.py       USD/KES loader: CBK export or rate implied by the WFP file
src/train.py         chronological split, Ridge vs Random Forest vs naive
src/compare_fx.py    controlled A/B: same rows and split, with vs without FX
src/diagnose.py      why series are dropped: rows surviving feature building per lag set
src/pooled.py        stacked log-return table across series, for the pooled model
src/train_pooled.py  offline pooled-model experiment (not used in production)
src/predict.py       recursive multi-month forecasting
src/explain.py       per-forecast feature contributions and labels
src/narrative.py     plain-English forecast summary, template-based
src/api.py           FastAPI endpoints
web/index.html       test console
```

## Known limits

- Forecasts are recursive: month 2 is built on month 1's prediction, so error
  compounds. Treat anything past three months as indicative.
- Price gaps of one or two months are linearly interpolated; longer gaps are
  left as gaps.
- Features are price history only. USD/KES was tested in `src/compare_fx.py`
  under a controlled A/B that held rows and test months fixed, found no
  improvement (median MASE change −0.004), and was therefore left out; the
  code is retained for reproducibility. Fuel and rainfall drivers are not in
  the model, so a driver-attribution screen cannot be built from it as it
  stands.
