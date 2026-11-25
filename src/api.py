"""BeiAlert prediction API.

Run locally:  uvicorn src.api:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import config, predict

app = FastAPI(
    title="BeiAlert API",
    version="0.1.0",
    description="Monthly commodity price forecasts for Kenyan markets, "
                "trained on WFP food price data.",
)

# The Android app and the test page call this from other origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health():
    """Liveness check, and whether a trained model is actually loaded."""
    try:
        series = predict.list_series()
        return {
            "status": "ok",
            "models_loaded": True,
            "series_count": len(series),
            "trained_at": predict.trained_at(),
        }
    except FileNotFoundError:
        return {"status": "degraded", "models_loaded": False,
                "detail": "No model artifact. Run python -m src.train"}


@app.get("/series", tags=["catalog"])
def series(market: str | None = None, commodity: str | None = None):
    """Everything forecastable, for populating the app's pickers."""
    try:
        rows = predict.list_series()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc

    if market:
        rows = [r for r in rows if r["market"].lower() == market.lower()]
    if commodity:
        rows = [r for r in rows if commodity.lower() in r["commodity"].lower()]
    return {"count": len(rows), "series": rows}


@app.get("/markets", tags=["catalog"])
def markets():
    try:
        rows = predict.list_series()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"markets": sorted({r["market"] for r in rows})}


@app.get("/forecast", tags=["forecast"])
def forecast(
    series_key: str = Query(..., description="Key from /series"),
    horizon: int = Query(1, ge=1, le=config.MAX_HORIZON),
):
    """Forecast one series up to MAX_HORIZON months ahead."""
    try:
        return predict.forecast(series_key, horizon)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except KeyError:
        raise HTTPException(404, f"Unknown series_key: {series_key}")


@app.get("/history", tags=["forecast"])
def history(series_key: str = Query(...), months: int = Query(24, ge=1, le=240)):
    """Observed monthly prices, for the chart behind the forecast."""
    try:
        return predict.history(series_key, months)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except KeyError:
        raise HTTPException(404, f"Unknown series_key: {series_key}")


@app.get("/", include_in_schema=False)
def index():
    page = config.ROOT / "web" / "index.html"
    if page.exists():
        return FileResponse(page)
    return {"service": "BeiAlert API", "docs": "/docs"}
