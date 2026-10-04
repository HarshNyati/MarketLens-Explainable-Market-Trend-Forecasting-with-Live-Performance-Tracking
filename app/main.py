import logging

from fastapi import FastAPI
from app.routes.market_routes import router as market_router
from app.routes.trend_routes import router as trend_router
from app.routes.predict import router as predict_router
from app.routes.performance import router as performance_router
from app.routes.explain import router as explain_router
from app.scheduler import start_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)

app = FastAPI(
    title="Market Trend Analysis API",
    version="1.0.0",
    description="FastAPI backend for market-trend predictions (crypto + stocks).",
)


@app.on_event("startup")
def startup_event() -> None:
    start_scheduler()


# ─── Health check ────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> dict:
    """Lightweight liveness probe used by Docker / load-balancers."""
    return {"status": "ok", "service": "market-trend-api"}


# ─── Domain routers ──────────────────────────────────────────────────────────

app.include_router(market_router, prefix="/api")
app.include_router(trend_router, prefix="/api")
app.include_router(predict_router, prefix="/api")
app.include_router(performance_router, prefix="/api")
app.include_router(explain_router, prefix="/api")

