from fastapi import APIRouter
from app.services.outcome_tracker import get_live_performance, resolve_outcomes

router = APIRouter()


@router.get("/performance/{symbol}", tags=["performance"])
def performance(symbol: str):
    """Return live performance metrics for a given asset.

    Calculates:
      • Number of resolved predictions
      • Live accuracy
      • Live accuracy compared with 'always NEUTRAL' over the same period
      • 'Not enough data yet' if fewer than 30 predictions are resolved
    """
    return get_live_performance(symbol)


@router.post("/performance/resolve", tags=["performance"])
def trigger_resolve():
    """Trigger an immediate run of the outcome resolution job."""
    resolved = resolve_outcomes()
    return {
        "status": "success",
        "resolved_count": len(resolved),
        "resolved_records": resolved,
    }
