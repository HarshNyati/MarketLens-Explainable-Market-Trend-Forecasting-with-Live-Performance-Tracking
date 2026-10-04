from fastapi import APIRouter
from app.services.explainer import explain_latest_prediction

router = APIRouter()


@router.get("/explain/{symbol}", tags=["explainability"])
def explain_symbol(symbol: str):
    """Return the top 5 features driving the latest prediction for *symbol* with SHAP values."""
    return explain_latest_prediction(symbol)
