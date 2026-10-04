from fastapi import APIRouter, Query
from app.services.ml_predictor import predict_symbol
from app.services.candle_ingester import ingest_asset_candles

router = APIRouter()

@router.get("/predict/{symbol}")
def predict(
    symbol: str,
    refresh: bool = Query(default=False, description="Fetch latest live market candle before predicting"),
):
    sym_upper = symbol.upper()
    if refresh:
        mtype = "crypto" if sym_upper in ("BITCOIN", "ETHEREUM") else "stock"
        ingest_asset_candles(sym_upper, mtype)
    return predict_symbol(sym_upper)

@router.post("/predict/{symbol}")
def trigger_prediction(symbol: str):
    """Trigger an immediate live candle fetch and prediction."""
    sym_upper = symbol.upper()
    mtype = "crypto" if sym_upper in ("BITCOIN", "ETHEREUM") else "stock"
    ingest_asset_candles(sym_upper, mtype)
    return predict_symbol(sym_upper)
