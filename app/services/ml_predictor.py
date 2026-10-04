"""
app/services/ml_predictor.py
──────────────────────────────
FastAPI service layer for live predictions.

Uses the ingested 1h candles from PostgreSQL (market_data table),
applies the identical feature engineering pipeline as training,
runs inference using the calibrated model, and persists predictions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.data_loader import load_live_data
from ml.feature_engineering import (
    FEATURE_COLS,
    TARGET_LABELS,
    create_features,
)
from ml.market_data_fetcher import DEFAULT_INTERVAL
from app.services.prediction_store import save_prediction
from app.services.candle_ingester import get_candle_count, ingest_asset_candles

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "ml" / "models"
DEFAULT_THRESHOLD = 0.40
MIN_CANDLES_REQUIRED = 50



def _load_meta(symbol: str) -> dict:
    """Load training metadata saved by train_offline.py."""
    meta_path = MODELS_DIR / f"{symbol.lower()}_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as fh:
                return json.load(fh)
        except Exception:
            pass
    return {
        "interval":         DEFAULT_INTERVAL,
        "horizon":          5,
        "up_thresh":        0.005,
        "dn_thresh":        0.005,
        "signal_threshold": DEFAULT_THRESHOLD,
    }


def is_us_equity_market_open(dt: datetime | None = None) -> bool:
    """Return True if US equity markets are open (Mon-Fri 09:30 - 16:00 US/Eastern)."""
    from zoneinfo import ZoneInfo
    if dt is None:
        dt = datetime.now(timezone.utc)
    try:
        et = dt.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        if dt.weekday() >= 5:
            return False
        return (dt.hour == 13 and dt.minute >= 30) or (14 <= dt.hour < 20) or (dt.hour == 20 and dt.minute == 0)

    if et.weekday() >= 5:
        return False
    market_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= et <= market_close


def predict_symbol(symbol: str, force: bool = False) -> dict:
    symbol = symbol.upper()

    # Skip IBM if US equity market is closed (unless forced)
    if symbol == "IBM" and not force and not is_us_equity_market_open():
        logger.info("Skipping IBM prediction: US stock market is closed.")
        return {
            "symbol": "IBM",
            "status": "market_closed",
            "message": "US Stock Market is currently closed. Predictions are skipped outside trading hours.",
            "direction": "NO_SIGNAL",
            "raw_direction": "NO_SIGNAL",
            "confidence": 0.0,
            "is_signal": False,
            "candle_timestamp": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model_path = MODELS_DIR / f"{symbol.lower()}_model.pkl"
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        return {"error": f"Model not found for {symbol}. Run offline training first."}

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        logger.error("Failed to load model %s: %s", model_path, exc)
        return {"error": f"Could not load model for {symbol}: {exc}"}

    # ── 2. Load meta ──────────────────────────────────────────────────────────
    meta      = _load_meta(symbol)
    horizon   = meta.get("horizon",          5)
    up_thresh = meta.get("up_thresh",        0.005)
    dn_thresh = meta.get("dn_thresh",        0.005)
    threshold = meta.get("signal_threshold", DEFAULT_THRESHOLD)

    # ── 3. Load ingested candles from DB ──────────────────────────────────────
    count = get_candle_count(symbol)
    if count < MIN_CANDLES_REQUIRED:
        # Auto-ingest if database hasn't been backfilled yet
        mtype = "crypto" if symbol in ("BITCOIN", "ETHEREUM") else "stock"
        logger.info("%s has only %d candles in DB; auto-ingesting backfill...", symbol, count)
        ingest_asset_candles(symbol, mtype)

    df = load_live_data(symbol, limit=500)

    if df.empty or len(df) < MIN_CANDLES_REQUIRED:
        return {
            "symbol": symbol,
            "error": f"Not enough candle data for {symbol} (found {len(df)}, need at least {MIN_CANDLES_REQUIRED}). Ingest candles first.",
        }

    # Ensure we only predict on fully closed candles:
    # A candle timestamped at T covers [T, T + 1h), so it is closed only after now >= T + 1h
    if "recorded_at" in df.columns:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
        now_utc = datetime.now(timezone.utc)
        closed_df = df[df["recorded_at"] + pd.Timedelta(hours=1) <= now_utc]
        if len(closed_df) >= MIN_CANDLES_REQUIRED:
            df = closed_df.copy()
        elif len(df) > 1:
            # Fallback: drop the latest active candle
            df = df.iloc[:-1].copy()

    # ── 4. Feature engineering (same logic as training) ───────────────────────
    try:
        enriched_df, _ = create_features(
            df,
            horizon=horizon,
            up_thresh=up_thresh,
            dn_thresh=dn_thresh,
            for_inference=True,
        )
    except Exception as exc:
        logger.error("Feature engineering failed for %s: %s", symbol, exc)
        return {"symbol": symbol, "error": f"Feature engineering failed: {exc}"}

    if enriched_df.empty:
        return {
            "symbol": symbol,
            "error": "Not enough candles after feature engineering windows. Please ingest more candles.",
        }

    latest = enriched_df[FEATURE_COLS].iloc[-1:]

    # ── 5. Model inference (CalibratedClassifierCV) ───────────────────────────
    try:
        proba = model.predict_proba(latest)[0]
        classes = model.classes_
        max_idx = int(np.argmax(proba))
        max_class = int(classes[max_idx])
        max_prob = float(proba[max_idx])
    except Exception as exc:
        logger.error("Inference failed for %s: %s", symbol, exc)
        return {"symbol": symbol, "error": f"Model inference failed: {exc}"}

    raw_direction = TARGET_LABELS.get(max_class, str(max_class))
    confidence = round(max_prob * 100, 2)

    # ── 6. Confidence gating ──────────────────────────────────────────────────
    if max_prob < threshold:
        direction = "NO_SIGNAL"
        is_signal = False
    else:
        direction = raw_direction
        is_signal = True

    # ── 7. Save prediction to DB (deduped by candle timestamp, ON CONFLICT DO NOTHING) ─
    candle_ts = enriched_df["recorded_at"].iloc[-1] if "recorded_at" in enriched_df.columns and not enriched_df.empty else (df["recorded_at"].iloc[-1] if "recorded_at" in df.columns and not df.empty else None)
    model_version = meta.get("model_version", f"{meta.get('best_model', 'model')}_{meta.get('interval', '1h')}_v1")
    try:
        save_prediction(
            symbol,
            direction,
            confidence,
            candle_timestamp=candle_ts,
            model_version=model_version,
        )
    except Exception as exc:
        logger.error("Failed to save prediction for %s: %s", symbol, exc)

    probs_dict = {
        TARGET_LABELS.get(int(c), str(c)): round(float(p), 4)
        for c, p in zip(classes, proba)
    }

    result = {
        "symbol":           symbol,
        "direction":        direction,
        "raw_direction":    raw_direction,
        "is_signal":        is_signal,
        "confidence":       confidence,
        "threshold":        round(threshold * 100, 1),
        "probabilities":    probs_dict,
        "candles_used":     len(enriched_df),
        "candle_timestamp": candle_ts.isoformat() if hasattr(candle_ts, "isoformat") else str(candle_ts),
        "model_version":    model_version,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }

    if not is_signal:
        result["reason"] = f"Max probability {max_prob:.3f} < threshold {threshold:.2f}"

    return result

