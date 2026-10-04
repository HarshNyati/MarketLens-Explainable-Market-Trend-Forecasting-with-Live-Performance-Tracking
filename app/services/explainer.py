"""
app/services/explainer.py
─────────────────────────
SHAP explainability service using TreeExplainer on trained models.

Caches TreeExplainer instances per asset so they are not rebuilt on each request.
Correctly extracts SHAP values for the predicted class (DOWN=0, NEUTRAL=1, UP=2)
across all model architectures (RandomForest, XGBoost, LightGBM).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from ml.data_loader import load_live_data
from ml.feature_engineering import (
    FEATURE_COLS,
    TARGET_CLASSES,
    create_features,
)
from app.services.ml_predictor import predict_symbol

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "ml" / "models"

# ─── Explainer Cache ──────────────────────────────────────────────────────────
# Cache TreeExplainer instances in memory to avoid rebuilding on every request
_EXPLAINER_CACHE: dict[str, Any] = {}
_TREE_MODEL_CACHE: dict[str, Any] = {}


def _get_tree_estimator(symbol: str) -> tuple[Any, str]:
    """Retrieve and cache the underlying tree-based estimator from the saved model."""
    sym = symbol.lower()
    if sym in _TREE_MODEL_CACHE:
        return _TREE_MODEL_CACHE[sym]

    model_path = MODELS_DIR / f"{sym}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    # If wrapped in CalibratedClassifierCV, unwrap to the underlying estimator
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        sub = model.calibrated_classifiers_[0]
        tree_est = getattr(sub, "estimator", getattr(sub, "base_estimator", None))
    else:
        tree_est = model

    model_type = type(tree_est).__name__
    _TREE_MODEL_CACHE[sym] = (tree_est, model_type)
    return tree_est, model_type


def get_explainer(symbol: str) -> tuple[shap.TreeExplainer, str]:
    """Retrieve or create a cached shap.TreeExplainer for the specified asset."""
    sym = symbol.lower()
    if sym in _EXPLAINER_CACHE:
        explainer, model_type = _EXPLAINER_CACHE[sym]
        return explainer, model_type

    tree_est, model_type = _get_tree_estimator(sym)
    logger.info("Initializing shap.TreeExplainer for %s (%s)...", sym.upper(), model_type)
    explainer = shap.TreeExplainer(tree_est)
    _EXPLAINER_CACHE[sym] = (explainer, model_type)
    return explainer, model_type


def _extract_class_shap(shap_output: Any, class_idx: int) -> np.ndarray:
    """Extract a 1D array of feature SHAP values for class_idx from various shap formats."""
    # List of arrays: [array(n_samples, n_features), ...]
    if isinstance(shap_output, list):
        arr = shap_output[class_idx]
        return arr[0] if arr.ndim == 2 else arr

    # 3D numpy array: (n_samples, n_features, n_classes) or (n_samples, n_classes, n_features)
    if isinstance(shap_output, np.ndarray):
        if shap_output.ndim == 3:
            if shap_output.shape[2] == 3:
                return shap_output[0, :, class_idx]
            elif shap_output.shape[1] == 3:
                return shap_output[0, class_idx, :]
        elif shap_output.ndim == 2:
            return shap_output[0]

    # Explanation object
    if hasattr(shap_output, "values"):
        v = shap_output.values
        if v.ndim == 3:
            if v.shape[2] == 3:
                return v[0, :, class_idx]
            elif v.shape[1] == 3:
                return v[0, class_idx, :]
        elif v.ndim == 2:
            return v[0]

    raise ValueError(f"Unrecognized SHAP output format or dimensions: {type(shap_output)}")


def explain_latest_prediction(symbol: str, top_k: int = 5) -> dict[str, Any]:
    """Explain the latest prediction for *symbol* using SHAP.

    Returns the top K features driving the prediction for the predicted class,
    including feature values, SHAP values, and direction of impact.
    """
    sym = symbol.strip().upper()

    # 1. Run / load live prediction to get the predicted class and timestamp
    pred = predict_symbol(sym, force=True)
    if "error" in pred:
        return {"error": pred["error"]}

    predicted_class = pred.get("direction", "NEUTRAL")
    confidence = pred.get("confidence", 0.0)
    candle_ts = pred.get("candle_timestamp")

    if predicted_class not in TARGET_CLASSES:
        predicted_class = pred.get("raw_direction", "NEUTRAL")
    class_idx = TARGET_CLASSES.get(predicted_class, 1)

    # 2. Reconstruct features for the latest closed candle
    df = load_live_data(sym, limit=200)
    if df.empty or len(df) < 50:
        return {"error": f"Not enough candles to compute SHAP features for {sym}"}

    # Ensure closed candles only
    if "recorded_at" in df.columns:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)
        now_utc = pd.Timestamp.now(tz="UTC")
        closed_df = df[df["recorded_at"] + pd.Timedelta(hours=1) <= now_utc]
        if len(closed_df) >= 50:
            df = closed_df.copy()
        elif len(df) > 1:
            df = df.iloc[:-1].copy()

    enriched_df, _ = create_features(df, for_inference=True)
    latest_row = enriched_df.iloc[-1]
    candle_ts = latest_row["recorded_at"].isoformat() if "recorded_at" in latest_row else candle_ts
    X_sample = latest_row[FEATURE_COLS].values.reshape(1, -1)

    # 3. Get cached explainer and compute SHAP values
    explainer, model_type = get_explainer(sym)
    shap_vals = explainer.shap_values(X_sample)

    # 4. Extract class SHAP values for the predicted class
    class_shap = _extract_class_shap(shap_vals, class_idx)

    # 5. Build feature breakdown
    feature_drivers = []
    for i, col_name in enumerate(FEATURE_COLS):
        f_val = float(latest_row[col_name])
        s_val = float(class_shap[i])
        direction = "pushes_toward" if s_val > 0 else ("pushes_against" if s_val < 0 else "neutral")

        feature_drivers.append({
            "feature": col_name,
            "feature_value": round(f_val, 4),
            "shap_value": round(s_val, 4),
            "abs_impact": round(abs(s_val), 4),
            "direction": direction,
        })

    # 6. Sort by absolute SHAP impact and take top_k
    feature_drivers.sort(key=lambda x: x["abs_impact"], reverse=True)
    top_drivers = feature_drivers[:top_k]

    return {
        "symbol": sym,
        "model_type": model_type,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "candle_timestamp": str(candle_ts) if candle_ts else None,
        "top_drivers": top_drivers,
    }
