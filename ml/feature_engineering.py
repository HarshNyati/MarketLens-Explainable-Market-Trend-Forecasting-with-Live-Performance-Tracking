"""
ml/feature_engineering.py
──────────────────────────
Single source-of-truth for feature construction.

Both training (train_offline.py) and live inference (ml_predictor.py /
predict_live.py) call `create_features()` so there is no train/serve skew.

All features are scale-free (returns, ratios, oscillators) so the same
code works for assets with very different price magnitudes (BTC vs IBM).

Target definition (ternary, look-ahead over `horizon` bars)
────────────────────────────────────────────────────────────
  UP      (2) – max forward return over [1..horizon] bars ≥ +UP_THRESH
  DOWN    (0) – min forward return over [1..horizon] bars ≤ -DN_THRESH
  NEUTRAL (1) – neither threshold crossed  → "no clear signal"

When both thresholds are crossed the stronger signal wins:
  abs(max_ret) ≥ abs(min_ret)  →  UP, else DOWN.

Configurable via module-level defaults or per-call kwargs.
Default: horizon=5 bars, thresholds ±0.5 %.

Expected input columns
──────────────────────
  Close   – float, adjusted close price  (required)
  Volume  – float, traded volume         (optional; falls back to 0)

Public names
────────────
  FEATURE_COLS    – list[str]  canonical feature column names
  TARGET_CLASSES  – dict       {label_str: int_code}  e.g. {"UP":2,"NEUTRAL":1,"DOWN":0}
  DEFAULT_HORIZON, DEFAULT_UP_THRESH, DEFAULT_DN_THRESH – configurable defaults
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─── Target configuration (override per call or change defaults here) ─────────

DEFAULT_HORIZON:   int   = 5      # bars to look ahead
DEFAULT_UP_THRESH: float = 0.005  # +0.5 %  → UP
DEFAULT_DN_THRESH: float = 0.005  # −0.5 %  → DOWN

TARGET_CLASSES: dict[str, int] = {
    "DOWN":    0,
    "NEUTRAL": 1,
    "UP":      2,
}
# Inverse map for display
TARGET_LABELS: dict[int, str] = {v: k for k, v in TARGET_CLASSES.items()}

# ─── Hyper-parameters for indicator windows ───────────────────────────────────

_SHORT_WIN = 5
_MID_WIN   = 10
_LONG_WIN  = 20
_VOL_WIN   = 10       # rolling std window for realised volatility
_RSI_WIN   = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIG  = 9
_LAG_STEPS = [1, 2, 3, 5]   # lagged-return lags

# ─── Canonical feature list ───────────────────────────────────────────────────
# Any code that needs to index X must do: df[FEATURE_COLS]
# Never hard-code the column names elsewhere.

FEATURE_COLS: list[str] = [
    # Multi-period returns
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    # Lagged returns
    *[f"ret_lag_{k}" for k in _LAG_STEPS],
    # Rolling realised volatility (std of daily / bar returns)
    f"vol_{_VOL_WIN}",
    # RSI (Wilder)
    f"rsi_{_RSI_WIN}",
    # MACD histogram  (MACD line – signal line)
    "macd_hist",
    # Price relative to moving averages  (close / MA − 1)
    f"close_vs_ma{_SHORT_WIN}",
    f"close_vs_ma{_MID_WIN}",
    f"close_vs_ma{_LONG_WIN}",
    # Volume change  (log ratio of consecutive bar volumes)
    "vol_change",
]


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _rsi(series: pd.Series, window: int) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ─── Target builder (separated so it can be called independently) ─────────────

def make_target(
    close: pd.Series,
    horizon:   int   = DEFAULT_HORIZON,
    up_thresh: float = DEFAULT_UP_THRESH,
    dn_thresh: float = DEFAULT_DN_THRESH,
) -> pd.Series:
    """Build the ternary forward-return target.

    For each bar t we look at the next `horizon` bars:
      max_ret = max(close[t+1..t+horizon] / close[t] − 1)
      min_ret = min(close[t+1..t+horizon] / close[t] − 1)

    Then:
      UP      (2) if  max_ret ≥ +up_thresh  AND  |max_ret| ≥ |min_ret|
              (strong up move dominates)
      DOWN    (0) if  min_ret ≤ −dn_thresh  AND  |min_ret| >  |max_ret|
              (strong down move dominates)
      NEUTRAL (1) otherwise

    The last `horizon` rows will be NaN (no future bars) and are dropped
    by create_features().

    Returns
    -------
    pd.Series of dtype int with values in {0, 1, 2}, aligned with *close*.
    """
    n   = len(close)
    arr = close.to_numpy(dtype=float)
    out = np.full(n, np.nan)

    for t in range(n - horizon):
        current   = arr[t]
        future    = arr[t + 1 : t + 1 + horizon]
        max_ret   = float(np.max(future) / current - 1)
        min_ret   = float(np.min(future) / current - 1)

        if max_ret >= up_thresh and abs(max_ret) >= abs(min_ret):
            out[t] = TARGET_CLASSES["UP"]
        elif min_ret <= -dn_thresh and abs(min_ret) > abs(max_ret):
            out[t] = TARGET_CLASSES["DOWN"]
        else:
            out[t] = TARGET_CLASSES["NEUTRAL"]

    return pd.Series(out, index=close.index, name="target")


# ─── Public API ───────────────────────────────────────────────────────────────

def create_features(
    df: pd.DataFrame,
    horizon:   int   = DEFAULT_HORIZON,
    up_thresh: float = DEFAULT_UP_THRESH,
    dn_thresh: float = DEFAULT_DN_THRESH,
    for_inference: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Add all features and the ternary `target` column to *df*.

    Parameters
    ----------
    df            : DataFrame with a ``Close`` column (float).
                    Optional ``Volume`` column; falls back to 0 if absent.
    horizon       : Number of forward bars for target labelling.
    up_thresh     : Fractional return threshold for UP   (default 0.005 = 0.5 %).
    dn_thresh     : Fractional return threshold for DOWN (default 0.005 = 0.5 %).
    for_inference : If True, does not drop rows where ``target`` is NaN, preserving
                    the most recent closed candles for live prediction.

    Returns
    -------
    (enriched_df, FEATURE_COLS)
        ``enriched_df`` has NaN feature rows dropped and includes ``target``.
        ``FEATURE_COLS`` is the module-level canonical list.
    """
    df = df.copy()

    # ── Normalise column names ────────────────────────────────────────────────
    rename_map = {
        "close": "Close", "Price": "Close", "Value": "Close",
        "volume": "Volume", "vol": "Volume",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
              inplace=True)

    if "Close" not in df.columns:
        raise ValueError("DataFrame must have a 'Close' column.")

    df["Close"] = df["Close"].astype(float)

    has_volume = "Volume" in df.columns and df["Volume"].notna().any()
    if has_volume:
        df["Volume"] = df["Volume"].astype(float)

    # ── Returns ───────────────────────────────────────────────────────────────
    df["ret_1"]  = df["Close"].pct_change(1)
    df["ret_3"]  = df["Close"].pct_change(3)
    df["ret_5"]  = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    # ── Lagged returns ────────────────────────────────────────────────────────
    for k in _LAG_STEPS:
        df[f"ret_lag_{k}"] = df["ret_1"].shift(k)

    # ── Rolling realised volatility ───────────────────────────────────────────
    df[f"vol_{_VOL_WIN}"] = df["ret_1"].rolling(_VOL_WIN).std()

    # ── RSI ───────────────────────────────────────────────────────────────────
    df[f"rsi_{_RSI_WIN}"] = _rsi(df["Close"], _RSI_WIN)

    # ── MACD histogram ────────────────────────────────────────────────────────
    macd_line      = _ema(df["Close"], _MACD_FAST) - _ema(df["Close"], _MACD_SLOW)
    signal         = _ema(macd_line, _MACD_SIG)
    df["macd_hist"] = macd_line - signal

    # ── Price / MA ratios ─────────────────────────────────────────────────────
    for win in [_SHORT_WIN, _MID_WIN, _LONG_WIN]:
        ma = df["Close"].rolling(win).mean()
        df[f"close_vs_ma{win}"] = df["Close"] / ma - 1

    # ── Volume change ─────────────────────────────────────────────────────────
    if has_volume:
        vol_clean = df["Volume"].astype(float).replace(0, np.nan).ffill().bfill()
        if vol_clean.isna().all():
            df["vol_change"] = 0.0
        else:
            vol_clean = vol_clean.fillna(1.0)
            ratio = vol_clean / vol_clean.shift(1)
            df["vol_change"] = np.log(ratio.replace(0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            df["vol_change"] = df["vol_change"].clip(-5.0, 5.0)
    else:
        df["vol_change"] = 0.0

    # ── Ternary target ────────────────────────────────────────────────────────
    df["target"] = make_target(df["Close"], horizon, up_thresh, dn_thresh)

    # ── Clean up ──────────────────────────────────────────────────────────────
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if for_inference:
        df.dropna(subset=FEATURE_COLS, inplace=True)
    else:
        df.dropna(subset=FEATURE_COLS + ["target"], inplace=True)
        df["target"] = df["target"].astype(int)
    df.reset_index(drop=True, inplace=True)

    return df, FEATURE_COLS
