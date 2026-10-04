"""
ml/market_data_fetcher.py
──────────────────────────
Intraday candle downloader backed by yfinance.

Used by both:
  • train_offline.py   – download historical candles for training
  • ml_predictor.py / predict_live.py – download recent candles for inference

Public API
──────────
  fetch_candles(ticker, interval, period)  → pd.DataFrame  [Close, Volume, …]

Supported intervals (yfinance)
──────────────────────────────
  "5m"   → max 60 d of history
  "1h"   → max 730 d of history    ← recommended default
  "1d"   → full history

ticker mapping (yfinance symbols)
──────────────────────────────────
  BITCOIN  → BTC-USD
  ETHEREUM → ETH-USD
  IBM      → IBM

Configurable via .env
──────────────────────
  ML_CANDLE_INTERVAL  – e.g. "1h" or "5m"   (default: "1h")
  ML_CANDLE_PERIOD    – yfinance period str  (default: "730d")
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Defaults (overridden by .env) ───────────────────────────────────────────

DEFAULT_INTERVAL = os.getenv("ML_CANDLE_INTERVAL", "1h")
DEFAULT_PERIOD   = os.getenv("ML_CANDLE_PERIOD",   "730d")

# ─── Ticker map ──────────────────────────────────────────────────────────────

TICKER_MAP: dict[str, str] = {
    "BITCOIN":  "BTC-USD",
    "ETHEREUM": "ETH-USD",
    "IBM":      "IBM",
}


def _resolve_ticker(symbol: str) -> str:
    """Translate our internal symbol names to yfinance tickers."""
    return TICKER_MAP.get(symbol.upper(), symbol.upper())


def fetch_candles(
    symbol:   str,
    interval: str = DEFAULT_INTERVAL,
    period:   str = DEFAULT_PERIOD,
    *,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download OHLCV candles for *symbol* from Yahoo Finance.

    Parameters
    ----------
    symbol      : Internal symbol name, e.g. "BITCOIN", "IBM".
    interval    : Bar size – "5m", "15m", "1h", "1d", …
    period      : How far back to fetch – "60d", "730d", "5y", …
    auto_adjust : Apply dividend/split adjustment (default True).

    Returns
    -------
    DataFrame with columns [Open, High, Low, Close, Volume] and a
    DatetimeIndex.  Returns an empty DataFrame on failure.

    Notes
    ─────
    • yfinance returns at most 60 days for 5 m data; use interval="1h"
      for longer histories.
    • For 5 m data, period is automatically clamped to "60d".
    """
    ticker = _resolve_ticker(symbol)

    # yfinance hard limit for sub-hourly data
    if interval in ("1m", "2m", "5m", "15m", "30m") and period not in (
        "1d", "5d", "7d", "30d", "60d"
    ):
        effective_period = "60d"
        logger.info(
            "fetch_candles: clamping period to '60d' for interval '%s'", interval
        )
    else:
        effective_period = period

    try:
        logger.info(
            "fetch_candles: downloading %s (%s) interval=%s period=%s",
            symbol, ticker, interval, effective_period,
        )
        raw = yf.download(
            ticker,
            period=effective_period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            # silence the new multi-level column warning
            group_by="ticker",
        )
    except Exception as exc:
        logger.error("fetch_candles: yfinance error for %s – %s", symbol, exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        logger.warning("fetch_candles: no data returned for %s", symbol)
        return pd.DataFrame()

    # Flatten multi-level columns that modern yfinance produces
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            raw.columns = raw.columns.get_level_values(0)
        elif "Close" in raw.columns.get_level_values(1):
            raw.columns = raw.columns.get_level_values(1)
        elif "close" in [str(c).lower() for c in raw.columns.get_level_values(0)]:
            raw.columns = raw.columns.get_level_values(0)
        elif "close" in [str(c).lower() for c in raw.columns.get_level_values(1)]:
            raw.columns = raw.columns.get_level_values(1)

    # Normalize column names
    rename_map = {"close": "Close", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"}
    raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns}, inplace=True)

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    if "Close" not in keep:
        logger.error("fetch_candles: 'Close' column missing in %s for %s", list(raw.columns), symbol)
        return pd.DataFrame()

    df = raw[keep].copy()
    df.dropna(subset=["Close"], inplace=True)

    logger.info("fetch_candles: %s rows fetched for %s", len(df), symbol)
    return df



def fetch_recent_candles(
    symbol:   str,
    n_bars:   int = 200,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Fetch the most recent *n_bars* candles for live inference.

    Uses a short trailing period so the download stays fast.
    """
    # Choose a period that is safely longer than n_bars candles
    if interval in ("1m", "2m", "5m", "15m", "30m"):
        period = "7d"
    elif interval == "1h":
        period = "30d"
    else:
        period = "90d"

    df = fetch_candles(symbol, interval=interval, period=period)
    if df.empty:
        return df
    return df.tail(n_bars)
