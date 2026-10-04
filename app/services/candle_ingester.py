"""
app/services/candle_ingester.py
───────────────────────────────
Unified candle ingestion service.

Uses the same source (yfinance via ml.market_data_fetcher) and interval (1h)
as training.

Behavior:
  - On first run: backfills the last ~500 1h candles for each asset.
  - On subsequent runs: fetches the latest candles and appends them.
  - Uses ON CONFLICT (symbol, recorded_at) DO NOTHING so duplicates are
    never inserted.
"""

from __future__ import annotations

import logging
import pandas as pd

from app.database import get_db_connection
from ml.market_data_fetcher import DEFAULT_INTERVAL, fetch_recent_candles

logger = logging.getLogger(__name__)

ASSETS = [
    {"symbol": "BITCOIN",  "market_type": "crypto"},
    {"symbol": "ETHEREUM", "market_type": "crypto"},
    {"symbol": "IBM",      "market_type": "stock"},
]

BACKFILL_COUNT = 500


def get_candle_count(symbol: str) -> int:
    """Return count of existing candles for *symbol* in market_data."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM market_data WHERE symbol = %s", (symbol,))
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


def insert_candles_df(df: pd.DataFrame, symbol: str, market_type: str) -> int:
    """Insert a candle DataFrame into market_data, skipping duplicates.

    Returns the number of new candles inserted.
    """
    if df is None or df.empty:
        return 0

    df = df.copy()

    # Ensure DatetimeIndex has UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Normalize column names
    rename_map = {"close": "Close", "Price": "Close", "volume": "Volume"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    if "Close" not in df.columns:
        return 0

    # Filter to strictly insert fully closed candles (start_time + 1h <= now)
    now_utc = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(hours=1) <= now_utc]
    if df.empty:
        return 0

    has_volume = "Volume" in df.columns

    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        INSERT INTO market_data (symbol, price, volume, market_type, recorded_at)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (symbol, recorded_at)
        DO UPDATE SET price = EXCLUDED.price, volume = EXCLUDED.volume
    """

    inserted = 0
    for ts, row in df.iterrows():
        price = float(row["Close"])
        vol = float(row["Volume"]) if has_volume and pd.notna(row["Volume"]) else 0.0
        recorded_at = ts.to_pydatetime()
        cur.execute(query, (symbol, price, vol, market_type, recorded_at))
        if cur.rowcount > 0:
            inserted += 1

    conn.commit()
    cur.close()
    conn.close()

    return inserted


def ingest_asset_candles(symbol: str, market_type: str, interval: str = DEFAULT_INTERVAL) -> int:
    """Backfill ~500 candles if table is empty/sparse, or append latest candles."""
    current_count = get_candle_count(symbol)

    if current_count < BACKFILL_COUNT:
        logger.info("Backfilling %s (current count: %d, target: %d)...", symbol, current_count, BACKFILL_COUNT)
        df = fetch_recent_candles(symbol, n_bars=BACKFILL_COUNT + 50, interval=interval)
    else:
        # Just fetch recent 24 bars to capture any new closed candles
        df = fetch_recent_candles(symbol, n_bars=24, interval=interval)

    new_rows = insert_candles_df(df, symbol, market_type)
    total = get_candle_count(symbol)
    logger.info("Ingested %d new candles for %s (total in DB: %d)", new_rows, symbol, total)
    return new_rows


def run_candle_ingestion(interval: str = DEFAULT_INTERVAL) -> dict[str, int]:
    """Ingest candles for all tracked assets."""
    results = {}
    for item in ASSETS:
        symbol = item["symbol"]
        mtype = item["market_type"]
        try:
            inserted = ingest_asset_candles(symbol, mtype, interval=interval)
            results[symbol] = inserted
        except Exception as exc:
            logger.error("Failed candle ingestion for %s: %s", symbol, exc)
            results[symbol] = 0
    return results
