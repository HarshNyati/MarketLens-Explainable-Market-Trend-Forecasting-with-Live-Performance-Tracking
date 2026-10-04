import sqlite3
from unittest.mock import MagicMock, patch
import pytest

from app.services.prediction_store import save_prediction


def test_prediction_store_query_has_conflict_do_nothing():
    """Verify save_prediction uses ON CONFLICT (symbol, candle_timestamp) DO NOTHING."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("app.services.prediction_store.get_db_connection", return_value=mock_conn):
        save_prediction(
            symbol="BITCOIN",
            direction="UP",
            confidence=85.0,
            candle_timestamp="2026-10-01 10:00:00",
            model_version="v1.0",
        )

    assert mock_cursor.execute.called
    query, params = mock_cursor.execute.call_args[0]
    normalized_query = " ".join(query.split())

    assert "ON CONFLICT (symbol, candle_timestamp) DO NOTHING" in normalized_query
    assert params == ("BITCOIN", "UP", 85.0, "2026-10-01 10:00:00", "v1.0")


def test_prediction_store_deduplication_semantics():
    """Verify duplicate candle timestamps do not overwrite existing predictions."""
    # Test deduplication semantics using an in-memory SQLite table mimicking the unique constraint
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE prediction_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            direction TEXT,
            confidence REAL,
            candle_timestamp TEXT,
            model_version TEXT,
            UNIQUE(symbol, candle_timestamp)
        )
    """)

    # Function executing the exact same SQL logic with ON CONFLICT DO NOTHING
    def execute_insert(sym, dir_, conf, ts, ver):
        cur.execute("""
            INSERT INTO prediction_results (symbol, direction, confidence, candle_timestamp, model_version)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (symbol, candle_timestamp)
            DO NOTHING
        """, (sym, dir_, conf, ts, ver))
        conn.commit()

    # 1. First prediction for BITCOIN at 10:00
    execute_insert("BITCOIN", "UP", 82.5, "2026-10-01 10:00:00", "v1.0")

    # 2. Second (duplicate) prediction attempt for BITCOIN at identical timestamp with different direction/confidence
    execute_insert("BITCOIN", "DOWN", 95.0, "2026-10-01 10:00:00", "v1.0")

    # 3. Query records
    cur.execute("SELECT symbol, direction, confidence, candle_timestamp FROM prediction_results")
    rows = cur.fetchall()

    # Assert exactly 1 row remains, retaining original "UP" and 82.5% confidence
    assert len(rows) == 1
    assert rows[0] == ("BITCOIN", "UP", 82.5, "2026-10-01 10:00:00")
