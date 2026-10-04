from unittest.mock import MagicMock, patch
import pytest

from app.services.outcome_tracker import resolve_outcomes


def test_outcome_tracker_labeling_and_immutability():
    """Verify outcome_tracker labels correctly and never modifies original prediction fields."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    # Scenario: Prediction was UP.
    # P0 = 100.0. Future 5 bars: [100.2, 100.3, 101.2, 100.5, 100.6] -> max return +1.2% -> actual UP
    pending_records = [
        (101, "BITCOIN", "UP", "2026-10-01 00:00:00"),
        (102, "ETHEREUM", "DOWN", "2026-10-01 00:00:00"),
        (103, "IBM", "NEUTRAL", "2026-10-01 00:00:00"),
    ]

    # For BITCOIN: UP prediction, actual is UP (+1.2%) -> was_correct = True
    # For ETHEREUM: DOWN prediction, actual is UP (+1.2%) -> was_correct = False
    # For IBM: NEUTRAL prediction, actual is NEUTRAL (+0.1%) -> was_correct = True
    row_p0 = (100.0, "2026-10-01 00:00:00")
    subsequent_up = [
        (100.2, "t1"),
        (100.3, "t2"),
        (101.2, "t3"),
        (100.5, "t4"),
        (100.6, "t5"),
    ]
    subsequent_neutral = [
        (100.1, "t1"),
        (100.0, "t2"),
        (99.9, "t3"),
        (100.2, "t4"),
        (100.1, "t5"),
    ]

    def mock_execute(query, params=None):
        pass

    mock_cursor.execute.side_effect = mock_execute

    # Track updates executed by outcome_tracker
    update_calls = []

    def mock_execute_capture(query, params=None):
        if "UPDATE prediction_results" in query:
            update_calls.append((query, params))

    mock_cursor.execute.side_effect = mock_execute_capture

    # Sequence of fetchall / fetchone calls:
    # 1. fetchall() -> pending_records
    # 2. For record 101:
    #    - fetchone() -> row_p0
    #    - fetchall() -> subsequent_up
    # 3. For record 102:
    #    - fetchone() -> row_p0
    #    - fetchall() -> subsequent_up
    # 4. For record 103:
    #    - fetchone() -> row_p0
    #    - fetchall() -> subsequent_neutral
    mock_cursor.fetchall.side_effect = [
        pending_records,
        subsequent_up,
        subsequent_up,
        subsequent_neutral,
    ]
    mock_cursor.fetchone.side_effect = [
        row_p0,
        row_p0,
        row_p0,
    ]

    with patch("app.services.outcome_tracker.get_db_connection", return_value=mock_conn):
        results = resolve_outcomes(horizon_bars=5, up_thresh=0.005, dn_thresh=0.005)

    assert len(results) == 3

    # Check Record 101 (BITCOIN: UP prediction, actual UP)
    r101 = results[0]
    assert r101["actual_label"] == "UP"
    assert r101["was_correct"] is True
    assert r101["direction"] == "UP"

    # Check Record 102 (ETHEREUM: DOWN prediction, actual UP)
    r102 = results[1]
    assert r102["actual_label"] == "UP"
    assert r102["was_correct"] is False
    assert r102["direction"] == "DOWN"

    # Check Record 103 (IBM: NEUTRAL prediction, actual NEUTRAL)
    r103 = results[2]
    assert r103["actual_label"] == "NEUTRAL"
    assert r103["was_correct"] is True
    assert r103["direction"] == "NEUTRAL"

    # Verify SQL updates never modify original prediction fields (direction, confidence, candle_timestamp)
    assert len(update_calls) == 3
    for query, params in update_calls:
        # Check SET clause
        set_clause = query.split("SET")[1].split("WHERE")[0].strip()
        set_columns = [col.split("=")[0].strip() for col in set_clause.split(",")]
        # Only actual_return, actual_label, and was_correct should be modified
        assert set(set_columns) == {"actual_return", "actual_label", "was_correct"}
        # Ensure direction, confidence, candle_timestamp are not touched
        assert "direction" not in set_columns
        assert "confidence" not in set_columns
        assert "candle_timestamp" not in set_columns
