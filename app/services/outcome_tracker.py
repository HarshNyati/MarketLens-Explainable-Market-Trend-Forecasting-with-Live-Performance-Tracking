"""
app/services/outcome_tracker.py
───────────────────────────────
Live outcome tracking for market trend predictions.

Periodically (or on demand) checks stored predictions in `prediction_results`
whose horizon (5 bars) has passed. Looks up the actual price movement in
`market_data`, resolves the outcome, and updates:
  • actual_return
  • actual_label  (UP if > +0.5%, DOWN if < -0.5%, else NEUTRAL)
  • was_correct   (true if predicted label == actual_label)

Never modifies the original prediction, direction, or confidence.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from app.database import get_db_connection

logger = logging.getLogger(__name__)

HORIZON_BARS = 5
UP_THRESH = 0.005   # +0.5%
DN_THRESH = 0.005   # -0.5%
MIN_RESOLVED_THRESHOLD = 30


def resolve_outcomes(
    horizon_bars: int = HORIZON_BARS,
    up_thresh: float = UP_THRESH,
    dn_thresh: float = DN_THRESH,
) -> list[dict[str, Any]]:
    """Look up actual price movement for pending predictions and record outcomes.

    For each unresolved prediction with a known `candle_timestamp`, finds the
    price at `candle_timestamp` (P0) and the price at the N-th subsequent bar (P_N)
    in `market_data`. If the horizon has passed (at least `horizon_bars` bars recorded
    after the prediction candle), computes the return, determines the actual label,
    and updates `prediction_results`.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Find predictions that have not been resolved yet
    query = """
        SELECT id, symbol, direction, candle_timestamp
        FROM prediction_results
        WHERE actual_label IS NULL
          AND candle_timestamp IS NOT NULL
        ORDER BY candle_timestamp ASC;
    """
    cur.execute(query)
    pending = cur.fetchall()

    resolved_records = []

    for pred_id, symbol, direction, candle_ts in pending:
        # 1. Look up starting price P0 at or nearest to candle_timestamp
        cur.execute(
            """
            SELECT price, recorded_at
            FROM market_data
            WHERE symbol = %s
              AND recorded_at = %s
            LIMIT 1;
            """,
            (symbol, candle_ts),
        )
        row_p0 = cur.fetchone()

        if not row_p0:
            # Fallback: find nearest recorded_at on or immediately after candle_ts
            cur.execute(
                """
                SELECT price, recorded_at
                FROM market_data
                WHERE symbol = %s
                  AND recorded_at >= %s
                ORDER BY recorded_at ASC
                LIMIT 1;
                """,
                (symbol, candle_ts),
            )
            row_p0 = cur.fetchone()

        if not row_p0:
            continue

        p0 = float(row_p0[0])

        # 2. Fetch the next `horizon_bars` prices to evaluate the full forward horizon
        cur.execute(
            """
            SELECT price, recorded_at
            FROM market_data
            WHERE symbol = %s
              AND recorded_at > %s
            ORDER BY recorded_at ASC
            LIMIT %s;
            """,
            (symbol, candle_ts, horizon_bars),
        )
        subsequent_rows = cur.fetchall()

        if len(subsequent_rows) < horizon_bars:
            # Horizon has not elapsed yet; keep nullable
            continue

        future_prices = [float(r[0]) for r in subsequent_rows]
        horizon_ts = subsequent_rows[-1][1]
        p_horizon = future_prices[-1]

        # 3. Calculate actual return and label matching make_target() in ml/feature_engineering.py
        max_ret = float(np.max(future_prices) / p0 - 1) if p0 > 0 else 0.0
        min_ret = float(np.min(future_prices) / p0 - 1) if p0 > 0 else 0.0
        end_ret = float((p_horizon - p0) / p0) if p0 > 0 else 0.0

        if max_ret >= up_thresh and abs(max_ret) >= abs(min_ret):
            actual_label = "UP"
            actual_return = max_ret
        elif min_ret <= -dn_thresh and abs(min_ret) > abs(max_ret):
            actual_label = "DOWN"
            actual_return = min_ret
        else:
            actual_label = "NEUTRAL"
            actual_return = end_ret

        was_correct = bool(str(direction).strip().upper() == actual_label)

        # 4. Update the prediction record without altering original prediction
        cur.execute(
            """
            UPDATE prediction_results
            SET actual_return = %s,
                actual_label = %s,
                was_correct = %s
            WHERE id = %s;
            """,
            (round(actual_return, 6), actual_label, was_correct, pred_id),
        )

        resolved_info = {
            "id": pred_id,
            "symbol": symbol,
            "direction": direction,
            "candle_timestamp": str(candle_ts),
            "horizon_timestamp": str(horizon_ts),
            "p0": p0,
            "p_horizon": p_horizon,
            "actual_return": round(actual_return, 6),
            "actual_label": actual_label,
            "was_correct": was_correct,
        }
        resolved_records.append(resolved_info)
        logger.info(
            "Resolved outcome for %s [id=%d, candle=%s]: actual=%s (ret=%.4f%%) predicted=%s was_correct=%s",
            symbol,
            pred_id,
            candle_ts,
            actual_label,
            actual_return * 100,
            direction,
            was_correct,
        )

    conn.commit()
    cur.close()
    conn.close()

    return resolved_records


def get_live_performance(symbol: str) -> dict[str, Any]:
    """Calculate live performance metrics for a given asset symbol.

    Returns:
      • resolved_count: total number of resolved predictions
      • live_accuracy: accuracy of the ML model (% correct)
      • always_neutral_accuracy: accuracy of an 'always NEUTRAL' baseline over the same bars
      • accuracy_difference: live_accuracy - always_neutral_accuracy
      • has_enough_data: True if resolved_count >= 30, False otherwise
    """
    sym = symbol.strip().upper()

    conn = get_db_connection()
    cur = conn.cursor()

    # Query resolved predictions
    cur.execute(
        """
        SELECT direction, actual_return, actual_label, was_correct, candle_timestamp
        FROM prediction_results
        WHERE symbol = %s
          AND actual_label IS NOT NULL
        ORDER BY candle_timestamp ASC;
        """,
        (sym,),
    )
    resolved_rows = cur.fetchall()

    # Query count of pending (unresolved) predictions
    cur.execute(
        """
        SELECT COUNT(*)
        FROM prediction_results
        WHERE symbol = %s
          AND actual_label IS NULL
          AND candle_timestamp IS NOT NULL;
        """,
        (sym,),
    )
    pending_count = cur.fetchone()[0]

    cur.close()
    conn.close()

    resolved_count = len(resolved_rows)
    has_enough_data = resolved_count >= MIN_RESOLVED_THRESHOLD

    if resolved_count == 0:
        return {
            "symbol": sym,
            "resolved_count": 0,
            "unresolved_count": pending_count,
            "correct_count": 0,
            "neutral_correct_count": 0,
            "live_accuracy": None,
            "live_accuracy_pct": None,
            "always_neutral_accuracy": None,
            "always_neutral_accuracy_pct": None,
            "accuracy_difference": None,
            "accuracy_difference_pct": None,
            "has_enough_data": False,
            "status": "Not enough data yet",
            "message": f"Not enough data yet ({resolved_count}/{MIN_RESOLVED_THRESHOLD} resolved)",
        }

    correct_count = sum(1 for r in resolved_rows if r[3] is True)
    neutral_correct_count = sum(1 for r in resolved_rows if str(r[2]).strip().upper() == "NEUTRAL")

    live_acc = correct_count / resolved_count
    neutral_acc = neutral_correct_count / resolved_count
    diff = live_acc - neutral_acc

    return {
        "symbol": sym,
        "resolved_count": resolved_count,
        "unresolved_count": pending_count,
        "correct_count": correct_count,
        "neutral_correct_count": neutral_correct_count,
        "live_accuracy": round(live_acc, 4),
        "live_accuracy_pct": round(live_acc * 100, 2),
        "always_neutral_accuracy": round(neutral_acc, 4),
        "always_neutral_accuracy_pct": round(neutral_acc * 100, 2),
        "accuracy_difference": round(diff, 4),
        "accuracy_difference_pct": round(diff * 100, 2),
        "has_enough_data": has_enough_data,
        "status": "Sufficient data" if has_enough_data else "Not enough data yet",
        "message": f"Evaluated on {resolved_count} resolved predictions"
        if has_enough_data
        else f"Not enough data yet ({resolved_count}/{MIN_RESOLVED_THRESHOLD} resolved)",
    }


def recompute_stored_outcomes(
    horizon_bars: int = HORIZON_BARS,
    up_thresh: float = UP_THRESH,
    dn_thresh: float = DN_THRESH,
) -> int:
    """Recompute all stored outcomes in prediction_results to match make_target() logic.

    Never modifies original prediction, direction, or confidence.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        SELECT id, symbol, direction, candle_timestamp
        FROM prediction_results
        WHERE candle_timestamp IS NOT NULL
        ORDER BY candle_timestamp ASC;
    """
    cur.execute(query)
    all_preds = cur.fetchall()

    updated_count = 0
    for pred_id, symbol, direction, candle_ts in all_preds:
        cur.execute(
            """
            SELECT price FROM market_data
            WHERE symbol = %s AND recorded_at = %s LIMIT 1;
            """,
            (symbol, candle_ts),
        )
        row_p0 = cur.fetchone()
        if not row_p0:
            continue
        p0 = float(row_p0[0])

        cur.execute(
            """
            SELECT price FROM market_data
            WHERE symbol = %s AND recorded_at > %s
            ORDER BY recorded_at ASC LIMIT %s;
            """,
            (symbol, candle_ts, horizon_bars),
        )
        sub_rows = cur.fetchall()
        if len(sub_rows) < horizon_bars:
            continue

        future_prices = [float(r[0]) for r in sub_rows]
        max_ret = float(np.max(future_prices) / p0 - 1) if p0 > 0 else 0.0
        min_ret = float(np.min(future_prices) / p0 - 1) if p0 > 0 else 0.0
        end_ret = float((future_prices[-1] - p0) / p0) if p0 > 0 else 0.0

        if max_ret >= up_thresh and abs(max_ret) >= abs(min_ret):
            actual_label = "UP"
            actual_return = max_ret
        elif min_ret <= -dn_thresh and abs(min_ret) > abs(max_ret):
            actual_label = "DOWN"
            actual_return = min_ret
        else:
            actual_label = "NEUTRAL"
            actual_return = end_ret

        was_correct = bool(str(direction).strip().upper() == actual_label)

        cur.execute(
            """
            UPDATE prediction_results
            SET actual_return = %s,
                actual_label = %s,
                was_correct = %s
            WHERE id = %s;
            """,
            (round(actual_return, 6), actual_label, was_correct, pred_id),
        )
        updated_count += 1

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Recomputed %d stored outcomes matching make_target() logic", updated_count)
    return updated_count

