from app.database import get_db_connection


def save_prediction(
    symbol: str,
    direction: str,
    confidence: float,
    candle_timestamp=None,
    model_version: str = "v1.0",
):
    """Persist prediction to database.

    Uses ON CONFLICT (symbol, candle_timestamp) DO NOTHING so only one prediction
    per asset per candle is ever stored.
    """
    symbol = str(symbol).upper()
    direction = str(direction)
    confidence = float(confidence)
    model_version = str(model_version)

    conn = get_db_connection()
    cur = conn.cursor()

    if candle_timestamp is not None:
        query = """
            INSERT INTO prediction_results (symbol, direction, confidence, predicted_at, candle_timestamp, model_version)
            VALUES (%s, %s, %s, NOW(), %s, %s)
            ON CONFLICT (symbol, candle_timestamp)
            DO NOTHING
        """
        cur.execute(query, (symbol, direction, confidence, candle_timestamp, model_version))
    else:
        query = """
            INSERT INTO prediction_results (symbol, direction, confidence, model_version)
            VALUES (%s, %s, %s, %s)
        """
        cur.execute(query, (symbol, direction, confidence, model_version))

    conn.commit()
    cur.close()
    conn.close()
