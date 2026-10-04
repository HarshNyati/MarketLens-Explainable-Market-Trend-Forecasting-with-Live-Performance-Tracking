import pandas as pd
from app.database import get_db_connection

def analyze_trends():
    conn = get_db_connection()

    query = """
        SELECT symbol, price, recorded_at
        FROM market_data
        ORDER BY symbol, recorded_at
    """

    df = pd.read_sql(query, conn)
    conn.close()

    trends = []

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]

        if len(symbol_df) < 2:
            continue

        old_price = symbol_df.iloc[-2]['price']
        new_price = symbol_df.iloc[-1]['price']

        if new_price > old_price:
            trend = "UP"
        elif new_price < old_price:
            trend = "DOWN"
        else:
            trend = "STABLE"

        trends.append({
            "symbol": symbol,
            "trend": trend,
            "confidence": "MEDIUM",
            "time_window": "short"
        })

    return trends
