import pandas as pd
from app.database import get_db_connection


def load_live_data(symbol: str, limit: int = 500) -> pd.DataFrame:
    """Load ingested candles for *symbol* from PostgreSQL."""
    conn = get_db_connection()

    query = """
        SELECT price, volume, recorded_at
        FROM market_data
        WHERE symbol = %s
        ORDER BY recorded_at ASC
    """

    df = pd.read_sql(query, conn, params=(symbol.upper(),))
    conn.close()

    if not df.empty:
        df.rename(columns={"price": "Close", "volume": "Volume"}, inplace=True)
        if limit and len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

    return df
