from fastapi import APIRouter, Query
from app.database import get_db_connection

router = APIRouter()

@router.get("/markets")
def get_market_data(
    symbol: str | None = None,
    market_type: str | None = None,
    limit: int = Query(50, le=200)
):
    conn = get_db_connection()
    cur = conn.cursor()

    query = "SELECT symbol, price, volume, market_type, recorded_at FROM market_data"
    conditions = []
    values = []

    if symbol:
        conditions.append("symbol = %s")
        values.append(symbol.upper())

    if market_type:
        conditions.append("market_type = %s")
        values.append(market_type)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY recorded_at DESC LIMIT %s"
    values.append(limit)

    cur.execute(query, values)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "symbol": r[0],
            "price": r[1],
            "volume": r[2],
            "market_type": r[3],
            "recorded_at": r[4]
        }
        for r in rows
    ]
