from fastapi import APIRouter
from app.database import get_db_connection

router = APIRouter()

@router.get("/trends")
def get_trends(symbol: str | None = None):
    conn = get_db_connection()
    cur = conn.cursor()

    query = "SELECT symbol, trend, confidence, time_window, generated_at FROM trend_insights"
    values = []

    if symbol:
        query += " WHERE symbol = %s"
        values.append(symbol.upper())

    query += " ORDER BY generated_at DESC"

    cur.execute(query, values)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "symbol": r[0],
            "trend": r[1],
            "confidence": r[2],
            "time_window": r[3],
            "generated_at": r[4]
        }
        for r in rows
    ]
