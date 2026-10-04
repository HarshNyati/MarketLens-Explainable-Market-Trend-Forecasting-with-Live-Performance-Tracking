from app.database import get_db_connection
from app.analysis.trend_engine import analyze_trends

def save_trends():
    trends = analyze_trends()
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        INSERT INTO trend_insights (symbol, trend, confidence, time_window)
        VALUES (%s, %s, %s, %s)
    """

    for t in trends:
        cur.execute(query, (
            t["symbol"],
            t["trend"],
            t["confidence"],
            t["time_window"]
        ))

    conn.commit()
    cur.close()
    conn.close()

    print("✅ Trend analysis saved successfully")
