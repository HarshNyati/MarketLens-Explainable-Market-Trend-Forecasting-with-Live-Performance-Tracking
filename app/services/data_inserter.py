from app.database import get_db_connection

def insert_market_data(data):
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        INSERT INTO market_data (symbol, price, volume, market_type)
        VALUES (%s, %s, %s, %s)
    """

    for item in data:
        cur.execute(query, (
            item["symbol"],
            item["price"],
            item["volume"],
            item["market_type"]
        ))

    conn.commit()
    cur.close()
    conn.close()
