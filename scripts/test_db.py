import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import get_db_connection

try:
    conn = get_db_connection()
    print("✅ PostgreSQL connected successfully!")
    conn.close()
except Exception as e:
    print("❌ Connection failed:", e)
