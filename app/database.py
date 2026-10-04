import os
import psycopg2
from dotenv import load_dotenv

# Load .env when imported directly (e.g. by the dashboard or scripts)
load_dotenv()


def get_db_connection() -> psycopg2.extensions.connection:
    """Return a new psycopg2 connection using settings from the environment.

    All variables are read from the environment (or a .env file):
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "market_trends"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )
