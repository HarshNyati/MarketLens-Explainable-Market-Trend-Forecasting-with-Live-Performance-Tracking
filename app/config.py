import os
from dotenv import load_dotenv

load_dotenv()

MARKET_API_KEY = os.getenv("MARKET_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
