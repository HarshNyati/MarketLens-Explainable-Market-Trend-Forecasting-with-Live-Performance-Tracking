import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

STOCK_API_KEY = os.getenv("STOCK_API_KEY")


def fetch_stock_data(symbol: str = "IBM"):
    """Fetch the latest quote for *symbol* from Alpha Vantage.

    Returns a dict on success, or None if the request fails or the
    response contains no usable data – so a failure never raises an
    exception up to the scheduler.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": STOCK_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.error("fetch_stock_data: request timed out for symbol=%s", symbol)
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("fetch_stock_data: request failed for symbol=%s – %s", symbol, exc)
        return None
    except ValueError as exc:
        logger.error("fetch_stock_data: failed to parse JSON for symbol=%s – %s", symbol, exc)
        return None

    quote = data.get("Global Quote", {})

    if not quote:
        logger.warning(
            "fetch_stock_data: empty 'Global Quote' in response for symbol=%s. "
            "Check your STOCK_API_KEY or rate limits.",
            symbol,
        )
        return None

    try:
        return {
            "symbol": symbol,
            "price": float(quote["05. price"]),
            "volume": float(quote["06. volume"]),
            "market_type": "stock",
        }
    except (KeyError, ValueError) as exc:
        logger.error(
            "fetch_stock_data: unexpected quote format for symbol=%s – %s", symbol, exc
        )
        return None
