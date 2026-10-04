import logging
import requests

logger = logging.getLogger(__name__)

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"
COINS = "bitcoin,ethereum"


def fetch_crypto_data() -> list[dict]:
    """Fetch current USD price and 24 h volume for BTC and ETH from CoinGecko.

    Returns a list of dicts on success, or an empty list on any failure –
    so a network error never propagates up to the scheduler.
    """
    params = {
        "ids": COINS,
        "vs_currencies": "usd",
        "include_24hr_vol": "true",
    }

    try:
        response = requests.get(COINGECKO_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        logger.error("fetch_crypto_data: request timed out")
        return []
    except requests.exceptions.RequestException as exc:
        logger.error("fetch_crypto_data: request failed – %s", exc)
        return []
    except ValueError as exc:
        logger.error("fetch_crypto_data: failed to parse JSON – %s", exc)
        return []

    crypto_list: list[dict] = []

    for coin, values in data.items():
        try:
            crypto_list.append(
                {
                    "symbol": coin.upper(),
                    "price": float(values["usd"]),
                    "volume": float(values.get("usd_24h_vol", 0)),
                    "market_type": "crypto",
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("fetch_crypto_data: skipping %s – unexpected format: %s", coin, exc)

    return crypto_list
