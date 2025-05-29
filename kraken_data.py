# kraken_data.py – árlekérdezés logika

import requests

KRAKEN_API_URL = "https://api.kraken.com/0/public/Ticker"


def get_current_price(symbol):
    # Kraken szimbólumformátum pl. BTC/USD -> XBTUSD
    symbol_map = {
        "BTC/USD": "XXBTZUSD",
        "ETH/USD": "XETHZUSD",
        # Add hozzá a használt párokat itt
    }

    kraken_symbol = symbol_map.get(symbol)
    if not kraken_symbol:
        return None

    try:
        response = requests.get(KRAKEN_API_URL, params={"pair": kraken_symbol})
        result = response.json()["result"]
        price = float(next(iter(result.values()))["c"][0])
        return price
    except Exception as e:
        print(f"Hiba az árlekérdezéskor: {e}")
        return None

