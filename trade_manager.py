import requests
import time
import hmac
import hashlib
import base64
import urllib.parse as urlparse
import os
from dotenv import load_dotenv

load_dotenv()

class TradeManager:
    def __init__(self):
        self.api_key = os.getenv("KRAKEN_API_KEY")
        self.api_sec = os.getenv("KRAKEN_API_SECRET")
        self.base_url = "https://api.kraken.com"

    def _sign(self, urlpath, data):
        postdata = urlparse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()

        mac = hmac.new(base64.b64decode(self.api_sec), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()

    def _post(self, urlpath, data):
        headers = {
            'API-Key': self.api_key,
            'API-Sign': self._sign(urlpath, data)
        }
        response = requests.post(self.base_url + urlpath, headers=headers, data=data)
        return response.json()

    def place_order(self, pair, side, volume, price=None):
        urlpath = '/0/private/AddOrder'
        data = {
            'nonce': int(1000*time.time()),
            'ordertype': 'market' if not price else 'limit',
            'type': side.lower(),
            'volume': volume,
            'pair': pair
        }
        if price:
            data['price'] = price

        return self._post(urlpath, data)

    def close_market_position(self, pair, side, volume):
        # zárás ellentétes iránnyal
        close_side = "sell" if side.lower() == "buy" else "buy"
        return self.place_order(pair, close_side, volume)

