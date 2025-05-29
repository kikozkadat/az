import requests
import time
import hmac
import hashlib
import base64
import urllib.parse as urlparse
import os
from dotenv import load_dotenv

load_dotenv()

class WalletTracker:
    def __init__(self):
        self.api_key = os.getenv("KRAKEN_API_KEY")
        self.api_sec = base64.b64decode(os.getenv("KRAKEN_API_SECRET"))
        self.base_url = "https://api.kraken.com"
        self.nonce = int(time.time() * 1000000)

    def _get_nonce(self):
        self.nonce += 1
        return str(self.nonce)

    def _sign(self, urlpath, data):
        postdata = urlparse.urlencode(data)
        encoded = (data['nonce'] + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(self.api_sec, message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _private_post(self, urlpath, data):
        data['nonce'] = self._get_nonce()
        headers = {
            'API-Key': self.api_key,
            'API-Sign': self._sign(urlpath, data)
        }
        response = requests.post(self.base_url + urlpath, headers=headers, data=data)
        return response.json()

    def get_balance(self):
        return self._private_post("/0/private/Balance", {})

    def get_trade_balance(self, asset='ZUSD'):
        return self._private_post("/0/private/TradeBalance", {'asset': asset})

