import requests
import time
import hmac
import hashlib
import base64
import urllib.parse as urlparse
import os
from dotenv import load_dotenv
from core.kraken_nonce import nonce_manager
# Feltételezve, hogy a PositionManager itt vagy egy importálható helyen van definiálva, ha típusellenőrzésre lenne szükség.
# from core.position_manager import PositionManager # Példa import, ha szükséges

load_dotenv()

class TradeManager:
    def __init__(self, api_client, position_manager=None):
        """
        TradeManager inicializálása.

        Args:
            api_client: Az API kliens objektum a kommunikációhoz (pl. KrakenAPIClient).
            position_manager (PositionManager, optional): A pozíciókezelő objektum. Defaults to None.
        """
        self.api_client = api_client
        self.position_manager = position_manager
        
        # A meglévő API kulcsok és URL betöltése továbbra is megmarad,
        # ha a belső _sign és _post metódusok ezt használják.
        # Ha a cél az, hogy a TradeManager teljes mértékben az api_client-en keresztül kommunikáljon,
        # akkor a _sign, _post és place_order metódusokat refaktorálni kellene,
        # hogy az self.api_client képességeit használják.
        self.api_key = os.getenv("KRAKEN_API_KEY")
        self.api_sec = os.getenv("KRAKEN_API_SECRET")
        self.base_url = "https://api.kraken.com"
        
        # Logolás, hogy lássuk, milyen kliensekkel lett inicializálva
        # Egy logger példány jobb lenne itt, de az egyszerűség kedvéért print-et használok
        # A logger használata jobb lenne a production kódban
        # import logging
        # logger = logging.getLogger(__name__)
        # logger.info(f"TradeManager initialized with api_client: {type(api_client)}")
        print(f"TradeManager initialized with api_client: {type(api_client)}")
        if position_manager:
            # logger.info(f"TradeManager initialized with position_manager: {type(position_manager)}")
            print(f"TradeManager initialized with position_manager: {type(position_manager)}")
        else:
            # logger.info("TradeManager initialized without a specific position_manager.")
            print("TradeManager initialized without a specific position_manager.")


    def _sign(self, urlpath, data):
        """Aláírja a kérést a Kraken API számára."""
        # Ez a metódus a self.api_sec-et használja, ami a .env fájlból jön.
        # Ha az api_client tartalmazza ezt a logikát, akkor ezt a metódust
        # az api_client megfelelő metódusára kellene cserélni.
        if not self.api_sec:
            # logger.error("API secret is not configured for TradeManager._sign")
            raise ValueError("API secret is not configured for TradeManager._sign")
            
        postdata = urlparse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()

        mac = hmac.new(base64.b64decode(self.api_sec), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()

    def _post(self, urlpath, data):
        """POST kérést küld a Kraken API-nak."""
        # Ez a metódus a self.api_key-t és a self._sign-t használja.
        # Ha az api_client tartalmazza ezt a logikát, akkor ezt a metódust
        # az api_client megfelelő metódusára kellene cserélni.
        if not self.api_key:
            # logger.error("API key is not configured for TradeManager._post")
            raise ValueError("API key is not configured for TradeManager._post")

        headers = {
            'API-Key': self.api_key,
            'API-Sign': self._sign(urlpath, data)
        }
        # logger.debug(f"Sending POST request to {self.base_url + urlpath} with data: {data} and headers: {headers}")
        response = requests.post(self.base_url + urlpath, headers=headers, data=data)
        # logger.debug(f"Received response: {response.status_code} - {response.text}")
        return response.json()

    def place_order(self, pair, side, volume, price=None, ordertype=None, leverage=None, close_data=None):
        """
        Megbízás elhelyezése.

        Args:
            pair (str): A kereskedési pár (pl. 'XBTUSD').
            side (str): 'buy' vagy 'sell'.
            volume (float): A kereskedési volumen.
            price (float, optional): Limitár. Ha None, market order.
            ordertype (str, optional): Megbízás típusa (pl. 'limit', 'market', 'stop-loss'). Alapértelmezetten 'market' vagy 'limit' a price alapján.
            leverage (str, optional): Tőkeáttétel (pl. '2:1', '5:1').
            close_data (dict, optional): Záró megbízás adatai (pl. stop-loss, take-profit).
                Példa: {'ordertype': 'stop-loss', 'price': stop_loss_price, 'price2': trigger_price}

        Returns:
            dict: Az API válasza.
        """
        urlpath = '/0/private/AddOrder'
        
        # Nonce generálása a nonce_manager segítségével
        current_nonce = nonce_manager.get_nonce()

        data = {
            'nonce': current_nonce,
            'ordertype': ordertype if ordertype else ('market' if not price else 'limit'),
            'type': side.lower(),
            'volume': str(volume), # A Kraken API stringként várja a volument
            'pair': pair
        }
        if price:
            data['price'] = str(price) # Az árat is stringként várhatja
        
        if leverage:
            data['leverage'] = leverage

        if close_data and isinstance(close_data, dict):
            # A záró megbízás adatait a 'close[prefix]' formátumban kell átadni
            for key, value in close_data.items():
                data[f'close[{key}]'] = str(value)
        
        # Itt használhatnánk a self.api_client.place_order() metódust, ha az létezik és megfelelő
        # Jelenleg a belső _post metódust használja, ami a .env fájlból olvassa a kulcsokat.
        # print(f"TradeManager: Placing order with data: {data}") # Debugoláshoz
        # logger.info(f"TradeManager: Placing order for {pair}, side: {side}, volume: {volume}, data: {data}")
        return self._post(urlpath, data)

    def close_market_position(self, pair, side_of_position_to_close, volume):
        """
        Nyitott pozíció zárása market orderrel.

        Args:
            pair (str): A kereskedési pár.
            side_of_position_to_close (str): Az eredeti pozíció iránya ('buy' vagy 'sell').
            volume (float): A zárandó volumen.

        Returns:
            dict: Az API válasza.
        """
        # A záráshoz az eredeti pozícióval ellentétes irányú megbízást kell adni.
        close_side = "sell" if side_of_position_to_close.lower() == "buy" else "buy"
        
        # Logolás a jobb követhetőség érdekében
        # logger.info(f"TradeManager: Attempting to close {volume} of {pair} (original side: {side_of_position_to_close}) with a {close_side} market order.")
        print(f"TradeManager: Attempting to close {volume} of {pair} (original side: {side_of_position_to_close}) with a {close_side} market order.")
        
        return self.place_order(pair=pair, side=close_side, volume=volume, ordertype='market')

    # TODO: További metódusok (pl. meglévő megbízások lekérdezése, törlése stb.)
    # ezeket is az self.api_client-en keresztül kellene implementálni ideális esetben.


