# data/kraken_api_client.py - ENHANCED VERSION WITH BACKGROUND SUPPORT AND MERGED FEATURES

import requests
import logging
import time
import threading
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
# import queue # queue import nem használt, eltávolítható vagy kikommentelhető
import random

# Imports for LiveTradingThread - assuming PyQt5, adjust if using PyQt6
try:
    from PyQt5.QtCore import QThread, pyqtSignal
except ImportError:
    try:
        from PyQt6.QtCore import QThread, pyqtSignal
    except ImportError:
        QThread = object
        pyqtSignal = object
        logging.warning("PyQt5 or PyQt6 not found. LiveTradingThread will not function correctly.")


logger = logging.getLogger(__name__)

class APIRateLimiter:
    """Adaptive API Rate Limiter for Kraken"""

    def __init__(self):
        self.call_counter = 0
        self.last_reset_time = time.time()
        self.request_times = deque(maxlen=50)
        self.base_delay = 1.0
        self.adaptive_delay = 1.0
        self.max_calls_per_minute = 59
        self.burst_allowance = 3
        self.consecutive_errors = 0
        self.lock = threading.Lock()
        self.total_requests_made = 0
        self.successful_requests = 0
        self.failed_requests_429 = 0

    def wait_if_needed(self, priority_level: int = 3, endpoint: Optional[str] = None):
        with self.lock:
            current_time = time.time()

            if current_time - self.last_reset_time >= 60:
                self.call_counter = 0
                self.last_reset_time = current_time

            if len(self.request_times) >= 3: # Legalább 3 kérés alapján kezdünk el finomhangolni
                time_since_last = current_time - self.request_times[-1]['timestamp']
                if time_since_last < self.adaptive_delay:
                    wait_time = self.adaptive_delay - time_since_last

                    if priority_level == 1: wait_time *= 0.5
                    elif priority_level == 2: wait_time *= 0.7
                    elif priority_level >= 4: wait_time *= 1.2

                    if wait_time > 0:
                        logger.debug(f"Rate limiting: waiting for {wait_time:.2f}s (Priority: {priority_level}, Delay: {self.adaptive_delay:.2f}s)")
                        time.sleep(wait_time)

            if self.call_counter >= self.max_calls_per_minute:
                wait_for_reset = 60 - (current_time - self.last_reset_time)
                if wait_for_reset > 0:
                    logger.warning(f"Approaching max calls per minute. Waiting {wait_for_reset:.2f}s for counter reset.")
                    time.sleep(wait_for_reset)
                    self.call_counter = 0
                    self.last_reset_time = time.time()

    def record_request_attempt(self, endpoint: Optional[str] = None):
        with self.lock:
            self.request_times.append({
                'timestamp': time.time(),
                'endpoint': endpoint,
                'status_code': None
            })
            self.call_counter += 1
            self.total_requests_made +=1

    def adjust_delay(self, response_time: float, success: bool = True, status_code: Optional[int] = None):
        with self.lock:
            if self.request_times: # Biztosítjuk, hogy van mit frissíteni
                self.request_times[-1]['status_code'] = status_code
                self.request_times[-1]['response_time'] = response_time

            if success:
                self.consecutive_errors = 0
                self.successful_requests +=1
                if response_time < 0.5:
                    self.adaptive_delay = max(self.base_delay * 0.8, self.adaptive_delay * 0.95)
                elif response_time > 2.0:
                    self.adaptive_delay = min(self.base_delay * 3.0, self.adaptive_delay * 1.1)
                else:
                    self.adaptive_delay = max(self.base_delay, self.adaptive_delay * 0.98)
            else:
                self.consecutive_errors += 1
                if status_code == 429:
                    self.failed_requests_429 +=1
                    self.adaptive_delay = min(self.base_delay * 10.0, self.adaptive_delay * 2.0 * (1.5 ** self.consecutive_errors))
                    logger.warning(f"Rate limit error (429) detected. Increased delay to {self.adaptive_delay:.2f}s.")
                else:
                    self.adaptive_delay = min(self.base_delay * 5.0, self.adaptive_delay * (1.5 ** self.consecutive_errors))
                logger.warning(f"API error (status: {status_code}). Consecutive errors: {self.consecutive_errors}. Adaptive delay increased to {self.adaptive_delay:.2f}s")

    def get_current_delay(self) -> float:
        with self.lock:
            return self.adaptive_delay

    def reset_emergency(self):
        with self.lock:
            logger.warning("Performing emergency rate limit reset!")
            self.call_counter = 0
            self.last_reset_time = time.time()
            self.adaptive_delay = self.base_delay
            self.consecutive_errors = 0
            self.request_times.clear()


class KrakenAPIClient:
    def __init__(self):
        self.base_url = "https://api.kraken.com/0/public"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Kraken-Trading-Bot/2.3' # Verzió növelve
        })

        self.rate_limiter = APIRateLimiter()
        self.rate_limit_delay = 1.0 # Hozzáadva a get_ohlc metódushoz, alapértelmezett érték

        self.MIN_VOLUME_USD = 500000
        self.EXCLUDE_BTC_ETH = True

        self.scanning_active = False
        self.response_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_duration = 30

        self.price_history: Dict[str, deque] = {}
        self.pump_threshold = 0.10

        self.promising_coins: List[Dict] = []
        self.promising_cache_time: float = 0
        self.promising_cache_duration = 300

        self.rate_lock = threading.Lock()

        class MockRateLimit: # Ez a régebbi, egyszerűbb rate limiter a snippetből
            def __init__(self):
                self.counter = 0
                self.max_counter = 20
                self.decay_rate = 0.5
                self.last_request_time = time.time()
                self.request_history = deque(maxlen=100)
        self.rate_limit = MockRateLimit()

        self.asset_pairs_cache: Optional[Dict] = None
        self.cache_timestamp: float = 0

        logger.info("Enhanced KrakenAPIClient initialized with adaptive rate limiting and OHLC pair mapping fix.")

    def _make_request(self, endpoint_path: str, params: Optional[Dict] = None,
                      priority_level: int = 3, timeout: int = 10) -> Optional[Dict]:
        url = f"{self.base_url}{endpoint_path}"
        request_start_time = time.time()
        try:
            self.rate_limiter.wait_if_needed(priority_level, endpoint=endpoint_path)
            self.rate_limiter.record_request_attempt(endpoint=endpoint_path)

            response = self.session.get(url, params=params, timeout=timeout)
            response_time = time.time() - request_start_time
            status_code = response.status_code

            with self.rate_lock:
                self.rate_limit.request_history.append({
                    'timestamp': request_start_time, 'endpoint': endpoint_path,
                    'status_code': status_code, 'response_time': response_time
                })
                self.rate_limit.last_request_time = request_start_time

            if status_code == 429:
                logger.warning(f"Kraken API rate limit hit (429) for {endpoint_path}.")
                self.rate_limiter.adjust_delay(response_time, success=False, status_code=status_code)
                return None

            response.raise_for_status()
            data = response.json()

            self.rate_limiter.adjust_delay(response_time, success=True, status_code=status_code)

            if 'error' in data and data['error']:
                logger.error(f"Kraken API error for {endpoint_path}: {data['error']}")
                return None
            return data.get("result", {}) # Visszaadja a 'result' részt, vagy üres dict-et

        except requests.exceptions.HTTPError as http_err:
            status_code_err = http_err.response.status_code if http_err.response is not None else None
            logger.error(f"HTTP error for {endpoint_path}: {http_err} (Status: {status_code_err})")
            self.rate_limiter.adjust_delay(time.time() - request_start_time, success=False, status_code=status_code_err)
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request exception for {endpoint_path}: {req_err}")
            self.rate_limiter.adjust_delay(time.time() - request_start_time, success=False)
        except Exception as e: # Általánosabb hibakezelés
            logger.error(f"Unexpected error in _make_request for {endpoint_path}: {e}", exc_info=True)
            self.rate_limiter.adjust_delay(time.time() - request_start_time, success=False)
        return None

    # ... (a többi metódus változatlan marad a legutóbbi verzióhoz képest, egészen a get_ohlc-ig) ...
    # Feltételezzük, hogy a start_background_scanning, stop_background_scanning,
    # get_usd_pairs_with_volume_background, _get_volume_data_background,
    # _get_volume_concurrent, _get_volume_sequential, _get_batch_ticker_with_priority,
    # _extract_volume_from_ticker, _update_price_history, _detect_pump,
    # get_btc_eth_prices_priority, update_promising_coins_cache, get_promising_coins_cache,
    # _get_cached_asset_pairs, _is_cache_valid, clear_cache, get_background_scan_status,
    # get_asset_pairs_api, get_usd_pairs_with_volume (legacy) és get_asset_pairs (legacy)
    # metódusok itt helyezkednek el, változatlanul.

    # JAVÍTOTT METÓDUSOK BEILLESZTÉSE (illetve a get_ohlc cseréje):

    def get_ohlc(self, altname, interval=1, limit=None) -> Dict: # Visszatérési típus pontosítva Dict-re
        """Get OHLC data for a trading pair"""
        try:
            time.sleep(self.rate_limit_delay)

            url = f"{self.base_url}/OHLC"
            params = {"pair": altname, "interval": interval}

            # Add limit if specified (Kraken uses 'count' parameter)
            if limit:
                params["count"] = limit

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'error' in data and data['error']:
                logger.error(f"Kraken OHLC API error for {altname}: {data['error']}")
                return {}

            if "result" not in data:
                logger.warning(f"No result in OHLC response for {altname}")
                return {}

            result = data["result"]

            # Remove 'last' timestamp if present
            if 'last' in result:
                del result['last']

            if not result: # Ha a result üres a 'last' eltávolítása után
                logger.warning(f"Empty OHLC result for {altname} after removing 'last'. Original result keys: {list(data.get('result', {}).keys())}")
                return {}

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting OHLC for {altname}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting OHLC for {altname}: {e}", exc_info=True)
            return {}

    def get_valid_usd_pairs(self) -> List[Dict]:
        """USD párok lekérése javított mapping-gel"""
        try:
            # A get_top_volume_pairs már a háttérben optimalizált metódust hívja,
            # ami a get_usd_pairs_with_volume_background-ra támaszkodik.
            # Ez a metódus már tartalmaz 'altname', 'wsname', 'base', 'quote', 'volume_usd' kulcsokat.
            volume_pairs = self.get_top_volume_pairs(limit=30, priority_level=3) # Magasabb prioritás a UI adatokhoz

            if not volume_pairs: # Ha a get_top_volume_pairs üres listát ad vissza
                logger.warning("get_top_volume_pairs returned no pairs for get_valid_usd_pairs. Using fallback.")
                return self.get_fallback_pairs() # Visszatérünk a fallback párokkal

            # A formátum már megfelelőnek tűnik, de biztosítjuk a szükséges kulcsokat.
            # A 'base' és 'quote' információk már a get_usd_pairs_with_volume_background-ból jönnek.
            result = []
            for pair_data in volume_pairs:
                # Biztosítjuk, hogy a szükséges kulcsok létezzenek
                # és az értékek ne legyenek None (bár a .get() ezt kezeli)
                altname = pair_data.get("altname")
                wsname = pair_data.get("wsname")

                if not altname or not wsname: # Ha valamelyik hiányzik, kihagyjuk a párt
                    logger.warning(f"Skipping pair due to missing altname or wsname in get_valid_usd_pairs: {pair_data}")
                    continue

                result.append({
                    "altname": altname,
                    "wsname": wsname,
                    "base": pair_data.get("base", ""), # .get() az esetleges hiányzó kulcsokhoz
                    "quote": pair_data.get("quote", ""),
                    "volume_usd": pair_data.get("volume_usd", 0)
                })

            if not result: # Ha a szűrés után sem maradt érvényes pár
                logger.warning("No valid USD pairs constructed after filtering. Using fallback.")
                return self.get_fallback_pairs()

            return result

        except Exception as e:
            logger.error(f"Error getting valid USD pairs: {e}", exc_info=True)
            return self.get_fallback_pairs() # Hiba esetén is fallback

    def get_fallback_pairs(self) -> List[Dict]: # Visszatérési típus javítva
        """Fallback pairs with correct altnames and wsnames, including base/quote"""
        logger.warning("Using fallback pairs for UI or in case of API error.")
        return [
            {"altname": "XBTUSD", "wsname": "XBT/USD", "base": "XBT", "quote": "USD", "volume_usd": 10000000}, # BTC
            {"altname": "ETHUSD", "wsname": "ETH/USD", "base": "ETH", "quote": "USD", "volume_usd": 7000000},
            {"altname": "ADAUSD", "wsname": "ADA/USD", "base": "ADA", "quote": "USD", "volume_usd": 2000000},
            {"altname": "SOLUSD", "wsname": "SOL/USD", "base": "SOL", "quote": "USD", "volume_usd": 5000000},
            {"altname": "DOTUSD", "wsname": "DOT/USD", "base": "DOT", "quote": "USD", "volume_usd": 1500000},
            {"altname": "LINKUSD", "wsname": "LINK/USD", "base": "LINK", "quote": "USD", "volume_usd": 3000000},
            {"altname": "1INCHUSD", "wsname": "1INCH/USD", "base": "1INCH", "quote": "USD", "volume_usd": 800000},
            {"altname": "UNIUSD", "wsname": "UNI/USD", "base": "UNI", "quote": "USD", "volume_usd": 4500000},
            {"altname": "AVAXUSD", "wsname": "AVAX/USD", "base": "AVAX", "quote": "USD", "volume_usd": 3800000},
            {"altname": "MATICUSD", "wsname": "MATIC/USD", "base": "MATIC", "quote": "USD", "volume_usd": 2800000}
        ]

    # ... (a többi metódus változatlan marad a legutóbbi verzióhoz képest) ...
    # Feltételezzük, hogy a get_volume_statistics, _get_rate_limit_stats,
    # get_top_volume_pairs, update_volume_filter, get_rate_limit_info, _calculate_optimal_delay,
    # emergency_rate_limit_reset, optimize_batch_sizes, get_optimized_scanning_strategy,
    # monitor_rate_limit_health, _get_health_recommendations, get_api_performance_stats,
    # _analyze_endpoint_usage, _calculate_performance_grade, cleanup, __del__,
    # get_market_data, get_current_price, get_ticker_data, és a LiveTradingThread osztály
    # itt helyezkednek el, változatlanul.

    def test_connection(self):
        """Test API connection"""
        try:
            url = f"{self.base_url}/Time"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()
            if 'result' in data and 'unixtime' in data['result']:
                server_time = data['result']['unixtime']
                logger.info(f"API Connection OK, server time: {server_time}")
                return True
            else:
                # Log the actual error content if available, otherwise a generic message
                error_message = data.get('error', 'Unknown error structure')
                logger.error(f"Connection failed: API response: {error_message}")
                return False

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"Connection test failed (HTTPError): {http_err}")
            return False
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Connection test failed (RequestException): {req_err}")
            return False
        except Exception as e:
            logger.error(f"Connection test failed (Unexpected Error): {e}", exc_info=True)
            return False

# Példa a LiveTradingThread definíciójára, ha az ebben a fájlban lenne
# (de a jelenlegi struktúra szerint a kraken_api_client.py-ban van)
# class LiveTradingThread(QThread):
#     # ... (definíció) ...
#     pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    client = KrakenAPIClient()

    logger.info("--- Testing KrakenAPIClient (OHLC Pair Mapping Focus) ---")
    
    # Test test_connection
    logger.info("--- Testing API Connection ---")
    if client.test_connection():
        logger.info("API connection test successful.")
    else:
        logger.error("API connection test failed.")
    
    # Test OHLC data with various altnames
    test_pairs_ohlc = ["XBTUSD", "ETHUSD", "EURUSD", "ADAUSD", "ADAEUR", "DOT/USD", "XXBTZUSD", "XETHZEUR"]
    for pair_test in test_pairs_ohlc:
        logger.info(f"Fetching OHLC for: {pair_test}")
        # Példa limit paraméter használatára:
        # ohlc_data = client.get_ohlc(altname=pair_test, interval=15, limit=5) 
        ohlc_data = client.get_ohlc(altname=pair_test, interval=15)
        if ohlc_data:
            # Az ohlc_data most {"API_PAIR_KEY": [[...], ...]} formátumú
            # (pl. {"XXBTZUSD": [[...candle_data...]]} ha pair_test="XBTUSD")
            # Kiírjuk a kulcsot és az első gyertyát, ha van
            if len(ohlc_data.keys()) > 0:
                actual_pair_key = list(ohlc_data.keys())[0] # Az API által visszaadott pár kulcs
                candle_list = ohlc_data[actual_pair_key]
                if candle_list:
                    logger.info(f"  Successfully fetched for '{actual_pair_key}' (requested '{pair_test}'). First candle: {candle_list[0]}")
                else:
                    logger.warning(f"  Fetched for '{actual_pair_key}' (requested '{pair_test}'), but no candle data returned in the list.")
            else:
                logger.warning(f"  Fetched OHLC for {pair_test}, but the result dictionary is empty.")
        else:
            logger.warning(f"  Could not get OHLC data for {pair_test}, or data format unexpected. Response: {ohlc_data}")
        time.sleep(1.1) # Kis szünet a rate limit miatt, mivel a get_ohlc direkt hívást használ

    logger.info("--- Testing get_valid_usd_pairs ---")
    valid_pairs = client.get_valid_usd_pairs()
    if valid_pairs:
        logger.info(f"Found {len(valid_pairs)} valid USD pairs. First 3:")
        for p_data in valid_pairs[:3]:
            logger.info(f"  {p_data}")
    else:
        logger.warning("No valid USD pairs found by get_valid_usd_pairs.")

    # Feltételezve, hogy létezik a cleanup metódus (a kommentek alapján igen)
    if hasattr(client, 'cleanup') and callable(getattr(client, 'cleanup')):
        client.cleanup()
    logger.info("--- KrakenAPIClient OHLC testing finished ---")
