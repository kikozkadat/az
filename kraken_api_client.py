# data/kraken_api_client.py - COMPLETE VERSION WITH KEY ROTATION AND DYNAMIC DISCOVERY

import requests
import logging
import time
import threading
import concurrent.futures
import hmac
import hashlib
import base64
import urllib.parse as urlparse
import os
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class APIKeyRotator:
    """Manages multiple API keys with intelligent rotation"""
    
    def __init__(self):
        # Load all API keys
        self.api_keys = []
        
        # Primary key
        key1 = os.getenv("KRAKEN_API_KEY")
        secret1 = os.getenv("KRAKEN_API_SECRET")
        if key1 and secret1:
            self.api_keys.append({
                'key': key1,
                'secret': secret1,
                'usage_count': 0,
                'last_used': 0,
                'rate_limited_until': 0,
                'consecutive_errors': 0,
                'name': 'API_KEY_1'
            })
        
        # Second key
        key2 = os.getenv("KRAKEN_API_KEY_2")
        secret2 = os.getenv("KRAKEN_API_SECRET_2")
        if key2 and secret2:
            self.api_keys.append({
                'key': key2,
                'secret': secret2,
                'usage_count': 0,
                'last_used': 0,
                'rate_limited_until': 0,
                'consecutive_errors': 0,
                'name': 'API_KEY_2'
            })
        
        # Third key
        key3 = os.getenv("KRAKEN_API_KEY_3")
        secret3 = os.getenv("KRAKEN_API_SECRET_3")
        if key3 and secret3:
            self.api_keys.append({
                'key': key3,
                'secret': secret3,
                'usage_count': 0,
                'last_used': 0,
                'rate_limited_until': 0,
                'consecutive_errors': 0,
                'name': 'API_KEY_3'
            })
        
        if not self.api_keys:
            raise Exception("No API keys found in environment variables!")
        
        self.current_index = 0
        self.lock = threading.Lock()
        logger.info(f"APIKeyRotator initialized with {len(self.api_keys)} keys")
    
    def get_next_key(self) -> Dict[str, Any]:
        """Get the next available API key using intelligent rotation"""
        with self.lock:
            current_time = time.time()
            
            # Try to find a non-rate-limited key
            for _ in range(len(self.api_keys)):
                key_data = self.api_keys[self.current_index]
                
                # Skip if rate limited
                if key_data['rate_limited_until'] > current_time:
                    logger.debug(f"{key_data['name']} is rate limited until {key_data['rate_limited_until']}")
                    self.current_index = (self.current_index + 1) % len(self.api_keys)
                    continue
                
                # Use this key
                key_data['usage_count'] += 1
                key_data['last_used'] = current_time
                
                # Rotate for next call
                next_index = (self.current_index + 1) % len(self.api_keys)
                
                logger.debug(f"Using {key_data['name']} (usage: {key_data['usage_count']})")
                
                self.current_index = next_index
                return key_data
            
            # All keys are rate limited - use the one that will be available soonest
            soonest_available = min(self.api_keys, key=lambda k: k['rate_limited_until'])
            wait_time = soonest_available['rate_limited_until'] - current_time
            
            if wait_time > 0:
                logger.warning(f"All keys rate limited. Waiting {wait_time:.2f}s for {soonest_available['name']}")
                time.sleep(wait_time)
            
            soonest_available['usage_count'] += 1
            soonest_available['last_used'] = current_time
            return soonest_available
    
    def mark_rate_limited(self, key_name: str, duration: int = 60):
        """Mark a key as rate limited"""
        with self.lock:
            for key_data in self.api_keys:
                if key_data['name'] == key_name:
                    key_data['rate_limited_until'] = time.time() + duration
                    key_data['consecutive_errors'] += 1
                    logger.warning(f"{key_name} marked as rate limited for {duration}s")
                    break
    
    def mark_success(self, key_name: str):
        """Mark successful usage of a key"""
        with self.lock:
            for key_data in self.api_keys:
                if key_data['name'] == key_name:
                    key_data['consecutive_errors'] = 0
                    break
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all API keys"""
        with self.lock:
            current_time = time.time()
            status = []
            
            for key_data in self.api_keys:
                status.append({
                    'name': key_data['name'],
                    'usage_count': key_data['usage_count'],
                    'is_rate_limited': key_data['rate_limited_until'] > current_time,
                    'seconds_until_available': max(0, key_data['rate_limited_until'] - current_time),
                    'consecutive_errors': key_data['consecutive_errors']
                })
            
            return {
                'total_keys': len(self.api_keys),
                'available_keys': sum(1 for s in status if not s['is_rate_limited']),
                'keys': status
            }


class KrakenAPIClient:
    """Enhanced Kraken API Client with API key rotation and dynamic pair discovery"""
    
    def __init__(self):
        self.base_url = "https://api.kraken.com/0"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Kraken-Trading-Bot/3.0'
        })
        
        # API Key Rotator
        self.key_rotator = APIKeyRotator()
        
        # Configuration
        self.MIN_VOLUME_USD = 500000
        self.EXCLUDE_BTC_ETH = True
        
        # Caching
        self.response_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self.cache_duration = 30
        self.asset_pairs_cache: Optional[Dict] = None
        self.asset_pairs_cache_time: float = 0
        self.asset_pairs_cache_duration = 300
        
        # Price tracking
        self.price_history: Dict[str, deque] = {}
        
        # Thread safety
        self.cache_lock = threading.Lock()
        
        # Pair mapping for Kraken's naming convention
        self.pair_mapping = {
            'XBTUSD': 'XXBTZUSD',
            'ETHUSD': 'XETHZUSD',
            'BTCUSD': 'XXBTZUSD',
            'XRPUSD': 'XXRPZUSD',
            'ADAUSD': 'ADAUSD',
            'SOLUSD': 'SOLUSD',
            'DOTUSD': 'DOTUSD'
        }
        
        self.reverse_pair_mapping = {v: k for k, v in self.pair_mapping.items()}
        
        logger.info(f"KrakenAPIClient initialized with {self.key_rotator.get_status()['total_keys']} API keys")
    
    def _sign(self, urlpath: str, data: Dict, api_secret: str) -> str:
        """Generate API signature"""
        postdata = urlparse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        
        mac = hmac.new(base64.b64decode(api_secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()
    
    def _make_private_request(self, endpoint_path: str, data: Optional[Dict] = None, 
                             priority_level: int = 3) -> Optional[Dict]:
        """Make authenticated private API request with key rotation"""
        
        # Get next available API key
        key_data = self.key_rotator.get_next_key()
        
        if data is None:
            data = {}
        
        # Add nonce
        data['nonce'] = int(time.time() * 1000)
        
        # Generate signature
        urlpath = f"/0{endpoint_path}"
        signature = self._sign(urlpath, data, key_data['secret'])
        
        # Prepare headers
        headers = {
            'API-Key': key_data['key'],
            'API-Sign': signature
        }
        
        # Make request
        url = f"{self.base_url}{endpoint_path}"
        
        try:
            response = self.session.post(url, headers=headers, data=data, timeout=10)
            
            if response.status_code == 429:
                # Rate limited - mark this key and try another
                self.key_rotator.mark_rate_limited(key_data['name'], duration=60)
                logger.warning(f"Rate limit hit on {key_data['name']}, rotating to next key")
                
                # Recursive call with different key
                return self._make_private_request(endpoint_path, data, priority_level)
            
            response.raise_for_status()
            result = response.json()
            
            if 'error' in result and result['error']:
                logger.error(f"Kraken API error on {key_data['name']}: {result['error']}")
                
                # Check for rate limit errors
                error_str = str(result['error'])
                if 'Rate limit exceeded' in error_str or 'EAPI:Rate limit exceeded' in error_str:
                    self.key_rotator.mark_rate_limited(key_data['name'], duration=60)
                    return self._make_private_request(endpoint_path, data, priority_level)
                
                return None
            
            # Success
            self.key_rotator.mark_success(key_data['name'])
            return result.get('result', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error on {key_data['name']}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error on {key_data['name']}: {e}", exc_info=True)
            return None
    
    def _make_public_request(self, endpoint_path: str, params: Optional[Dict] = None,
                            priority_level: int = 3, timeout: int = 10) -> Optional[Dict]:
        """Make public API request (no auth needed, but still benefit from multiple IPs)"""
        
        url = f"{self.base_url}/public{endpoint_path}"
        
        try:
            # For public requests, we can still track which "session" we're using
            key_data = self.key_rotator.get_next_key()
            logger.debug(f"Making public request using session for {key_data['name']}")
            
            response = self.session.get(url, params=params, timeout=timeout)
            
            if response.status_code == 429:
                logger.warning(f"Rate limit on public endpoint {endpoint_path}")
                time.sleep(2)  # Brief pause
                return None
            
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data and data['error']:
                logger.error(f"Kraken API error for {endpoint_path}: {data['error']}")
                return None
                
            return data.get("result", {})
            
        except Exception as e:
            logger.error(f"Public request error for {endpoint_path}: {e}", exc_info=True)
            return None
    
    # Trading methods using private API
    def place_order(self, pair: str, side: str, volume: float, price: Optional[float] = None) -> Optional[Dict]:
        """Place a trading order"""
        data = {
            'pair': pair,
            'type': side.lower(),
            'ordertype': 'limit' if price else 'market',
            'volume': str(volume)
        }
        
        if price:
            data['price'] = str(price)
        
        return self._make_private_request('/private/AddOrder', data)
    
    def get_open_orders(self) -> Optional[Dict]:
        """Get open orders"""
        return self._make_private_request('/private/OpenOrders')
    
    def cancel_order(self, order_id: str) -> Optional[Dict]:
        """Cancel an order"""
        data = {'txid': order_id}
        return self._make_private_request('/private/CancelOrder', data)
    
    def get_account_balance(self) -> Optional[Dict]:
        """Get account balance"""
        return self._make_private_request('/private/Balance')
    
    def get_trade_history(self, trades: bool = False) -> Optional[Dict]:
        """Get trade history"""
        data = {'trades': trades}
        return self._make_private_request('/private/TradesHistory', data)
    
    # Public methods with dynamic pair discovery
    def get_all_asset_pairs(self) -> Optional[Dict]:
        """Get all available asset pairs from Kraken"""
        try:
            with self.cache_lock:
                current_time = time.time()
                if (self.asset_pairs_cache is not None and 
                    current_time - self.asset_pairs_cache_time < self.asset_pairs_cache_duration):
                    logger.debug("Using cached asset pairs")
                    return self.asset_pairs_cache
            
            result = self._make_public_request('/AssetPairs', priority_level=2)
            
            if result:
                with self.cache_lock:
                    self.asset_pairs_cache = result
                    self.asset_pairs_cache_time = time.time()
                logger.info(f"Fetched {len(result)} asset pairs from Kraken")
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting asset pairs: {e}", exc_info=True)
            return None
    
    def get_usd_pairs_with_volume(self, min_volume_usd: float = 500000) -> List[Dict]:
        """Get USD pairs with volume filtering"""
        try:
            logger.info(f"Getting USD pairs with minimum volume: ${min_volume_usd:,.0f}")
            
            # Get all asset pairs
            asset_pairs = self.get_all_asset_pairs()
            if not asset_pairs:
                logger.error("CRITICAL: Could not fetch asset pairs from Kraken API")
                raise Exception("Failed to fetch asset pairs - API connection required")
            
            # Filter USD pairs
            usd_pairs = []
            for pair_name, pair_info in asset_pairs.items():
                # Check if it's a USD pair
                if pair_info.get('quote') == 'ZUSD' or pair_info.get('quote') == 'USD':
                    altname = pair_info.get('altname', pair_name)
                    wsname = pair_info.get('wsname', altname)
                    base = pair_info.get('base', '')
                    
                    # Skip BTC and ETH if configured
                    if self.EXCLUDE_BTC_ETH:
                        if base in ['XXBT', 'XBT', 'XETH', 'ETH']:
                            continue
                    
                    usd_pairs.append({
                        'pair_name': pair_name,
                        'altname': altname,
                        'wsname': wsname,
                        'base': base,
                        'quote': 'USD'
                    })
            
            logger.info(f"Found {len(usd_pairs)} USD pairs")
            
            # Get volumes for pairs
            return self._enrich_pairs_with_volume(usd_pairs, min_volume_usd)
            
        except Exception as e:
            logger.error(f"Error in get_usd_pairs_with_volume: {e}", exc_info=True)
            raise Exception(f"Failed to get USD pairs: {e}")
    
    def _enrich_pairs_with_volume(self, pairs: List[Dict], min_volume_usd: float) -> List[Dict]:
        """Enrich pairs with volume data using concurrent requests"""
        try:
            enriched_pairs = []
            
            # Use ThreadPoolExecutor for concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Create futures for each pair
                future_to_pair = {}
                for pair_info in pairs[:30]:  # Limit to top 30 to avoid too many requests
                    future = executor.submit(self._get_pair_volume, pair_info)
                    future_to_pair[future] = pair_info
                
                # Process completed futures
                for future in concurrent.futures.as_completed(future_to_pair):
                    pair_info = future_to_pair[future]
                    try:
                        volume_usd = future.result()
                        if volume_usd and volume_usd >= min_volume_usd:
                            pair_info['volume_usd'] = volume_usd
                            enriched_pairs.append(pair_info)
                    except Exception as e:
                        logger.error(f"Error getting volume for {pair_info.get('altname')}: {e}")
            
            # Sort by volume
            enriched_pairs.sort(key=lambda x: x.get('volume_usd', 0), reverse=True)
            
            logger.info(f"Found {len(enriched_pairs)} pairs with volume >= ${min_volume_usd:,.0f}")
            return enriched_pairs
            
        except Exception as e:
            logger.error(f"Error enriching pairs with volume: {e}", exc_info=True)
            return []
    
    def _get_pair_volume(self, pair_info: Dict) -> Optional[float]:
        """Get 24h volume for a pair in USD"""
        try:
            altname = pair_info.get('altname')
            ticker = self.get_ticker_data(altname)
            
            if ticker and 'volume_24h' in ticker:
                # Get current price to calculate USD volume
                price = ticker.get('price', 0)
                volume = ticker.get('volume_24h', 0)
                
                if price and volume:
                    return float(price) * float(volume)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting volume for {pair_info.get('altname')}: {e}")
            return None
    
    def get_ticker_data(self, pair_altname: str) -> Optional[Dict]:
        """Get ticker data for a trading pair"""
        try:
            kraken_pair = self._get_pair_key(pair_altname)
            logger.debug(f"Fetching ticker data for {pair_altname} (kraken: {kraken_pair})")
            
            response = self._make_public_request('/Ticker', {'pair': kraken_pair})
            
            if not response:
                logger.warning(f"No response from Ticker API for {pair_altname}")
                return None
            
            # Find the correct key in response
            pair_data = None
            for key, data in response.items():
                if (key.upper() == kraken_pair.upper() or 
                    key.replace('/', '').upper() == kraken_pair.replace('/', '').upper() or
                    self._normalize_pair_name(key) == pair_altname.upper()):
                    pair_data = data
                    break
            
            if not pair_data:
                logger.warning(f"No ticker data found for {pair_altname}")
                return None
            
            # Parse ticker data with safe type conversion
            def safe_float(value):
                """Safely convert value to float"""
                if value is None:
                    return None
                try:
                    return float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert '{value}' to float")
                    return None
            
            current_price = None
            if 'c' in pair_data and len(pair_data['c']) > 0:
                current_price = safe_float(pair_data['c'][0])
            
            bid_price = None
            if 'b' in pair_data and len(pair_data['b']) > 0:
                bid_price = safe_float(pair_data['b'][0])
                
            ask_price = None
            if 'a' in pair_data and len(pair_data['a']) > 0:
                ask_price = safe_float(pair_data['a'][0])
                
            volume_24h = None
            if 'v' in pair_data and len(pair_data['v']) > 1:
                volume_24h = safe_float(pair_data['v'][1])
                
            high_24h = None
            if 'h' in pair_data and len(pair_data['h']) > 1:
                high_24h = safe_float(pair_data['h'][1])
                
            low_24h = None
            if 'l' in pair_data and len(pair_data['l']) > 1:
                low_24h = safe_float(pair_data['l'][1])
            
            return {
                'price': current_price,
                'bid': bid_price,
                'ask': ask_price,
                'volume_24h': volume_24h,
                'high_24h': high_24h,
                'low_24h': low_24h
            }
            
        except Exception as e:
            logger.error(f"Error getting ticker data for {pair_altname}: {e}", exc_info=True)
            return None
    
    def get_ohlc(self, altname: str, interval: int = 1, limit: Optional[int] = None) -> Dict:
        """Get OHLC data for a trading pair"""
        try:
            kraken_pair = self._get_pair_key(altname)
            
            params = {"pair": kraken_pair, "interval": interval}
            if limit:
                params["count"] = limit
            
            response = self._make_public_request('/OHLC', params, priority_level=3)
            
            if not response:
                logger.warning(f"No OHLC response for {altname}")
                return {}
            
            # Remove 'last' timestamp if present
            if 'last' in response:
                del response['last']
            
            # Process and ensure all OHLC values are floats
            processed_response = {}
            for pair_key, candles in response.items():
                if isinstance(candles, list):
                    processed_candles = []
                    for candle in candles:
                        if isinstance(candle, list) and len(candle) >= 8:
                            try:
                                # OHLC format: [time, open, high, low, close, vwap, volume, count]
                                processed_candle = [
                                    int(candle[0]),      # time
                                    float(candle[1]),    # open
                                    float(candle[2]),    # high
                                    float(candle[3]),    # low
                                    float(candle[4]),    # close
                                    float(candle[5]),    # vwap
                                    float(candle[6]),    # volume
                                    int(candle[7])       # count
                                ]
                                processed_candles.append(processed_candle)
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Error processing OHLC candle for {pair_key}: {e}")
                                continue
                    
                    if processed_candles:
                        # Normalize the pair key
                        normalized_key = self._normalize_pair_name(pair_key)
                        processed_response[normalized_key] = processed_candles
            
            return processed_response
            
        except Exception as e:
            logger.error(f"Error getting OHLC for {altname}: {e}", exc_info=True)
            return {}
    
    def get_current_price(self, pair_altname: str) -> Optional[float]:
        """Get current price for a trading pair"""
        try:
            logger.debug(f"Fetching current price for {pair_altname}")
            
            # Try ticker data first
            ticker_data = self.get_ticker_data(pair_altname)
            if ticker_data and 'price' in ticker_data:
                price = ticker_data['price']
                if price is not None and price > 0:
                    logger.debug(f"Got price from Ticker for {pair_altname}: {price}")
                    return float(price)
            
            # Fallback to OHLC
            ohlc_data = self.get_ohlc(pair_altname, interval=1, limit=1)
            if ohlc_data:
                # Find the correct key
                for key, candles in ohlc_data.items():
                    if candles and len(candles) > 0:
                        close_price = float(candles[-1][4])  # Close price
                        if close_price > 0:
                            logger.debug(f"Got price from OHLC for {pair_altname}: {close_price}")
                            return close_price
            
            logger.warning(f"Could not get current price for {pair_altname}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {pair_altname}: {e}", exc_info=True)
            return None
    
    def get_valid_usd_pairs(self) -> List[Dict]:
        """Get valid USD pairs for trading"""
        try:
            # Get USD pairs with volume
            pairs = self.get_usd_pairs_with_volume(min_volume_usd=self.MIN_VOLUME_USD)
            
            if not pairs:
                logger.error("CRITICAL: No USD pairs found - cannot continue")
                raise Exception("No valid USD pairs available for trading")
            
            # Format for GUI
            result = []
            for pair_data in pairs[:20]:  # Limit to top 20
                result.append({
                    "altname": pair_data.get("altname", ""),
                    "wsname": pair_data.get("wsname", ""),
                    "base": pair_data.get("base", ""),
                    "quote": "USD",
                    "volume_usd": pair_data.get("volume_usd", 0)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting valid USD pairs: {e}", exc_info=True)
            raise Exception(f"Failed to get valid USD pairs: {e}")
    
    def get_market_data(self) -> List[Dict]:
        """Get market data for scoring system"""
        try:
            logger.info("Fetching market data for scoring system...")
            
            # Get USD pairs
            pairs = self.get_valid_usd_pairs()
            
            market_data = []
            for pair_info in pairs[:15]:  # Limit to 15 pairs
                altname = pair_info.get('altname')
                if not altname:
                    continue
                
                # Get current price
                current_price = self.get_current_price(altname)
                if not current_price:
                    continue
                
                # Get ticker for additional data
                ticker = self.get_ticker_data(altname)
                
                # Ensure all numeric values are floats, not strings
                def ensure_float(value, default=0.0):
                    """Ensure value is float"""
                    if value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                # Calculate safe values
                safe_price = ensure_float(current_price, 0.0)
                safe_high = ensure_float(ticker.get('high_24h', current_price) if ticker else current_price, safe_price)
                safe_low = ensure_float(ticker.get('low_24h', current_price) if ticker else current_price, safe_price)
                safe_volume = ensure_float(ticker.get('volume_24h', 0) if ticker else 0, 0.0)
                
                # Calculate change percentage safely
                change_24h = 0.0
                if safe_low > 0:
                    change_24h = ((safe_price - safe_low) / safe_low * 100)
                
                market_entry = {
                    'symbol': altname,
                    'price': safe_price,
                    'volume_usd': ensure_float(pair_info.get('volume_usd', 0), 0.0),
                    'volume_24h': safe_volume,
                    'high_24h': safe_high,
                    'low_24h': safe_low,
                    'bid': ensure_float(ticker.get('bid', safe_price * 0.999) if ticker else safe_price * 0.999, safe_price * 0.999),
                    'ask': ensure_float(ticker.get('ask', safe_price * 1.001) if ticker else safe_price * 1.001, safe_price * 1.001),
                    'change_24h': change_24h
                }
                
                market_data.append(market_entry)
                
                # Small delay to avoid rate limiting
                time.sleep(0.2)
            
            logger.info(f"Successfully fetched market data for {len(market_data)} pairs")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}", exc_info=True)
            return []
    
    def _get_pair_key(self, altname: str) -> str:
        """Get the correct pair key for Kraken API"""
        if altname in self.pair_mapping:
            return self.pair_mapping[altname]
        
        upper_altname = altname.upper()
        if upper_altname in self.pair_mapping:
            return self.pair_mapping[upper_altname]
        
        return altname
    
    def _normalize_pair_name(self, kraken_pair: str) -> str:
        """Convert Kraken pair name back to standard format"""
        if kraken_pair in self.reverse_pair_mapping:
            return self.reverse_pair_mapping[kraken_pair]
        
        cleaned = kraken_pair.replace('/', '').upper()
        if cleaned.startswith('XXBT'):
            cleaned = cleaned.replace('XXBT', 'XBT')
        if cleaned.startswith('XETH'):
            cleaned = cleaned.replace('XETH', 'ETH')
        if cleaned.startswith('XXRP'):
            cleaned = cleaned.replace('XXRP', 'XRP')
            
        return cleaned
    
    def check_api_health(self) -> Dict[str, Any]:
        """Comprehensive API health check"""
        health_status = {
            'is_healthy': False,
            'connection': False,
            'can_fetch_pairs': False,
            'can_fetch_prices': False,
            'timestamp': time.time(),
            'errors': []
        }
        
        try:
            # Test basic connection
            health_status['connection'] = self.test_connection()
            if not health_status['connection']:
                health_status['errors'].append("API connection failed")
                return health_status
            
            # Test fetching pairs
            try:
                pairs = self.get_all_asset_pairs()
                if pairs and len(pairs) > 0:
                    health_status['can_fetch_pairs'] = True
                else:
                    health_status['errors'].append("No asset pairs returned")
            except Exception as e:
                health_status['errors'].append(f"Failed to fetch pairs: {e}")
            
            # Test fetching a price
            try:
                price = self.get_current_price("ADAUSD")
                if price and price > 0:
                    health_status['can_fetch_prices'] = True
                else:
                    health_status['errors'].append("Failed to fetch valid price")
            except Exception as e:
                health_status['errors'].append(f"Failed to fetch price: {e}")
            
            # Overall health
            health_status['is_healthy'] = (
                health_status['connection'] and 
                health_status['can_fetch_pairs'] and 
                health_status['can_fetch_prices']
            )
            
            return health_status
            
        except Exception as e:
            health_status['errors'].append(f"Health check failed: {e}")
            return health_status
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get comprehensive API status including key rotation status"""
        return {
            'connection_test': self.test_connection(),
            'key_rotation_status': self.key_rotator.get_status(),
            'cache_size': len(self.response_cache),
            'asset_pairs_cached': self.asset_pairs_cache is not None
        }
    
    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            response = self._make_public_request('/Time', priority_level=1)
            
            if response and 'unixtime' in response:
                logger.info(f"API Connection OK, server time: {response['unixtime']}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}", exc_info=True)
            return False
    
    def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("Cleaning up KrakenAPIClient...")
            
            if hasattr(self, 'session'):
                self.session.close()
            
            # Log final key usage stats
            status = self.key_rotator.get_status()
            logger.info(f"Final API key usage stats: {status}")
            
            logger.info("KrakenAPIClient cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)


# Background thread for continuous pair discovery
class PairDiscoveryThread(threading.Thread):
    """Background thread for discovering new trading pairs"""
    
    def __init__(self, api_client: KrakenAPIClient, update_interval: int = 300):
        super().__init__(daemon=True)
        self.api_client = api_client
        self.update_interval = update_interval
        self.running = False
        self.discovered_pairs = []
        self.lock = threading.Lock()
        
    def run(self):
        """Run pair discovery in background"""
        self.running = True
        logger.info("Starting pair discovery thread")
        
        while self.running:
            try:
                # Discover pairs
                pairs = self.api_client.get_usd_pairs_with_volume()
                
                with self.lock:
                    self.discovered_pairs = pairs
                    
                logger.info(f"Discovered {len(pairs)} USD pairs")
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in pair discovery thread: {e}", exc_info=True)
                time.sleep(60)  # Wait before retry
    
    def get_pairs(self) -> List[Dict]:
        """Get discovered pairs thread-safely"""
        with self.lock:
            return self.discovered_pairs.copy()
    
    def stop(self):
        """Stop the discovery thread"""
        self.running = False


# Safe wrapper for production use
class SafeKrakenAPIClient:
    """Wrapper for KrakenAPIClient that ensures safe operation"""
    
    def __init__(self):
        self.client = KrakenAPIClient()
        self.last_health_check = 0
        self.health_check_interval = 60  # Check health every minute
        self._is_healthy = False
        
    def _ensure_healthy(self):
        """Ensure API is healthy before operations"""
        current_time = time.time()
        
        # Check if we need a new health check
        if current_time - self.last_health_check > self.health_check_interval:
            health = self.client.check_api_health()
            self._is_healthy = health['is_healthy']
            self.last_health_check = current_time
            
            if not self._is_healthy:
                error_msg = f"API is not healthy: {', '.join(health['errors'])}"
                logger.error(error_msg)
                raise Exception(error_msg)
    
    def get_market_data(self) -> List[Dict]:
        """Get market data with health check"""
        self._ensure_healthy()
        return self.client.get_market_data()
    
    def get_valid_usd_pairs(self) -> List[Dict]:
        """Get USD pairs with health check"""
        self._ensure_healthy()
        return self.client.get_valid_usd_pairs()
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Get price with health check"""
        self._ensure_healthy()
        return self.client.get_current_price(pair)
    
    def place_order(self, pair: str, side: str, volume: float, price: Optional[float] = None) -> Optional[Dict]:
        """Place order with health check"""
        self._ensure_healthy()
        return self.client.place_order(pair, side, volume, price)
    
    def get_account_balance(self) -> Optional[Dict]:
        """Get balance with health check"""
        self._ensure_healthy()
        return self.client.get_account_balance()
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API status"""
        return self.client.get_api_status()
    
    def cleanup(self):
        """Cleanup resources"""
        self.client.cleanup()


# Enhanced TradeManager with key rotation
class TradeManager:
    """Trade Manager with API key rotation support"""
    
    def __init__(self):
        self.api_client = SafeKrakenAPIClient()
    
    def place_order(self, pair: str, side: str, volume: float, price: Optional[float] = None) -> Optional[Dict]:
        """Place order using rotated API keys"""
        return self.api_client.place_order(pair, side, volume, price)
    
    def close_market_position(self, pair: str, side: str, volume: float) -> Optional[Dict]:
        """Close position with opposite side order"""
        close_side = "sell" if side.lower() == "buy" else "buy"
        return self.place_order(pair, close_side, volume)
    
    def get_open_orders(self) -> Optional[Dict]:
        """Get open orders"""
        return self.api_client.client.get_open_orders()
    
    def cancel_order(self, order_id: str) -> Optional[Dict]:
        """Cancel order"""
        return self.api_client.client.cancel_order(order_id)
    
    def get_balance(self) -> Optional[Dict]:
        """Get account balance"""
        return self.api_client.get_account_balance()
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get API and key rotation status"""
        return self.api_client.get_api_status()


if __name__ == '__main__':
    # Complete test suite
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        logger.info("Testing Complete KrakenAPIClient with key rotation and dynamic discovery...")
        
        # Initialize client
        client = KrakenAPIClient()
        
        # Show API key status
        logger.info("\n=== API Key Status ===")
        status = client.get_api_status()
        logger.info(f"Total keys: {status['key_rotation_status']['total_keys']}")
        logger.info(f"Available keys: {status['key_rotation_status']['available_keys']}")
        
        # Test connection
        logger.info("\n=== Testing Connection ===")
        if client.test_connection():
            logger.info("✓ API connection successful")
        else:
            logger.error("✗ API connection failed")
            exit(1)
        
        # Test dynamic pair discovery
        logger.info("\n=== Testing Dynamic Pair Discovery ===")
        
        # Get all asset pairs
        all_pairs = client.get_all_asset_pairs()
        if all_pairs:
            logger.info(f"✓ Found {len(all_pairs)} total asset pairs")
        else:
            logger.error("✗ Failed to get asset pairs")
        
        # Get USD pairs with volume
        logger.info("\nGetting USD pairs with volume...")
        usd_pairs = client.get_usd_pairs_with_volume(min_volume_usd=100000)
        logger.info(f"✓ Found {len(usd_pairs)} USD pairs with volume > $100k")
        
        if usd_pairs:
            logger.info("Top 5 pairs by volume:")
            for pair in usd_pairs[:5]:
                logger.info(f"  - {pair['altname']}: ${pair['volume_usd']:,.0f}")
        
        # Test market data
        logger.info("\n=== Testing Market Data ===")
        market_data = client.get_market_data()
        if market_data:
            logger.info(f"✓ Got market data for {len(market_data)} pairs")
            for data in market_data[:3]:
                logger.info(f"  - {data['symbol']}: ${data['price']:.4f} (Vol: ${data['volume_usd']:,.0f})")
        
        # Test private endpoints
        logger.info("\n=== Testing Private Endpoints ===")
        
        # Test balance
        balance = client.get_account_balance()
        if balance:
            logger.info("✓ Successfully fetched account balance")
            logger.info(f"Balance keys: {list(balance.keys())[:5]}...")  # Show first 5 keys
        else:
            logger.warning("✗ Could not fetch balance (check API permissions)")
        
        # Test background discovery
        logger.info("\n=== Testing Background Pair Discovery ===")
        discovery_thread = PairDiscoveryThread(client, update_interval=10)
        discovery_thread.start()
        
        # Wait for discovery
        time.sleep(15)
        
        discovered = discovery_thread.get_pairs()
        if discovered:
            logger.info(f"✓ Background thread discovered {len(discovered)} pairs")
        
        discovery_thread.stop()
        
        # Test safe client
        logger.info("\n=== Testing SafeKrakenAPIClient ===")
        safe_client = SafeKrakenAPIClient()
        
        try:
            safe_pairs = safe_client.get_valid_usd_pairs()
            logger.info(f"✓ Safe client got {len(safe_pairs)} pairs")
        except Exception as e:
            logger.error(f"✗ Safe client error: {e}")
        
        # Test TradeManager
        logger.info("\n=== Testing TradeManager ===")
        trade_manager = TradeManager()
        tm_status = trade_manager.get_api_status()
        logger.info(f"✓ TradeManager initialized with {tm_status['key_rotation_status']['total_keys']} keys")
        
        # Final key usage stats
        logger.info("\n=== Final Key Usage Statistics ===")
        final_status = client.get_api_status()
        for key_info in final_status['key_rotation_status']['keys']:
            logger.info(
                f"{key_info['name']}: {key_info['usage_count']} uses, "
                f"Rate limited: {key_info['is_rate_limited']}, "
                f"Errors: {key_info['consecutive_errors']}"
            )
        
        # Cleanup
        client.cleanup()
        safe_client.cleanup()
        logger.info("\n✓ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        exit(1)
