# data/kraken_api_client.py - FORGALOM ALAPÚ SZŰRÉSSEL

import requests
import logging
import time
import concurrent.futures
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class KrakenAPIClient:
    def __init__(self):
        self.base_url = "https://api.kraken.com/0/public"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Kraken-Trading-Bot/1.0'
        })
        self.rate_limit_delay = 0.5  # Reduced for faster scanning
        
        # Volume filtering settings
        self.MIN_VOLUME_USD = 500000  # $500K minimum volume
        self.EXCLUDE_BTC_ETH = True   # Exclude BTC/ETH pairs
        
    def get_usd_pairs_with_volume(self, min_volume_usd: float = None) -> List[Dict]:
        """
        Lekéri az összes USD párt és szűri a forgalom alapján
        """
        try:
            if min_volume_usd is None:
                min_volume_usd = self.MIN_VOLUME_USD
                
            logger.info(f"🔍 Scanning USD pairs with min volume ${min_volume_usd:,.0f}...")
            
            # 1. Get all asset pairs
            all_pairs = self.get_asset_pairs()
            if not all_pairs:
                logger.error("No asset pairs received")
                return self.get_fallback_pairs()
                
            # 2. Filter USD pairs
            usd_pairs = []
            for altname, details in all_pairs.items():
                wsname = details.get("wsname", "")
                
                # USD pairs check
                if wsname.endswith("/USD") and ".d" not in altname:
                    # Exclude BTC/ETH if configured
                    if self.EXCLUDE_BTC_ETH:
                        if altname in ['XBTUSD', 'ETHUSD', 'XXBTZUSD', 'XETHZUSD']:
                            continue
                    
                    usd_pairs.append({
                        "altname": altname,
                        "wsname": wsname,
                        "base": details.get("base", ""),
                        "quote": details.get("quote", ""),
                        "volume_usd": 0  # Will be filled later
                    })
            
            logger.info(f"📊 Found {len(usd_pairs)} USD pairs (BTC/ETH {'excluded' if self.EXCLUDE_BTC_ETH else 'included'})")
            
            # 3. Get volume data in batches
            pairs_with_volume = self._get_volume_data_batch(usd_pairs)
            
            # 4. Filter by volume
            filtered_pairs = [
                pair for pair in pairs_with_volume 
                if pair['volume_usd'] >= min_volume_usd
            ]
            
            # 5. Sort by volume (highest first)
            filtered_pairs.sort(key=lambda x: x['volume_usd'], reverse=True)
            
            logger.info(f"✅ Found {len(filtered_pairs)} pairs with volume ≥ ${min_volume_usd:,.0f}")
            
            # Log top pairs
            for i, pair in enumerate(filtered_pairs[:10]):
                logger.info(f"  #{i+1}: {pair['wsname']} - ${pair['volume_usd']:,.0f}")
                
            return filtered_pairs
            
        except Exception as e:
            logger.error(f"Error getting USD pairs with volume: {e}")
            return self.get_fallback_pairs()
    
    def _get_volume_data_batch(self, pairs: List[Dict]) -> List[Dict]:
        """
        Batch-ben lekéri a forgalmi adatokat
        """
        try:
            # Prepare pair names for ticker API
            pair_names = [pair['altname'] for pair in pairs]
            
            # Split into batches (Kraken API limitation)
            batch_size = 20
            batches = [pair_names[i:i + batch_size] for i in range(0, len(pair_names), batch_size)]
            
            volume_data = {}
            
            for batch_num, batch in enumerate(batches):
                logger.info(f"📈 Getting volume data for batch {batch_num + 1}/{len(batches)} ({len(batch)} pairs)")
                
                try:
                    # Get ticker data for batch
                    ticker_data = self._get_batch_ticker(batch)
                    
                    for pair_name, data in ticker_data.items():
                        try:
                            # Extract volume and price
                            volume_24h = float(data.get('v', [0, 0])[1])  # 24h volume
                            last_price = float(data.get('c', [0])[0])     # Last price
                            
                            # Calculate USD volume
                            volume_usd = volume_24h * last_price
                            volume_data[pair_name] = volume_usd
                            
                        except (ValueError, IndexError, TypeError) as e:
                            logger.warning(f"Error parsing volume for {pair_name}: {e}")
                            volume_data[pair_name] = 0
                            
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"Error getting ticker data for batch {batch_num + 1}: {e}")
                    # Fill with zeros for this batch
                    for pair_name in batch:
                        volume_data[pair_name] = 0
            
            # Update pairs with volume data
            for pair in pairs:
                pair['volume_usd'] = volume_data.get(pair['altname'], 0)
                
            return pairs
            
        except Exception as e:
            logger.error(f"Batch volume data retrieval failed: {e}")
            return pairs
    
    def _get_batch_ticker(self, pair_names: List[str]) -> Dict:
        """
        Egy batch pár ticker adatainak lekérése
        """
        try:
            url = f"{self.base_url}/Ticker"
            
            # Join pair names with comma
            pairs_param = ','.join(pair_names)
            params = {"pair": pairs_param}
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data and data['error']:
                logger.error(f"Kraken Ticker API error: {data['error']}")
                return {}
                
            return data.get("result", {})
            
        except Exception as e:
            logger.error(f"Error getting batch ticker: {e}")
            return {}
    
    def get_top_volume_pairs(self, limit: int = 50) -> List[Dict]:
        """
        Top volume párok lekérése
        """
        try:
            all_volume_pairs = self.get_usd_pairs_with_volume()
            
            # Return top N pairs
            top_pairs = all_volume_pairs[:limit]
            
            logger.info(f"🏆 Top {len(top_pairs)} volume pairs selected")
            
            return top_pairs
            
        except Exception as e:
            logger.error(f"Error getting top volume pairs: {e}")
            return self.get_fallback_pairs()
    
    def update_volume_filter(self, min_volume_usd: float, exclude_btc_eth: bool = True):
        """
        Forgalom szűrő beállítások frissítése
        """
        self.MIN_VOLUME_USD = min_volume_usd
        self.EXCLUDE_BTC_ETH = exclude_btc_eth
        
        logger.info(f"🎛️ Volume filter updated: min ${min_volume_usd:,.0f}, BTC/ETH excluded: {exclude_btc_eth}")
    
    def get_valid_usd_pairs(self):
        """
        Visszafelé kompatibilitás - forgalom alapú szűréssel
        """
        volume_pairs = self.get_top_volume_pairs(limit=30)
        
        # Convert to old format
        result = []
        for pair in volume_pairs:
            result.append({
                "altname": pair["altname"],
                "wsname": pair["wsname"],
                "base": pair.get("base", ""),
                "quote": pair.get("quote", ""),
                "volume_usd": pair.get("volume_usd", 0)
            })
            
        return result
    
    def get_asset_pairs(self):
        """Get all available asset pairs"""
        try:
            url = f"{self.base_url}/AssetPairs"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'error' in data and data['error']:
                logger.error(f"Kraken API error: {data['error']}")
                return {}
                
            return data.get("result", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"AssetPairs fetch error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error in get_asset_pairs: {e}")
            return {}

    def get_fallback_pairs(self):
        """Fallback pairs if API fails"""
        return [
            {"altname": "ADAUSD", "wsname": "ADA/USD", "volume_usd": 2000000},
            {"altname": "SOLUSD", "wsname": "SOL/USD", "volume_usd": 5000000},
            {"altname": "DOTUSD", "wsname": "DOT/USD", "volume_usd": 1500000},
            {"altname": "LINKUSD", "wsname": "LINK/USD", "volume_usd": 3000000},
            {"altname": "1INCHUSD", "wsname": "1INCH/USD", "volume_usd": 800000}
        ]

    def get_ohlc(self, altname, interval=1):
        """Get OHLC data for a trading pair"""
        try:
            # Add rate limiting
            time.sleep(self.rate_limit_delay)
            
            url = f"{self.base_url}/OHLC"
            params = {"pair": altname, "interval": interval}
            
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
                
            if not result:
                logger.warning(f"Empty OHLC result for {altname}")
                return {}
                
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting OHLC for {altname}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting OHLC for {altname}: {e}")
            return {}

    def test_connection(self):
        """Test API connection"""
        try:
            url = f"{self.base_url}/Time"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if 'result' in data:
                server_time = data['result']['unixtime']
                logger.info(f"API Connection OK, server time: {server_time}")
                return True
            else:
                logger.error(f"Connection failed: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_volume_statistics(self) -> Dict:
        """
        Forgalom statisztikák lekérése
        """
        try:
            pairs = self.get_usd_pairs_with_volume(min_volume_usd=0)  # Get all pairs
            
            if not pairs:
                return {}
                
            volumes = [pair['volume_usd'] for pair in pairs if pair['volume_usd'] > 0]
            
            if not volumes:
                return {}
                
            return {
                'total_pairs': len(pairs),
                'pairs_with_volume': len(volumes),
                'total_volume_usd': sum(volumes),
                'avg_volume_usd': sum(volumes) / len(volumes),
                'max_volume_usd': max(volumes),
                'min_volume_usd': min(volumes),
                'above_500k': len([v for v in volumes if v >= 500000]),
                'above_1m': len([v for v in volumes if v >= 1000000]),
                'above_5m': len([v for v in volumes if v >= 5000000])
            }
            
        except Exception as e:
            logger.error(f"Error getting volume statistics: {e}")
            return {}
