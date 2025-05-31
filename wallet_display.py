# core/wallet_display.py - ÉLES KERESKEDÉSRE JAVÍTVA

import requests
import time
from typing import Dict, Optional

class WalletManager:
    def __init__(self):
        try:
            from core.wallet_tracker import WalletTracker
            self.wallet = WalletTracker()
            self.has_real_wallet = True
            # ✅ ÉLES MÓD: Ellenőrizzük az API kulcsokat
            if not self.wallet.api_key or not self.wallet.api_sec:
                print("[WALLET] ❌ NO API KEYS - Cannot use real wallet")
                self.has_real_wallet = False
            else:
                print("[WALLET] ✅ Real wallet tracker initialized with API keys")
        except ImportError:
            self.wallet = None
            self.has_real_wallet = False
            print("[WALLET] ❌ WalletTracker import failed - using simulation mode")
        
        # Simulation data CSAK fallback-ként
        self.simulation_balance = 750.0
        self.simulation_crypto = {
            'BTC': {'amount': 0.001, 'value_usd': 50.0},
            'ETH': {'amount': 0.05, 'value_usd': 150.0},
            'ADA': {'amount': 100.0, 'value_usd': 75.0}
        }
        
        # Cache for API calls - ÉLES módban rövidebb cache
        self.balance_cache = None
        self.balance_cache_time = 0
        self.cache_duration = 30  # ✅ ÉLES: 30 sec cache (volt 60)
        
        # ✅ ÉLES: API retry settings
        self.api_retry_count = 3
        self.api_retry_delay = 5

    def get_usd_balance(self) -> Optional[float]:
        """
        USD egyenleg lekérdezése ÉLES KERESKEDÉSHEZ
        """
        try:
            # Check cache first - de ÉLES módban rövidebb cache
            if self.balance_cache is not None and time.time() - self.balance_cache_time < self.cache_duration:
                return self.balance_cache
            
            # ✅ ÉLES: Prioritás a valós wallet-nek
            if self.has_real_wallet and self.wallet:
                # ✅ ÉLES: Többszöri próbálkozás API hibáknál
                for attempt in range(self.api_retry_count):
                    try:
                        print(f"[WALLET] 🔄 API attempt {attempt + 1}/{self.api_retry_count}")
                        data = self.wallet.get_trade_balance(asset="ZUSD")
                        
                        if data and 'result' in data and 'eb' in data['result']:
                            balance = float(data['result']['eb'])
                            
                            # Cache the result
                            self.balance_cache = round(balance, 2)
                            self.balance_cache_time = time.time()
                            
                            print(f"[WALLET] ✅ REAL BALANCE: ${balance:.2f}")
                            return self.balance_cache
                        else:
                            print(f"[WALLET] ⚠️ Unexpected API response: {data}")
                            if attempt < self.api_retry_count - 1:
                                time.sleep(self.api_retry_delay)
                                continue
                            
                    except Exception as e:
                        print(f"[WALLET] ❌ API attempt {attempt + 1} failed: {e}")
                        if attempt < self.api_retry_count - 1:
                            time.sleep(self.api_retry_delay)
                            continue
                        else:
                            print(f"[WALLET] 🚨 All API attempts failed!")
                            break
            
            # ✅ ÉLES: CSAK akkor használjon szimulációt, ha tényleg nincs API vagy minden kísérlet sikertelen
            if not self.has_real_wallet or not hasattr(self.wallet, 'api_key'):
                print(f"[WALLET] ⚠️ NO API KEYS - using simulation: ${self.simulation_balance:.2f}")
                self.balance_cache = self.simulation_balance
                self.balance_cache_time = time.time()
                return self.simulation_balance
            else:
                # API kulcsok vannak, de valami hiba van
                print(f"[WALLET] 🚨 API ERROR - This is CRITICAL in live trading!")
                print(f"[WALLET] 🔄 Will retry in {self.cache_duration} seconds")
                # ✅ ÉLES: Visszaadjuk a cache-elt értéket ha van, különben None
                if self.balance_cache is not None:
                    print(f"[WALLET] 📋 Using cached balance: ${self.balance_cache:.2f}")
                    return self.balance_cache
                else:
                    print(f"[WALLET] ❌ NO CACHED BALANCE - Cannot trade safely!")
                    return None  # ✅ ÉLES: None = ne kereskedjen
                    
        except Exception as e:
            print(f"[WALLET] 🚨 CRITICAL ERROR getting USD balance: {e}")
            # ✅ ÉLES: Kritikus hiba esetén ne kereskedjen
            return None

    def get_crypto_holdings(self, min_value_usd: float = 1.0) -> Dict[str, float]:
        """
        Crypto holdings lekérdezése ÉLES KERESKEDÉSHEZ
        """
        try:
            # ✅ ÉLES: Prioritás a valós wallet-nek
            if self.has_real_wallet and self.wallet:
                # ✅ ÉLES: Retry logika
                for attempt in range(self.api_retry_count):
                    try:
                        print(f"[WALLET] 🔄 Getting crypto holdings, attempt {attempt + 1}")
                        
                        # Try real wallet
                        data = self.wallet.get_balance()
                        if not data or 'result' not in data:
                            print(f"[WALLET] ⚠️ Invalid balance API response, attempt {attempt + 1}")
                            if attempt < self.api_retry_count - 1:
                                time.sleep(self.api_retry_delay)
                                continue
                            else:
                                break
                        
                        # Get ticker data for USD conversion
                        ticker_response = requests.get(
                            "https://api.kraken.com/0/public/Ticker",
                            params={"pair": "BTCUSD,ETHUSD,XRPUSD,ADAUSD,SOLUSD"},
                            timeout=15  # ✅ ÉLES: Hosszabb timeout
                        )
                        
                        if ticker_response.status_code != 200:
                            print(f"[WALLET] ⚠️ Ticker API failed, attempt {attempt + 1}")
                            if attempt < self.api_retry_count - 1:
                                time.sleep(self.api_retry_delay)
                                continue
                            else:
                                break
                        
                        ticker_data = ticker_response.json()
                        if 'result' not in ticker_data:
                            print(f"[WALLET] ⚠️ Invalid ticker response, attempt {attempt + 1}")
                            if attempt < self.api_retry_count - 1:
                                time.sleep(self.api_retry_delay)
                                continue
                            else:
                                break
                                
                        usd_ticker = ticker_data['result']
                        
                        # Process real balance data
                        result = {}
                        balance_result = data['result']
                        
                        symbol_map = {
                            "XXBT": "XBTUSD", 
                            "XETH": "ETHUSD", 
                            "XXRP": "XRPUSD",
                            "ADA": "ADAUSD", 
                            "SOL": "SOLUSD"
                        }
                        
                        for asset, amount_str in balance_result.items():
                            try:
                                amount = float(amount_str)
                                if amount <= 0:
                                    continue
                                    
                                if asset in symbol_map:
                                    usd_pair = symbol_map[asset]
                                    if usd_pair in usd_ticker:
                                        last_price = float(usd_ticker[usd_pair]['c'][0])
                                        value_usd = amount * last_price
                                        
                                        if value_usd >= min_value_usd:
                                            result[asset] = round(value_usd, 2)
                                            
                            except (ValueError, KeyError, IndexError) as e:
                                print(f"[WALLET] ⚠️ Error processing {asset}: {e}")
                                continue
                        
                        if result:
                            print(f"[WALLET] ✅ REAL crypto holdings: {result}")
                            return result
                        else:
                            print(f"[WALLET] ✅ No significant real holdings (< ${min_value_usd})")
                            return {}
                            
                    except Exception as e:
                        print(f"[WALLET] ❌ Crypto holdings attempt {attempt + 1} failed: {e}")
                        if attempt < self.api_retry_count - 1:
                            time.sleep(self.api_retry_delay)
                            continue
                        else:
                            break
            
            # ✅ ÉLES: CSAK fallback-ként használjon szimulációt
            print(f"[WALLET] ⚠️ Using simulation crypto holdings - THIS IS NOT IDEAL FOR LIVE TRADING")
            return self._get_simulation_crypto_holdings(min_value_usd)
            
        except Exception as e:
            print(f"[WALLET] 🚨 CRITICAL ERROR getting crypto holdings: {e}")
            return {}

    def _get_simulation_crypto_holdings(self, min_value_usd: float) -> Dict[str, float]:
        """Simulation crypto holdings - CSAK fallback"""
        try:
            result = {}
            for symbol, data in self.simulation_crypto.items():
                value = data['value_usd']
                if value >= min_value_usd:
                    result[f"{symbol}_USD_VALUE"] = value
            
            print(f"[WALLET] 📋 Simulation crypto holdings: {result}")
            return result
            
        except Exception as e:
            print(f"[WALLET] ❌ Simulation holdings error: {e}")
            return {}

    def update_simulation_balance(self, new_balance: float):
        """Update simulation balance - ÉLES módban minimális használat"""
        try:
            if self.has_real_wallet:
                print(f"[WALLET] ⚠️ Simulation balance update in LIVE MODE - this should not happen often")
            
            old_balance = self.simulation_balance
            self.simulation_balance = max(0, new_balance)
            
            # Clear cache to force refresh
            self.balance_cache = None
            
            print(f"[WALLET] 📋 Simulation balance updated: ${old_balance:.2f} -> ${self.simulation_balance:.2f}")
            
        except Exception as e:
            print(f"[WALLET] ❌ Balance update error: {e}")

    def add_trade_profit(self, profit_usd: float):
        """Add trading profit - ÉLES módban real tracking"""
        try:
            if self.has_real_wallet:
                print(f"[WALLET] ✅ LIVE TRADING PROFIT: ${profit_usd:.2f} (tracked in real account)")
                # ✅ ÉLES: Cache törölése hogy fresh balance-t kapjunk
                self.balance_cache = None
                self.balance_cache_time = 0
            else:
                print(f"[WALLET] 📋 Simulation profit: ${profit_usd:.2f}")
                self.simulation_balance += profit_usd
                self.balance_cache = None  # Clear cache
                
        except Exception as e:
            print(f"[WALLET] ❌ Add profit error: {e}")

    def get_portfolio_summary(self) -> Dict:
        """Get complete portfolio summary - ÉLES kereskedéshez"""
        try:
            usd_balance = self.get_usd_balance()
            
            # ✅ ÉLES: Ha nincs balance, ne kereskedjen
            if usd_balance is None:
                return {
                    'error': 'Cannot get USD balance - trading not safe',
                    'usd_balance': 0,
                    'crypto_value': 0,
                    'total_value': 0,
                    'trading_safe': False,
                    'is_simulation': not self.has_real_wallet
                }
            
            crypto_holdings = self.get_crypto_holdings()
            
            total_crypto_value = sum(crypto_holdings.values())
            total_portfolio_value = usd_balance + total_crypto_value
            
            return {
                'usd_balance': usd_balance,
                'crypto_value': total_crypto_value,
                'total_value': total_portfolio_value,
                'crypto_holdings': crypto_holdings,
                'portfolio_allocation': {
                    'usd_percent': (usd_balance / total_portfolio_value * 100) if total_portfolio_value > 0 else 0,
                    'crypto_percent': (total_crypto_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
                },
                'trading_safe': True,  # ✅ ÉLES: Ha idáig eljutunk, biztonságos
                'is_simulation': not self.has_real_wallet
            }
            
        except Exception as e:
            print(f"[WALLET] 🚨 Portfolio summary error: {e}")
            return {
                'error': str(e),
                'usd_balance': 0,
                'crypto_value': 0.0,
                'total_value': 0,
                'crypto_holdings': {},
                'portfolio_allocation': {'usd_percent': 0, 'crypto_percent': 0},
                'trading_safe': False,  # ✅ ÉLES: Hiba esetén ne kereskedjen
                'is_simulation': True
            }

    def is_sufficient_balance(self, required_usd: float) -> bool:
        """Check if sufficient balance for trade - ÉLES kereskedéshez"""
        try:
            current_balance = self.get_usd_balance()
            
            # ✅ ÉLES: Ha nincs balance info, ne engedélyezzen kereskedést
            if current_balance is None:
                print(f"[WALLET] 🚨 Cannot verify balance - blocking trade for ${required_usd:.2f}")
                return False
                
            is_sufficient = current_balance >= required_usd
            print(f"[WALLET] 💰 Balance check: ${current_balance:.2f} >= ${required_usd:.2f} = {is_sufficient}")
            return is_sufficient
        except Exception as e:
            print(f"[WALLET] ❌ Balance check error: {e}")
            return False  # ✅ ÉLES: Hiba esetén ne engedélyezzen kereskedést

    def get_available_for_trading(self, reserve_percent: float = 10.0) -> float:
        """Get available balance for trading (with reserve) - ÉLES kereskedéshez"""
        try:
            total_balance = self.get_usd_balance()
            
            # ✅ ÉLES: Ha nincs balance, 0 elérhető
            if total_balance is None:
                print(f"[WALLET] 🚨 No balance available - cannot calculate trading amount")
                return 0.0
                
            reserve_amount = total_balance * (reserve_percent / 100)
            available = max(0, total_balance - reserve_amount)
            
            print(f"[WALLET] 💰 Available for trading: ${available:.2f} (reserve: ${reserve_amount:.2f})")
            return available
            
        except Exception as e:
            print(f"[WALLET] ❌ Available calculation error: {e}")
            return 0.0  # ✅ ÉLES: Hiba esetén 0

    def get_connection_status(self) -> Dict:
        """Get wallet connection status - ÉLES kereskedéshez"""
        try:
            status = {
                'has_real_wallet': self.has_real_wallet,
                'cache_active': self.balance_cache is not None,
                'cache_age_seconds': time.time() - self.balance_cache_time if self.balance_cache_time > 0 else 0,
                'simulation_mode': not self.has_real_wallet,
                'api_retry_count': self.api_retry_count,
                'cache_duration': self.cache_duration
            }
            
            if self.has_real_wallet and self.wallet:
                # ✅ ÉLES: Test connection
                try:
                    test_response = self.wallet.get_balance()
                    status['api_connection'] = 'connected' if test_response and 'result' in test_response else 'failed'
                    status['last_api_test'] = time.time()
                except Exception as e:
                    status['api_connection'] = 'error'
                    status['api_error'] = str(e)
            else:
                status['api_connection'] = 'simulation'
                
            return status
            
        except Exception as e:
            print(f"[WALLET] ❌ Status check error: {e}")
            return {'error': str(e), 'trading_safe': False}

    def clear_cache(self):
        """Clear balance cache to force refresh - ÉLES kereskedéshez"""
        try:
            self.balance_cache = None
            self.balance_cache_time = 0
            print("[WALLET] 🔄 Cache cleared - next call will fetch fresh data")
        except Exception as e:
            print(f"[WALLET] ❌ Cache clear error: {e}")

    def validate_trading_readiness(self) -> Dict:
        """✅ ÚJ FUNKCIÓ: Validate if ready for live trading"""
        try:
            readiness = {
                'ready': False,
                'issues': [],
                'warnings': []
            }
            
            # Check API connection
            if not self.has_real_wallet:
                readiness['issues'].append("No real wallet connection")
            
            # Check balance access
            balance = self.get_usd_balance()
            if balance is None:
                readiness['issues'].append("Cannot access USD balance")
            elif balance < 50:
                readiness['warnings'].append(f"Low balance: ${balance:.2f}")
            
            # Check API keys
            if self.has_real_wallet and self.wallet:
                if not hasattr(self.wallet, 'api_key') or not self.wallet.api_key:
                    readiness['issues'].append("Missing API key")
                if not hasattr(self.wallet, 'api_sec') or not self.wallet.api_sec:
                    readiness['issues'].append("Missing API secret")
            
            readiness['ready'] = len(readiness['issues']) == 0
            
            if readiness['ready']:
                print("[WALLET] ✅ READY FOR LIVE TRADING")
            else:
                print(f"[WALLET] ❌ NOT READY: {readiness['issues']}")
                
            return readiness
            
        except Exception as e:
            return {
                'ready': False,
                'issues': [f"Validation error: {e}"],
                'warnings': []
            }

    def reset_simulation(self, initial_balance: float = 750.0):
        """Reset simulation to initial state - ÉLES módban limitált használat"""
        try:
            if self.has_real_wallet:
                print(f"[WALLET] ⚠️ Simulation reset in LIVE MODE - should only be used for testing")
                
            self.simulation_balance = initial_balance
            self.simulation_crypto = {
                'BTC': {'amount': 0.001, 'value_usd': 50.0},
                'ETH': {'amount': 0.05, 'value_usd': 150.0},
                'ADA': {'amount': 100.0, 'value_usd': 75.0}
            }
            self.clear_cache()
            print(f"[WALLET] 📋 Simulation reset to ${initial_balance:.2f}")
            
        except Exception as e:
            print(f"[WALLET] ❌ Simulation reset error: {e}")
