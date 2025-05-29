# core/wallet_display.py - Javított egyenleg kezelés

import requests
import time
from typing import Dict, Optional

class WalletManager:
    def __init__(self):
        try:
            from core.wallet_tracker import WalletTracker
            self.wallet = WalletTracker()
            self.has_real_wallet = True
            print("[WALLET] Real wallet tracker initialized")
        except ImportError:
            self.wallet = None
            self.has_real_wallet = False
            print("[WALLET] Using simulation mode - no real wallet connection")
        
        # Simulation data for testing
        self.simulation_balance = 750.0
        self.simulation_crypto = {
            'BTC': {'amount': 0.001, 'value_usd': 50.0},
            'ETH': {'amount': 0.05, 'value_usd': 150.0},
            'ADA': {'amount': 100.0, 'value_usd': 75.0}
        }
        
        # Cache for API calls
        self.balance_cache = None
        self.balance_cache_time = 0
        self.cache_duration = 60  # 1 minute cache

    def get_usd_balance(self) -> Optional[float]:
        """
        USD egyenleg lekérdezése fejlesztett hibakezeléssel
        """
        try:
            # Check cache first
            if self.balance_cache is not None and time.time() - self.balance_cache_time < self.cache_duration:
                return self.balance_cache
            
            if self.has_real_wallet and self.wallet:
                # Try real wallet first
                try:
                    data = self.wallet.get_trade_balance(asset="ZUSD")
                    if data and 'result' in data and 'eb' in data['result']:
                        balance = float(data['result']['eb'])
                        
                        # Cache the result
                        self.balance_cache = round(balance, 2)
                        self.balance_cache_time = time.time()
                        
                        print(f"[WALLET] Real balance retrieved: ${balance:.2f}")
                        return self.balance_cache
                    else:
                        print(f"[WALLET] Unexpected API response format: {data}")
                        
                except Exception as e:
                    print(f"[WALLET] Real wallet API failed: {e}")
                    # Fall through to simulation
            
            # Use simulation balance
            print(f"[WALLET] Using simulation balance: ${self.simulation_balance:.2f}")
            self.balance_cache = self.simulation_balance
            self.balance_cache_time = time.time()
            return self.simulation_balance
            
        except Exception as e:
            print(f"[WALLET] Critical error getting USD balance: {e}")
            # Emergency fallback
            return 500.0

    def get_crypto_holdings(self, min_value_usd: float = 1.0) -> Dict[str, float]:
        """
        Crypto holdings lekérdezése fejlesztett hibakezeléssel
        """
        try:
            if self.has_real_wallet and self.wallet:
                try:
                    # Try real wallet
                    data = self.wallet.get_balance()
                    if not data or 'result' not in data:
                        print("[WALLET] Invalid balance API response")
                        return self._get_simulation_crypto_holdings(min_value_usd)
                    
                    # Get ticker data for USD conversion
                    ticker_response = requests.get(
                        "https://api.kraken.com/0/public/Ticker",
                        params={"pair": "BTCUSD,ETHUSD,XRPUSD,ADAUSD,SOLUSD"},
                        timeout=10
                    )
                    
                    if ticker_response.status_code != 200:
                        print("[WALLET] Ticker API failed, using simulation")
                        return self._get_simulation_crypto_holdings(min_value_usd)
                    
                    ticker_data = ticker_response.json()
                    if 'result' not in ticker_data:
                        print("[WALLET] Invalid ticker response")
                        return self._get_simulation_crypto_holdings(min_value_usd)
                        
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
                            print(f"[WALLET] Error processing {asset}: {e}")
                            continue
                    
                    if result:
                        print(f"[WALLET] Real crypto holdings: {result}")
                        return result
                    else:
                        print("[WALLET] No significant real holdings, using simulation")
                        
                except Exception as e:
                    print(f"[WALLET] Real crypto holdings failed: {e}")
            
            # Fallback to simulation
            return self._get_simulation_crypto_holdings(min_value_usd)
            
        except Exception as e:
            print(f"[WALLET] Critical error getting crypto holdings: {e}")
            return {}

    def _get_simulation_crypto_holdings(self, min_value_usd: float) -> Dict[str, float]:
        """Simulation crypto holdings"""
        try:
            result = {}
            for symbol, data in self.simulation_crypto.items():
                value = data['value_usd']
                if value >= min_value_usd:
                    result[f"{symbol}_USD_VALUE"] = value
            
            print(f"[WALLET] Simulation crypto holdings: {result}")
            return result
            
        except Exception as e:
            print(f"[WALLET] Simulation holdings error: {e}")
            return {}

    def update_simulation_balance(self, new_balance: float):
        """Update simulation balance"""
        try:
            old_balance = self.simulation_balance
            self.simulation_balance = max(0, new_balance)
            
            # Clear cache to force refresh
            self.balance_cache = None
            
            print(f"[WALLET] Simulation balance updated: ${old_balance:.2f} -> ${self.simulation_balance:.2f}")
            
        except Exception as e:
            print(f"[WALLET] Balance update error: {e}")

    def add_trade_profit(self, profit_usd: float):
        """Add trading profit to simulation balance"""
        try:
            if self.has_real_wallet:
                print(f"[WALLET] Real trading - profit ${profit_usd:.2f} (not added to simulation)")
            else:
                self.simulation_balance += profit_usd
                self.balance_cache = None  # Clear cache
                print(f"[WALLET] Added profit ${profit_usd:.2f} to simulation balance")
                
        except Exception as e:
            print(f"[WALLET] Add profit error: {e}")

    def get_portfolio_summary(self) -> Dict:
        """Get complete portfolio summary"""
        try:
            usd_balance = self.get_usd_balance() or 0
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
                'is_simulation': not self.has_real_wallet
            }
            
        except Exception as e:
            print(f"[WALLET] Portfolio summary error: {e}")
            return {
                'usd_balance': 500.0,
                'crypto_value': 0.0,
                'total_value': 500.0,
                'crypto_holdings': {},
                'portfolio_allocation': {'usd_percent': 100, 'crypto_percent': 0},
                'is_simulation': True
            }

    def is_sufficient_balance(self, required_usd: float) -> bool:
        """Check if sufficient balance for trade"""
        try:
            current_balance = self.get_usd_balance() or 0
            return current_balance >= required_usd
        except Exception as e:
            print(f"[WALLET] Balance check error: {e}")
            return False

    def get_available_for_trading(self, reserve_percent: float = 10.0) -> float:
        """Get available balance for trading (with reserve)"""
        try:
            total_balance = self.get_usd_balance() or 0
            reserve_amount = total_balance * (reserve_percent / 100)
            available = max(0, total_balance - reserve_amount)
            
            print(f"[WALLET] Available for trading: ${available:.2f} (reserve: ${reserve_amount:.2f})")
            return available
            
        except Exception as e:
            print(f"[WALLET] Available calculation error: {e}")
            return 0.0

    def get_connection_status(self) -> Dict:
        """Get wallet connection status"""
        try:
            status = {
                'has_real_wallet': self.has_real_wallet,
                'cache_active': self.balance_cache is not None,
                'cache_age_seconds': time.time() - self.balance_cache_time if self.balance_cache_time > 0 else 0,
                'simulation_mode': not self.has_real_wallet
            }
            
            if self.has_real_wallet and self.wallet:
                # Test connection
                try:
                    test_response = self.wallet.get_balance()
                    status['api_connection'] = 'connected' if test_response else 'failed'
                except Exception:
                    status['api_connection'] = 'error'
            else:
                status['api_connection'] = 'simulation'
                
            return status
            
        except Exception as e:
            print(f"[WALLET] Status check error: {e}")
            return {'error': str(e)}

    def clear_cache(self):
        """Clear balance cache to force refresh"""
        try:
            self.balance_cache = None
            self.balance_cache_time = 0
            print("[WALLET] Cache cleared")
        except Exception as e:
            print(f"[WALLET] Cache clear error: {e}")

    def reset_simulation(self, initial_balance: float = 750.0):
        """Reset simulation to initial state"""
        try:
            self.simulation_balance = initial_balance
            self.simulation_crypto = {
                'BTC': {'amount': 0.001, 'value_usd': 50.0},
                'ETH': {'amount': 0.05, 'value_usd': 150.0},
                'ADA': {'amount': 100.0, 'value_usd': 75.0}
            }
            self.clear_cache()
            print(f"[WALLET] Simulation reset to ${initial_balance:.2f}")
            
        except Exception as e:
            print(f"[WALLET] Simulation reset error: {e}")
