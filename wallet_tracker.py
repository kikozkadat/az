# core/wallet_tracker.py - Kraken API Wallet Integration for Live Trading

import os
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
from typing import Dict, Optional, Any
import json
from core.kraken_nonce import nonce_manager  # KÖZÖS NONCE HASZNÁLATA

class WalletTracker:
    """
    Live wallet tracker for Kraken API integration.
    Handles real balance queries and trade balance operations.
    """
    
    def __init__(self):
        """Initialize with API credentials from environment"""
        # Get API credentials from environment variables
        self.api_key = os.getenv('KRAKEN_API_KEY')
        self.api_sec = os.getenv('KRAKEN_API_SECRET')
        
        # API endpoints
        self.api_url = "https://api.kraken.com"
        self.api_version = "0"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Validate credentials
        if not self.api_key or not self.api_sec:
            print("[WALLET_TRACKER] ⚠️ WARNING: API credentials not found in environment!")
            print("[WALLET_TRACKER] Set KRAKEN_API_KEY and KRAKEN_API_SECRET environment variables")
        else:
            print(f"[WALLET_TRACKER] ✅ Initialized with API key: {self.api_key[:8]}...")
    
    def _get_kraken_signature(self, urlpath: str, data: Dict[str, Any], nonce: str) -> str:
        """Generate Kraken API signature"""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(nonce) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        
        mac = hmac.new(base64.b64decode(self.api_sec), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()
    
    def _kraken_request(self, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated request to Kraken API"""
        try:
            # Rate limiting
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            
            # Prepare request
            urlpath = f"/{self.api_version}/private/{endpoint}"
            url = f"{self.api_url}{urlpath}"
            
            # Get nonce from shared manager
            nonce = nonce_manager.get_nonce()  # KÖZÖS NONCE
            
            if data is None:
                data = {}
            data['nonce'] = nonce
            
            # Generate signature
            signature = self._get_kraken_signature(urlpath, data, nonce)
            
            # Headers
            headers = {
                'API-Key': self.api_key,
                'API-Sign': signature,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # Make request
            response = requests.post(url, headers=headers, data=data, timeout=15)
            self.last_request_time = time.time()
            
            if response.status_code != 200:
                print(f"[WALLET_TRACKER] ❌ API Error: Status {response.status_code}")
                return None
            
            result = response.json()
            
            # Check for API errors
            if result.get('error'):
                print(f"[WALLET_TRACKER] ❌ API Error: {result['error']}")
                return None
            
            return result
            
        except Exception as e:
            print(f"[WALLET_TRACKER] ❌ Request failed: {e}")
            return None
    
    def get_balance(self) -> Optional[Dict]:
        """
        Get account balance
        Returns: {'result': {'XXBT': '0.1234', 'ZUSD': '1000.00', ...}}
        """
        try:
            print("[WALLET_TRACKER] 📊 Fetching account balance...")
            result = self._kraken_request('Balance')
            
            if result and 'result' in result:
                # Log non-zero balances
                balances = result['result']
                non_zero = {k: v for k, v in balances.items() if float(v) > 0}
                if non_zero:
                    print(f"[WALLET_TRACKER] ✅ Non-zero balances: {non_zero}")
                else:
                    print("[WALLET_TRACKER] 📋 No non-zero balances found")
                    
            return result
            
        except Exception as e:
            print(f"[WALLET_TRACKER] ❌ Balance fetch error: {e}")
            return None
    
    def get_trade_balance(self, asset: str = "ZUSD") -> Optional[Dict]:
        """
        Get trade balance (available for trading)
        
        Args:
            asset: Base asset to show balance in (default: ZUSD for USD)
            
        Returns: {
            'result': {
                'eb': '1000.00',  # Equivalent balance
                'tb': '1000.00',  # Trade balance (available for trading)
                'fb': '900.00',   # Free balance
                'm': '100.00',    # Margin used
                'n': '0.00',      # Unrealized P&L
                'c': '0.00',      # Cost basis
                'v': '0.00',      # Floating valuation
                'e': '1000.00',   # Equity
                'mf': '900.00'    # Free margin
            }
        }
        """
        try:
            print(f"[WALLET_TRACKER] 💰 Fetching trade balance in {asset}...")
            data = {'asset': asset}
            result = self._kraken_request('TradeBalance', data)
            
            if result and 'result' in result:
                balance_info = result['result']
                print(f"[WALLET_TRACKER] ✅ Trade balance: ${float(balance_info.get('eb', 0)):.2f}")
                print(f"[WALLET_TRACKER] 📋 Available for trading: ${float(balance_info.get('fb', 0)):.2f}")
                
            return result
            
        except Exception as e:
            print(f"[WALLET_TRACKER] ❌ Trade balance error: {e}")
            return None
    
    def get_open_orders(self) -> Optional[Dict]:
        """Get open orders"""
        try:
            print("[WALLET_TRACKER] 📋 Fetching open orders...")
            result = self._kraken_request('OpenOrders')
            
            if result and 'result' in result:
                open_orders = result['result'].get('open', {})
                if open_orders:
                    print(f"[WALLET_TRACKER] 📊 Found {len(open_orders)} open orders")
                    for order_id, order_info in list(open_orders.items())[:3]:  # Show max 3
                        desc = order_info.get('descr', {})
                        print(f"  - {desc.get('pair', 'N/A')}: {desc.get('type', 'N/A')} "
                              f"{desc.get('ordertype', 'N/A')} @ {desc.get('price', 'market')}")
                else:
                    print("[WALLET_TRACKER] ✅ No open orders")
                    
            return result
            
        except Exception as e:
            print(f"[WALLET_TRACKER] ❌ Open orders error: {e}")
            return None
    
    def get_trades_history(self, trades_type: str = "all", limit: int = 50) -> Optional[Dict]:
        """
        Get trades history
        
        Args:
            trades_type: Type of trades ('all', 'any position', 'closed position', 'closing position', 'no position')
            limit: Maximum number of trades to return
        """
        try:
            print(f"[WALLET_TRACKER] 📜 Fetching last {limit} trades...")
            data = {
                'type': trades_type,
                'trades': True,
                'ofs': 0,
                'limit': limit
            }
            
            result = self._kraken_request('TradesHistory', data)
            
            if result and 'result' in result:
                trades = result['result'].get('trades', {})
                if trades:
                    print(f"[WALLET_TRACKER] 📊 Found {len(trades)} trades")
                    # Show summary of recent trades
                    for trade_id, trade_info in list(trades.items())[:5]:  # Show max 5
                        pair = trade_info.get('pair', 'N/A')
                        trade_type = trade_info.get('type', 'N/A')
                        vol = float(trade_info.get('vol', 0))
                        price = float(trade_info.get('price', 0))
                        print(f"  - {pair}: {trade_type} {vol:.6f} @ ${price:.2f}")
                else:
                    print("[WALLET_TRACKER] 📋 No trades found")
                    
            return result
            
        except Exception as e:
            print(f"[WALLET_TRACKER] ❌ Trades history error: {e}")
            return None
    
    def get_positions(self) -> Optional[Dict]:
        """Get open positions"""
        try:
            print("[WALLET_TRACKER] 📊 Fetching open positions...")
            result = self._kraken_request('OpenPositions')
            
            if result and 'result' in result:
                positions = result['result']
                if positions:
                    print(f"[WALLET_TRACKER] 📈 Found {len(positions)} open positions")
                    for pos_id, pos_info in positions.items():
                        pair = pos_info.get('pair', 'N/A')
                        pos_type = pos_info.get('type', 'N/A')
                        vol = float(pos_info.get('vol', 0))
                        net = float(pos_info.get('net', 0))
                        print(f"  - {pair}: {pos_type} {vol:.6f}, P&L: ${net:.2f}")
                else:
                    print("[WALLET_TRACKER] ✅ No open positions")
                    
            return result
            
        except Exception as e:
            print(f"[WALLET_TRACKER] ❌ Positions error: {e}")
            return None
    
    def add_order(self, pair: str, type: str, ordertype: str, volume: float, 
                  price: Optional[float] = None, leverage: Optional[str] = None,
                  oflags: Optional[str] = None, validate: bool = True) -> Optional[Dict]:
        """
        Add new order
        
        Args:
            pair: Trading pair (e.g., 'XBTUSD')
            type: Order type ('buy' or 'sell')
            ordertype: Order type ('market', 'limit', 'stop-loss', 'take-profit')
            volume: Order volume
            price: Price (required for limit orders)
            leverage: Leverage amount (optional)
            oflags: Order flags (e.g., 'post' for post-only)
            validate: If True, validate only without placing order
        """
        try:
            print(f"[WALLET_TRACKER] 📝 {'Validating' if validate else 'Placing'} order: "
                  f"{type} {volume} {pair} @ {price if price else 'market'}")
            
            data = {
                'pair': pair,
                'type': type,
                'ordertype': ordertype,
                'volume': str(volume)
            }
            
            if price is not None:
                data['price'] = str(price)
            
            if leverage:
                data['leverage'] = leverage
                
            if oflags:
                data['oflags'] = oflags
                
            if validate:
                data['validate'] = True
            
            result = self._kraken_request('AddOrder', data)
            
            if result and 'result' in result:
                if validate:
                    print("[WALLET_TRACKER] ✅ Order validation successful")
                else:
                    order_info = result['result']
                    txid = order_info.get('txid', [])
                    if txid:
                        print(f"[WALLET_TRACKER] ✅ Order placed successfully: {txid}")
                    
            return result
            
        except Exception as e:
            print(f"[WALLET_TRACKER] ❌ Add order error: {e}")
            return None
    
    def cancel_order(self, txid: str) -> Optional[Dict]:
        """Cancel an open order"""
        try:
            print(f"[WALLET_TRACKER] 🚫 Cancelling order: {txid}")
            data = {'txid': txid}
            result = self._kraken_request('CancelOrder', data)
            
            if result and 'result' in result:
                print(f"[WALLET_TRACKER] ✅ Order cancelled: {result['result']}")
                
            return result
            
        except Exception as e:
            print(f"[WALLET_TRACKER] ❌ Cancel order error: {e}")
            return None
    
    def get_account_info(self) -> Optional[Dict]:
        """Get general account information"""
        try:
            # Get balance
            balance = self.get_balance()
            
            # Get trade balance
            trade_balance = self.get_trade_balance()
            
            # Get open orders count
            open_orders = self.get_open_orders()
            open_count = len(open_orders.get('result', {}).get('open', {})) if open_orders else 0
            
            # Get positions
            positions = self.get_positions()
            position_count = len(positions.get('result', {})) if positions else 0
            
            print("\n[WALLET_TRACKER] 📊 ACCOUNT SUMMARY:")
            print("=" * 50)
            
            if trade_balance and 'result' in trade_balance:
                tb = trade_balance['result']
                print(f"💰 Equity: ${float(tb.get('eb', 0)):.2f}")
                print(f"💵 Free Balance: ${float(tb.get('fb', 0)):.2f}")
                print(f"📊 Margin Used: ${float(tb.get('m', 0)):.2f}")
                
            print(f"📋 Open Orders: {open_count}")
            print(f"📈 Open Positions: {position_count}")
            print("=" * 50)
            
            return {
                'balance': balance,
                'trade_balance': trade_balance,
                'open_orders': open_count,
                'positions': position_count
            }
            
        except Exception as e:
            print(f"[WALLET_TRACKER] ❌ Account info error: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test API connection and credentials"""
        try:
            print("[WALLET_TRACKER] 🔌 Testing API connection...")
            
            if not self.api_key or not self.api_sec:
                print("[WALLET_TRACKER] ❌ No API credentials configured")
                return False
            
            # Try to get balance as connection test
            result = self.get_balance()
            
            if result and 'result' in result:
                print("[WALLET_TRACKER] ✅ API connection successful")
                return True
            else:
                print("[WALLET_TRACKER] ❌ API connection failed")
                return False
                
        except Exception as e:
            print(f"[WALLET_TRACKER] ❌ Connection test error: {e}")
            return False


# Quick test if run directly
if __name__ == "__main__":
    print("\n🚀 Testing Kraken Wallet Tracker...")
    print("=" * 60)
    
    # Create tracker instance
    tracker = WalletTracker()
    
    # Test connection
    if tracker.test_connection():
        print("\n✅ Connection test passed!")
        
        # Get account info
        tracker.get_account_info()
    else:
        print("\n❌ Connection test failed!")
        print("Please ensure KRAKEN_API_KEY and KRAKEN_API_SECRET are set correctly")
