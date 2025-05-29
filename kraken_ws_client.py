# data/kraken_ws_client.py - JAVÍTOTT VERZIÓ

import websocket
import json
import threading
import time
import logging

logger = logging.getLogger(__name__)

class KrakenWSClient:
    def __init__(self, pairs, data_handler):
        self.pairs = pairs
        self.data_handler = data_handler
        self.ws = None
        self.running = False
        self.thread = None
        self.reconnect_interval = 5
        self.max_reconnects = 10
        self.reconnect_count = 0
        
    def start(self):
        """Start the WebSocket connection"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_forever)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"WebSocket client started for {len(self.pairs)} pairs")
        
    def stop(self):
        """Stop the WebSocket connection"""
        logger.info("Stopping WebSocket client...")
        self.running = False
        
        # Unsubscribe from all pairs
        if self.ws and hasattr(self.ws, 'sock') and self.ws.sock:
            try:
                unsubscribe_msg = {
                    "event": "unsubscribe",
                    "pair": self.pairs,
                    "subscription": {
                        "name": "ticker"
                    }
                }
                self.ws.send(json.dumps(unsubscribe_msg))
                logger.info("Unsubscribed from all pairs")
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error during unsubscribe: {e}")
        
        if self.ws:
            self.ws.close()
            
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            
        logger.info("WebSocket client stopped")
        
    def _run_forever(self):
        """Main WebSocket loop with reconnection logic"""
        while self.running and self.reconnect_count < self.max_reconnects:
            try:
                self._connect()
                self.reconnect_count = 0  # Reset on successful connection
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                self.reconnect_count += 1
                if self.running:
                    logger.info(f"Reconnecting in {self.reconnect_interval}s (attempt {self.reconnect_count})")
                    time.sleep(self.reconnect_interval)
                    
    def _connect(self):
        """Establish WebSocket connection"""
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            "wss://ws.kraken.com/",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.ws.run_forever()
        
    def _on_open(self, ws):
        """Handle WebSocket connection opened"""
        logger.info("WebSocket connection opened")
        self._subscribe_to_pairs()
        
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages - JAVÍTOTT"""
        try:
            data = json.loads(message)
            
            # Handle subscription status
            if isinstance(data, dict) and 'event' in data:
                if data['event'] == 'subscriptionStatus':
                    status = data.get('status', 'unknown')
                    pair = data.get('pair', 'unknown')
                    if status == 'subscribed':
                        logger.info(f"Subscription {status} for {pair}")
                    elif status == 'unsubscribed':
                        logger.info(f"Subscription {status} for {pair}")
                    return
                elif data['event'] == 'systemStatus':
                    logger.info(f"Kraken system status: {data.get('status', 'unknown')}")
                    return
                    
            # Handle ticker data - JAVÍTOTT LOGIKA
            if isinstance(data, list) and len(data) >= 4:
                try:
                    channel_id = data[0]
                    ticker_data = data[1]
                    channel_name = data[2]
                    pair = data[3]
                    
                    if channel_name == "ticker":
                        self._handle_ticker_data_fixed(pair, ticker_data)
                except Exception as e:
                    logger.error(f"Error parsing ticker message: {e}")
                    
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
        except Exception as e:
            logger.error(f"Message handling failed: {e}")
            
    def _handle_ticker_data_fixed(self, pair, ticker_data):
        """JAVÍTOTT ticker adat feldolgozás"""
        try:
            # Kraken WebSocket ticker formátum:
            # {"a":["price","whole_lot_volume","lot_volume"],
            #  "b":["price","whole_lot_volume","lot_volume"], 
            #  "c":["price","lot_volume"],
            #  "v":["today","last_24_hours"],
            #  "p":["today","last_24_hours"],
            #  "t":[today,last_24_hours],
            #  "l":["today","last_24_hours"],
            #  "h":["today","last_24_hours"],
            #  "o":"today_opening_price"}
            
            if not isinstance(ticker_data, dict):
                logger.error(f"Unexpected ticker data format for {pair}: {type(ticker_data)}")
                return
                
            # BIZTONSÁGOS ADAT KINYERÉS
            def safe_extract_price(data_field, index=0, default=0.0):
                """Biztonságosan kinyeri az ár adatot"""
                try:
                    if isinstance(data_field, list) and len(data_field) > index:
                        return float(data_field[index])
                    elif isinstance(data_field, (str, int, float)):
                        return float(data_field)
                    return default
                except (ValueError, TypeError):
                    return default
            
            def safe_extract_volume(data_field, index=0, default=0.0):
                """Biztonságosan kinyeri a volume adatot"""
                try:
                    if isinstance(data_field, list) and len(data_field) > index:
                        value = data_field[index]
                        if isinstance(value, (str, int, float)):
                            return float(value)
                    return default
                except (ValueError, TypeError):
                    return default
            
            # Processed ticker data létrehozása
            processed_data = {
                'pair': pair,
                'timestamp': time.time()
            }
            
            # Ask price (a field)
            if 'a' in ticker_data:
                processed_data['ask'] = safe_extract_price(ticker_data['a'], 0)
            else:
                processed_data['ask'] = 0.0
                
            # Bid price (b field)  
            if 'b' in ticker_data:
                processed_data['bid'] = safe_extract_price(ticker_data['b'], 0)
            else:
                processed_data['bid'] = 0.0
                
            # Last trade price (c field)
            if 'c' in ticker_data:
                processed_data['last'] = safe_extract_price(ticker_data['c'], 0)
            else:
                processed_data['last'] = processed_data['ask']  # Fallback to ask
                
            # Volume (v field) - 24h volume
            if 'v' in ticker_data:
                processed_data['volume'] = safe_extract_volume(ticker_data['v'], 1)  # 24h volume
                processed_data['volume_today'] = safe_extract_volume(ticker_data['v'], 0)  # Today volume
            else:
                processed_data['volume'] = 0.0
                processed_data['volume_today'] = 0.0
                
            # High (h field)
            if 'h' in ticker_data:
                processed_data['high'] = safe_extract_price(ticker_data['h'], 1)  # 24h high
                processed_data['high_today'] = safe_extract_price(ticker_data['h'], 0)  # Today high
            else:
                processed_data['high'] = processed_data['last']
                processed_data['high_today'] = processed_data['last']
                
            # Low (l field)
            if 'l' in ticker_data:
                processed_data['low'] = safe_extract_price(ticker_data['l'], 1)  # 24h low  
                processed_data['low_today'] = safe_extract_price(ticker_data['l'], 0)  # Today low
            else:
                processed_data['low'] = processed_data['last']
                processed_data['low_today'] = processed_data['last']
                
            # Opening price (o field)
            if 'o' in ticker_data:
                processed_data['open'] = safe_extract_price(ticker_data['o'])
            else:
                processed_data['open'] = processed_data['last']
                
            # Volume weighted average price (p field)
            if 'p' in ticker_data:
                processed_data['vwap'] = safe_extract_price(ticker_data['p'], 1)  # 24h VWAP
                processed_data['vwap_today'] = safe_extract_price(ticker_data['p'], 0)  # Today VWAP
            else:
                processed_data['vwap'] = processed_data['last']
                processed_data['vwap_today'] = processed_data['last']
                
            # Trade count (t field)
            if 't' in ticker_data:
                processed_data['count'] = safe_extract_volume(ticker_data['t'], 1)  # 24h count
                processed_data['count_today'] = safe_extract_volume(ticker_data['t'], 0)  # Today count
            else:
                processed_data['count'] = 0
                processed_data['count_today'] = 0
                
            # Calculate volume in USD for filtering
            if processed_data['last'] > 0 and processed_data['volume'] > 0:
                processed_data['volume_usd'] = processed_data['last'] * processed_data['volume']
            else:
                processed_data['volume_usd'] = 0.0
                
            # Validate essential data
            if processed_data['last'] <= 0:
                logger.warning(f"Invalid price data for {pair}: {processed_data['last']}")
                return
                
            # Forward to data handler
            if self.data_handler:
                self.data_handler(processed_data)
                
            # Log successful processing (occasionally)
            if int(time.time()) % 30 == 0:  # Every 30 seconds
                logger.info(f"✅ {pair}: ${processed_data['last']:.4f}, Vol: ${processed_data['volume_usd']:,.0f}")
                
        except Exception as e:
            logger.error(f"Ticker data processing failed for {pair}: {e}")
            
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed"""
        if close_status_code is not None or close_msg is not None:
            logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        else:
            logger.info("WebSocket connection closed")
        
    def _subscribe_to_pairs(self):
        """Subscribe to ticker data for specified pairs in batches"""
        if not self.pairs:
            logger.warning("No pairs to subscribe to")
            return
            
        # Subscribe in smaller batches to avoid overwhelming
        batch_size = 10
        
        for i in range(0, len(self.pairs), batch_size):
            batch = self.pairs[i:i + batch_size]
            
            subscription_msg = {
                "event": "subscribe",
                "pair": batch,
                "subscription": {
                    "name": "ticker"
                }
            }
            
            try:
                self.ws.send(json.dumps(subscription_msg))
                logger.info(f"Subscribed to batch: {batch}")
                time.sleep(0.5)  # Small delay between batches
            except Exception as e:
                logger.error(f"Subscription failed for batch {batch}: {e}")
                
    def get_connection_status(self):
        """Get current connection status"""
        return {
            'running': self.running,
            'connected': self.ws is not None and hasattr(self.ws, 'sock') and self.ws.sock and self.ws.sock.connected,
            'reconnect_count': self.reconnect_count,
            'subscribed_pairs': len(self.pairs) if self.pairs else 0
        }
