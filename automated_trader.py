# strategy/automated_trader.py - JAVÍTOTT AUTOMATIKUS KERESKEDŐ

import time
from data.kraken_api_client import KrakenAPIClient
from strategy.indicator_engine import IndicatorEngine
from core.scoring_scheduler import ScoringScheduler
from core.trade_manager import TradeManager
from core.position_manager import PositionManager
from utils.trade_logger import TradeLogger
from utils.history_analyzer import HistoryAnalyzer
from strategy.market_scanner import MarketScanner
from config import TRADE_ENABLED, MAX_PARALLEL_POSITIONS, MICRO_POSITION_SIZE, MICRO_PROFIT_TARGET
import pandas as pd
from utils.logger import logger
from typing import List, Optional # Added Optional for type hinting

class AutomatedTrader:
    def __init__(self):
        # Core components
        self.api = KrakenAPIClient()
        self.indicators = IndicatorEngine()
        self.scanner = MarketScanner(self.api)  # Connect API to scanner
        self.trader = TradeManager()
        self.position_manager = PositionManager()
        self.logger = TradeLogger()
        self.analyzer = HistoryAnalyzer()
        
        # Trading settings - MIKRO KERESKEDÉS
        self.trading_enabled = TRADE_ENABLED
        self.max_positions = MAX_PARALLEL_POSITIONS
        self.position_size_usd = MICRO_POSITION_SIZE  # $50
        self.profit_target_usd = MICRO_PROFIT_TARGET  # $0.15
        
        # Scanning settings
        self.last_scan_time = 0
        self.scan_interval = 120  # 2 perc
        self.min_volume_usd = 500000  # $500K minimum forgalom
        
        # Performance tracking
        self.trades_today = 0
        self.profit_today = 0.0
        self.last_trade_time = 0
        self.simulation_mode = True # Initialize simulation_mode attribute
        
        logger.info(f"AutomatedTrader initialized - ${self.position_size_usd} positions, ${self.profit_target_usd} target")
        
    def set_trading_enabled(self, enabled: bool):
        """Enable or disable automated trading"""
        self.trading_enabled = enabled
        logger.info(f"Automated trading {'enabled' if enabled else 'disabled'}")
        
    def set_max_positions(self, max_pos: int):
        """Set maximum number of parallel positions"""
        self.max_positions = max_pos
        logger.info(f"Max positions set to: {max_pos}")
        
    def run_trading_cycle(self):
        """
        Főbb kereskedési ciklus - VOLUME ALAPÚ SZŰRÉSSEL
        """
        if not self.trading_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_scan_time < self.scan_interval:
            return
            
        self.last_scan_time = current_time
        
        try:
            logger.info("🚀 Starting automated trading cycle...")
            
            # 1. CHECK EXISTING POSITIONS
            self._monitor_existing_positions()
            
            # 2. CHECK IF WE CAN OPEN NEW POSITIONS
            current_positions = len(self.position_manager.positions)
            if current_positions >= self.max_positions:
                logger.info(f"Max positions ({self.max_positions}) reached")
                return
                
            # 3. SCAN FOR HIGH VOLUME OPPORTUNITIES
            opportunities = self._scan_for_volume_opportunities()
            
            if not opportunities:
                logger.info("No volume-based opportunities found")
                return
                
            # 4. SELECT AND EXECUTE BEST OPPORTUNITY
            best_opportunity = self._select_best_opportunity(opportunities)
            
            if best_opportunity:
                self._execute_trade_opportunity(best_opportunity)
                
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            
    def _scan_for_volume_opportunities(self) -> List: # Assuming Opportunity is a defined class/type
        """
        Volume alapú lehetőségek szkenelése
        """
        try:
            logger.info("🔍 Scanning for high-volume opportunities...")
            
            # Get high volume opportunities from scanner
            # Assuming get_top_opportunities returns a list of objects with attributes like volume_usd, rsi, btc_correlation, final_score, pair
            opportunities = self.scanner.get_top_opportunities(min_score=0.5) 
            
            if not opportunities:
                logger.warning("Scanner returned no opportunities")
                return []
                
            # Filter by volume requirement
            volume_filtered = [
                opp for opp in opportunities 
                if hasattr(opp, 'volume_usd') and opp.volume_usd >= self.min_volume_usd # Check attribute existence
            ]
            
            logger.info(f"📊 Found {len(volume_filtered)} opportunities with volume ≥ ${self.min_volume_usd:,}")
            
            # Additional technical filtering
            technical_filtered = []
            
            for opp in volume_filtered:
                # Technical criteria for micro trading
                # Ensure all attributes exist before accessing
                if (hasattr(opp, 'rsi') and hasattr(opp, 'btc_correlation') and hasattr(opp, 'volume_usd') and
                    25 <= opp.rsi <= 40 and  # Oversold but not extreme
                    opp.btc_correlation >= 0.8 and  # Strong BTC correlation
                    opp.volume_usd >= 1000000):  # At least $1M volume for best opportunities
                    
                    technical_filtered.append(opp)
                    
            logger.info(f"🎯 {len(technical_filtered)} opportunities passed technical filter")
            
            return technical_filtered[:5]  # Top 5
            
        except Exception as e:
            logger.error(f"Volume opportunity scan failed: {e}")
            return []

    def _select_best_opportunity(self, opportunities: List) -> Optional: # Assuming Opportunity is a defined class/type
        """
        Legjobb lehetőség kiválasztása
        """
        try:
            if not opportunities:
                return None
                
            # Sort by combined score (volume + technical)
            scored_opportunities = []
            
            for opp in opportunities:
                # Combined score calculation
                # Ensure all attributes exist before accessing
                if not (hasattr(opp, 'volume_usd') and hasattr(opp, 'final_score') and hasattr(opp, 'pair')):
                    logger.warning(f"Opportunity object missing required attributes: {opp}")
                    continue

                volume_score = min(1.0, opp.volume_usd / 10000000)  # Normalize to 10M
                technical_score = opp.final_score
                
                # Weight: 60% technical, 40% volume
                combined_score = (technical_score * 0.6) + (volume_score * 0.4)
                
                scored_opportunities.append({
                    'opportunity': opp,
                    'combined_score': combined_score
                })
                
            if not scored_opportunities: # If all opportunities were filtered out
                logger.warning("No opportunities left after attribute check in _select_best_opportunity")
                return None

            # Sort by combined score
            scored_opportunities.sort(key=lambda x: x['combined_score'], reverse=True)
            
            best = scored_opportunities[0]
            
            logger.info(f"🏆 Best opportunity: {best['opportunity'].pair} "
                       f"(score: {best['combined_score']:.3f}, "
                       f"volume: ${best['opportunity'].volume_usd:,.0f})")
            
            return best['opportunity']
            
        except Exception as e:
            logger.error(f"Opportunity selection failed: {e}")
            return None

    def _execute_trade_opportunity(self, opportunity): # Assuming Opportunity is a defined class/type
        """
        Kereskedési lehetőség végrehajtása
        """
        try:
            if not hasattr(opportunity, 'pair') or not hasattr(opportunity, 'volume_usd'): # Check attributes
                logger.error(f"Opportunity object missing required attributes for execution: {opportunity}")
                return

            pair = opportunity.pair
            
            logger.info(f"💰 Executing trade for {pair}...")
            
            # Check if position already exists
            if self.position_manager.get_position(pair):
                logger.warning(f"Position already exists for {pair}")
                return
                
            # Get current price
            current_price = self._get_current_price(pair)
            if not current_price: # current_price can be 0.0, which is falsy. Explicitly check for None.
                logger.error(f"Could not get current price for {pair}")
                return
                
            # Calculate position parameters
            if current_price == 0: # Avoid division by zero
                logger.error(f"Current price for {pair} is zero, cannot calculate volume.")
                return
            volume = self.position_size_usd / current_price
            
            # Calculate SL/TP for micro trading
            stop_loss_pct = 0.04  # 4% stop loss
            target_profit_pct = self.profit_target_usd / self.position_size_usd if self.position_size_usd > 0 else 0.003 # Avoid division by zero, default to 0.3%
            
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + target_profit_pct)
            
            logger.info(f"📊 Trade parameters:")
            logger.info(f"   Price: ${current_price:.6f}")
            logger.info(f"   Volume: {volume:.6f}")
            logger.info(f"   Position size: ${self.position_size_usd}")
            logger.info(f"   Stop loss: ${stop_loss:.6f} (-{stop_loss_pct*100:.1f}%)")
            logger.info(f"   Take profit: ${take_profit:.6f} (+{target_profit_pct*100:.2f}%)")
            
            if self.simulation_mode:
                logger.info("🎮 SIMULATION MODE - Trade not executed")
                # Simulate success for position manager in simulation mode
                success_pm = True 
            else:
                # ÉLES KERESKEDÉS - Valós order leadás
                result = self.trader.place_order(pair, "buy", volume)
                if result and 'error' not in result:
                    logger.info(f"✅ Real order placed: {pair} {volume:.6f} @ ${current_price:.6f}")
                    success_pm = True # Assume success if order placement is confirmed
                else:
                    logger.error(f"❌ Real order placement failed for {pair}: {result.get('error', 'Unknown error') if isinstance(result, dict) else result}")
                    success_pm = False

            if success_pm: # Proceed if simulated or real order was (assumed) successful
                # Open position in position manager
                success_open = self.position_manager.open_position(
                    pair=pair,
                    side="buy",
                    entry_price=current_price,
                    volume=volume,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
                if success_open:
                    # Log the trade
                    self.logger.log(
                        pair=pair,
                        side="buy",
                        entry_price=current_price,
                        exit_price=None,
                        volume=volume,
                        pnl=0.0,
                        reason=f"AUTO_OPEN_VOLUME_{opportunity.volume_usd:.0f}"
                    )
                    
                    self.trades_today += 1
                    self.last_trade_time = time.time()
                    
                    logger.info(f"✅ Position opened successfully in manager for {pair}")
                else:
                    logger.error(f"Failed to open position in manager for {pair}")
            # Removed the original success variable as it was tied only to position manager
            # and did not reflect the actual trade execution result.
            # The logic now depends on success_pm which is True in simulation or if real order is placed.
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

    def _monitor_existing_positions(self):
        """
        Meglévő pozíciók monitorozása
        """
        try:
            positions = self.position_manager.get_all_positions() # Returns a dict
            
            if not positions:
                return
                
            logger.info(f"📋 Monitoring {len(positions)} active positions...")
            
            # Iterate over a copy of items if modifying the dict during iteration,
            # though here we are calling _close_position which modifies position_manager.positions
            for pair, position_details in list(positions.items()): 
                try:
                    current_price = self._get_current_price(pair)
                    if current_price is None: # Explicitly check for None
                        logger.warning(f"Could not get current price for {pair} during monitoring. Skipping.")
                        continue
                        
                    # Check SL/TP
                    # Ensure position_details is the actual dictionary, not just the key 'pair'
                    should_close, reason = self._should_close_position(position_details, current_price)
                    
                    if should_close:
                        self._close_position(pair, position_details, current_price, reason) # Pass pair and position_details
                        
                except Exception as e:
                    logger.error(f"Position monitoring failed for {pair}: {e}")
                    
        except Exception as e:
            logger.error(f"Position monitoring failed: {e}")

    def _should_close_position(self, position: dict, current_price: float) -> tuple[bool, str]:
        """
        Pozíció zárás ellenőrzése
        """
        try:
            entry_price = position.get("entry_price")
            stop_loss = position.get("stop_loss")
            take_profit = position.get("take_profit")
            side = position.get("side")
            open_time = position.get("open_time", time.time()) # Default to now if not set
            volume = position.get("volume", 0) # Get volume for profit calculation

            if None in [entry_price, side, volume]: # Essential attributes
                logger.error(f"Position data incomplete for close check: {position}")
                return False, "INCOMPLETE_DATA"

            # Time-based exit (max 2 hours for micro trading)
            max_hold_time = 7200  # 2 hours
            if time.time() - open_time > max_hold_time:
                return True, "TIME_LIMIT"
            
            if side.lower() == "buy":
                # Stop loss check
                if stop_loss and current_price <= stop_loss:
                    return True, "STOP_LOSS"
                    
                # Take profit check
                if take_profit and current_price >= take_profit:
                    return True, "TAKE_PROFIT"
                    
                # Quick profit check (if we have $0.10+ profit after 15 minutes)
                if time.time() - open_time > 900:  # 15 minutes
                    if volume > 0: # Ensure volume is positive for calculation
                        profit_usd = (current_price - entry_price) * volume
                        if profit_usd >= 0.10:  # $0.10 quick profit
                            return True, "QUICK_PROFIT"
            # Add logic for "sell" side if applicable in your strategy
            # elif side.lower() == "sell":
            #     if stop_loss and current_price >= stop_loss:
            #         return True, "STOP_LOSS"
            #     if take_profit and current_price <= take_profit:
            #         return True, "TAKE_PROFIT"
            #     if time.time() - open_time > 900:
            #         if volume > 0:
            #             profit_usd = (entry_price - current_price) * volume
            #             if profit_usd >= 0.10:
            #                 return True, "QUICK_PROFIT"

            return False, ""
            
        except Exception as e:
            logger.error(f"Position close check failed: {e}")
            return False, "ERROR"

    def _close_position(self, pair: str, position: dict, exit_price: float, reason: str):
        """
        Pozíció zárása
        """
        try:
            side = position.get("side")
            volume = position.get("volume")
            entry_price = position.get("entry_price")

            if None in [side, volume, entry_price]:
                logger.error(f"Cannot close position {pair}, essential data missing: {position}")
                return
            
            logger.info(f"🚪 Closing position: {pair} @ ${exit_price:.6f} ({reason})")
            
            if self.simulation_mode:
                logger.info("🎮 SIMULATION MODE - Position close not executed")
                closed_successfully_broker = True # Simulate success
            else:
                # ÉLES ZÁRÁS - Valós order leadás
                # Ensure self.trader.close_market_position exists and handles side correctly
                # The original code used 'side' from position, assuming it's 'buy' or 'sell'
                # If closing a 'buy' position, you 'sell'. If closing 'sell', you 'buy'.
                # Adjust 'side' for closing order if necessary, or ensure trader.close_market_position handles it.
                # For simplicity, assuming trader.close_market_position knows what to do with the original side.
                result = self.trader.close_market_position(pair, side, volume) # Pass original side
                if result and 'error' not in result:
                    logger.info(f"✅ Real close order placed for {pair} {volume:.6f} @ ${exit_price:.6f}")
                    closed_successfully_broker = True
                else:
                    logger.error(f"❌ Real close order placement failed for {pair}: {result.get('error', 'Unknown error') if isinstance(result, dict) else result}")
                    closed_successfully_broker = False
            
            if closed_successfully_broker: # If simulated or real close order was successful
                # Calculate P&L
                if side.lower() == "buy":
                    pnl = (exit_price - entry_price) * volume
                elif side.lower() == "sell": # Added P&L for sell side
                    pnl = (entry_price - exit_price) * volume
                else:
                    logger.error(f"Unknown side '{side}' for P&L calculation on {pair}")
                    pnl = 0.0
                    
                # Close position in manager
                # PositionManager.close_position might need the exit price
                self.position_manager.close_position(pair, exit_price) 
                
                # Log the trade
                self.logger.log(
                    pair=pair,
                    side=side, # Log the original side of the position
                    entry_price=entry_price,
                    exit_price=exit_price,
                    volume=volume,
                    pnl=pnl,
                    reason=reason
                )
                
                self.profit_today += pnl
                
                logger.info(f"💰 Position closed in manager - P&L: ${pnl:.2f}")
            else:
                logger.error(f"Broker did not confirm close for {pair}. Position might still be open with broker.")

        except Exception as e:
            logger.error(f"Position closing failed for {pair}: {e}")

    def _get_current_price(self, pair: str) -> Optional[float]:
        """
        Aktuális ár lekérése
        """
        try:
            # Try to get price from API
            # Ensure get_ohlc is robust and handles API errors
            ohlc_data = self.api.get_ohlc(pair, interval=1) # interval=1 minute
            
            if ohlc_data:
                # Get the first (and usually only) pair data from the dict
                # The structure of ohlc_data might be {'PAIR_NAME': [[timestamp, open, high, low, close, volume], ...]}
                # or directly the list of candles if the API client processes it.
                # Assuming it's a dict where keys are pair names.
                if pair in ohlc_data and ohlc_data[pair]:
                    last_candle = ohlc_data[pair][-1]
                    # Candle format: [time, open, high, low, close, vwap, volume, count] for Kraken
                    # Or more commonly: [timestamp, open, high, low, close, volume]
                    # Assuming close is at index 4
                    if len(last_candle) > 4:
                        return float(last_candle[4])  # Close price
                    else:
                        logger.warning(f"OHLC candle data for {pair} has unexpected format: {last_candle}")
                        return None
                elif list(ohlc_data.values()) and list(ohlc_data.values())[0]: # Fallback if pair key is not direct
                    pair_data_list = list(ohlc_data.values())[0]
                    if pair_data_list:
                        last_candle = pair_data_list[-1]
                        if len(last_candle) > 4:
                             return float(last_candle[4])
                        else:
                            logger.warning(f"OHLC candle data for {pair} (fallback) has unexpected format: {last_candle}")
                            return None

            logger.warning(f"Could not get OHLC price data for {pair} from API.")
            return None # Return None if price cannot be determined
            
        except Exception as e:
            logger.error(f"Price fetch failed for {pair}: {e}")
            return None
            
    def get_status(self) -> dict:
        """
        Trader státusz lekérése
        """
        return {
            "trading_enabled": self.trading_enabled,
            "simulation_mode": self.simulation_mode, # Added simulation mode status
            "max_positions": self.max_positions,
            "position_size_usd": self.position_size_usd,
            "profit_target_usd": self.profit_target_usd,
            "current_positions": len(self.position_manager.positions),
            "open_positions_details": self.position_manager.get_all_positions(), # More details
            "trades_today": self.trades_today,
            "profit_today": self.profit_today,
            "last_scan": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_scan_time)) if self.last_scan_time else "N/A",
            "last_trade_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_trade_time)) if self.last_trade_time else "N/A",
            "min_volume_requirement": f"${self.min_volume_usd:,}",
            "scan_interval_seconds": self.scan_interval
        }
    
    def update_settings(self, settings: dict):
        """
        Trading beállítások frissítése
        """
        try:
            if 'position_size_usd' in settings:
                new_size = float(settings['position_size_usd'])
                if new_size > 0: self.position_size_usd = new_size
                else: logger.warning("Invalid position_size_usd provided.")
                
            if 'profit_target_usd' in settings:
                new_target = float(settings['profit_target_usd'])
                if new_target > 0: self.profit_target_usd = new_target
                else: logger.warning("Invalid profit_target_usd provided.")

            if 'max_positions' in settings:
                new_max_pos = int(settings['max_positions'])
                if new_max_pos >= 0: self.max_positions = new_max_pos
                else: logger.warning("Invalid max_positions provided.")
                
            if 'min_volume_usd' in settings:
                new_min_vol = float(settings['min_volume_usd'])
                if new_min_vol >=0: self.min_volume_usd = new_min_vol
                else: logger.warning("Invalid min_volume_usd provided.")
                
            if 'scan_interval' in settings:
                new_interval = int(settings['scan_interval'])
                if new_interval > 0: self.scan_interval = new_interval
                else: logger.warning("Invalid scan_interval provided.")

            if 'trading_enabled' in settings: # Allow enabling/disabling via settings
                self.set_trading_enabled(bool(settings['trading_enabled']))

            if 'simulation_mode' in settings: # Allow changing simulation mode
                self.set_simulation_mode(bool(settings['simulation_mode']))
                
            logger.info(f"Trading settings updated: {settings}")
            logger.info(f"Current effective settings: {self.get_status()}") # Log current state
            
        except ValueError as ve:
            logger.error(f"Settings update failed due to invalid value type: {ve}")
        except Exception as e:
            logger.error(f"Settings update failed: {e}")
    
    def force_scan(self):
        """
        Kényszerített scan futtatása
        """
        try:
            logger.info("🔄 Force scanning market...")
            self.last_scan_time = 0  # Reset scan time to allow immediate scan
            self.run_trading_cycle()
            
        except Exception as e:
            logger.error(f"Force scan failed: {e}")
    
    def emergency_stop(self):
        """
        Vészleállítás - összes pozíció zárása és kereskedés letiltása
        """
        try:
            logger.warning("🚨 EMERGENCY STOP ACTIVATED - Disabling trading and closing all positions")
            
            self.set_trading_enabled(False) # Disable further automated trading
            
            # Make a copy of position keys to iterate over, as closing modifies the underlying dict
            positions_to_close = list(self.position_manager.get_all_positions().keys())
            closed_count = 0

            if not positions_to_close:
                logger.info("🚨 No open positions to close during emergency stop.")
                return

            logger.info(f"🚨 Attempting to close {len(positions_to_close)} positions...")
            
            for pair in positions_to_close:
                try:
                    # Position manager should still have the details even if we iterate by keys
                    position_details = self.position_manager.get_position(pair) 
                    if position_details: # Ensure position still exists before trying to close
                        current_price = self._get_current_price(pair)
                        if current_price is not None:
                            logger.info(f"🚨 Emergency closing {pair} at market price ${current_price:.6f}")
                            self._close_position(pair, position_details, current_price, "EMERGENCY_STOP")
                            closed_count +=1
                        else:
                            logger.error(f"🚨 Could not get price for {pair} during emergency stop. Manual check required.")
                    else:
                        logger.warning(f"🚨 Position {pair} disappeared before emergency close. Already closed?")
                            
                except Exception as e:
                    logger.error(f"🚨 Emergency close failed for {pair}: {e}")
                    
            logger.warning(f"🚨 Emergency stop completed. Attempted to close {len(positions_to_close)} positions, {closed_count} were processed.")
            
        except Exception as e:
            logger.error(f"🚨 Emergency stop procedure failed: {e}")
    
    def get_volume_statistics(self) -> dict:
        """
        Forgalom statisztikák lekérése (feltételezve, hogy az API kliens rendelkezik ilyen funkcióval)
        """
        try:
            if hasattr(self.api, 'get_volume_statistics') and callable(self.api.get_volume_statistics):
                return self.api.get_volume_statistics()
            else:
                logger.warning("API client does not have 'get_volume_statistics' method.")
                return {"error": "Volume statistics not available via API client"}
                
        except Exception as e:
            logger.error(f"Volume statistics fetch failed: {e}")
            return {"error": str(e)}
    
    def get_performance_summary(self) -> dict:
        """
        Teljesítmény összefoglaló
        """
        try:
            # Get position manager statistics
            stats = self.position_manager.get_statistics() # Assuming this method exists and returns a dict
            
            # Add trader-specific metrics
            stats.update({
                "trades_today": self.trades_today,
                "profit_today_usd": self.profit_today, # Clarify currency
                "current_position_size_usd": self.position_size_usd,
                "current_profit_target_usd": self.profit_target_usd,
                "avg_profit_per_trade_usd": (self.profit_today / self.trades_today) if self.trades_today > 0 else 0.0,
                "last_trade_time_readable": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_trade_time)) if self.last_trade_time else "N/A",
                "trading_enabled": self.trading_enabled,
                "simulation_mode": self.simulation_mode
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {"error": f"Failed to generate performance summary: {str(e)}"}
    
    def analyze_market_conditions(self) -> dict:
        """
        Piaci körülmények elemzése
        """
        try:
            # Get scanner status
            scanner_status = {}
            if hasattr(self.scanner, 'get_scanner_status') and callable(self.scanner.get_scanner_status):
                scanner_status = self.scanner.get_scanner_status()
            else:
                logger.warning("MarketScanner does not have 'get_scanner_status' method.")

            # Get volume statistics
            volume_stats = self.get_volume_statistics()
            
            # Combine analysis
            api_connected_status = self.api.test_connection() if hasattr(self.api, 'test_connection') else False

            market_analysis = {
                "scanner_mode": scanner_status.get('mode', 'UNKNOWN'),
                "pairs_available_scanner": scanner_status.get('cached_results', 0), # From scanner
                "min_volume_filter_scanner": scanner_status.get('min_volume_usd', 'N/A'), # From scanner
                "api_connection_ok": api_connected_status,
                "volume_statistics_general": volume_stats, # General market volume if available
                "overall_trading_conditions": "GOOD" if (
                    api_connected_status and 
                    scanner_status.get('cached_results', 0) > 0 # Example condition
                ) else "LIMITED"
            }
            
            logger.info(f"📊 Market conditions: {market_analysis['overall_trading_conditions']}")
            
            return market_analysis
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return {"overall_trading_conditions": "ERROR", "error_message": str(e)}
    
    def set_simulation_mode(self, simulation_enabled: bool):
        """
        Szimuláció mód be/kikapcsolása.
        Fontos: Éles kereskedésnél ez False kell legyen!
        """
        self.simulation_mode = simulation_enabled
        mode_text = "SIMULATION" if simulation_enabled else "LIVE TRADING"
        logger.warning(f"🎮 Trading mode explicitly set to: {mode_text}")
        
        if not simulation_enabled:
            logger.critical("⚠️ LIVE TRADING MODE HAS BEEN ENABLED! REAL MONEY IS AT RISK! ⚠️")
            # Potentially add more safeguards or confirmations here if needed
    
    def validate_trading_conditions(self) -> dict:
        """
        Kereskedési feltételek validálása a bot indítása előtt vagy közben.
        """
        try:
            issues = []
            
            # API connection check
            if hasattr(self.api, 'test_connection') and not self.api.test_connection():
                issues.append("API connection failed or not responsive.")
            
            # Position size validation (example ranges)
            if not (10 <= self.position_size_usd <= 1000): # Adjusted reasonable range
                issues.append(f"Position size ${self.position_size_usd:.2f} is outside recommended range ($10-$1000).")
            
            # Profit target validation (example ranges)
            if not (0.05 <= self.profit_target_usd <= self.position_size_usd * 0.1): # e.g. max 10% of position size
                issues.append(f"Profit target ${self.profit_target_usd:.2f} seems unreasonable for position size ${self.position_size_usd:.2f}.")
            
            # Volume requirement check
            if self.min_volume_usd < 100000: # Minimum $100k
                issues.append(f"Minimum volume requirement ${self.min_volume_usd:,.0f} might be too low (recommended: $100K+).")
            
            # Scanner connection/status
            if hasattr(self.scanner, 'api_client') and not self.scanner.api_client: # Assuming direct api_client attribute
                issues.append("Market scanner does not appear to be connected to an API client.")
            elif hasattr(self.scanner, 'get_scanner_status'):
                 scanner_stat = self.scanner.get_scanner_status()
                 if not scanner_stat.get('api_connected', True): # If scanner reports API issue
                     issues.append("Market scanner reports API connection issue.")

            # Check if simulation mode is on when trading is enabled (as a warning)
            if self.trading_enabled and self.simulation_mode:
                logger.info("FYI: Trading is enabled, but currently in SIMULATION MODE. No real trades will be placed.")
            elif self.trading_enabled and not self.simulation_mode:
                logger.critical("CRITICAL: Trading is enabled and in LIVE TRADING MODE. Real funds are active.")


            validation_result = {
                "is_valid_for_trading": len(issues) == 0, # Renamed for clarity
                "issues_found": issues,
                "validation_timestamp": time.time()
            }
            
            if validation_result["is_valid_for_trading"]:
                logger.info("✅ Trading conditions and settings validation passed.")
            else:
                logger.warning(f"⚠️ Trading validation found issues: {issues}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Trading conditions validation process failed: {e}")
            return {
                "is_valid_for_trading": False,
                "issues_found": [f"Validation process error: {str(e)}"],
                "validation_timestamp": time.time()
            }

# Example of how you might run this (for testing purposes, not part of the class)
if __name__ == '__main__':
    logger.info("Initializing AutomatedTrader for a test run...")
    trader_bot = AutomatedTrader()

    # Setup for a test
    trader_bot.set_simulation_mode(True) # Start in simulation
    logger.info(f"Initial simulation mode: {trader_bot.simulation_mode}")
    
    trader_bot.update_settings({
        "position_size_usd": 30,
        "profit_target_usd": 0.10,
        "max_positions": 2,
        "min_volume_usd": 600000,
        "scan_interval": 60 # Scan every minute for testing
    })

    # Validate conditions
    validation = trader_bot.validate_trading_conditions()
    logger.info(f"Validation Result: {validation}")

    if validation["is_valid_for_trading"]:
        trader_bot.set_trading_enabled(True)
        logger.info("Trading enabled for test.")
        
        # Simulate a few cycles
        for i in range(3):
            logger.info(f"--- Test Cycle {i+1} ---")
            trader_bot.run_trading_cycle()
            logger.info(f"Trader Status after cycle: {trader_bot.get_status()}")
            logger.info(f"Performance after cycle: {trader_bot.get_performance_summary()}")
            if i < 2: # Don't sleep after the last cycle
                time.sleep(10) # Short sleep for test
    else:
        logger.error("Cannot start test trading due to validation issues.")

    # Test emergency stop
    # trader_bot.emergency_stop()
    # logger.info(f"Status after emergency stop: {trader_bot.get_status()}")

    # Test switching to live (hypothetically, ensure API keys etc. are placeholders if real)
    # logger.info("Attempting to switch to LIVE mode (hypothetical test)")
    # trader_bot.set_simulation_mode(False)
    # validation_live = trader_bot.validate_trading_conditions()
    # if trader_bot.trading_enabled and not trader_bot.simulation_mode:
    #    logger.critical("NOW IN (HYPOTHETICAL) LIVE MODE")


    logger.info("Test run finished.")

