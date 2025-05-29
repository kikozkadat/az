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
        
        logger.info(f"AutomatedTrader initialized - ${self.position_size_usd} positions, ${self.profit_target_usd} target")
        
    def set_trading_enabled(self, enabled):
        """Enable or disable automated trading"""
        self.trading_enabled = enabled
        logger.info(f"Automated trading {'enabled' if enabled else 'disabled'}")
        
    def set_max_positions(self, max_pos):
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
            
    def _scan_for_volume_opportunities(self) -> List:
        """
        Volume alapú lehetőségek szkenelése
        """
        try:
            logger.info("🔍 Scanning for high-volume opportunities...")
            
            # Get high volume opportunities from scanner
            opportunities = self.scanner.get_top_opportunities(min_score=0.5)
            
            if not opportunities:
                logger.warning("Scanner returned no opportunities")
                return []
                
            # Filter by volume requirement
            volume_filtered = [
                opp for opp in opportunities 
                if opp.volume_usd >= self.min_volume_usd
            ]
            
            logger.info(f"📊 Found {len(volume_filtered)} opportunities with volume ≥ ${self.min_volume_usd:,}")
            
            # Additional technical filtering
            technical_filtered = []
            
            for opp in volume_filtered:
                # Technical criteria for micro trading
                if (25 <= opp.rsi <= 40 and  # Oversold but not extreme
                    opp.btc_correlation >= 0.8 and  # Strong BTC correlation
                    opp.volume_usd >= 1000000):  # At least $1M volume for best opportunities
                    
                    technical_filtered.append(opp)
                    
            logger.info(f"🎯 {len(technical_filtered)} opportunities passed technical filter")
            
            return technical_filtered[:5]  # Top 5
            
        except Exception as e:
            logger.error(f"Volume opportunity scan failed: {e}")
            return []

    def _select_best_opportunity(self, opportunities) -> Optional:
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
                volume_score = min(1.0, opp.volume_usd / 10000000)  # Normalize to 10M
                technical_score = opp.final_score
                
                # Weight: 60% technical, 40% volume
                combined_score = (technical_score * 0.6) + (volume_score * 0.4)
                
                scored_opportunities.append({
                    'opportunity': opp,
                    'combined_score': combined_score
                })
                
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

    def _execute_trade_opportunity(self, opportunity):
        """
        Kereskedési lehetőség végrehajtása
        """
        try:
            pair = opportunity.pair
            
            logger.info(f"💰 Executing trade for {pair}...")
            
            # Check if position already exists
            if self.position_manager.get_position(pair):
                logger.warning(f"Position already exists for {pair}")
                return
                
            # Get current price
            current_price = self._get_current_price(pair)
            if not current_price:
                logger.error(f"Could not get current price for {pair}")
                return
                
            # Calculate position parameters
            volume = self.position_size_usd / current_price
            
            # Calculate SL/TP for micro trading
            stop_loss_pct = 0.04  # 4% stop loss
            target_profit_pct = self.profit_target_usd / self.position_size_usd  # ~0.3%
            
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + target_profit_pct)
            
            logger.info(f"📊 Trade parameters:")
            logger.info(f"   Price: ${current_price:.6f}")
            logger.info(f"   Volume: {volume:.6f}")
            logger.info(f"   Position size: ${self.position_size_usd}")
            logger.info(f"   Stop loss: ${stop_loss:.6f} (-{stop_loss_pct*100:.1f}%)")
            logger.info(f"   Take profit: ${take_profit:.6f} (+{target_profit_pct*100:.2f}%)")
            
            # SIMULATION MODE - In real trading, uncomment the trade execution
            logger.info("🎮 SIMULATION MODE - Trade not executed")
            
            # Real trading code (uncomment for live trading):
            # result = self.trader.place_order(pair, "buy", volume)
            # if result and 'error' not in result:
            
            # Open position in position manager
            success = self.position_manager.open_position(
                pair=pair,
                side="buy",
                entry_price=current_price,
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if success:
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
                
                logger.info(f"✅ Position opened successfully for {pair}")
            else:
                logger.error(f"Failed to open position for {pair}")
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")

    def _monitor_existing_positions(self):
        """
        Meglévő pozíciók monitorozása
        """
        try:
            positions = self.position_manager.get_all_positions()
            
            if not positions:
                return
                
            logger.info(f"📋 Monitoring {len(positions)} active positions...")
            
            for pair, position in list(positions.items()):
                try:
                    current_price = self._get_current_price(pair)
                    if not current_price:
                        continue
                        
                    # Check SL/TP
                    should_close, reason = self._should_close_position(position, current_price)
                    
                    if should_close:
                        self._close_position(pair, position, current_price, reason)
                        
                except Exception as e:
                    logger.error(f"Position monitoring failed for {pair}: {e}")
                    
        except Exception as e:
            logger.error(f"Position monitoring failed: {e}")

    def _should_close_position(self, position, current_price) -> tuple:
        """
        Pozíció zárás ellenőrzése
        """
        try:
            entry_price = position["entry_price"]
            stop_loss = position.get("stop_loss")
            take_profit = position.get("take_profit")
            side = position["side"]
            open_time = position.get("open_time", time.time())
            
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
                    volume = position.get("volume", 0)
                    profit_usd = (current_price - entry_price) * volume
                    if profit_usd >= 0.10:  # $0.10 quick profit
                        return True, "QUICK_PROFIT"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Position close check failed: {e}")
            return False, "ERROR"

    def _close_position(self, pair, position, current_price, reason):
        """
        Pozíció zárása
        """
        try:
            side = position["side"]
            volume = position["volume"]
            entry_price = position["entry_price"]
            
            logger.info(f"🚪 Closing position: {pair} @ ${current_price:.6f} ({reason})")
            
            # SIMULATION MODE - In real trading, uncomment the trade execution
            logger.info("🎮 SIMULATION MODE - Position close not executed")
            
            # Real trading code (uncomment for live trading):
            # result = self.trader.close_market_position(pair, side, volume)
            
            # Calculate P&L
            if side.lower() == "buy":
                pnl = (current_price - entry_price) * volume
            else:
                pnl = (entry_price - current_price) * volume
                
            # Close position in manager
            self.position_manager.close_position(pair, current_price)
            
            # Log the trade
            self.logger.log(
                pair=pair,
                side=side,
                entry_price=entry_price,
                exit_price=current_price,
                volume=volume,
                pnl=pnl,
                reason=reason
            )
            
            self.profit_today += pnl
            
            logger.info(f"💰 Position closed - P&L: ${pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Position closing failed for {pair}: {e}")

    def _get_current_price(self, pair):
        """
        Aktuális ár lekérése
        """
        try:
            # Try to get price from API
            ohlc_data = self.api.get_ohlc(pair, interval=1)
            
            if ohlc_data:
                # Get the first (and usually only) pair data
                pair_data = list(ohlc_data.values())[0]
                if pair_data:
                    # Get last candle close price
                    last_candle = pair_data[-1]
                    return float(last_candle[4])  # Close price
                    
            logger.warning(f"Could not get price for {pair}")
            return None
            
        except Exception as e:
            logger.error(f"Price fetch failed for {pair}: {e}")
            return None
            
    def get_status(self):
        """
        Trader státusz lekérése
        """
        return {
            "trading_enabled": self.trading_enabled,
            "max_positions": self.max_positions,
            "position_size_usd": self.position_size_usd,
            "profit_target_usd": self.profit_target_usd,
            "current_positions": len(self.position_manager.positions),
            "trades_today": self.trades_today,
            "profit_today": self.profit_today,
            "last_scan": self.last_scan_time,
            "min_volume_requirement": f"${self.min_volume_usd:,}",
            "scan_interval_seconds": self.scan_interval
        }
    
    def update_settings(self, settings):
        """
        Trading beállítások frissítése
        """
        try:
            if 'position_size_usd' in settings:
                self.position_size_usd = settings['position_size_usd']
                
            if 'profit_target_usd' in settings:
                self.profit_target_usd = settings['profit_target_usd']
                
            if 'max_positions' in settings:
                self.max_positions = settings['max_positions']
                
            if 'min_volume_usd' in settings:
                self.min_volume_usd = settings['min_volume_usd']
                
            if 'scan_interval' in settings:
                self.scan_interval = settings['scan_interval']
                
            logger.info(f"Trading settings updated: {settings}")
            
        except Exception as e:
            logger.error(f"Settings update failed: {e}")
    
    def force_scan(self):
        """
        Kényszerített scan futtatása
        """
        try:
            logger.info("🔄 Force scanning market...")
            self.last_scan_time = 0  # Reset scan time
            self.run_trading_cycle()
            
        except Exception as e:
            logger.error(f"Force scan failed: {e}")
    
    def emergency_stop(self):
        """
        Vészleállítás - összes pozíció zárása
        """
        try:
            logger.warning("🚨 EMERGENCY STOP - Closing all positions")
            
            self.trading_enabled = False
            positions = list(self.position_manager.get_all_positions().keys())
            
            for pair in positions:
                try:
                    current_price = self._get_current_price(pair)
                    if current_price:
                        position = self.position_manager.get_position(pair)
                        if position:
                            self._close_position(pair, position, current_price, "EMERGENCY_STOP")
                            
                except Exception as e:
                    logger.error(f"Emergency close failed for {pair}: {e}")
                    
            logger.warning(f"🚨 Emergency stop completed - {len(positions)} positions closed")
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
    
    def get_volume_statistics(self):
        """
        Forgalom statisztikák lekérése
        """
        try:
            if hasattr(self.api, 'get_volume_statistics'):
                return self.api.get_volume_statistics()
            else:
                return {"error": "Volume statistics not available"}
                
        except Exception as e:
            logger.error(f"Volume statistics failed: {e}")
            return {"error": str(e)}
    
    def get_performance_summary(self):
        """
        Teljesítmény összefoglaló
        """
        try:
            # Get position manager statistics
            stats = self.position_manager.get_statistics()
            
            # Add trader-specific metrics
            stats.update({
                "trades_today": self.trades_today,
                "profit_today": self.profit_today,
                "position_size_usd": self.position_size_usd,
                "profit_target_usd": self.profit_target_usd,
                "avg_profit_per_trade": (self.profit_today / max(self.trades_today, 1)),
                "last_trade_time": self.last_trade_time,
                "trading_enabled": self.trading_enabled
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Performance summary failed: {e}")
            return {}
    
    def analyze_market_conditions(self):
        """
        Piaci körülmények elemzése
        """
        try:
            # Get scanner status
            scanner_status = self.scanner.get_scanner_status()
            
            # Get volume statistics
            volume_stats = self.get_volume_statistics()
            
            # Combine analysis
            market_analysis = {
                "scanner_mode": scanner_status.get('mode', 'UNKNOWN'),
                "pairs_available": scanner_status.get('cached_results', 0),
                "min_volume_filter": scanner_status.get('min_volume_usd', 'N/A'),
                "api_connected": scanner_status.get('api_connected', False),
                "volume_statistics": volume_stats,
                "trading_conditions": "GOOD" if (
                    scanner_status.get('api_connected', False) and 
                    scanner_status.get('cached_results', 0) > 0
                ) else "LIMITED"
            }
            
            logger.info(f"📊 Market conditions: {market_analysis['trading_conditions']}")
            
            return market_analysis
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return {"trading_conditions": "ERROR", "error": str(e)}
    
    def set_simulation_mode(self, simulation_enabled=True):
        """
        Szimuláció mód be/kikapcsolása
        """
        self.simulation_mode = simulation_enabled
        mode_text = "SIMULATION" if simulation_enabled else "LIVE TRADING"
        logger.warning(f"🎮 Trading mode set to: {mode_text}")
        
        if not simulation_enabled:
            logger.warning("⚠️ LIVE TRADING MODE ENABLED - Real money at risk!")
    
    def validate_trading_conditions(self):
        """
        Kereskedési feltételek validálása
        """
        try:
            issues = []
            
            # API connection check
            if not self.api.test_connection():
                issues.append("API connection failed")
            
            # Position size validation
            if self.position_size_usd < 25 or self.position_size_usd > 100:
                issues.append(f"Position size ${self.position_size_usd} outside recommended range ($25-$100)")
            
            # Profit target validation
            if self.profit_target_usd < 0.05 or self.profit_target_usd > 2.0:
                issues.append(f"Profit target ${self.profit_target_usd} outside reasonable range ($0.05-$2.00)")
            
            # Volume requirement check
            if self.min_volume_usd < 100000:
                issues.append(f"Volume requirement ${self.min_volume_usd:,} too low (recommended: $500K+)")
            
            # Scanner connection
            if not hasattr(self.scanner, 'api_client') or not self.scanner.api_client:
                issues.append("Market scanner not connected to API")
            
            validation_result = {
                "valid": len(issues) == 0,
                "issues": issues,
                "validation_time": time.time()
            }
            
            if validation_result["valid"]:
                logger.info("✅ Trading conditions validation passed")
            else:
                logger.warning(f"⚠️ Trading validation issues: {issues}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Trading validation failed: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {e}"],
                "validation_time": time.time()
            }
