# strategy/intelligent_trader.py - Javított verzió

import time
from datetime import datetime
from typing import Dict, List, Optional
# from utils.logger import logger # Feltételezve, hogy ez a logger import helyes
import logging # Standard logging használata, ha a utils.logger nem specifikus
logger = logging.getLogger(__name__)


# Fallback PositionManager definíciója, ha a valós nem érhető el vagy demo módban van.
class PositionManagerFallback:
    """Fallback position manager for simulation or when real one is not available."""
    
    def __init__(self):
        self.positions: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []
        self.max_positions = 3 # Szimulációhoz
        logger.info("PositionManagerFallback initialized.")

    def get_all_positions(self) -> Dict[str, Dict]:
        return self.positions.copy()
    
    def get_position(self, pair: str) -> Optional[Dict]:
        return self.positions.get(pair)
    
    def open_position(self, pair: str, side: str, entry_price: float, volume: float, stop_loss: Optional[float] = None, take_profit: Optional[float] = None, **kwargs) -> bool:
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Fallback PM: Max positions ({self.max_positions}) reached. Cannot open {pair}.")
            return False
        
        position_data = {
            'pair': pair,
            'side': side,
            'entry_price': entry_price,
            'volume': volume,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'open_time': time.time(),
            'status': 'open',
            **kwargs
        }
        self.positions[pair] = position_data
        logger.info(f"Fallback PM: Opened simulated position for {pair} ({side}) @ {entry_price}")
        return True

    def close_position(self, pair: str, exit_price: float, reason: str = "Simulated Close") -> Optional[Dict]:
        if pair in self.positions:
            position = self.positions.pop(pair)
            entry_price = position['entry_price']
            volume = position['volume']
            side = position['side']
            
            pnl = 0.0
            if side == 'buy':
                pnl = (exit_price - entry_price) * volume
            else: # sell
                pnl = (entry_price - exit_price) * volume
            
            closed_trade_info = {**position, 'exit_price': exit_price, 'pnl': pnl, 'reason': reason, 'status': 'closed', 'close_time': time.time()}
            self.position_history.append(closed_trade_info)
            logger.info(f"Fallback PM: Closed simulated position for {pair}. P&L: {pnl:.2f}. Reason: {reason}")
            return closed_trade_info
        logger.warning(f"Fallback PM: No position found for {pair} to close.")
        return None

    def get_history(self) -> List[Dict]:
        return self.position_history[:]

    def get_statistics(self) -> Dict:
        total_trades = len(self.position_history)
        wins = sum(1 for trade in self.position_history if trade.get('pnl', 0) > 0)
        win_rate = (wins / total_trades) if total_trades > 0 else 0.0
        total_profit = sum(trade.get('pnl', 0) for trade in self.position_history)
        return {'total_trades': total_trades, 'win_rate': win_rate, 'total_profit': total_profit, 'active_positions': len(self.positions)}


class IntelligentTrader:
    """AI-vezérelt trading rendszer $50 mikro-pozíciókhoz"""
    
    def __init__(self, main_window=None, simulation_mode: bool = False):  # simulation_mode hozzáadva
        self.main_window = main_window
        self.simulation_mode = simulation_mode # Éles (False) vagy Demo (True) mód
        
        # 🎯 MIKRO-TRADING BEÁLLÍTÁSOK
        self.DEFAULT_POSITION_SIZE = 50.0      # $50 pozíciók
        self.MIN_PROFIT_TARGET = 0.15          # $0.15 minimum profit
        self.MAX_LOSS_PER_POSITION = 2.0       # $2 maximum loss
        self.POSITION_SIZE_RANGE = (25.0, 75.0)  # $25-75 között
        
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'active_positions': 0,
            'win_rate': 0.0
        }
        self.learning_data = []
        
        # Position manager beállítása a mód alapján
        if not self.simulation_mode and self.main_window and hasattr(self.main_window, 'position_manager') and self.main_window.position_manager:
            self.position_manager = self.main_window.position_manager
            logger.info("IntelligentTrader initialized in LIVE mode, using main_window.position_manager.")
        else:
            if self.simulation_mode:
                logger.info("IntelligentTrader initialized in SIMULATION mode, using PositionManagerFallback.")
            else: # Éles mód, de a main_window.position_manager nem elérhető
                logger.warning("IntelligentTrader: main_window.position_manager not available in LIVE mode. Falling back to PositionManagerFallback. THIS IS UNEXPECTED FOR LIVE TRADING.")
            self.position_manager = PositionManagerFallback()
        
        self.min_confidence_threshold = 0.65
        self.max_concurrent_trades = 3
        
        self.micro_trading_settings = {
            'target_profit_pct': 0.30,
            'max_loss_pct': 4.0,
            'min_hold_time_minutes': 5,
            'max_hold_time_hours': 2,
            'fee_aware_targets': True,
            'quick_profit_enabled': True,
        }
        
        # logger.info(f"IntelligentTrader initialized. Simulation mode: {self.simulation_mode}") # Már fentebb logolva

    def get_trading_status(self) -> Dict:
        """Trading státusz lekérdezése"""
        try:
            # Mindig a self.position_manager-t használjuk, ami vagy a valós, vagy a fallback
            active_positions = len(self.position_manager.get_all_positions())
            
            # A performance_metrics-et a close_position_with_learning frissíti
            # A win_rate-et is ott lehetne, vagy a get_statistics-ből venni
            stats = self.position_manager.get_statistics() if hasattr(self.position_manager, 'get_statistics') else self.performance_metrics

            return {
                'total_trades': stats.get('total_trades', self.performance_metrics['total_trades']),
                'winning_trades': stats.get('winning_trades', self.performance_metrics['winning_trades']),
                'total_profit': stats.get('total_profit', self.performance_metrics['total_profit']),
                'active_positions': active_positions, # Ez mindig friss
                'win_rate': stats.get('win_rate', self.performance_metrics['win_rate'])
            }
        except Exception as e:
            logger.error(f"Error getting trading status: {e}", exc_info=True)
            # Visszaadjuk a belső metrikákat fallbackként
            copied_metrics = self.performance_metrics.copy()
            copied_metrics['active_positions'] = len(self.position_manager.get_all_positions()) if self.position_manager else 0
            return copied_metrics

    def run_micro_trading_cycle(self, available_balance: float = 750.0):
        """Mikro-trading ciklus futtatása"""
        try:
            logger.info(f"🎯 Starting micro-trading cycle ($50 positions). Simulation: {self.simulation_mode}")
            
            self.monitor_micro_positions() # Ez a szimulált pozíciókat kezeli, ha fallback PM van
            
            current_positions = len(self.position_manager.get_all_positions())
            max_positions = self.max_concurrent_trades
            
            if current_positions < max_positions:
                logger.info(f"Looking for new positions ({current_positions}/{max_positions})")
                
                if self.simulation_mode:
                    logger.info("Simulation mode is active. Attempting to simulate position opening.")
                    self.simulate_position_opening() 
                else: # Éles mód
                    logger.info("Live mode is active. Attempting to find and execute real opportunity.")
                    can_trade_live = (
                        self.main_window and
                        hasattr(self.main_window, 'api') and self.main_window.api and
                        hasattr(self.main_window, 'position_manager') and self.main_window.position_manager and
                        # Ellenőrizzük, hogy a self.position_manager a valós-e (nem a fallback)
                        not isinstance(self.position_manager, PositionManagerFallback) and
                        hasattr(self.main_window, 'trade_logger') and self.main_window.trade_logger
                    )

                    if can_trade_live:
                        opportunity = self.find_real_opportunity()
                        if opportunity:
                            self.execute_real_position(opportunity)
                        else:
                            logger.info("No real opportunity found in live mode.")
                    else:
                        logger.error("Cannot trade live: main_window or its essential components (api, real position_manager, trade_logger) are not available or not configured for live trading.")
            
            self.update_performance_metrics() # Ez a self.position_manager alapján frissít
            
        except Exception as e:
            logger.error(f"Error in micro-trading cycle: {e}", exc_info=True)

    def find_real_opportunity(self):
        """Valós kereskedési lehetőség keresése"""
        try:
            # Éles módban vagyunk, ha ezt a metódust hívjuk a run_micro_trading_cycle-ból
            if self.simulation_mode:
                logger.info("find_real_opportunity called in simulation_mode. This should ideally be find_simulated_opportunity. Returning None.")
                return None

            if not self.main_window or \
               not hasattr(self.main_window, 'api') or not self.main_window.api or \
               not hasattr(self.main_window, 'pair_list') or not self.main_window.pair_list or \
               not hasattr(self.main_window, 'position_manager') or not self.main_window.position_manager or \
               isinstance(self.position_manager, PositionManagerFallback): # Biztosítjuk, hogy ne a fallback PM-et használjuk élesben
                logger.error("Cannot find real opportunity: main_window or its essential components (api, pair_list, real position_manager) are not available for live trading.")
                return None
                
            logger.info("🔥 Finding REAL trading opportunity using main_window components...")
            
            import random
            if self.main_window.pair_list.count() > 0:
                random_index = random.randint(0, self.main_window.pair_list.count() - 1)
                selected_pair_display = self.main_window.pair_list.item(random_index).text()
                selected_pair_altname = selected_pair_display.replace("/", "")
                
                current_price = self.main_window.get_current_price_for_pair(selected_pair_altname)
                
                if current_price and current_price > 0:
                    opportunity = {
                        'pair': selected_pair_altname,
                        'entry_price': current_price,
                        'side': 'buy', 
                        'position_size_usd': self.DEFAULT_POSITION_SIZE,
                        'confidence': 0.75, # Ezt egy valós stratégia adná
                        'reason': 'REAL_MARKET_OPPORTUNITY_PLACEHOLDER_STRATEGY'
                    }
                    logger.info(f"🎯 REAL opportunity found: {selected_pair_display} @ ${current_price:.6f}")
                    return opportunity
                else:
                    logger.warning(f"Could not get a valid price for {selected_pair_altname} to find real opportunity.")
                    return None
            else:
                logger.warning("No valid pairs available in main_window.pair_list for finding real opportunity.")
                return None
                
        except Exception as e:
            logger.error(f"Error finding real opportunity: {e}", exc_info=True)
            return None

    def execute_real_position(self, opportunity: Dict):
        """Valós pozíció végrehajtás"""
        try:
            if self.simulation_mode:
                logger.info("execute_real_position called in simulation_mode. This is incorrect. Returning False.")
                return False

            if not self.main_window or \
               not hasattr(self.main_window, 'position_manager') or not self.main_window.position_manager or \
               isinstance(self.position_manager, PositionManagerFallback) or \
               not hasattr(self.main_window, 'trade_logger') or not self.main_window.trade_logger:
                logger.error("Cannot execute real position: main_window or its essential components (real position_manager, trade_logger) are not available for live trading.")
                return False
                
            if not opportunity:
                logger.warning("No opportunity provided for real execution.")
                return False
                
            logger.info(f"🔥 Executing REAL position via main_window.position_manager: {opportunity.get('pair')}")
            
            pair = opportunity.get('pair')
            entry_price = opportunity.get('entry_price')
            side = opportunity.get('side', 'buy')
            position_size_usd = opportunity.get('position_size_usd', self.DEFAULT_POSITION_SIZE)
            
            if not pair or not entry_price or entry_price <= 0:
                logger.error(f"Invalid opportunity data for real execution: Pair or Price missing/invalid. Pair: {pair}, Price: {entry_price}")
                return False
                
            volume = position_size_usd / entry_price
            
            if side == 'buy':
                stop_loss = entry_price * (1 - self.micro_trading_settings['max_loss_pct'] / 100.0)
                take_profit = entry_price * (1 + self.micro_trading_settings['target_profit_pct'] / 100.0)
            else: # sell
                stop_loss = entry_price * (1 + self.micro_trading_settings['max_loss_pct'] / 100.0)
                take_profit = entry_price * (1 - self.micro_trading_settings['target_profit_pct'] / 100.0)
            
            # A self.position_manager itt már a main_window.position_manager-re mutat (éles módban)
            success = self.position_manager.open_position(
                pair=pair, side=side, entry_price=entry_price, volume=volume,
                stop_loss=stop_loss, take_profit=take_profit,
                entry_time_unix=time.time(), # main_window ismeri ezt? Vagy a PM?
                # További kwargs, ha a valós PM igényli, pl. 'reason'
                reason=opportunity.get('reason', 'INTELLIGENT_TRADER_REAL')
            )
            
            if success:
                self.main_window.trade_logger.log(
                    pair=pair, side=side, entry_price=entry_price,
                    exit_price=None, volume=volume, pnl=0.0, 
                    reason=opportunity.get('reason', "INTELLIGENT_TRADER_REAL")
                )
                logger.info(f"🎯 REAL position opened: {side.upper()} {pair} ${position_size_usd:.0f} @ {entry_price:.6f}")
                
                if hasattr(self.main_window, 'refresh_open_trades'): self.main_window.refresh_open_trades()
                if hasattr(self.main_window, 'refresh_balance'): self.main_window.refresh_balance()
                
                # A performance metrikákat a valós PositionManager-nek kellene kezelnie,
                # vagy a close_position_with_learning-hez hasonló logika kellene a valós zárásokhoz is.
                # Itt most nem frissítjük a self.performance_metrics-et, mert az a fallback PM-hez kötődik.
                return True
            else:
                logger.error(f"main_window.position_manager.open_position failed for {pair}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing real position: {e}", exc_info=True)
            return False

    def simulate_position_opening(self):
        """Szimulált pozíció nyitás (csak ha self.position_manager a FallbackPM)"""
        if not isinstance(self.position_manager, PositionManagerFallback):
            logger.warning("Attempted to simulate position opening, but not using FallbackPositionManager. Skipping.")
            return

        try:
            import random
            pairs = ['XBTUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD']
            pair = random.choice(pairs)
            base_prices = {'XBTUSD': 50000, 'ETHUSD': 3000, 'SOLUSD': 100, 'ADAUSD': 0.5, 'XRPUSD': 0.6}
            entry_price = base_prices.get(pair, 1000) * (0.95 + random.random() * 0.1)
            position_size = self.DEFAULT_POSITION_SIZE
            volume = position_size / entry_price
            stop_loss = entry_price * (1 - self.micro_trading_settings['max_loss_pct'] / 100.0)
            take_profit = entry_price * (1 + self.micro_trading_settings['target_profit_pct'] / 100.0)
            
            # A FallbackPositionManager open_position metódusát használjuk
            success = self.position_manager.open_position(
                pair=pair, side='buy', entry_price=entry_price, volume=volume,
                stop_loss=stop_loss, take_profit=take_profit,
                position_size_usd=position_size, # Ezt a fallback PM ismeri
                reason="SIMULATED_OPEN"
            )
            if success:
                logger.info(f"🎯 Simulated position opened via FallbackPM: {pair} ${position_size:.0f}")
            else:
                logger.warning(f"Failed to open simulated position for {pair} via FallbackPM.")
            
        except Exception as e:
            logger.error(f"Error simulating position opening: {e}", exc_info=True)

    def monitor_micro_positions(self):
        """Mikro pozíciók monitoring. Csak a FallbackPositionManager-rel működik helyesen jelenleg."""
        if not isinstance(self.position_manager, PositionManagerFallback):
            # logger.info("Skipping micro position monitoring as not using FallbackPositionManager.")
            # Ha éles módban vagyunk, a valós pozíciók monitorozása a MainWindow.monitor_positions_with_ai-ban történik.
            # Itt lehetne egy ellenőrzés, hogy ne duplán monitorozzunk, vagy ez a metódus csak szimulációhoz legyen.
            # Jelenlegi formájában ez a metódus szimulált árakkal és zárásokkal dolgozik.
            if not self.simulation_mode:
                 logger.debug("monitor_micro_positions called in live mode, but it's designed for simulation with FallbackPM. Real position monitoring is in MainWindow.")
                 return 
            # Ha simulation_mode True, de valamiért mégsem FallbackPM (pl. hiba a __init__-ben), akkor itt nem fut le.
            else: # simulation_mode True, de nem FallbackPM
                 logger.warning("monitor_micro_positions called in simulation_mode, but self.position_manager is not PositionManagerFallback. Skipping simulated monitoring.")
                 return


        try:
            positions_to_close = []
            # logger.debug(f"Monitoring {len(self.position_manager.get_all_positions())} simulated positions.")

            for pair, position in list(self.position_manager.get_all_positions().items()): # list() a RuntimeError elkerülésére
                import random
                entry_price = position.get('entry_price', 1000)
                # Szimulált ár: az entry price körül mozogjon, de néha érje el a SL/TP-t
                price_movement = random.uniform(-0.05, 0.05) # +/- 5% max mozgás
                current_price = entry_price * (1 + price_movement)
                
                # Biztosítjuk, hogy néha a SL/TP is teljesüljön
                if random.random() < 0.1: # 10% eséllyel
                    if random.random() < 0.5 and position.get('stop_loss'): # SL trigger
                        current_price = position['stop_loss'] * (0.999 if position.get('side','buy') == 'buy' else 1.001)
                    elif position.get('take_profit'): # TP trigger
                        current_price = position['take_profit'] * (1.001 if position.get('side','buy') == 'buy' else 0.999)

                open_time = position.get('open_time', time.time())
                hold_time_minutes = (time.time() - open_time) / 60
                
                should_close = False
                close_reason = ""
                
                stop_loss = position.get('stop_loss')
                take_profit = position.get('take_profit')
                side = position.get('side', 'buy')

                if side == 'buy':
                    if stop_loss and current_price <= stop_loss:
                        should_close = True; close_reason = "SIM_STOP_LOSS"
                    elif take_profit and current_price >= take_profit:
                        should_close = True; close_reason = "SIM_TAKE_PROFIT"
                else: # sell
                    if stop_loss and current_price >= stop_loss:
                        should_close = True; close_reason = "SIM_STOP_LOSS"
                    elif take_profit and current_price <= take_profit:
                        should_close = True; close_reason = "SIM_TAKE_PROFIT"
                
                if not should_close: # Csak akkor ellenőrizzük a többi feltételt, ha SL/TP nem teljesült
                    if hold_time_minutes > self.micro_trading_settings['max_hold_time_hours'] * 60:
                        should_close = True; close_reason = "SIM_TIME_LIMIT"
                    elif random.random() < 0.01: # 1% random close chance per cycle
                        should_close = True; close_reason = "SIM_RANDOM_EXIT"
                
                if should_close:
                    logger.info(f"Simulated position {pair} marked for closure. Reason: {close_reason}, Current Price: {current_price:.4f}")
                    positions_to_close.append((pair, current_price, close_reason))
            
            for pair, exit_price, reason in positions_to_close:
                self.close_position_with_learning(pair, exit_price, reason)
                
        except Exception as e:
            logger.error(f"Error monitoring micro positions (simulation): {e}", exc_info=True)

    def close_position_with_learning(self, pair: str, exit_price: float, reason: str):
        """Pozíció zárás tanulással (ez a metódus a self.position_manager-t használja, ami lehet valós vagy fallback)"""
        try:
            # A self.position_manager.close_position metódusnak kellene visszaadnia a zárt pozíció adatait
            closed_position_info = self.position_manager.close_position(pair, exit_price, reason=reason)
            
            if not closed_position_info:
                logger.warning(f"Could not close position {pair} via position_manager, or already closed.")
                return

            pnl = closed_position_info.get('pnl', 0.0)
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['total_profit'] += pnl
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            
            # Update win_rate (itt is lehet, vagy a get_statistics-ben)
            if self.performance_metrics['total_trades'] > 0:
                 self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
            else:
                self.performance_metrics['win_rate'] = 0.0

            # Add to learning data
            # A closed_position_info már tartalmazza a szükséges adatokat
            trade_data_for_learning = {
                'pair': closed_position_info.get('pair'),
                'entry_price': closed_position_info.get('entry_price'),
                'exit_price': closed_position_info.get('exit_price'),
                'pnl': pnl,
                'side': closed_position_info.get('side'),
                'volume': closed_position_info.get('volume'),
                'reason': reason,
                'open_time': closed_position_info.get('open_time'),
                'close_time': closed_position_info.get('close_time', time.time()),
                'timestamp': datetime.now().isoformat()
            }
            self.learning_data.append(trade_data_for_learning)
            
            logger.info(f"📚 Position closed and logged for learning: {pair} P&L: ${pnl:.2f} ({reason})")
            
            # Ha a main_window-ban van GUI frissítés, azt is meg lehetne hívni, de ez a trader felelőssége lehet
            if not self.simulation_mode and self.main_window:
                if hasattr(self.main_window, 'refresh_open_trades'): self.main_window.refresh_open_trades()
                if hasattr(self.main_window, 'refresh_balance'): self.main_window.refresh_balance()
                if hasattr(self.main_window, 'update_ai_performance_display'): self.main_window.update_ai_performance_display()


        except Exception as e:
            logger.error(f"Error closing position {pair} with learning: {e}", exc_info=True)

    def update_performance_metrics(self):
        """Performance metrikák frissítése (főleg a win_rate és active_positions)"""
        try:
            # A legtöbb metrikát (total_trades, winning_trades, total_profit)
            # a close_position_with_learning frissíti.
            # Itt frissíthetjük az aktív pozíciók számát és a win rate-et, ha az nem történt meg máshol.
            
            if hasattr(self.position_manager, 'get_statistics'):
                stats = self.position_manager.get_statistics()
                self.performance_metrics['active_positions'] = stats.get('active_positions', len(self.position_manager.get_all_positions()))
                self.performance_metrics['total_trades'] = stats.get('total_trades', self.performance_metrics['total_trades'])
                self.performance_metrics['winning_trades'] = stats.get('winning_trades', self.performance_metrics['winning_trades'])
                self.performance_metrics['total_profit'] = stats.get('total_profit', self.performance_metrics['total_profit'])
                self.performance_metrics['win_rate'] = stats.get('win_rate', self.performance_metrics['win_rate'])
            else: # Fallback, ha nincs get_statistics
                self.performance_metrics['active_positions'] = len(self.position_manager.get_all_positions())
                if self.performance_metrics['total_trades'] > 0:
                    self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
                else:
                    self.performance_metrics['win_rate'] = 0.0
            
            # logger.debug(f"Performance metrics updated: {self.performance_metrics}")

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}", exc_info=True)

    def learn_and_adapt(self):
        """AI tanulás és adaptáció"""
        try:
            if len(self.learning_data) < 5: # Legalább 5 trade kell a tanuláshoz
                # logger.info("Not enough data for learning and adaptation.")
                return
            
            recent_trades = self.learning_data[-10:] # Utolsó 10 trade elemzése
            if not recent_trades: return

            recent_pnls = [trade.get('pnl', 0) for trade in recent_trades]
            
            if not recent_pnls: return

            avg_pnl = sum(recent_pnls) / len(recent_pnls)
            
            # Egyszerű adaptációs logika:
            if avg_pnl > self.MIN_PROFIT_TARGET * 0.1: # Ha az átlagos PnL pozitív (a min target 10%-a felett)
                # Jól teljesít, esetleg csökkenthetjük a bizalmi küszöböt (óvatosabban)
                self.min_confidence_threshold = max(0.60, self.min_confidence_threshold - 0.005) # Nagyon kis lépés
                logger.info(f"📈 Learning: Performance good (avg PnL ${avg_pnl:.2f}). Confidence threshold slightly lowered to {self.min_confidence_threshold:.3f}")
            elif avg_pnl < -(self.MAX_LOSS_PER_POSITION * 0.05): # Ha az átlagos PnL negatív (a max loss 5%-a alatt)
                # Rosszul teljesít, növeljük a bizalmi küszöböt
                self.min_confidence_threshold = min(0.85, self.min_confidence_threshold + 0.01)
                logger.info(f"📉 Learning: Performance poor (avg PnL ${avg_pnl:.2f}). Confidence threshold increased to {self.min_confidence_threshold:.3f}")
            
            # TODO: Komplexebb tanulási algoritmusok implementálása
            # Pl. sikeres/sikertelen trade-ek paramétereinek elemzése, stratégia finomítása.

        except Exception as e:
            logger.error(f"Error in learning and adaptation: {e}", exc_info=True)

    def run_intelligent_trading_cycle(self, available_balance: float):
        """IntelligentTrader fő ciklusa, a run_micro_trading_cycle aliasaként"""
        self.run_micro_trading_cycle(available_balance)


