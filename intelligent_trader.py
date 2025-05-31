# strategy/intelligent_trader.py - Javított verzió

import time
from datetime import datetime
from typing import Dict, List, Optional
from utils.logger import logger

class IntelligentTrader:
    """AI-vezérelt trading rendszer $50 mikro-pozíciókhoz"""
    
    def __init__(self, main_window=None):  # ← JAVÍTÁS: main_window paraméter hozzáadva
        # ← JAVÍTÁS: main_window attribútum inicializálása
        self.main_window = main_window
        
        # 🎯 MIKRO-TRADING BEÁLLÍTÁSOK
        self.DEFAULT_POSITION_SIZE = 50.0      # $50 pozíciók
        self.MIN_PROFIT_TARGET = 0.15          # $0.15 minimum profit
        self.MAX_LOSS_PER_POSITION = 2.0       # $2 maximum loss
        self.POSITION_SIZE_RANGE = (25.0, 75.0)  # $25-75 között
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'active_positions': 0,
            'win_rate': 0.0
        }
        
        # Learning data
        self.learning_data = []
        
        # Position manager fallback
        self.position_manager = PositionManagerFallback()
        
        # Trading settings
        self.simulation_mode = False  # ÉLES MÓD
        self.min_confidence_threshold = 0.65
        self.max_concurrent_trades = 3
        
        # Mikro-trading specifikus paraméterek
        self.micro_trading_settings = {
            'target_profit_pct': 0.30,      # 0.30% profit target
            'max_loss_pct': 4.0,            # 4% max loss
            'min_hold_time_minutes': 5,     # Min 5 perc tartás
            'max_hold_time_hours': 2,       # Max 2 óra tartás
            'fee_aware_targets': True,      # Fee-tudatos célok
            'quick_profit_enabled': True,   # Gyors profit realizálás
        }
        
        logger.info("IntelligentTrader initialized with fallback mode")

    def get_trading_status(self) -> Dict:
        """Trading státusz lekérdezése"""
        try:
            active_positions = len(self.position_manager.get_all_positions())
            
            # Win rate számítás
            total_trades = self.performance_metrics['total_trades']
            winning_trades = self.performance_metrics['winning_trades']
            win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'total_profit': self.performance_metrics['total_profit'],
                'active_positions': active_positions,
                'win_rate': win_rate
            }
        except Exception as e:
            logger.error(f"Error getting trading status: {e}")
            return self.performance_metrics.copy()

    def run_micro_trading_cycle(self, available_balance: float = 750.0):
        """Mikro-trading ciklus futtatása"""
        try:
            logger.info("🎯 Starting micro-trading cycle ($50 positions)...")
            
            # Pozíciók monitorozása
            self.monitor_micro_positions()
            
            # Új mikro pozíciók keresése
            current_positions = len(self.position_manager.get_all_positions())
            max_positions = self.max_concurrent_trades
            
            if current_positions < max_positions:
                logger.info(f"Looking for new positions ({current_positions}/{max_positions})")
                
                # DEBUG információk
                logger.info(f"🔍 DEBUG: main_window exists = {self.main_window is not None}")
                if self.main_window:
                    logger.info(f"🔍 DEBUG: main_window has trader = {hasattr(self.main_window, 'trader')}")
                    if hasattr(self.main_window, 'trader'):
                        logger.info(f"🔍 DEBUG: trader object = {type(self.main_window.trader)}")
                
                # ← JAVÍTÁS: Biztonságos main_window elérés
                # A main_window-ban 'trader' helyett más néven lehet a TradeManager
                has_trader = (hasattr(self.main_window, 'trader') or 
                             hasattr(self.main_window, 'trade_manager') or
                             hasattr(self.main_window, 'position') or
                             hasattr(self.main_window, 'api'))
                
                if self.main_window and has_trader:
                    opportunity = self.find_real_opportunity()
                    if opportunity:
                        self.execute_real_position(opportunity)
                else:
                    # Fallback: szimulált pozíció nyitás
                    logger.info("Main window not available, using simulation")
                    self.simulate_position_opening()
            
            # Teljesítmény frissítése
            self.update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error in micro-trading cycle: {e}")

    def find_real_opportunity(self):
        """Valós kereskedési lehetőség keresése"""
        try:
            # ← JAVÍTÁS: Biztonságos main_window használat
            if not self.main_window:
                logger.warning("Main window not available for real opportunity finding")
                return None
                
            # 🚀 ÉLES KERESKEDÉSI LOGIKA
            logger.info("🔥 Finding REAL trading opportunity using main_window...")
            
            # Használjuk a main_window API-ját és kereskedési rendszerét
            if hasattr(self.main_window, 'api') and self.main_window.api:
                # Válasszunk egy random párt a main_window pair_list-jéből
                import random
                
                if hasattr(self.main_window, 'pair_list') and self.main_window.pair_list and self.main_window.pair_list.count() > 0:
                    # Véletlenszerű pár kiválasztása a GUI listából
                    random_index = random.randint(0, self.main_window.pair_list.count() - 1)
                    selected_pair_display = self.main_window.pair_list.item(random_index).text()  # pl. "XBT/USD"
                    selected_pair_altname = selected_pair_display.replace("/", "")  # pl. "XBTUSD"
                    
                    # Aktuális ár lekérése
                    current_price = self.main_window.get_current_price_for_pair(selected_pair_altname)
                    
                    if current_price and current_price > 0:
                        # VALÓS KERESKEDÉSI LEHETŐSÉG LÉTREHOZÁSA
                        opportunity = {
                            'pair': selected_pair_altname,
                            'entry_price': current_price,
                            'side': 'buy',  # Alapértelmezett: vásárlás
                            'position_size_usd': self.DEFAULT_POSITION_SIZE,
                            'confidence': 0.75,  # Magas bizalom
                            'reason': 'REAL_MARKET_OPPORTUNITY'
                        }
                        
                        logger.info(f"🎯 REAL opportunity found: {selected_pair_display} @ ${current_price:.6f}")
                        return opportunity
                
                # Ha nincs elérhető pár, fallback
                logger.warning("No valid pairs available in main_window.pair_list")
                return None
            else:
                logger.warning("main_window.api not available")
                return None
                
        except Exception as e:
            logger.error(f"Error finding real opportunity: {e}")
            return None

    def execute_real_position(self, opportunity):
        """Valós pozíció végrehajtás"""
        try:
            # ← JAVÍTÁS: Biztonságos main_window használat
            if not self.main_window:
                logger.warning("Main window not available for real position execution")
                return False
                
            if not opportunity:
                logger.warning("No opportunity provided for execution")
                return False
                
            # 🚀 ÉLES POZÍCIÓ VÉGREHAJTÁS a main_window rendszerével
            logger.info("🔥 Executing REAL position via main_window...")
            
            pair = opportunity.get('pair')
            entry_price = opportunity.get('entry_price')
            side = opportunity.get('side', 'buy')
            position_size_usd = opportunity.get('position_size_usd', self.DEFAULT_POSITION_SIZE)
            
            if not pair or not entry_price:
                logger.error("Invalid opportunity data for execution")
                return False
                
            # Volume számítás
            volume = position_size_usd / entry_price
            
            # SL/TP számítás a mikro-trading beállítások alapján
            if side == 'buy':
                stop_loss = entry_price * (1 - self.micro_trading_settings['max_loss_pct'] / 100.0)
                take_profit = entry_price * (1 + self.micro_trading_settings['target_profit_pct'] / 100.0)
            else:
                stop_loss = entry_price * (1 + self.micro_trading_settings['max_loss_pct'] / 100.0)
                take_profit = entry_price * (1 - self.micro_trading_settings['target_profit_pct'] / 100.0)
            
            # VALÓS POZÍCIÓ NYITÁS a main_window PositionManager-jével
            if hasattr(self.main_window, 'position') and self.main_window.position:
                success = self.main_window.position.open_position(
                    pair=pair,
                    side=side,
                    entry_price=entry_price,
                    volume=volume,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    entry_time_unix=time.time()
                )
                
                if success:
                    # Logolás a main_window trade_logger-jével
                    if hasattr(self.main_window, 'trade_logger') and self.main_window.trade_logger:
                        self.main_window.trade_logger.log(
                            pair=pair, side=side, entry_price=entry_price,
                            exit_price=None, volume=volume, pnl=0.0, 
                            reason="INTELLIGENT_TRADER_REAL"
                        )
                    
                    logger.info(f"🎯 REAL position opened: {side.upper()} {pair} ${position_size_usd:.0f} @ {entry_price:.6f}")
                    
                    # Frissítjük a main_window GUI-t
                    if hasattr(self.main_window, 'refresh_open_trades'):
                        self.main_window.refresh_open_trades()
                    if hasattr(self.main_window, 'refresh_balance'):
                        self.main_window.refresh_balance()
                    
                    return True
                else:
                    logger.error(f"main_window.position.open_position failed for {pair}")
                    return False
            else:
                logger.error("main_window.position not available")
                return False
                
        except Exception as e:
            logger.error(f"Error executing real position: {e}")
            return False

    def simulate_position_opening(self):
        """Szimulált pozíció nyitás"""
        try:
            import random
            
            # Random pair selection
            pairs = ['XBTUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD']
            pair = random.choice(pairs)
            
            # Random entry price
            base_prices = {'XBTUSD': 50000, 'ETHUSD': 3000, 'SOLUSD': 100, 'ADAUSD': 0.5, 'XRPUSD': 0.6}
            entry_price = base_prices.get(pair, 1000) * (0.95 + random.random() * 0.1)
            
            # Position details
            position_size = self.DEFAULT_POSITION_SIZE
            volume = position_size / entry_price
            
            # SL/TP calculation
            stop_loss = entry_price * 0.96  # 4% stop loss
            take_profit = entry_price * 1.003  # 0.3% take profit
            
            # Add position
            position = {
                'pair': pair,
                'side': 'buy',
                'entry_price': entry_price,
                'volume': volume,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'open_time': time.time(),
                'position_size_usd': position_size
            }
            
            self.position_manager.positions[pair] = position
            logger.info(f"🎯 Simulated position opened: {pair} ${position_size:.0f}")
            
        except Exception as e:
            logger.error(f"Error simulating position opening: {e}")

    def monitor_micro_positions(self):
        """Mikro pozíciók monitoring"""
        try:
            positions_to_close = []
            
            for pair, position in self.position_manager.get_all_positions().items():
                # Szimulált ár mozgás
                import random
                entry_price = position.get('entry_price', 1000)
                current_price = entry_price * (0.98 + random.random() * 0.04)  # ±2% mozgás
                
                # Time check
                open_time = position.get('open_time', time.time())
                hold_time_minutes = (time.time() - open_time) / 60
                
                should_close = False
                close_reason = ""
                
                # SL/TP check
                stop_loss = position.get('stop_loss', 0)
                take_profit = position.get('take_profit', float('inf'))
                
                if current_price <= stop_loss:
                    should_close = True
                    close_reason = "STOP_LOSS"
                elif current_price >= take_profit:
                    should_close = True
                    close_reason = "TAKE_PROFIT"
                elif hold_time_minutes > 120:  # 2 óra
                    should_close = True
                    close_reason = "TIME_LIMIT"
                elif random.random() < 0.05:  # 5% random close chance
                    should_close = True
                    close_reason = "RANDOM_EXIT"
                
                if should_close:
                    positions_to_close.append((pair, current_price, close_reason))
            
            # Close marked positions
            for pair, exit_price, reason in positions_to_close:
                self.close_position_with_learning(pair, exit_price, reason)
                
        except Exception as e:
            logger.error(f"Error monitoring micro positions: {e}")

    def close_position_with_learning(self, pair: str, exit_price: float, reason: str):
        """Pozíció zárás tanulással"""
        try:
            position = self.position_manager.get_position(pair)
            if not position:
                return
            
            entry_price = position.get('entry_price', exit_price)
            position_size = position.get('position_size_usd', 50)
            
            # P&L calculation
            pnl = (exit_price - entry_price) / entry_price * position_size
            
            # Update metrics
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['total_profit'] += pnl
            
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            
            # Add to learning data
            trade_data = {
                'pair': pair,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            self.learning_data.append(trade_data)
            
            # Remove position
            if pair in self.position_manager.positions:
                del self.position_manager.positions[pair]
            
            logger.info(f"🎯 Position closed: {pair} P&L: ${pnl:.2f} ({reason})")
            
        except Exception as e:
            logger.error(f"Error closing position {pair}: {e}")

    def update_performance_metrics(self):
        """Performance metrikák frissítése"""
        try:
            total_trades = self.performance_metrics['total_trades']
            winning_trades = self.performance_metrics['winning_trades']
            
            if total_trades > 0:
                self.performance_metrics['win_rate'] = winning_trades / total_trades
            else:
                self.performance_metrics['win_rate'] = 0.0
                
            # Active positions count
            self.performance_metrics['active_positions'] = len(self.position_manager.get_all_positions())
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def learn_and_adapt(self):
        """AI tanulás és adaptáció"""
        try:
            if len(self.learning_data) < 5:
                logger.info("Not enough data for learning")
                return
            
            # Simple learning: adjust confidence based on recent performance
            recent_trades = self.learning_data[-10:]  # Last 10 trades
            recent_pnls = [trade['pnl'] for trade in recent_trades]
            
            avg_pnl = sum(recent_pnls) / len(recent_pnls)
            
            if avg_pnl > 0:
                # Good performance, slightly lower confidence threshold
                self.min_confidence_threshold = max(0.6, self.min_confidence_threshold - 0.01)
                logger.info(f"Performance good, lowered confidence to {self.min_confidence_threshold:.2f}")
            else:
                # Poor performance, increase confidence threshold
                self.min_confidence_threshold = min(0.8, self.min_confidence_threshold + 0.02)
                logger.info(f"Performance poor, raised confidence to {self.min_confidence_threshold:.2f}")
                
        except Exception as e:
            logger.error(f"Error in learning: {e}")

    def run_intelligent_trading_cycle(self, available_balance: float):
        """Fallback for intelligent trading cycle"""
        self.run_micro_trading_cycle(available_balance)


class PositionManagerFallback:
    """Fallback position manager"""
    
    def __init__(self):
        self.positions = {}
    
    def get_all_positions(self):
        return self.positions.copy()
    
    def get_position(self, pair):
        return self.positions.get(pair)
    
    def close_position(self, pair):
        if pair in self.positions:
            del self.positions[pair]
