# strategy/micro_strategy_engine.py

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from utils.logger import logger

class TradeSignal(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class MicroTradeSetup:
    """Mikro trade setup configuration"""
    pair: str
    signal: TradeSignal
    confidence: float
    entry_price: float
    
    # Position sizing
    position_size_usd: float
    volume: float
    
    # Risk management
    stop_loss: float
    take_profit: float
    trailing_stop_activation: float  # 0.6 dollár
    trailing_stop_distance: float   # 0.1 dollár (0.6-ról 0.5-re)
    
    # Timing
    max_hold_time: int  # seconds
    entry_urgency: float  # 0-1, how quickly we need to enter
    
    # Scoring
    bollinger_score: float
    correlation_score: float
    volume_score: float
    total_score: float
    
    # Expected outcomes
    expected_profit: float
    risk_reward_ratio: float
    success_probability: float

@dataclass 
class TradeExecution:
    """Trade execution eredmény"""
    setup: MicroTradeSetup
    executed: bool
    execution_time: float
    execution_price: float
    slippage: float
    reason: str

class MicroStrategyEngine:
    """2 perces scalping optimalizált strategy engine"""
    
    def __init__(self):
        # 🎯 MIKRO-TRADING CORE SETTINGS
        self.POSITION_SIZE = 50.0  # $50 pozíció
        self.PROFIT_TARGET = 0.20  # $0.20 profit cél
        self.MAX_LOSS = 2.50      # $2.50 max veszteség
        self.TRAILING_ACTIVATION = 0.60  # $0.60-nál indul a trailing
        self.TRAILING_DISTANCE = 0.10   # $0.10 távolság
        
        # Scalping timeframes (2 perc fókusz)
        self.SCALPING_TIMEFRAME = 120  # 2 perc seconds
        self.MAX_HOLD_TIME = 300      # 5 perc max
        self.MIN_HOLD_TIME = 30       # 30 sec min
        
        # Signal thresholds (BOLLINGER FÓKUSZ!)
        self.MIN_BOLLINGER_SCORE = 0.6   # Min Bollinger kitörési score
        self.MIN_CORRELATION = 0.8       # Min BTC/ETH korreláció
        self.MIN_TOTAL_SCORE = 0.65      # Min össz pontszám
        
        # Entry timing
        self.ENTRY_TIMEOUT = 10  # 10 sec max entry window
        self.SLIPPAGE_TOLERANCE = 0.001  # 0.1% max slippage
        
        # Performance tracking
        self.recent_trades = []
        self.success_rate = 0.0
        self.total_profit = 0.0
        
        logger.info("MicroStrategyEngine initialized for 2-minute scalping")

    def analyze_micro_opportunity(self, coin_analysis, ml_prediction) -> Optional[MicroTradeSetup]:
        """
        Mikro lehetőség elemzés BOLLINGER + 2 PERC FÓKUSSZAL
        """
        try:
            pair = coin_analysis.pair
            current_price = coin_analysis.price
            
            # 1. BOLLINGER BREAKOUT CHECK (KRITIKUS!)
            if coin_analysis.bb_breakout_potential < self.MIN_BOLLINGER_SCORE:
                logger.debug(f"{pair}: Bollinger score too low: {coin_analysis.bb_breakout_potential:.3f}")
                return None
            
            # 2. CORRELATION CHECK (KRITIKUS!)
            max_correlation = max(coin_analysis.btc_correlation, coin_analysis.eth_correlation)
            if max_correlation < self.MIN_CORRELATION:
                logger.debug(f"{pair}: Correlation too low: {max_correlation:.3f}")
                return None
            
            # 3. ML CONFIDENCE CHECK
            if ml_prediction.probability < 0.6:
                logger.debug(f"{pair}: ML probability too low: {ml_prediction.probability:.3f}")
                return None
            
            # 4. SIGNAL GENERATION
            signal = self._generate_scalping_signal(coin_analysis, ml_prediction)
            if signal == TradeSignal.HOLD:
                return None
            
            # 5. POSITION SIZING (ATR alapú)
            position_config = self._calculate_position_size(coin_analysis, current_price)
            
            # 6. RISK MANAGEMENT LEVELS
            risk_levels = self._calculate_micro_risk_levels(
                current_price, position_config['size_usd'], coin_analysis
            )
            
            # 7. TIMING ANALYSIS
            timing = self._analyze_entry_timing(coin_analysis)
            
            # 8. SCORING
            scores = self._calculate_micro_scores(coin_analysis, ml_prediction)
            
            # Final score check
            if scores['total'] < self.MIN_TOTAL_SCORE:
                logger.debug(f"{pair}: Total score too low: {scores['total']:.3f}")
                return None
            
            # 9. CREATE TRADE SETUP
            setup = MicroTradeSetup(
                pair=pair,
                signal=signal,
                confidence=ml_prediction.confidence,
                entry_price=current_price,
                
                position_size_usd=position_config['size_usd'],
                volume=position_config['volume'],
                
                stop_loss=risk_levels['stop_loss'],
                take_profit=risk_levels['take_profit'],
                trailing_stop_activation=risk_levels['trailing_activation'],
                trailing_stop_distance=risk_levels['trailing_distance'],
                
                max_hold_time=timing['max_hold'],
                entry_urgency=timing['urgency'],
                
                bollinger_score=scores['bollinger'],
                correlation_score=scores['correlation'],
                volume_score=scores['volume'],
                total_score=scores['total'],
                
                expected_profit=ml_prediction.expected_return,
                risk_reward_ratio=risk_levels['risk_reward'],
                success_probability=ml_prediction.probability
            )
            
            logger.info(
                f"🎯 Micro opportunity: {pair} - BB:{scores['bollinger']:.2f}, "
                f"Corr:{scores['correlation']:.2f}, Total:{scores['total']:.2f}"
            )
            
            return setup
            
        except Exception as e:
            logger.error(f"Micro opportunity analysis failed for {coin_analysis.pair}: {e}")
            return None

    def _generate_scalping_signal(self, coin_analysis, ml_prediction) -> TradeSignal:
        """2 perces scalping signal generálás"""
        
        # STRONG BUY feltételek (BOLLINGER FÓKUSZ!)
        if (coin_analysis.bb_breakout_potential > 0.8 and
            coin_analysis.bb_position > 0.95 and  # Felső sáv közelében
            coin_analysis.volume_ratio > 2.0 and
            max(coin_analysis.btc_correlation, coin_analysis.eth_correlation) > 0.9 and
            ml_prediction.probability > 0.8):
            return TradeSignal.STRONG_BUY
        
        # BUY feltételek
        if (coin_analysis.bb_breakout_potential > 0.6 and
            (coin_analysis.bb_position > 0.85 or coin_analysis.bb_position < 0.15) and
            coin_analysis.volume_ratio > 1.5 and
            max(coin_analysis.btc_correlation, coin_analysis.eth_correlation) > 0.8 and
            ml_prediction.probability > 0.65):
            return TradeSignal.BUY
        
        # Oversold bounce (alsó Bollinger sáv)
        if (coin_analysis.bb_position < 0.1 and
            coin_analysis.rsi_3m < 35 and
            coin_analysis.volume_ratio > 1.3 and
            ml_prediction.probability > 0.6):
            return TradeSignal.BUY
        
        return TradeSignal.HOLD

    def _calculate_position_size(self, coin_analysis, current_price: float) -> Dict:
        """ATR alapú pozíció méret számítás"""
        
        # Base position size
        base_size = self.POSITION_SIZE
        
        # ATR adjustment (ha elérhető)
        atr_multiplier = 1.0
        if hasattr(coin_analysis, 'atr_ratio'):
            # Ha magas a volatilitás, kisebb pozíció
            if coin_analysis.atr_ratio > 1.5:
                atr_multiplier = 0.8
            elif coin_analysis.atr_ratio < 0.8:
                atr_multiplier = 1.2
        
        # Confidence adjustment
        confidence_multiplier = 0.8 + (coin_analysis.bb_breakout_potential * 0.4)
        
        # Final size
        final_size = base_size * atr_multiplier * confidence_multiplier
        final_size = max(25.0, min(75.0, final_size))  # $25-75 között
        
        volume = final_size / current_price
        
        return {
            'size_usd': final_size,
            'volume': volume,
            'atr_multiplier': atr_multiplier,
            'confidence_multiplier': confidence_multiplier
        }

    def _calculate_micro_risk_levels(self, entry_price: float, position_size: float, 
                                   coin_analysis) -> Dict:
        """Mikro-trading risk level számítás"""
        
        # 1. STOP LOSS (% based for micro-trading)
        if hasattr(coin_analysis, 'atr_ratio') and coin_analysis.atr_ratio > 0:
            # ATR alapú SL (szigorúbb mikro-tradinghez)
            sl_distance = entry_price * 0.035 * coin_analysis.atr_ratio  # 3.5% * ATR
            sl_distance = min(sl_distance, entry_price * 0.05)  # Max 5%
        else:
            # Fix 4% SL
            sl_distance = entry_price * 0.04
        
        stop_loss = entry_price - sl_distance
        
        # 2. TAKE PROFIT (BOLLINGER pozíció alapján)
        if coin_analysis.bb_position > 0.9:
            # Felső sáv közelében - konzervatívabb TP
            tp_target = 0.15  # $0.15
        elif coin_analysis.bb_position < 0.2:
            # Alsó sáv közelében - agresszívebb TP oversold bounce-nál
            tp_target = 0.25  # $0.25
        else:
            # Standard TP
            tp_target = self.PROFIT_TARGET  # $0.20
        
        tp_distance = tp_target / (position_size / entry_price)  # Convert to price
        take_profit = entry_price + tp_distance
        
        # 3. TRAILING STOP
        trailing_activation = entry_price + (self.TRAILING_ACTIVATION / (position_size / entry_price))
        trailing_distance_price = self.TRAILING_DISTANCE / (position_size / entry_price)
        
        # 4. RISK/REWARD
        potential_loss = sl_distance * (position_size / entry_price)
        potential_gain = tp_distance * (position_size / entry_price)
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_activation': trailing_activation,
            'trailing_distance': trailing_distance_price,
            'risk_reward': risk_reward,
            'potential_loss_usd': potential_loss,
            'potential_gain_usd': potential_gain
        }

    def _analyze_entry_timing(self, coin_analysis) -> Dict:
        """Entry timing elemzés 2 perc fókusszal"""
        
        # Base urgency
        urgency = 0.5
        
        # Bollinger breakout urgency (FOKOZOTT!)
        if coin_analysis.bb_breakout_potential > 0.8:
            urgency += 0.3  # High urgency for strong breakouts
        elif coin_analysis.bb_breakout_potential > 0.6:
            urgency += 0.2
        
        # Volume spike urgency
        if coin_analysis.volume_ratio > 3.0:
            urgency += 0.2
        elif coin_analysis.volume_ratio > 2.0:
            urgency += 0.1
        
        # Squeeze urgency
        if coin_analysis.bb_squeeze:
            urgency += 0.15
        
        # Correlation momentum urgency
        if (coin_analysis.btc_correlation > 0.9 or coin_analysis.eth_correlation > 0.9):
            urgency += 0.1
        
        urgency = min(1.0, urgency)
        
        # Max hold time based on volatility and signal strength
        if coin_analysis.bb_breakout_potential > 0.8:
            max_hold = self.SCALPING_TIMEFRAME  # 2 minutes for strong signals
        elif coin_analysis.bb_breakout_potential > 0.6:
            max_hold = self.SCALPING_TIMEFRAME * 1.5  # 3 minutes
        else:
            max_hold = self.MAX_HOLD_TIME  # 5 minutes
        
        return {
            'urgency': urgency,
            'max_hold': int(max_hold),
            'entry_window': int(self.ENTRY_TIMEOUT / urgency) if urgency > 0 else self.ENTRY_TIMEOUT
        }

    def _calculate_micro_scores(self, coin_analysis, ml_prediction) -> Dict:
        """Mikro-trading scoring BOLLINGER FÓKUSSZAL"""
        
        # 1. BOLLINGER SCORE (40% súly!)
        bollinger_score = coin_analysis.bb_breakout_potential
        
        # Bollinger position bonus
        if coin_analysis.bb_position > 0.95 or coin_analysis.bb_position < 0.05:
            bollinger_score += 0.1
        
        # Squeeze bonus
        if coin_analysis.bb_squeeze:
            bollinger_score += 0.15
        
        bollinger_score = min(1.0, bollinger_score)
        
        # 2. CORRELATION SCORE (30% súly)
        correlation_score = max(coin_analysis.btc_correlation, coin_analysis.eth_correlation)
        
        # High correlation bonus
        if correlation_score > 0.95:
            correlation_score += 0.05
        
        correlation_score = min(1.0, correlation_score)
        
        # 3. VOLUME SCORE (20% súly)
        volume_score = min(1.0, coin_analysis.volume_ratio / 4.0)  # 4x volume = max score
        
        # Volume trend bonus
        if hasattr(coin_analysis, 'volume_trend') and coin_analysis.volume_trend > 0.1:
            volume_score += 0.1
        
        volume_score = min(1.0, volume_score)
        
        # 4. ML SCORE (10% súly)
        ml_score = ml_prediction.probability
        
        # TOTAL SCORE (súlyozott)
        total_score = (
            bollinger_score * 0.40 +
            correlation_score * 0.30 +
            volume_score * 0.20 +
            ml_score * 0.10
        )
        
        return {
            'bollinger': bollinger_score,
            'correlation': correlation_score,
            'volume': volume_score,
            'ml': ml_score,
            'total': total_score
        }

    def execute_micro_trade(self, setup: MicroTradeSetup, market_data: Dict) -> TradeExecution:
        """
        Mikro trade végrehajtás (szimuláció)
        """
        try:
            start_time = time.time()
            
            # 1. ENTRY WINDOW CHECK
            if setup.entry_urgency > 0.8:
                max_wait = 5  # 5 sec for urgent entries
            else:
                max_wait = self.ENTRY_TIMEOUT
            
            # 2. SLIPPAGE SIMULATION
            current_price = market_data.get('current_price', setup.entry_price)
            price_movement = abs(current_price - setup.entry_price) / setup.entry_price
            
            if price_movement > self.SLIPPAGE_TOLERANCE:
                return TradeExecution(
                    setup=setup,
                    executed=False,
                    execution_time=start_time,
                    execution_price=current_price,
                    slippage=price_movement,
                    reason=f"Excessive slippage: {price_movement:.4f}"
                )
            
            # 3. LIQUIDITY CHECK
            if not self._check_liquidity(setup.pair, setup.volume):
                return TradeExecution(
                    setup=setup,
                    executed=False,
                    execution_time=start_time,
                    execution_price=current_price,
                    slippage=0.0,
                    reason="Insufficient liquidity"
                )
            
            # 4. FINAL CONFIRMATION
            if not self._final_entry_confirmation(setup, current_price):
                return TradeExecution(
                    setup=setup,
                    executed=False,
                    execution_time=start_time,
                    execution_price=current_price,
                    slippage=price_movement,
                    reason="Final confirmation failed"
                )
            
            # 5. EXECUTE TRADE
            execution_price = current_price * (1 + price_movement * 0.5)  # Partial slippage
            
            logger.info(
                f"🎯 EXECUTED: {setup.pair} {setup.signal.value} ${setup.position_size_usd:.0f} "
                f"@ {execution_price:.6f} (score: {setup.total_score:.3f})"
            )
            
            return TradeExecution(
                setup=setup,
                executed=True,
                execution_time=start_time,
                execution_price=execution_price,
                slippage=price_movement,
                reason="Successfully executed"
            )
            
        except Exception as e:
            logger.error(f"Trade execution failed for {setup.pair}: {e}")
            return TradeExecution(
                setup=setup,
                executed=False,
                execution_time=time.time(),
                execution_price=setup.entry_price,
                slippage=0.0,
                reason=f"Execution error: {e}"
            )

    def _check_liquidity(self, pair: str, volume: float) -> bool:
        """Liquidity check (mock)"""
        # Mock implementation - replace with real order book analysis
        return volume < 10.0  # Assume liquidity OK for small volumes

    def _final_entry_confirmation(self, setup: MicroTradeSetup, current_price: float) -> bool:
        """Final entry confirmation"""
        
        # Price movement check
        price_change = abs(current_price - setup.entry_price) / setup.entry_price
        if price_change > 0.005:  # 0.5% max movement
            return False
        
        # Time-based confirmation (example: don't enter near resistance without strong signal)
        if setup.signal == TradeSignal.BUY and setup.bollinger_score < 0.7:
            # Need higher confidence for regular BUY signals
            return setup.total_score > 0.7
        
        return True

    def monitor_active_position(self, position_data: Dict, current_price: float) -> Dict:
        """
        Aktív pozíció monitoring trailing stop-pal
        """
        try:
            entry_price = position_data['entry_price']
            position_size = position_data['position_size_usd']
            stop_loss = position_data['stop_loss']
            take_profit = position_data['take_profit']
            trailing_activation = position_data.get('trailing_activation', 0)
            trailing_distance = position_data.get('trailing_distance', 0)
            
            # Current P&L
            current_pnl = (current_price - entry_price) / entry_price * position_size
            
            # 1. STOP LOSS CHECK
            if current_price <= stop_loss:
                return {
                    'action': 'CLOSE',
                    'reason': 'STOP_LOSS',
                    'pnl': current_pnl,
                    'exit_price': current_price
                }
            
            # 2. TAKE PROFIT CHECK
            if current_price >= take_profit:
                return {
                    'action': 'CLOSE',
                    'reason': 'TAKE_PROFIT',
                    'pnl': current_pnl,
                    'exit_price': current_price
                }
            
            # 3. TRAILING STOP LOGIC
            max_price = position_data.get('max_price_seen', entry_price)
            if current_price > max_price:
                max_price = current_price
                position_data['max_price_seen'] = max_price
            
            # Trailing stop activation ($0.60 profit)
            if max_price >= trailing_activation:
                trailing_stop_price = max_price - trailing_distance
                
                if current_price <= trailing_stop_price:
                    return {
                        'action': 'CLOSE',
                        'reason': 'TRAILING_STOP',
                        'pnl': current_pnl,
                        'exit_price': current_price,
                        'trailing_activated': True
                    }
                
                # Update stop loss to trailing level
                position_data['stop_loss'] = max(stop_loss, trailing_stop_price)
            
            # 4. TIME-BASED EXIT
            hold_time = time.time() - position_data.get('entry_time', time.time())
            max_hold = position_data.get('max_hold_time', self.MAX_HOLD_TIME)
            
            if hold_time > max_hold:
                return {
                    'action': 'CLOSE',
                    'reason': 'TIME_LIMIT',
                    'pnl': current_pnl,
                    'exit_price': current_price,
                    'hold_time': hold_time
                }
            
            # 5. PROFIT PROTECTION (ha eléri a 0.6-ot, ne essen 0.5 alá)
            profit_protection_threshold = 0.50  # $0.50
            if (max_price >= trailing_activation and 
                current_pnl < profit_protection_threshold):
                return {
                    'action': 'CLOSE',
                    'reason': 'PROFIT_PROTECTION',
                    'pnl': current_pnl,
                    'exit_price': current_price
                }
            
            return {
                'action': 'HOLD',
                'pnl': current_pnl,
                'max_price': max_price,
                'trailing_active': max_price >= trailing_activation,
                'hold_time': hold_time
            }
            
        except Exception as e:
            logger.error(f"Position monitoring failed: {e}")
            return {'action': 'HOLD', 'pnl': 0.0}

    def update_performance_metrics(self, trade_result: Dict):
        """Teljesítmény metrikák frissítése"""
        try:
            self.recent_trades.append({
                'timestamp': time.time(),
                'pair': trade_result.get('pair'),
                'pnl': trade_result.get('pnl', 0),
                'success': trade_result.get('pnl', 0) > 0,
                'reason': trade_result.get('reason'),
                'hold_time': trade_result.get('hold_time', 0)
            })
            
            # Keep only last 100 trades
            if len(self.recent_trades) > 100:
                self.recent_trades = self.recent_trades[-100:]
            
            # Calculate metrics
            if self.recent_trades:
                successful_trades = [t for t in self.recent_trades if t['success']]
                self.success_rate = len(successful_trades) / len(self.recent_trades)
                self.total_profit = sum(t['pnl'] for t in self.recent_trades)
            
            logger.info(
                f"📊 Performance: {len(self.recent_trades)} trades, "
                f"{self.success_rate:.1%} success, ${self.total_profit:.2f} total"
            )
            
        except Exception as e:
            logger.error(f"Performance update failed: {e}")

    def get_strategy_status(self) -> Dict:
        """Strategy státusz"""
        return {
            'name': 'MicroScalpingStrategy',
            'timeframe': '2min',
            'position_size': f"${self.POSITION_SIZE}",
            'profit_target': f"${self.PROFIT_TARGET}",
            'trailing_activation': f"${self.TRAILING_ACTIVATION}",
            'min_bollinger_score': self.MIN_BOLLINGER_SCORE,
            'min_correlation': self.MIN_CORRELATION,
            'recent_trades': len(self.recent_trades),
            'success_rate': f"{self.success_rate:.1%}",
            'total_profit': f"${self.total_profit:.2f}",
            'bollinger_focus': True
        }

    def optimize_parameters(self):
        """Paraméter optimalizálás based on recent performance"""
        try:
            if len(self.recent_trades) < 20:
                return
            
            # Analyze recent performance
            recent_20 = self.recent_trades[-20:]
            recent_success_rate = sum(1 for t in recent_20 if t['success']) / len(recent_20)
            
            # Adjust thresholds based on performance
            if recent_success_rate > 0.7:
                # Good performance - can be slightly more aggressive
                self.MIN_BOLLINGER_SCORE = max(0.5, self.MIN_BOLLINGER_SCORE - 0.05)
                self.MIN_TOTAL_SCORE = max(0.6, self.MIN_TOTAL_SCORE - 0.02)
            elif recent_success_rate < 0.4:
                # Poor performance - be more conservative
                self.MIN_BOLLINGER_SCORE = min(0.8, self.MIN_BOLLINGER_SCORE + 0.05)
                self.MIN_TOTAL_SCORE = min(0.8, self.MIN_TOTAL_SCORE + 0.02)
            
            logger.info(f"📈 Parameters optimized based on {recent_success_rate:.1%} recent success rate")
            
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")

    def backtest_strategy(self, historical_data: List[Dict]) -> Dict:
        """Strategy backtesting"""
        # Simplified backtest implementation
        simulated_trades = 0
        successful_trades = 0
        total_pnl = 0.0
        
        for data_point in historical_data:
            # Mock backtest logic
            if data_point.get('bb_breakout_potential', 0) > 0.6:
                simulated_trades += 1
                
                # Simulate trade outcome based on historical data
                profit = np.random.normal(0.15, 0.10)  # Mean $0.15, std $0.10
                total_pnl += profit
                
                if profit > 0:
                    successful_trades += 1
        
        return {
            'total_trades': simulated_trades,
            'successful_trades': successful_trades,
            'win_rate': successful_trades / simulated_trades if simulated_trades > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / simulated_trades if simulated_trades > 0 else 0
        }
