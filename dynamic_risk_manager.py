# core/dynamic_risk_manager.py

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math
from utils.logger import logger

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

@dataclass
class PositionRisk:
    """Pozíció kockázati adatok"""
    pair: str
    position_size_usd: float
    entry_price: float
    current_price: float
    
    # ATR alapú metrikák
    atr_value: float
    atr_ratio: float  # current ATR vs historical avg
    volatility_percentile: float
    
    # Correlation kockázat
    btc_correlation: float
    eth_correlation: float
    correlation_risk: float
    
    # Position specifikus
    unrealized_pnl: float
    unrealized_pnl_pct: float
    max_favorable_excursion: float
    max_adverse_excursion: float
    
    # Dynamic stops
    dynamic_stop_loss: float
    dynamic_take_profit: float
    trailing_stop_price: float
    
    # Risk metrics
    value_at_risk: float  # VaR
    risk_level: RiskLevel
    recommended_action: str

@dataclass
class PortfolioRisk:
    """Portfólió szintű kockázat"""
    total_exposure: float
    used_buying_power: float
    available_balance: float
    
    # Correlation matrix
    avg_correlation: float
    max_correlation: float
    correlation_concentration: float
    
    # Portfolio metrics
    portfolio_var: float
    max_drawdown_risk: float
    daily_pnl: float
    
    # Risk limits
    risk_utilization: float  # 0-1
    position_concentration: float
    
    overall_risk_level: RiskLevel
    risk_recommendations: List[str]

class DynamicRiskManager:
    """ATR alapú dinamikus kockázatkezelő"""
    
    def __init__(self):
        # 🎯 MIKRO-TRADING RISK LIMITS
        self.MAX_POSITION_SIZE = 66.0       # Max $66 pozíció ($66 = teljes egyenleg)
        self.MAX_PORTFOLIO_RISK = 0.15      # Max 15% portfolio kockázat
        self.MAX_DAILY_LOSS = 10.0          # Max $10 napi veszteség
        self.MAX_CORRELATION = 0.95         # Max korreláció párok között
        
        # ATR settings
        self.ATR_PERIOD = 14
        self.ATR_MULTIPLIER_SL = 2.2        # SL = 2.2 * ATR
        self.ATR_MULTIPLIER_TP = 1.8        # TP = 1.8 * ATR
        self.VOLATILITY_LOOKBACK = 30       # 30 day volatility lookback
        
        # Dynamic risk adjustment
        self.RISK_SCALE_FACTOR = 0.8        # Scale down position size in high vol
        self.CORRELATION_PENALTY = 0.2      # Reduce size by 20% for high correlation
        
        # Position limits
        self.MAX_CONCURRENT_POSITIONS = 1   # Only 1 position with $66
        self.MIN_POSITION_SIZE = 25.0       # Min $25
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.max_daily_drawdown = 0.0
        self.risk_events = []
        
        # Volatility cache
        self.volatility_cache = {}
        self.correlation_cache = {}
        
        logger.info("DynamicRiskManager initialized for $66 micro-trading")

    def calculate_position_size(self, pair: str, entry_price: float, 
                              account_balance: float, coin_analysis) -> Dict:
        """
        ATR alapú dinamikus pozíció méret számítás
        """
        try:
            # 1. BASE POSITION SIZE
            base_size = min(50.0, account_balance * 0.75)  # Max 75% of balance
            
            # 2. ATR ADJUSTMENT
            atr_data = self._calculate_atr_metrics(pair, entry_price, coin_analysis)
            
            # Volatility scaling
            vol_percentile = atr_data['volatility_percentile']
            if vol_percentile > 80:  # High volatility
                vol_multiplier = 0.7
            elif vol_percentile > 60:
                vol_multiplier = 0.85
            elif vol_percentile < 20:  # Low volatility
                vol_multiplier = 1.2
            else:
                vol_multiplier = 1.0
            
            # 3. CORRELATION ADJUSTMENT
            correlation_adjustment = self._calculate_correlation_adjustment(pair, coin_analysis)
            
            # 4. BOLLINGER CONFIDENCE BOOST
            confidence_multiplier = 1.0
            if hasattr(coin_analysis, 'bb_breakout_potential'):
                if coin_analysis.bb_breakout_potential > 0.8:
                    confidence_multiplier = 1.1  # 10% boost for strong signals
                elif coin_analysis.bb_breakout_potential > 0.6:
                    confidence_multiplier = 1.05
            
            # 5. ACCOUNT HEALTH CHECK
            account_multiplier = self._calculate_account_health_multiplier(account_balance)
            
            # 6. FINAL CALCULATION
            final_size = (base_size * 
                         vol_multiplier * 
                         correlation_adjustment * 
                         confidence_multiplier * 
                         account_multiplier)
            
            # 7. POSITION SIZE LIMITS
            final_size = max(self.MIN_POSITION_SIZE, 
                           min(self.MAX_POSITION_SIZE, final_size))
            
            # If balance is less than position size, use maximum safe amount
            if final_size > account_balance * 0.9:
                final_size = account_balance * 0.8
            
            volume = final_size / entry_price
            
            # 8. RISK METRICS
            stop_loss_distance = atr_data['atr'] * self.ATR_MULTIPLIER_SL
            stop_loss_price = entry_price - stop_loss_distance
            max_loss_usd = stop_loss_distance * volume
            
            risk_metrics = {
                'position_size_usd': final_size,
                'volume': volume,
                'max_loss_usd': max_loss_usd,
                'risk_pct': (max_loss_usd / account_balance) * 100,
                'vol_multiplier': vol_multiplier,
                'correlation_adj': correlation_adjustment,
                'confidence_boost': confidence_multiplier,
                'atr_value': atr_data['atr'],
                'stop_loss_price': stop_loss_price
            }
            
            logger.info(
                f"💰 Position sizing for {pair}: ${final_size:.0f} "
                f"(Vol: {vol_multiplier:.2f}, Corr: {correlation_adjustment:.2f}, "
                f"Risk: {risk_metrics['risk_pct']:.1f}%)"
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Position sizing failed for {pair}: {e}")
            # Fallback safe sizing
            safe_size = min(40.0, account_balance * 0.6)
            return {
                'position_size_usd': safe_size,
                'volume': safe_size / entry_price,
                'max_loss_usd': safe_size * 0.04,  # 4% max loss
                'risk_pct': 4.0,
                'error': str(e)
            }

    def calculate_dynamic_stops(self, pair: str, entry_price: float, 
                              position_size: float, coin_analysis) -> Dict:
        """
        ATR alapú dinamikus stop loss és take profit
        """
        try:
            # 1. ATR CALCULATION
            atr_data = self._calculate_atr_metrics(pair, entry_price, coin_analysis)
            atr = atr_data['atr']
            
            # 2. DYNAMIC STOP LOSS
            # Base ATR stop
            atr_stop_distance = atr * self.ATR_MULTIPLIER_SL
            
            # Bollinger adjustment
            if hasattr(coin_analysis, 'bb_position'):
                if coin_analysis.bb_position > 0.9:  # Near upper band
                    atr_stop_distance *= 0.8  # Tighter stop
                elif coin_analysis.bb_position < 0.1:  # Near lower band
                    atr_stop_distance *= 1.2  # Wider stop for bounce
            
            # Volatility adjustment
            if atr_data['volatility_percentile'] > 80:
                atr_stop_distance *= 1.3  # Wider stop in high vol
            elif atr_data['volatility_percentile'] < 20:
                atr_stop_distance *= 0.8  # Tighter stop in low vol
            
            stop_loss_price = entry_price - atr_stop_distance
            
            # 3. DYNAMIC TAKE PROFIT
            # Base take profit (targeting $0.20 profit)
            target_profit_usd = 0.20
            volume = position_size / entry_price
            tp_distance = target_profit_usd / volume
            
            # ATR-based adjustment
            atr_tp_distance = atr * self.ATR_MULTIPLIER_TP
            
            # Use larger of the two (ensure minimum profit target)
            final_tp_distance = max(tp_distance, atr_tp_distance)
            take_profit_price = entry_price + final_tp_distance
            
            # 4. TRAILING STOP PARAMETERS
            trailing_activation = entry_price + (0.60 / volume)  # $0.60 activation
            trailing_distance = atr * 1.5  # 1.5 ATR trailing distance
            
            # 5. RISK/REWARD VALIDATION
            potential_loss = atr_stop_distance * volume
            potential_gain = final_tp_distance * volume
            risk_reward_ratio = potential_gain / potential_loss if potential_loss > 0 else 0
            
            # If R/R is too poor, adjust
            if risk_reward_ratio < 0.3:  # Less than 1:3 R/R
                # Widen TP or tighten SL
                final_tp_distance = atr_stop_distance * 0.4  # At least 1:2.5 R/R
                take_profit_price = entry_price + final_tp_distance
                risk_reward_ratio = 0.4
            
            return {
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'trailing_activation': trailing_activation,
                'trailing_distance': trailing_distance,
                'atr_value': atr,
                'stop_distance_pct': (atr_stop_distance / entry_price) * 100,
                'tp_distance_pct': (final_tp_distance / entry_price) * 100,
                'risk_reward_ratio': risk_reward_ratio,
                'potential_loss_usd': potential_loss,
                'potential_gain_usd': potential_gain,
                'volatility_percentile': atr_data['volatility_percentile']
            }
            
        except Exception as e:
            logger.error(f"Dynamic stops calculation failed for {pair}: {e}")
            # Fallback fixed percentage stops
            return {
                'stop_loss': entry_price * 0.96,    # 4% SL
                'take_profit': entry_price * 1.004, # 0.4% TP
                'trailing_activation': entry_price * 1.012,  # 1.2% activation
                'trailing_distance': entry_price * 0.002,    # 0.2% distance
                'risk_reward_ratio': 0.1,
                'error': str(e)
            }

    def assess_position_risk(self, position_data: Dict, current_price: float) -> PositionRisk:
        """Pozíció kockázat értékelés"""
        try:
            pair = position_data['pair']
            entry_price = position_data['entry_price']
            position_size = position_data['position_size_usd']
            
            # P&L calculations
            unrealized_pnl = (current_price - entry_price) / entry_price * position_size
            unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100
            
            # Max excursions
            max_price = position_data.get('max_price_seen', current_price)
            min_price = position_data.get('min_price_seen', current_price)
            
            max_favorable = (max_price - entry_price) / entry_price * position_size
            max_adverse = (entry_price - min_price) / entry_price * position_size
            
            # ATR analysis
            atr_metrics = self._get_cached_atr(pair, current_price)
            
            # Correlation risk
            correlation_risk = self._assess_correlation_risk(pair)
            
            # VaR calculation (1-day, 95% confidence)
            daily_vol = atr_metrics.get('daily_volatility', 0.02)
            var_95 = position_size * daily_vol * 1.65  # 95% VaR
            
            # Risk level assessment
            risk_level = self._determine_risk_level(
                unrealized_pnl_pct, var_95, position_size, correlation_risk
            )
            
            # Recommended action
            action = self._get_risk_recommendation(
                risk_level, unrealized_pnl, unrealized_pnl_pct, position_data
            )
            
            return PositionRisk(
                pair=pair,
                position_size_usd=position_size,
                entry_price=entry_price,
                current_price=current_price,
                
                atr_value=atr_metrics.get('atr', 0),
                atr_ratio=atr_metrics.get('atr_ratio', 1.0),
                volatility_percentile=atr_metrics.get('volatility_percentile', 50),
                
                btc_correlation=correlation_risk.get('btc_correlation', 0.7),
                eth_correlation=correlation_risk.get('eth_correlation', 0.6),
                correlation_risk=correlation_risk.get('risk_score', 0.5),
                
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                max_favorable_excursion=max_favorable,
                max_adverse_excursion=max_adverse,
                
                dynamic_stop_loss=position_data.get('stop_loss', entry_price * 0.96),
                dynamic_take_profit=position_data.get('take_profit', entry_price * 1.004),
                trailing_stop_price=position_data.get('trailing_stop', 0),
                
                value_at_risk=var_95,
                risk_level=risk_level,
                recommended_action=action
            )
            
        except Exception as e:
            logger.error(f"Position risk assessment failed: {e}")
            # Return safe default
            return PositionRisk(
                pair=position_data.get('pair', 'UNKNOWN'),
                position_size_usd=position_data.get('position_size_usd', 50),
                entry_price=position_data.get('entry_price', current_price),
                current_price=current_price,
                atr_value=0, atr_ratio=1.0, volatility_percentile=50,
                btc_correlation=0.7, eth_correlation=0.6, correlation_risk=0.5,
                unrealized_pnl=0, unrealized_pnl_pct=0,
                max_favorable_excursion=0, max_adverse_excursion=0,
                dynamic_stop_loss=current_price * 0.96, dynamic_take_profit=current_price * 1.004,
                trailing_stop_price=0, value_at_risk=2.0, risk_level=RiskLevel.MEDIUM,
                recommended_action="MONITOR"
            )

    def assess_portfolio_risk(self, positions: List[Dict], account_balance: float) -> PortfolioRisk:
        """Portfólió szintű kockázat értékelés"""
        try:
            if not positions:
                return PortfolioRisk(
                    total_exposure=0, used_buying_power=0, available_balance=account_balance,
                    avg_correlation=0, max_correlation=0, correlation_concentration=0,
                    portfolio_var=0, max_drawdown_risk=0, daily_pnl=0,
                    risk_utilization=0, position_concentration=0,
                    overall_risk_level=RiskLevel.LOW, risk_recommendations=[]
                )
            
            # Portfolio metrics
            total_exposure = sum(pos.get('position_size_usd', 0) for pos in positions)
            used_buying_power = total_exposure / account_balance if account_balance > 0 else 0
            
            # Correlation analysis
            correlations = []
            for pos in positions:
                pair = pos.get('pair', '')
                corr_data = self._get_cached_correlation(pair)
                correlations.extend([corr_data.get('btc_correlation', 0.7), 
                                   corr_data.get('eth_correlation', 0.6)])
            
            avg_correlation = np.mean(correlations) if correlations else 0
            max_correlation = max(correlations) if correlations else 0
            
            # Portfolio VaR (simplified)
            portfolio_var = 0
            for pos in positions:
                pos_size = pos.get('position_size_usd', 0)
                daily_vol = 0.02  # 2% daily volatility assumption
                pos_var = pos_size * daily_vol * 1.65  # 95% VaR
                portfolio_var += pos_var
            
            # Diversification adjustment (if multiple positions)
            if len(positions) > 1:
                diversification_factor = 1 - (avg_correlation * 0.3)
                portfolio_var *= diversification_factor
            
            # Risk utilization
            risk_utilization = used_buying_power
            
            # Position concentration
            if positions:
                largest_position = max(pos.get('position_size_usd', 0) for pos in positions)
                position_concentration = largest_position / total_exposure if total_exposure > 0 else 0
            else:
                position_concentration = 0
            
            # Overall risk level
            overall_risk = self._determine_portfolio_risk_level(
                risk_utilization, position_concentration, avg_correlation, portfolio_var, account_balance
            )
            
            # Risk recommendations
            recommendations = self._generate_portfolio_recommendations(
                risk_utilization, position_concentration, avg_correlation, len(positions)
            )
            
            return PortfolioRisk(
                total_exposure=total_exposure,
                used_buying_power=used_buying_power,
                available_balance=account_balance - total_exposure,
                
                avg_correlation=avg_correlation,
                max_correlation=max_correlation,
                correlation_concentration=max_correlation,
                
                portfolio_var=portfolio_var,
                max_drawdown_risk=portfolio_var / account_balance if account_balance > 0 else 0,
                daily_pnl=self.daily_pnl,
                
                risk_utilization=risk_utilization,
                position_concentration=position_concentration,
                
                overall_risk_level=overall_risk,
                risk_recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            return PortfolioRisk(
                total_exposure=0, used_buying_power=0, available_balance=account_balance,
                avg_correlation=0, max_correlation=0, correlation_concentration=0,
                portfolio_var=0, max_drawdown_risk=0, daily_pnl=0,
                risk_utilization=0, position_concentration=0,
                overall_risk_level=RiskLevel.MEDIUM,
                risk_recommendations=["Error in risk assessment"]
            )

    def update_dynamic_stops(self, position_data: Dict, current_price: float, 
                           market_conditions: Dict) -> Dict:
        """Dinamikus stop-ok frissítése"""
        try:
            pair = position_data['pair']
            entry_price = position_data['entry_price']
            current_sl = position_data.get('stop_loss', entry_price * 0.96)
            current_tp = position_data.get('take_profit', entry_price * 1.004)
            
            # ATR-based adjustment
            atr_data = self._get_cached_atr(pair, current_price)
            atr = atr_data.get('atr', current_price * 0.02)
            
            # Dynamic SL adjustment based on market conditions
            vol_percentile = atr_data.get('volatility_percentile', 50)
            
            # If volatility increased significantly, widen stops
            if vol_percentile > 85:
                new_sl_distance = atr * 2.8  # Wider stop
            elif vol_percentile < 15:
                new_sl_distance = atr * 1.8  # Tighter stop
            else:
                new_sl_distance = atr * 2.2  # Standard
            
            new_stop_loss = current_price - new_sl_distance
            
            # Never move SL against us (only trail up)
            if new_stop_loss > current_sl:
                updated_sl = new_stop_loss
                sl_updated = True
            else:
                updated_sl = current_sl
                sl_updated = False
            
            # TP adjustment based on momentum
            momentum = market_conditions.get('momentum', 0)
            if momentum > 0.01:  # Strong momentum
                # Extend TP slightly
                tp_extension = atr * 0.5
                new_take_profit = current_tp + tp_extension
            else:
                new_take_profit = current_tp
            
            # Trailing stop logic
            max_price = position_data.get('max_price_seen', entry_price)
            if current_price > max_price:
                max_price = current_price
                position_data['max_price_seen'] = max_price
            
            # Calculate trailing stop
            profit_usd = (current_price - entry_price) / entry_price * position_data.get('position_size_usd', 50)
            
            trailing_stop_price = 0
            trailing_active = False
            
            if profit_usd >= 0.60:  # $0.60 activation threshold
                trailing_distance = 0.10 / (position_data.get('position_size_usd', 50) / current_price)
                trailing_stop_price = max_price - trailing_distance
                trailing_active = True
                
                # Use trailing stop if it's better than regular SL
                if trailing_stop_price > updated_sl:
                    updated_sl = trailing_stop_price
                    sl_updated = True
            
            return {
                'stop_loss': updated_sl,
                'take_profit': new_take_profit,
                'trailing_stop_price': trailing_stop_price,
                'trailing_active': trailing_active,
                'sl_updated': sl_updated,
                'max_price_seen': max_price,
                'atr_used': atr,
                'volatility_adjustment': vol_percentile
            }
            
        except Exception as e:
            logger.error(f"Dynamic stops update failed: {e}")
            return position_data  # Return unchanged

    # Helper methods...
    def _calculate_atr_metrics(self, pair: str, current_price: float, coin_analysis) -> Dict:
        """ATR metrikák számítás"""
        try:
            # Mock ATR calculation - replace with real data
            base_atr = current_price * 0.025  # 2.5% of price as base ATR
            
            # Adjust based on market conditions
            if hasattr(coin_analysis, 'volume_ratio'):
                if coin_analysis.volume_ratio > 3.0:
                    atr_multiplier = 1.3  # Higher volatility
                elif coin_analysis.volume_ratio < 0.7:
                    atr_multiplier = 0.8  # Lower volatility
                else:
                    atr_multiplier = 1.0
            else:
                atr_multiplier = 1.0
            
            atr = base_atr * atr_multiplier
            
            # Historical ATR comparison (mock)
            import random
            atr_ratio = random.uniform(0.8, 1.3)
            volatility_percentile = random.uniform(20, 80)
            daily_volatility = atr / current_price  # Daily vol as fraction
            
            result = {
                'atr': atr,
                'atr_ratio': atr_ratio,
                'volatility_percentile': volatility_percentile,
                'daily_volatility': daily_volatility,
                'atr_multiplier': atr_multiplier
            }
            
            # Cache result
            self.volatility_cache[pair] = {
                'data': result,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"ATR calculation failed for {pair}: {e}")
            return {
                'atr': current_price * 0.02,
                'atr_ratio': 1.0,
                'volatility_percentile': 50,
                'daily_volatility': 0.02
            }

    def _calculate_correlation_adjustment(self, pair: str, coin_analysis) -> float:
        """Korreláció alapú pozíció méret korrekció"""
        try:
            btc_corr = getattr(coin_analysis, 'btc_correlation', 0.7)
            eth_corr = getattr(coin_analysis, 'eth_correlation', 0.6)
            
            max_corr = max(btc_corr, eth_corr)
            
            # High correlation = reduce position size
            if max_corr > 0.95:
                return 0.7  # 30% reduction
            elif max_corr > 0.9:
                return 0.85  # 15% reduction
            elif max_corr > 0.85:
                return 0.95  # 5% reduction
            else:
                return 1.0  # No adjustment
                
        except Exception as e:
            logger.error(f"Correlation adjustment failed: {e}")
            return 1.0

    def _calculate_account_health_multiplier(self, account_balance: float) -> float:
        """Számla egészség alapú szorzó"""
        
        # Daily P&L impact
        daily_loss_pct = abs(self.daily_pnl) / account_balance if account_balance > 0 else 0
        
        if daily_loss_pct > 0.15:  # More than 15% daily loss
            return 0.5  # Halve position sizes
        elif daily_loss_pct > 0.10:  # More than 10% daily loss
            return 0.7  # Reduce position sizes
        elif daily_loss_pct > 0.05:  # More than 5% daily loss
            return 0.85  # Slightly reduce
        
        # Account size impact
        if account_balance < 50:
            return 0.8  # More conservative with small accounts
        elif account_balance > 200:
            return 1.1  # Slightly more aggressive with larger accounts
        
        return 1.0

    def _get_cached_atr(self, pair: str, current_price: float) -> Dict:
        """Cached ATR adatok lekérése"""
        cache_key = pair
        if cache_key in self.volatility_cache:
            cached = self.volatility_cache[cache_key]
            if time.time() - cached['timestamp'] < 300:  # 5 min cache
                return cached['data']
        
        # Generate fresh ATR data
        return self._calculate_atr_metrics(pair, current_price, None)

    def _assess_correlation_risk(self, pair: str) -> Dict:
        """Korreláció kockázat értékelés"""
        # Mock implementation
        import random
        return {
            'btc_correlation': random.uniform(0.6, 0.95),
            'eth_correlation': random.uniform(0.5, 0.9),
            'risk_score': random.uniform(0.3, 0.8)
        }

    def _get_cached_correlation(self, pair: str) -> Dict:
        """Cached korreláció adatok"""
        return self._assess_correlation_risk(pair)

    def _determine_risk_level(self, pnl_pct: float, var: float, 
                            position_size: float, corr_risk: float) -> RiskLevel:
        """Kockázati szint meghatározás"""
        
        # Multiple factors
        risk_score = 0
        
        # P&L based
        if pnl_pct < -8:  # More than 8% loss
            risk_score += 3
        elif pnl_pct < -5:
            risk_score += 2
        elif pnl_pct < -2:
            risk_score += 1
        
        # VaR based
        var_pct = (var / position_size) * 100
        if var_pct > 10:
            risk_score += 2
        elif var_pct > 5:
            risk_score += 1
        
        # Correlation risk
        if corr_risk > 0.8:
            risk_score += 1
        
        # Determine level
        if risk_score >= 4:
            return RiskLevel.EXTREME
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _get_risk_recommendation(self, risk_level: RiskLevel, pnl: float, 
                               pnl_pct: float, position_data: Dict) -> str:
        """Kockázat alapú ajánlás"""
        
        if risk_level == RiskLevel.EXTREME:
            return "CLOSE_IMMEDIATELY"
        elif risk_level == RiskLevel.HIGH:
            if pnl < -3.0:  # More than $3 loss
                return "CLOSE_POSITION"
            else:
                return "TIGHTEN_STOPS"
        elif risk_level == RiskLevel.MEDIUM:
            if pnl > 0:
                return "TRAIL_STOPS"
            else:
                return "MONITOR_CLOSELY"
        else:
            return "MONITOR"

    def _determine_portfolio_risk_level(self, risk_util: float, pos_conc: float, 
                                      avg_corr: float, portfolio_var: float, 
                                      account_balance: float) -> RiskLevel:
        """Portfólió kockázati szint"""
        
        risk_score = 0
        
        if risk_util > 0.9:  # More than 90% utilization
            risk_score += 2
        elif risk_util > 0.7:
            risk_score += 1
        
        if pos_conc > 0.8:  # High concentration
            risk_score += 1
        
        if avg_corr > 0.9:  # High correlation
            risk_score += 1
        
        var_pct = (portfolio_var / account_balance) * 100 if account_balance > 0 else 0
        if var_pct > 15:
            risk_score += 2
        elif var_pct > 10:
            risk_score += 1
        
        if risk_score >= 4:
            return RiskLevel.EXTREME
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_portfolio_recommendations(self, risk_util: float, pos_conc: float, 
                                          avg_corr: float, num_positions: int) -> List[str]:
        """Portfólió ajánlások generálás"""
        recommendations = []
        
        if risk_util > 0.85:
            recommendations.append("Reduce position sizes - high capital utilization")
        
        if pos_conc > 0.9:
            recommendations.append("Diversify positions - high concentration risk")
        
        if avg_corr > 0.9:
            recommendations.append("Reduce correlation - positions too similar")
        
        if num_positions == 0:
            recommendations.append("No positions - consider opportunities")
        
        if not recommendations:
            recommendations.append("Portfolio risk within acceptable limits")
        
        return recommendations

    def get_risk_status(self) -> Dict:
        """Risk manager státusz"""
        return {
            'name': 'DynamicRiskManager',
            'max_position_size': f"${self.MAX_POSITION_SIZE}",
            'max_daily_loss': f"${self.MAX_DAILY_LOSS}",
            'atr_period': self.ATR_PERIOD,
            'atr_sl_multiplier': self.ATR_MULTIPLIER_SL,
            'max_correlation': self.MAX_CORRELATION,
            'daily_pnl': f"${self.daily_pnl:.2f}",
            'risk_events_today': len([e for e in self.risk_events 
                                    if time.time() - e.get('timestamp', 0) < 86400]),
            'volatility_cache_size': len(self.volatility_cache),
            'dynamic_adjustment': True
        }
