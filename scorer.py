# strategy/scorer.py - Fejlesztett érme pontozó $50 mikro-tradinghez

import time
import numpy as np
from typing import Dict, List, Optional
import math

class CoinScorer:
    def __init__(self, blacklist=None, whitelist=None):
        self.blacklist = blacklist or {}
        self.whitelist = whitelist or {}
        self.last_check = {}
        
        # 🎯 MIKRO-TRADING SPECIFIC WEIGHTS
        self.weights = {
            'volume_spike': 0.25,      # Volume spike a legfontosabb
            'rsi_momentum': 0.20,      # RSI momentum
            'bollinger_breakout': 0.15, # Bollinger kitörés
            'price_action': 0.10,      # Ár akció
            'correlation': 0.10,       # BTC/ETH korreláció
            'volatility': 0.08,        # Volatilitás
            'historical_performance': 0.07, # Történeti teljesítmény
            'liquidity': 0.05          # Likviditás
        }
        
        # Mikro-trading thresholds
        self.thresholds = {
            'min_volume_24h': 100000,       # Min $100K volume
            'max_volume_24h': 50000000,     # Max $50M volume (túl nagy = manipulation risk)
            'ideal_volatility_range': (0.02, 0.08),  # 2-8% ideális volatilitás
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_neutral_zone': (40, 60),
            'volume_spike_threshold': 2.0,   # 2x volume spike
            'mega_pump_threshold': 5.0,      # 5x mega pump
            'correlation_strength': 0.7,     # Erős korreláció
            'price_momentum_threshold': 0.005, # 0.5% momentum
        }

    def score_coin(self, coin: Dict) -> float:
        """
        Fejlesztett érme pontozás mikro-tradinghez
        Cél: A legjobb 2-5 perces scalping lehetőségek azonosítása
        """
        try:
            total_score = 0.0
            
            # Blacklist check FIRST
            symbol = coin.get('symbol', '')
            if self._is_blacklisted(symbol):
                return -100  # Instant disqualification
            
            # 1. VOLUME SPIKE ANALYSIS (25% weight)
            volume_score = self._calculate_volume_score(coin)
            total_score += volume_score * self.weights['volume_spike']
            
            # 2. RSI MOMENTUM ANALYSIS (20% weight)
            rsi_score = self._calculate_rsi_momentum_score(coin)
            total_score += rsi_score * self.weights['rsi_momentum']
            
            # 3. BOLLINGER BREAKOUT ANALYSIS (15% weight)
            bb_score = self._calculate_bollinger_score(coin)
            total_score += bb_score * self.weights['bollinger_breakout']
            
            # 4. PRICE ACTION ANALYSIS (10% weight)
            price_score = self._calculate_price_action_score(coin)
            total_score += price_score * self.weights['price_action']
            
            # 5. CORRELATION ANALYSIS (10% weight)
            corr_score = self._calculate_correlation_score(coin)
            total_score += corr_score * self.weights['correlation']
            
            # 6. VOLATILITY ANALYSIS (8% weight)
            vol_score = self._calculate_volatility_score(coin)
            total_score += vol_score * self.weights['volatility']
            
            # 7. HISTORICAL PERFORMANCE (7% weight)
            hist_score = self._calculate_historical_score(coin)
            total_score += hist_score * self.weights['historical_performance']
            
            # 8. LIQUIDITY ANALYSIS (5% weight)
            liq_score = self._calculate_liquidity_score(coin)
            total_score += liq_score * self.weights['liquidity']
            
            # BONUS/PENALTY MODIFIERS
            total_score = self._apply_bonus_penalties(coin, total_score)
            
            # Final normalization and clamping
            final_score = max(0, min(10, total_score))
            
            # Store score in coin data for reference
            coin['score'] = final_score
            coin['score_breakdown'] = self._get_score_breakdown(coin)
            
            return final_score
            
        except Exception as e:
            print(f"[SCORER] Error scoring {coin.get('symbol', 'unknown')}: {e}")
            return 0.0

    def _calculate_volume_score(self, coin: Dict) -> float:
        """Volume spike elemzés - mikro-trading kulcsfontosságú"""
        try:
            volume_last = coin.get('volume_last', 0)
            volume_avg = coin.get('volume_15m_avg', 1)
            volume_24h = coin.get('volume_24h', volume_last * 96)  # Estimate if missing
            
            # Volume ratio calculation
            volume_ratio = volume_last / max(volume_avg, 1)
            
            score = 0.0
            
            # Basic volume requirements
            if volume_24h < self.thresholds['min_volume_24h']:
                return 0.0  # Too low volume
            
            if volume_24h > self.thresholds['max_volume_24h']:
                score -= 2.0  # Too high = manipulation risk
            
            # Volume spike scoring
            if volume_ratio >= self.thresholds['mega_pump_threshold']:
                score += 5.0  # Mega pump - nagy esély
            elif volume_ratio >= self.thresholds['volume_spike_threshold']:
                score += 3.0  # Jó spike
            elif volume_ratio >= 1.5:
                score += 1.5  # Mérsékelt spike
            elif volume_ratio < 0.5:
                score -= 1.0  # Túl alacsony
            
            # Sustained volume bonus
            if volume_ratio > 1.2:
                score += 0.5  # Tartós érdeklődés
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Volume score error: {e}")
            return 0.0

    def _calculate_rsi_momentum_score(self, coin: Dict) -> float:
        """RSI alapú momentum scoring"""
        try:
            rsi_3m = coin.get('rsi_3m', 50)
            rsi_15m = coin.get('rsi_15m', 50)
            
            score = 0.0
            
            # Multi-timeframe RSI analysis
            # 3-minute RSI (short-term momentum)
            if 25 <= rsi_3m <= 35:
                score += 3.0  # Oversold but not extreme
            elif 35 < rsi_3m <= 45:
                score += 2.0  # Mild oversold
            elif 45 < rsi_3m <= 55:
                score += 1.0  # Neutral
            elif rsi_3m > 75:
                score -= 2.0  # Dangerous overbought
            elif rsi_3m < 20:
                score -= 1.0  # Too extreme
                
            # 15-minute RSI (trend confirmation)
            if 30 <= rsi_15m <= 50:
                score += 1.5  # Good trend support
            elif rsi_15m > 70:
                score -= 1.0  # Trend resistance
                
            # RSI divergence bonus (if 3m > 15m = building momentum)
            if rsi_3m > rsi_15m + 5:
                score += 1.0  # Positive divergence
            elif rsi_3m < rsi_15m - 10:
                score -= 0.5  # Negative divergence
                
            return score
            
        except Exception as e:
            print(f"[SCORER] RSI score error: {e}")
            return 0.0

    def _calculate_bollinger_score(self, coin: Dict) -> float:
        """Bollinger Bands breakout analysis"""
        try:
            price = coin.get('close', 0)
            bb_upper = coin.get('boll_3m_upper', price * 1.02)
            bb_lower = coin.get('boll_3m_lower', price * 0.98)
            bb_middle = (bb_upper + bb_lower) / 2
            
            score = 0.0
            
            # Band position analysis
            if price > bb_upper * 0.995:
                score += 3.0  # Breakout imminent
            elif price > bb_upper * 0.98:
                score += 2.0  # Near breakout
            elif price < bb_lower * 1.005:
                score += 2.5  # Oversold reversal opportunity
            elif price < bb_lower * 1.02:
                score += 1.5  # Near oversold
            elif abs(price - bb_middle) / bb_middle < 0.005:
                score += 0.5  # Neutral - low volatility
            
            # Band width analysis (volatility indicator)
            band_width = (bb_upper - bb_lower) / bb_middle
            if 0.02 <= band_width <= 0.06:
                score += 1.0  # Ideal volatility range
            elif band_width > 0.08:
                score -= 0.5  # Too volatile
            elif band_width < 0.01:
                score -= 1.0  # Too low volatility
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Bollinger score error: {e}")
            return 0.0

    def _calculate_price_action_score(self, coin: Dict) -> float:
        """Price action and momentum analysis"""
        try:
            price = coin.get('close', 0)
            price_change = coin.get('price_change_5m', 0)  # 5-minute change
            
            score = 0.0
            
            # Short-term momentum
            if abs(price_change) > self.thresholds['price_momentum_threshold']:
                if price_change > 0:
                    score += 2.0  # Positive momentum
                else:
                    score += 1.0  # Negative momentum can be reversal opportunity
            
            # Price stability check (for entry timing)
            price_volatility = coin.get('price_volatility_1h', 0.02)
            if 0.01 <= price_volatility <= 0.05:
                score += 1.0  # Good volatility for scalping
            elif price_volatility > 0.10:
                score -= 1.0  # Too chaotic
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Price action score error: {e}")
            return 0.0

    def _calculate_correlation_score(self, coin: Dict) -> float:
        """BTC/ETH correlation analysis for market following"""
        try:
            btc_corr = coin.get('correl_btc', 0)
            eth_corr = coin.get('correl_eth', 0)
            btc_breaking = coin.get('btc_is_breaking', False)
            eth_breaking = coin.get('eth_is_breaking', False)
            
            score = 0.0
            
            # High correlation with breaking major coins = good opportunity
            if btc_breaking and btc_corr > self.thresholds['correlation_strength']:
                score += 2.0
            if eth_breaking and eth_corr > self.thresholds['correlation_strength']:
                score += 1.5
                
            # Moderate correlation is often ideal (not too dependent)
            if 0.3 <= abs(btc_corr) <= 0.8:
                score += 1.0
            if 0.2 <= abs(eth_corr) <= 0.7:
                score += 0.5
                
            # Too high correlation = risky
            if abs(btc_corr) > 0.9 or abs(eth_corr) > 0.9:
                score -= 1.0
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Correlation score error: {e}")
            return 0.0

    def _calculate_volatility_score(self, coin: Dict) -> float:
        """Volatility analysis for scalping suitability"""
        try:
            # Estimate volatility from volume ratio and price action
            volume_ratio = coin.get('volume_last', 0) / max(coin.get('volume_15m_avg', 1), 1)
            price_range = coin.get('price_range_1h', 0.02)  # 1-hour high-low range
            
            score = 0.0
            
            # Ideal volatility range for $50 positions
            ideal_min, ideal_max = self.thresholds['ideal_volatility_range']
            
            if ideal_min <= price_range <= ideal_max:
                score += 2.0  # Perfect volatility
            elif price_range < ideal_min:
                score += 0.5  # Low but manageable
            elif price_range <= ideal_max * 1.5:
                score += 1.0  # Moderate volatility
            else:
                score -= 1.0  # Too volatile for micro-trading
                
            # Volume-volatility relationship
            if volume_ratio > 2 and price_range > ideal_min:
                score += 1.0  # Good volume + volatility combo
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Volatility score error: {e}")
            return 0.0

    def _calculate_historical_score(self, coin: Dict) -> float:
        """Historical performance analysis"""
        try:
            recent_winrate = coin.get('recent_winrate', 0.5)
            
            score = 0.0
            
            # Historical win rate bonus
            if recent_winrate > 0.7:
                score += 2.0  # Excellent historical performance
            elif recent_winrate > 0.6:
                score += 1.0  # Good performance
            elif recent_winrate < 0.4:
                score -= 1.0  # Poor performance
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Historical score error: {e}")
            return 0.0

    def _calculate_liquidity_score(self, coin: Dict) -> float:
        """Liquidity analysis for easy entry/exit"""
        try:
            volume_24h = coin.get('volume_24h', coin.get('volume_last', 0) * 96)
            
            score = 0.0
            
            # Basic liquidity requirements for micro-trading
            if volume_24h > 1000000:  # $1M+
                score += 1.0
            elif volume_24h > 500000:  # $500K+
                score += 0.5
            else:
                score -= 0.5  # Too illiquid
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Liquidity score error: {e}")
            return 0.0

    def _apply_bonus_penalties(self, coin: Dict, base_score: float) -> float:
        """Apply additional bonuses and penalties"""
        try:
            score = base_score
            
            # Whitelist bonus
            if coin.get('symbol') in self.whitelist:
                score += 1.0
                
            # Recent success bonus
            if coin.get('recent_profit_streak', 0) > 3:
                score += 0.5
                
            # Micro-trading specific bonuses
            volume_ratio = coin.get('volume_last', 0) / max(coin.get('volume_15m_avg', 1), 1)
            
            # Perfect storm bonus (multiple positive indicators)
            perfect_storm_conditions = [
                volume_ratio > 2.0,  # Volume spike
                30 <= coin.get('rsi_3m', 50) <= 40,  # Good RSI
                coin.get('close', 0) > coin.get('boll_3m_upper', 0) * 0.98,  # Near breakout
                coin.get('btc_is_breaking', False) or coin.get('eth_is_breaking', False)  # Market support
            ]
            
            if sum(perfect_storm_conditions) >= 3:
                score += 2.0  # Perfect storm bonus
            elif sum(perfect_storm_conditions) >= 2:
                score += 1.0  # Good conditions bonus
                
            # Time-based penalties (avoid trading during low-activity periods)
            current_hour = time.localtime().tm_hour
            if 2 <= current_hour <= 6:  # Low activity hours
                score -= 1.0
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Bonus/penalty error: {e}")
            return base_score

    def _is_blacklisted(self, symbol: str) -> bool:
        """Check if symbol is blacklisted"""
        try:
            if symbol in self.blacklist:
                return self.blacklist[symbol] > time.time()
            return False
        except Exception:
            return False

    def _get_score_breakdown(self, coin: Dict) -> Dict:
        """Get detailed score breakdown for analysis"""
        try:
            return {
                'volume_score': self._calculate_volume_score(coin),
                'rsi_score': self._calculate_rsi_momentum_score(coin),
                'bollinger_score': self._calculate_bollinger_score(coin),
                'price_action_score': self._calculate_price_action_score(coin),
                'correlation_score': self._calculate_correlation_score(coin),
                'volatility_score': self._calculate_volatility_score(coin),
                'historical_score': self._calculate_historical_score(coin),
                'liquidity_score': self._calculate_liquidity_score(coin)
            }
        except Exception:
            return {}

    def select_best_coin(self, coin_list: List[Dict], min_score: float = 3.0) -> Optional[Dict]:
        """
        Select the best coin for micro-trading
        Enhanced algorithm for $50 position optimization
        """
        try:
            if not coin_list:
                return None
                
            # Score all coins
            scored_coins = []
            for coin in coin_list:
                score = self.score_coin(coin)
                if score >= min_score:
                    scored_coins.append((coin, score))
            
            if not scored_coins:
                return None
                
            # Sort by score descending
            scored_coins.sort(key=lambda x: x[1], reverse=True)
            
            # Apply additional selection criteria for micro-trading
            best_candidates = []
            
            for coin, score in scored_coins[:10]:  # Top 10 candidates
                # Additional micro-trading filters
                if self._is_suitable_for_micro_trading(coin):
                    best_candidates.append((coin, score))
                    
            if not best_candidates:
                # If no candidates pass strict filters, return best score
                return scored_coins[0][0]
                
            # Return the best candidate
            return best_candidates[0][0]
            
        except Exception as e:
            print(f"[SCORER] Best coin selection error: {e}")
            return None

    def _is_suitable_for_micro_trading(self, coin: Dict) -> bool:
        """Additional micro-trading suitability checks"""
        try:
            # Volume requirements
            volume_24h = coin.get('volume_24h', coin.get('volume_last', 0) * 96)
            if volume_24h < self.thresholds['min_volume_24h']:
                return False
                
            # RSI not in extreme zones
            rsi = coin.get('rsi_3m', 50)
            if rsi < 15 or rsi > 85:
                return False
                
            # Price stability check
            price_volatility = coin.get('price_volatility_1h', 0.02)
            if price_volatility > 0.15:  # Too volatile for $50 positions
                return False
                
            # Ensure we have minimum required data
            required_fields = ['symbol', 'close', 'volume_last']
            if not all(field in coin for field in required_fields):
                return False
                
            return True
            
        except Exception as e:
            print(f"[SCORER] Suitability check error: {e}")
            return False

    def get_top_opportunities(self, coin_list: List[Dict], top_n: int = 5) -> List[Dict]:
        """Get top N trading opportunities with scores"""
        try:
            opportunities = []
            
            for coin in coin_list:
                score = self.score_coin(coin)
                if score > 0:  # Only positive scores
                    opportunities.append({
                        'coin': coin,
                        'score': score,
                        'symbol': coin.get('symbol', 'UNKNOWN'),
                        'breakdown': coin.get('score_breakdown', {}),
                        'confidence': min(100, score * 20),  # Convert to percentage
                        'risk_level': self._assess_risk_level(coin, score)
                    })
            
            # Sort by score and return top N
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            return opportunities[:top_n]
            
        except Exception as e:
            print(f"[SCORER] Top opportunities error: {e}")
            return []

    def _assess_risk_level(self, coin: Dict, score: float) -> str:
        """Assess risk level for the coin"""
        try:
            volume_ratio = coin.get('volume_last', 0) / max(coin.get('volume_15m_avg', 1), 1)
            rsi = coin.get('rsi_3m', 50)
            
            risk_factors = 0
            
            # High volume spike = higher risk
            if volume_ratio > 5:
                risk_factors += 2
            elif volume_ratio > 3:
                risk_factors += 1
                
            # Extreme RSI = higher risk
            if rsi < 25 or rsi > 75:
                risk_factors += 1
                
            # Low score despite good indicators = hidden risk
            if score < 2:
                risk_factors += 1
                
            if risk_factors >= 3:
                return "HIGH"
            elif risk_factors >= 2:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception:
            return "UNKNOWN"

    def blacklist_coin(self, symbol: str, duration: int = 86400, reason: str = "Auto blacklist"):
        """Add coin to blacklist"""
        try:
            self.blacklist[symbol] = time.time() + duration
            print(f"[SCORER] Blacklisted {symbol} for {duration}s - {reason}")
        except Exception as e:
            print(f"[SCORER] Blacklist error: {e}")

    def whitelist_coin(self, symbol: str):
        """Add coin to whitelist"""
        try:
            self.whitelist[symbol] = True
            print(f"[SCORER] Whitelisted {symbol}")
        except Exception as e:
            print(f"[SCORER] Whitelist error: {e}")

    def get_blacklist_status(self, symbol: str) -> tuple:
        """Get blacklist status and expiry time"""
        try:
            if symbol in self.blacklist:
                expiry = self.blacklist[symbol]
                is_blacklisted = expiry > time.time()
                return is_blacklisted, expiry
            return False, 0
        except Exception:
            return False, 0

    def get_whitelist_status(self, symbol: str) -> bool:
        """Get whitelist status"""
        try:
            return symbol in self.whitelist
        except Exception:
            return False

    def cleanup_expired_blacklist(self):
        """Remove expired blacklist entries"""
        try:
            current_time = time.time()
            expired = [symbol for symbol, expiry in self.blacklist.items() if expiry <= current_time]
            for symbol in expired:
                del self.blacklist[symbol]
            if expired:
                print(f"[SCORER] Removed {len(expired)} expired blacklist entries")
        except Exception as e:
            print(f"[SCORER] Blacklist cleanup error: {e}")

    def update_weights(self, performance_data: Dict):
        """Update scoring weights based on performance"""
        try:
            # Simple adaptive weighting based on what indicators worked best
            if 'best_indicators' in performance_data:
                best = performance_data['best_indicators']
                
                # Increase weights for successful indicators
                for indicator in best:
                    if indicator in self.weights:
                        self.weights[indicator] = min(0.4, self.weights[indicator] * 1.1)
                        
                # Normalize weights to sum to 1.0
                total_weight = sum(self.weights.values())
                if total_weight > 0:
                    for key in self.weights:
                        self.weights[key] /= total_weight
                        
                print(f"[SCORER] Updated weights based on performance")
                
        except Exception as e:
            print(f"[SCORER] Weight update error: {e}")

    def get_scoring_stats(self) -> Dict:
        """Get scoring statistics"""
        try:
            current_time = time.time()
            active_blacklist = sum(1 for expiry in self.blacklist.values() if expiry > current_time)
            
            return {
                'active_blacklist_count': active_blacklist,
                'whitelist_count': len(self.whitelist),
                'current_weights': self.weights.copy(),
                'thresholds': self.thresholds.copy(),
                'last_cleanup': getattr(self, 'last_cleanup', 0)
            }
        except Exception as e:
            print(f"[SCORER] Stats error: {e}")
            return {}

    def score_coin_data(self, coin_data: Dict) -> tuple:
        """
        Score coin data and return (score, reasons) tuple
        Compatible with external interfaces
        """
        try:
            score = self.score_coin(coin_data)
            breakdown = coin_data.get('score_breakdown', {})
            
            # Generate human-readable reasons
            reasons = []
            
            if breakdown.get('volume_score', 0) > 2:
                reasons.append("Strong volume spike")
            if breakdown.get('rsi_score', 0) > 2:
                reasons.append("Good RSI momentum")
            if breakdown.get('bollinger_score', 0) > 2:
                reasons.append("Bollinger breakout potential")
            if breakdown.get('correlation_score', 0) > 1:
                reasons.append("Favorable market correlation")
                
            if not reasons:
                reasons.append("Standard market conditions")
                
            return score, reasons
            
        except Exception as e:
            print(f"[SCORER] Score coin data error: {e}")
            return 0.0, ["Scoring error"]
