# strategy/scorer.py - Optimalizált verzió FÜGGETLEN MOZGÁSOKKAL

import time
import numpy as np
from typing import Dict, List, Optional
import math

class CoinScorer:
    def __init__(self, blacklist=None, whitelist=None):
        self.blacklist = blacklist or {}
        self.whitelist = whitelist or {}
        self.last_check = {}
        
        # 🎯 OPTIMALIZÁLT SÚLYOK - FÜGGETLEN MOZGÁSOK FÓKUSZ
        self.weights = {
            'volume_spike': 0.30,           # 30% - Volume spike a legfontosabb
            'bollinger_breakout': 0.25,     # 25% - Bollinger kitörés
            'rsi_momentum': 0.18,           # 18% - RSI momentum
            'price_action': 0.12,           # 12% - Ár akció
            'volatility': 0.08,             # 8% - Volatilitás
            'correlation': 0.03,            # 3% - BTC/ETH korreláció (csökkentve!)
            'independence_bonus': 0.02,     # 2% - Független mozgás bónusz
            'liquidity': 0.02               # 2% - Likviditás
        }
        
        # Mikro-trading thresholds
        self.thresholds = {
            'min_volume_24h': 100000,           # Min $100K volume
            'max_volume_24h': 50000000,         # Max $50M volume
            'ideal_volatility_range': (0.02, 0.08),  # 2-8% volatilitás
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_neutral_zone': (40, 60),
            'volume_spike_threshold': 2.0,       # 2x volume spike
            'mega_pump_threshold': 5.0,          # 5x mega pump
            'correlation_strength': 0.6,         # CSÖKKENTETT korrelációs küszöb
            'price_momentum_threshold': 0.005,   # 0.5% momentum
            'independence_threshold': 0.4        # Független mozgás küszöb
        }

    def score_coin(self, coin: Dict) -> float:
        """
        Optimalizált pontozás FÜGGETLEN MOZGÁSOKKAL
        """
        try:
            total_score = 0.0
            
            # Blacklist check
            symbol = coin.get('symbol', '')
            if self._is_blacklisted(symbol):
                return -100
            
            # 1. VOLUME SPIKE ANALYSIS (30% súly - növelve!)
            volume_score = self._calculate_volume_score(coin)
            total_score += volume_score * self.weights['volume_spike']
            
            # 2. BOLLINGER BREAKOUT ANALYSIS (25% súly - növelve!)
            bb_score = self._calculate_bollinger_score(coin)
            total_score += bb_score * self.weights['bollinger_breakout']
            
            # 3. RSI MOMENTUM ANALYSIS (18% súly)
            rsi_score = self._calculate_rsi_momentum_score(coin)
            total_score += rsi_score * self.weights['rsi_momentum']
            
            # 4. PRICE ACTION ANALYSIS (12% súly)
            price_score = self._calculate_price_action_score(coin)
            total_score += price_score * self.weights['price_action']
            
            # 5. VOLATILITY ANALYSIS (8% súly)
            vol_score = self._calculate_volatility_score(coin)
            total_score += vol_score * self.weights['volatility']
            
            # 6. CORRELATION ANALYSIS (3% súly - DRASZTIKUSAN CSÖKKENTETT!)
            corr_score = self._calculate_correlation_score(coin)
            total_score += corr_score * self.weights['correlation']
            
            # 7. INDEPENDENCE BONUS (2% súly - ÚJ!)
            independence_score = self._calculate_independence_bonus(coin)
            total_score += independence_score * self.weights['independence_bonus']
            
            # 8. LIQUIDITY ANALYSIS (2% súly)
            liq_score = self._calculate_liquidity_score(coin)
            total_score += liq_score * self.weights['liquidity']
            
            # BONUS/PENALTY MODIFIERS
            total_score = self._apply_bonus_penalties(coin, total_score)
            
            # Final normalization
            final_score = max(0, min(10, total_score))
            
            coin['score'] = final_score
            coin['score_breakdown'] = self._get_score_breakdown(coin)
            
            return final_score
            
        except Exception as e:
            print(f"[SCORER] Error scoring {coin.get('symbol', 'unknown')}: {e}")
            return 0.0

    def _calculate_volume_score(self, coin: Dict) -> float:
        """Volume spike elemzés - még fontosabb lett"""
        try:
            volume_last = coin.get('volume_last', 0)
            volume_avg = coin.get('volume_15m_avg', 1)
            volume_24h = coin.get('volume_24h', volume_last * 96)
            
            volume_ratio = volume_last / max(volume_avg, 1)
            score = 0.0
            
            # Alapvető volume követelmények
            if volume_24h < self.thresholds['min_volume_24h']:
                return 0.0
            
            if volume_24h > self.thresholds['max_volume_24h']:
                score -= 1.0  # Csökkentett büntetés
            
            # Volume spike scoring - OPTIMALIZÁLT
            if volume_ratio >= self.thresholds['mega_pump_threshold']:
                score += 5.0  # Mega pump
            elif volume_ratio >= self.thresholds['volume_spike_threshold']:
                score += 4.0  # Jó spike - növelve
            elif volume_ratio >= 1.5:
                score += 2.5  # Mérsékelt spike - növelve
            elif volume_ratio >= 1.2:
                score += 1.0  # Kis spike - ÚJ kategória
            elif volume_ratio < 0.8:
                score -= 0.5  # Csökkentett büntetés
            
            # Tartós volume bónusz
            if volume_ratio > 1.3:
                score += 1.0  # Növelt bónusz
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Volume score error: {e}")
            return 0.0

    def _calculate_bollinger_score(self, coin: Dict) -> float:
        """Bollinger Bands breakout analysis - növelt súly"""
        try:
            price = coin.get('close', 0)
            bb_upper = coin.get('boll_3m_upper', price * 1.02)
            bb_lower = coin.get('boll_3m_lower', price * 0.98)
            bb_middle = (bb_upper + bb_lower) / 2
            
            score = 0.0
            
            # Band position analysis - OPTIMALIZÁLT
            if price > bb_upper * 0.99:
                score += 4.0  # Breakout imminent - növelve
            elif price > bb_upper * 0.97:
                score += 3.0  # Near breakout - növelve
            elif price < bb_lower * 1.01:
                score += 3.5  # Oversold reversal - növelve
            elif price < bb_lower * 1.03:
                score += 2.0  # Near oversold - növelve
            elif abs(price - bb_middle) / bb_middle < 0.005:
                score += 1.0  # Neutral - javítva
            
            # Band width analysis
            band_width = (bb_upper - bb_lower) / bb_middle
            if 0.02 <= band_width <= 0.06:
                score += 1.5  # Ideális volatilitás - növelve
            elif band_width > 0.08:
                score -= 0.3  # Csökkentett büntetés
            elif band_width < 0.01:
                score -= 0.5  # Csökkentett büntetés
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Bollinger score error: {e}")
            return 0.0

    def _calculate_correlation_score(self, coin: Dict) -> float:
        """BTC/ETH korreláció - DRASZTIKUSAN CSÖKKENTETT súly"""
        try:
            btc_corr = coin.get('correl_btc', 0)
            eth_corr = coin.get('correl_eth', 0)
            btc_breaking = coin.get('btc_is_breaking', False)
            eth_breaking = coin.get('eth_is_breaking', False)
            
            score = 0.0
            
            # CSAK akkor számít a korreláció, ha BTC/ETH aktívan mozog
            if btc_breaking and btc_corr > 0.7:  # Csökkentett küszöb
                score += 1.5  # Csökkentett bónusz
            if eth_breaking and eth_corr > 0.6:  # Csökkentett küszöb
                score += 1.0  # Csökkentett bónusz
                
            # Mérsékelt korreláció bónusz - ELTÁVOLÍTVA a túl szigorú követelmény
            if 0.2 <= abs(btc_corr) <= 0.8:  # Bővebb tartomány
                score += 0.5  # Kisebb bónusz
            if 0.1 <= abs(eth_corr) <= 0.7:  # Bővebb tartomány
                score += 0.3  # Kisebb bónusz
                
            # Túl magas korreláció büntetése - ENYHÍTVE
            if abs(btc_corr) > 0.95 or abs(eth_corr) > 0.95:
                score -= 0.5  # Csökkentett büntetés
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Correlation score error: {e}")
            return 0.0

    def _calculate_independence_bonus(self, coin: Dict) -> float:
        """ÚJ: Független mozgás bónusz számítás"""
        try:
            btc_corr = abs(coin.get('correl_btc', 0.7))
            eth_corr = abs(coin.get('correl_eth', 0.6))
            volume_ratio = coin.get('volume_last', 0) / max(coin.get('volume_15m_avg', 1), 1)
            price_change = coin.get('price_change_5m', 0)
            
            score = 0.0
            
            # Alacsony korreláció bónusz (független mozgás)
            avg_correlation = (btc_corr + eth_corr) / 2
            if avg_correlation < self.thresholds['independence_threshold']:
                independence_factor = (self.thresholds['independence_threshold'] - avg_correlation) * 5
                score += independence_factor
            
            # Volume spike független mozgással
            if volume_ratio > 2.0 and avg_correlation < 0.6:
                score += 2.0  # Nagy bónusz független volume spike-ért
            
            # Jelentős ármozgás alacsony korrelációval
            if abs(price_change) > 0.01 and avg_correlation < 0.5:
                score += 1.5  # Független price action bónusz
            
            # Anti-követés bónusz (amikor BTC/ETH nem mozog, de a coin igen)
            btc_momentum = coin.get('btc_momentum', 0)
            eth_momentum = coin.get('eth_momentum', 0)
            
            if (abs(btc_momentum) < 0.002 and abs(eth_momentum) < 0.002 and 
                abs(price_change) > 0.005):
                score += 2.0  # Nagy bónusz független mozgásért
                
            return min(5.0, score)  # Max 5.0 bónusz
            
        except Exception as e:
            print(f"[SCORER] Independence bonus error: {e}")
            return 0.0

    def _calculate_rsi_momentum_score(self, coin: Dict) -> float:
        """RSI alapú momentum scoring - kicsit módosítva"""
        try:
            rsi_3m = coin.get('rsi_3m', 50)
            rsi_15m = coin.get('rsi_15m', 50)
            
            score = 0.0
            
            # 3-minute RSI (short-term momentum)
            if 25 <= rsi_3m <= 35:
                score += 3.5  # Kicsit növelve
            elif 35 < rsi_3m <= 45:
                score += 2.5  # Növelve
            elif 45 < rsi_3m <= 55:
                score += 1.5  # Növelve
            elif rsi_3m > 75:
                score -= 1.5  # Csökkentett büntetés
            elif rsi_3m < 20:
                score -= 0.5  # Csökkentett büntetés
                
            # 15-minute RSI confirmation
            if 30 <= rsi_15m <= 50:
                score += 1.5
            elif rsi_15m > 70:
                score -= 0.5  # Csökkentett büntetés
                
            # RSI divergencia bónusz
            if rsi_3m > rsi_15m + 5:
                score += 1.0
            elif rsi_3m < rsi_15m - 10:
                score -= 0.3  # Csökkentett büntetés
                
            return score
            
        except Exception as e:
            print(f"[SCORER] RSI score error: {e}")
            return 0.0

    def _calculate_price_action_score(self, coin: Dict) -> float:
        """Ár akció elemzés - kicsit módosítva"""
        try:
            price = coin.get('close', 0)
            price_change = coin.get('price_change_5m', 0)
            
            score = 0.0
            
            # Short-term momentum - OPTIMALIZÁLT
            if abs(price_change) > self.thresholds['price_momentum_threshold']:
                if price_change > 0:
                    score += 2.5  # Pozitív momentum - növelve
                else:
                    score += 1.5  # Negatív momentum is lehet reversal
            
            # Kis momentum is számít
            if 0.002 <= abs(price_change) <= 0.005:
                score += 1.0  # Mérsékelt momentum bónusz
            
            # Price stability check
            price_volatility = coin.get('price_volatility_1h', 0.02)
            if 0.01 <= price_volatility <= 0.05:
                score += 1.2  # Növelt bónusz
            elif price_volatility > 0.10:
                score -= 0.5  # Csökkentett büntetés
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Price action score error: {e}")
            return 0.0

    def _calculate_volatility_score(self, coin: Dict) -> float:
        """Volatilitás elemzés - változatlan"""
        try:
            volume_ratio = coin.get('volume_last', 0) / max(coin.get('volume_15m_avg', 1), 1)
            price_range = coin.get('price_range_1h', 0.02)
            
            score = 0.0
            
            ideal_min, ideal_max = self.thresholds['ideal_volatility_range']
            
            if ideal_min <= price_range <= ideal_max:
                score += 2.0
            elif price_range < ideal_min:
                score += 0.5
            elif price_range <= ideal_max * 1.5:
                score += 1.0
            else:
                score -= 1.0
                
            if volume_ratio > 2 and price_range > ideal_min:
                score += 1.0
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Volatility score error: {e}")
            return 0.0

    def _calculate_liquidity_score(self, coin: Dict) -> float:
        """Likviditás elemzés - változatlan de csökkentett súly"""
        try:
            volume_24h = coin.get('volume_24h', coin.get('volume_last', 0) * 96)
            
            score = 0.0
            
            if volume_24h > 1000000:
                score += 1.0
            elif volume_24h > 500000:
                score += 0.5
            else:
                score -= 0.5
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Liquidity score error: {e}")
            return 0.0

    def _apply_bonus_penalties(self, coin: Dict, base_score: float) -> float:
        """Bónuszok és büntetések - OPTIMALIZÁLT"""
        try:
            score = base_score
            
            # Whitelist bónusz
            if coin.get('symbol') in self.whitelist:
                score += 1.0
                
            # Recent success bónusz
            if coin.get('recent_profit_streak', 0) > 3:
                score += 0.5
                
            volume_ratio = coin.get('volume_last', 0) / max(coin.get('volume_15m_avg', 1), 1)
            
            # Perfect storm bónusz - MÓDOSÍTOTT feltételek
            perfect_storm_conditions = [
                volume_ratio > 2.0,  # Volume spike
                25 <= coin.get('rsi_3m', 50) <= 45,  # Jó RSI tartomány - bővítve
                coin.get('close', 0) > coin.get('boll_3m_upper', 0) * 0.97,  # Near/above BB
                abs(coin.get('correl_btc', 0.7)) < 0.8 or abs(coin.get('correl_eth', 0.6)) < 0.7  # NEM túl korrelált
            ]
            
            if sum(perfect_storm_conditions) >= 3:
                score += 3.0  # Független perfect storm bónusz
            elif sum(perfect_storm_conditions) >= 2:
                score += 1.5
                
            # Független mozgás extra bónusz
            if (volume_ratio > 2.5 and 
                abs(coin.get('correl_btc', 0.7)) < 0.5 and 
                abs(coin.get('correl_eth', 0.6)) < 0.4):
                score += 2.0  # Nagy független mozgás bónusz
                
            # Időalapú büntetés - változatlan
            current_hour = time.localtime().tm_hour
            if 2 <= current_hour <= 6:
                score -= 0.5  # Csökkentett büntetés
                
            return score
            
        except Exception as e:
            print(f"[SCORER] Bonus/penalty error: {e}")
            return base_score

    # A többi metódus változatlan marad...
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
                'bollinger_score': self._calculate_bollinger_score(coin),
                'rsi_score': self._calculate_rsi_momentum_score(coin),
                'price_action_score': self._calculate_price_action_score(coin),
                'correlation_score': self._calculate_correlation_score(coin),
                'independence_score': self._calculate_independence_bonus(coin),
                'volatility_score': self._calculate_volatility_score(coin),
                'liquidity_score': self._calculate_liquidity_score(coin)
            }
        except Exception:
            return {}

    def select_best_coin(self, coin_list: List[Dict], min_score: float = 3.0) -> Optional[Dict]:
        """Legjobb coin kiválasztás - optimalizált logikával"""
        try:
            if not coin_list:
                return None
                
            scored_coins = []
            for coin in coin_list:
                score = self.score_coin(coin)
                if score >= min_score:
                    scored_coins.append((coin, score))
            
            if not scored_coins:
                return None
                
            # Rendezés pontszám szerint
            scored_coins.sort(key=lambda x: x[1], reverse=True)
            
            # Független mozgás preferencia
            best_candidates = []
            
            for coin, score in scored_coins[:10]:
                if self._is_suitable_for_micro_trading(coin):
                    # Extra pont független mozgásért
                    independence_bonus = self._calculate_independence_bonus(coin)
                    adjusted_score = score + (independence_bonus * 0.1)
                    best_candidates.append((coin, adjusted_score))
                    
            if not best_candidates:
                return scored_coins[0][0]
                
            # Újra rendezés az adjusted score alapján
            best_candidates.sort(key=lambda x: x[1], reverse=True)
            return best_candidates[0][0]
            
        except Exception as e:
            print(f"[SCORER] Best coin selection error: {e}")
            return None

    def _is_suitable_for_micro_trading(self, coin: Dict) -> bool:
        """Mikro-trading alkalmasság - enyhített feltételek"""
        try:
            volume_24h = coin.get('volume_24h', coin.get('volume_last', 0) * 96)
            if volume_24h < self.thresholds['min_volume_24h']:
                return False
                
            rsi = coin.get('rsi_3m', 50)
            if rsi < 10 or rsi > 90:  # Enyhített RSI feltételek
                return False
                
            price_volatility = coin.get('price_volatility_1h', 0.02)
            if price_volatility > 0.20:  # Enyhített volatilitás
                return False
                
            required_fields = ['symbol', 'close', 'volume_last']
            if not all(field in coin for field in required_fields):
                return False
                
            return True
            
        except Exception as e:
            print(f"[SCORER] Suitability check error: {e}")
            return False

    # Többi metódus változatlan...
    def get_top_opportunities(self, coin_list: List[Dict], top_n: int = 5) -> List[Dict]:
        """Top lehetőségek független mozgás prioritással"""
        try:
            opportunities = []
            
            for coin in coin_list:
                score = self.score_coin(coin)
                if score > 0:
                    independence_score = self._calculate_independence_bonus(coin)
                    
                    opportunities.append({
                        'coin': coin,
                        'score': score,
                        'independence_score': independence_score,
                        'symbol': coin.get('symbol', 'UNKNOWN'),
                        'breakdown': coin.get('score_breakdown', {}),
                        'confidence': min(100, score * 20),
                        'risk_level': self._assess_risk_level(coin, score),
                        'correlation_dependency': max(
                            abs(coin.get('correl_btc', 0)), 
                            abs(coin.get('correl_eth', 0))
                        )
                    })
            
            # Rendezés: független mozgás > magas pontszám
            opportunities.sort(
                key=lambda x: (x['independence_score'], x['score']), 
                reverse=True
            )
            
            return opportunities[:top_n]
            
        except Exception as e:
            print(f"[SCORER] Top opportunities error: {e}")
            return []

    def _assess_risk_level(self, coin: Dict, score: float) -> str:
        """Kockázat értékelés"""
        try:
            volume_ratio = coin.get('volume_last', 0) / max(coin.get('volume_15m_avg', 1), 1)
            rsi = coin.get('rsi_3m', 50)
            correlation = max(abs(coin.get('correl_btc', 0)), abs(coin.get('correl_eth', 0)))
            
            risk_factors = 0
            
            if volume_ratio > 5:
                risk_factors += 2
            elif volume_ratio > 3:
                risk_factors += 1
                
            if rsi < 20 or rsi > 80:
                risk_factors += 1
                
            if correlation > 0.9:  # Túl magas korreláció = kockázat
                risk_factors += 1
                
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

    # További metódusok változatlanul...
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

    def get_scoring_stats(self) -> Dict:
        """Scoring statisztikák"""
        return {
            'weights': self.weights.copy(),
            'correlation_weight': f"{self.weights['correlation']*100:.1f}%",
            'independence_bonus': f"{self.weights['independence_bonus']*100:.1f}%",
            'volume_weight': f"{self.weights['volume_spike']*100:.1f}%",
            'bollinger_weight': f"{self.weights['bollinger_breakout']*100:.1f}%",
            'thresholds': self.thresholds.copy()
        }
