# strategy/correlation_analyzer.py - OPTIMALIZÁLT FÜGGETLEN MOZGÁSOKHOZ

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')
from utils.logger import logger

@dataclass
class CorrelationData:
    """Korreláció adatok"""
    pair: str
    timeframe: str
    
    # BTC correlation
    btc_correlation: float
    btc_correlation_strength: str
    btc_momentum_aligned: bool
    
    # ETH correlation
    eth_correlation: float
    eth_correlation_strength: str
    eth_momentum_aligned: bool
    
    # Cross-correlation
    btc_eth_correlation: float
    
    # Dynamic correlations
    correlation_1h: float
    correlation_4h: float
    correlation_24h: float
    correlation_trend: str
    
    # Momentum analysis
    btc_momentum: float
    eth_momentum: float
    pair_momentum: float
    momentum_sync_score: float
    
    # Independence metrics (ÚJ!)
    independence_score: float
    is_independent_move: bool
    delayed_correlation: bool
    
    # Trading signals - MÓDOSÍTOTT logika
    correlation_signal: str
    confidence_score: float
    optimal_entry_window: bool

@dataclass
class MomentumEvent:
    """BTC/ETH momentum esemény"""
    timestamp: float
    source: str
    direction: str
    magnitude: float
    duration_estimate: float
    confidence: float
    triggered_pairs: List[str]

class CorrelationAnalyzer:
    """Optimalizált korreláció elemzés FÜGGETLEN MOZGÁSOKKAL"""
    
    def __init__(self):
        # 🎯 OPTIMALIZÁLT BEÁLLÍTÁSOK
        self.MIN_CORRELATION = 0.4      # 40% min (80%-ról csökkentve!)
        self.STRONG_CORRELATION = 0.65  # 65% erős (90%-ról csökkentve!)
        self.MOMENTUM_THRESHOLD = 0.0003  # 0.03% min momentum (0.06%-ról csökkentve!)
        self.SYNC_WINDOW = 300          # 5 perc szinkronizációs ablak
        self.INDEPENDENCE_THRESHOLD = 0.3  # 30% alatt már független
        
        # Timeframe weights - FÜGGETLEN MOZGÁS FÓKUSZ
        self.TIMEFRAME_WEIGHTS = {
            '1h': 0.6,    # 60% súly a friss adatoknak (növelve)
            '4h': 0.25,   # 25% súly
            '24h': 0.15   # 15% súly (csökkentve)
        }
        
        # Data storage
        self.btc_price_history = []
        self.eth_price_history = []
        self.correlation_cache = {}
        self.momentum_events = []
        
        # Independence tracking
        self.independent_moves = {}
        self.delayed_followers = {}
        
        # Performance tracking
        self.successful_signals = 0
        self.total_signals = 0
        self.last_update_time = 0
        
        logger.info("CorrelationAnalyzer optimized: 0.03% momentum, 40% min correlation, independence focus")

    def analyze_pair_correlation(self, pair: str, price_history: List[float], 
                                timeframe: str = "3m") -> CorrelationData:
        """
        Optimalizált korreláció elemzés FÜGGETLEN MOZGÁS FÓKUSSZAL
        """
        try:
            if len(price_history) < 20:
                logger.warning(f"Insufficient price history for {pair}")
                return self._create_default_correlation(pair, timeframe)
            
            # 1. BTC/ETH referencia adatok frissítése
            self._update_reference_data()
            
            # 2. Multi-timeframe korrelációk
            correlations = self._calculate_multi_timeframe_correlations(pair, price_history)
            
            # 3. Momentum elemzés - OPTIMALIZÁLT
            momentum_data = self._analyze_momentum_alignment(pair, price_history)
            
            # 4. FÜGGETLEN MOZGÁS ELEMZÉS (ÚJ!)
            independence_data = self._analyze_independence(pair, price_history, correlations)
            
            # 5. Dinamikus korreláció trend
            correlation_trend = self._detect_correlation_trend(pair, correlations)
            
            # 6. Trading signal generálás - MÓDOSÍTOTT logika
            signal_data = self._generate_independence_focused_signal(
                correlations, momentum_data, independence_data
            )
            
            # 7. Optimal entry window - FÜGGETLEN MOZGÁS alapján
            entry_window = self._detect_optimal_entry_window_v2(
                momentum_data, correlations, independence_data
            )
            
            correlation_data = CorrelationData(
                pair=pair,
                timeframe=timeframe,
                
                btc_correlation=correlations['btc']['weighted'],
                btc_correlation_strength=self._classify_correlation_strength(correlations['btc']['weighted']),
                btc_momentum_aligned=momentum_data['btc_aligned'],
                
                eth_correlation=correlations['eth']['weighted'],
                eth_correlation_strength=self._classify_correlation_strength(correlations['eth']['weighted']),
                eth_momentum_aligned=momentum_data['eth_aligned'],
                
                btc_eth_correlation=correlations['btc_eth'],
                
                correlation_1h=correlations['btc']['1h'],
                correlation_4h=correlations['btc']['4h'],
                correlation_24h=correlations['btc']['24h'],
                correlation_trend=correlation_trend,
                
                btc_momentum=momentum_data['btc_momentum'],
                eth_momentum=momentum_data['eth_momentum'],
                pair_momentum=momentum_data['pair_momentum'],
                momentum_sync_score=momentum_data['sync_score'],
                
                # ÚJ: Independence metrics
                independence_score=independence_data['independence_score'],
                is_independent_move=independence_data['is_independent'],
                delayed_correlation=independence_data['delayed_correlation'],
                
                correlation_signal=signal_data['signal'],
                confidence_score=signal_data['confidence'],
                optimal_entry_window=entry_window
            )
            
            # Cache eredmény
            self._cache_correlation_data(pair, correlation_data)
            
            logger.info(
                f"🔗 {pair}: BTC: {correlations['btc']['weighted']:.2f}, "
                f"ETH: {correlations['eth']['weighted']:.2f}, "
                f"Independence: {independence_data['independence_score']:.2f}, "
                f"Signal: {signal_data['signal']}"
            )
            
            return correlation_data
            
        except Exception as e:
            logger.error(f"Correlation analysis failed for {pair}: {e}")
            return self._create_default_correlation(pair, timeframe)

    def _analyze_independence(self, pair: str, price_history: List[float], 
                            correlations: Dict) -> Dict:
        """ÚJ: Független mozgás elemzés"""
        try:
            if len(price_history) < 20:
                return {
                    'independence_score': 0.5,
                    'is_independent': False,
                    'delayed_correlation': False,
                    'independent_strength': 0.0
                }
            
            # 1. Átlagos korreláció számítás
            avg_correlation = (
                abs(correlations['btc']['weighted']) + 
                abs(correlations['eth']['weighted'])
            ) / 2
            
            # 2. Alapvető függetlenségi pontszám (fordított korreláció)
            base_independence = 1.0 - min(avg_correlation, 1.0)
            
            # 3. Volume-based independence (ha nagy a volume, de alacsony a korreláció)
            volume_independence = 0.0
            if hasattr(self, '_get_volume_data'):
                volume_ratio = self._get_volume_ratio(pair)
                if volume_ratio > 2.0 and avg_correlation < 0.6:
                    volume_independence = min(0.3, (volume_ratio - 2.0) * 0.1)
            
            # 4. Momentum independence (saját momentum vs BTC/ETH momentum)
            momentum_independence = self._calculate_momentum_independence(pair, price_history)
            
            # 5. Time-lagged correlation check (késleltetett követés)
            delayed_correlation = self._check_delayed_correlation(pair, price_history)
            
            # 6. Végső independence score
            independence_score = min(1.0, 
                base_independence * 0.5 + 
                volume_independence + 
                momentum_independence * 0.3 + 
                (0.2 if not delayed_correlation else 0.0)
            )
            
            # 7. Independent move meghatározás
            is_independent = (
                independence_score > 0.6 and
                avg_correlation < self.INDEPENDENCE_THRESHOLD and
                not delayed_correlation
            )
            
            return {
                'independence_score': independence_score,
                'is_independent': is_independent,
                'delayed_correlation': delayed_correlation,
                'independent_strength': momentum_independence,
                'avg_correlation': avg_correlation
            }
            
        except Exception as e:
            logger.error(f"Independence analysis failed for {pair}: {e}")
            return {
                'independence_score': 0.5,
                'is_independent': False,
                'delayed_correlation': False,
                'independent_strength': 0.0
            }

    def _calculate_momentum_independence(self, pair: str, price_history: List[float]) -> float:
        """Momentum függetlenség számítás"""
        try:
            if len(price_history) < 20 or len(self.btc_price_history) < 20:
                return 0.5
            
            lookback = min(10, len(price_history))  # Rövidebb időtáv az azonnali mozgásokhoz
            
            # Pair momentum
            pair_momentum = (price_history[-1] - price_history[-lookback]) / price_history[-lookback]
            
            # BTC momentum
            btc_current = self.btc_price_history[-1]['price']
            btc_past = self.btc_price_history[-lookback]['price'] 
            btc_momentum = (btc_current - btc_past) / btc_past
            
            # ETH momentum
            eth_current = self.eth_price_history[-1]['price']
            eth_past = self.eth_price_history[-lookback]['price']
            eth_momentum = (eth_current - eth_past) / eth_past
            
            # Independence scoring
            independence = 0.0
            
            # 1. Ha a pair jelentősen mozog, de BTC/ETH nem
            if (abs(pair_momentum) > self.MOMENTUM_THRESHOLD * 2 and
                abs(btc_momentum) < self.MOMENTUM_THRESHOLD and 
                abs(eth_momentum) < self.MOMENTUM_THRESHOLD):
                independence += 0.8  # Nagy független mozgás
                
            # 2. Ha ellentétes irányú mozgás
            elif (abs(pair_momentum) > self.MOMENTUM_THRESHOLD and
                  (np.sign(pair_momentum) != np.sign(btc_momentum) or
                   np.sign(pair_momentum) != np.sign(eth_momentum))):
                independence += 0.6  # Ellentétes mozgás
                
            # 3. Ha aránytalanul nagy a mozgás a BTC/ETH-hoz képest
            elif (abs(pair_momentum) > self.MOMENTUM_THRESHOLD and
                  abs(pair_momentum) > abs(btc_momentum) * 2 and
                  abs(pair_momentum) > abs(eth_momentum) * 2):
                independence += 0.5  # Aránytalanul nagy mozgás
                
            return min(1.0, independence)
            
        except Exception as e:
            logger.error(f"Momentum independence calculation failed: {e}")
            return 0.5

    def _check_delayed_correlation(self, pair: str, price_history: List[float]) -> bool:
        """Késleltetett korreláció ellenőrzés"""
        try:
            if len(price_history) < 30 or len(self.btc_price_history) < 30:
                return False
            
            # Ellenőrizzük, hogy a coin követi-e a BTC/ETH mozgást 5-15 perc késéssel
            pair_returns = np.diff(price_history[-20:])
            btc_returns = np.diff([p['price'] for p in self.btc_price_history[-20:]])
            
            # 1-5 perces lag ellenőrzés
            for lag in range(1, 6):
                if len(pair_returns) > lag and len(btc_returns) > lag:
                    try:
                        # Lag correlation
                        corr, _ = pearsonr(pair_returns[lag:], btc_returns[:-lag])
                        if not np.isnan(corr) and abs(corr) > 0.7:
                            return True  # Késleltetett korreláció detektálva
                    except:
                        continue
                        
            return False
            
        except Exception as e:
            logger.error(f"Delayed correlation check failed: {e}")
            return False

    def _generate_independence_focused_signal(self, correlations: Dict, 
                                            momentum_data: Dict, 
                                            independence_data: Dict) -> Dict:
        """FÜGGETLEN MOZGÁS fókuszú signal generálás"""
        try:
            btc_corr = correlations['btc']['weighted']
            eth_corr = correlations['eth']['weighted']
            max_corr = max(abs(btc_corr), abs(eth_corr))
            
            independence_score = independence_data['independence_score']
            is_independent = independence_data['is_independent']
            
            # Base confidence számítás
            confidence = 0.3
            
            # FÜGGETLEN MOZGÁS PRIORITÁS
            if is_independent:
                # Független mozgás = erős signal
                if independence_score > 0.8:
                    signal = "STRONG_BUY"
                    confidence = 0.9
                elif independence_score > 0.6:
                    signal = "BUY" 
                    confidence = 0.8
                else:
                    signal = "HOLD"
                    confidence = 0.6
                    
            # Hagyományos korreláció alapú (ha nem független)
            elif max_corr >= self.STRONG_CORRELATION:
                if (momentum_data['btc_aligned'] or momentum_data['eth_aligned']):
                    signal = "BUY"
                    confidence = 0.7
                else:
                    signal = "HOLD"
                    confidence = 0.5
                    
            elif max_corr >= self.MIN_CORRELATION:
                if (momentum_data['btc_aligned'] and momentum_data['eth_aligned']):
                    signal = "BUY"
                    confidence = 0.6
                else:
                    signal = "HOLD"
                    confidence = 0.4
                    
            else:
                # Alacsony korreláció
                if independence_score > 0.5:
                    signal = "HOLD"  # Lehet független mozgás
                    confidence = 0.5
                else:
                    signal = "AVOID"
                    confidence = 0.2
            
            # Independence bonus a confidence-re
            if is_independent:
                confidence = min(1.0, confidence + independence_score * 0.2)
            
            # Delayed correlation penalty
            if independence_data['delayed_correlation']:
                confidence *= 0.7
                if signal == "STRONG_BUY":
                    signal = "BUY"
                elif signal == "BUY":
                    signal = "HOLD"
            
            return {
                'signal': signal,
                'confidence': confidence,
                'independence_factor': independence_score,
                'correlation_factor': max_corr
            }
            
        except Exception as e:
            logger.error(f"Independence-focused signal generation failed: {e}")
            return {'signal': 'HOLD', 'confidence': 0.5}

    def _detect_optimal_entry_window_v2(self, momentum_data: Dict, 
                                       correlations: Dict, 
                                       independence_data: Dict) -> bool:
        """Optimális belépési ablak - FÜGGETLEN MOZGÁS alapján"""
        try:
            conditions = []
            
            # 1. FÜGGETLEN MOZGÁS PRIORITÁS
            if independence_data['is_independent']:
                conditions.append(True)  # Független mozgás = jó entry
            else:
                conditions.append(False)
            
            # 2. High independence score (még ha nem teljesen független)
            conditions.append(independence_data['independence_score'] > 0.6)
            
            # 3. No delayed correlation (nem követő mozgás)
            conditions.append(not independence_data['delayed_correlation'])
            
            # 4. Hagyományos momentum (ha van korreláció)
            if not independence_data['is_independent']:
                momentum_condition = (momentum_data['btc_aligned'] or 
                                    momentum_data['eth_aligned'])
                conditions.append(momentum_condition)
            else:
                conditions.append(True)  # Független mozgásnál nem számít
            
            # 5. Reasonable correlation (se túl magas, se túl alacsony)
            max_corr = max(abs(correlations['btc']['weighted']), 
                          abs(correlations['eth']['weighted']))
            reasonable_corr = 0.2 <= max_corr <= 0.8  # Bővebb tartomány
            conditions.append(reasonable_corr)
            
            # Értékelés: legalább 3/5 feltétel kell
            score = sum(1 for c in conditions if c is True)
            return score >= 3
            
        except Exception as e:
            logger.error(f"Optimal entry window v2 detection failed: {e}")
            return False

    def find_correlated_opportunities(self, available_pairs: List[str]) -> List[Dict]:
        """FÜGGETLEN MOZGÁSÚ lehetőségek keresése (név megtartva kompatibilitásért)"""
        try:
            opportunities = []
            
            # Active momentum events (BTC/ETH mozgás)
            active_events = [e for e in self.momentum_events 
                           if time.time() - e.timestamp < 300]
            
            logger.info(f"Active momentum events: {len(active_events)}")
            
            # Minden pár elemzése
            for pair in available_pairs:
                if pair in ['XBTUSD', 'BTCUSD', 'ETHUSD']:
                    continue
                
                correlation_data = self._get_cached_correlation(pair)
                if not correlation_data:
                    continue
                
                # FÜGGETLEN MOZGÁS PRIORITÁS
                if correlation_data.is_independent_move:
                    priority_score = 1.0
                elif correlation_data.independence_score > 0.6:
                    priority_score = 0.8
                elif not correlation_data.delayed_correlation:
                    priority_score = 0.6
                else:
                    # Hagyományos korreláció alapú
                    max_corr = max(correlation_data.btc_correlation, 
                                  correlation_data.eth_correlation)
                    if max_corr < self.MIN_CORRELATION:
                        continue
                    priority_score = max_corr * 0.4
                
                # Opportunity score számítás
                opportunity_score = self._calculate_opportunity_score_v2(
                    correlation_data, active_events, priority_score
                )
                
                if opportunity_score > 0.4:  # Alacsonyabb küszöb
                    opportunities.append({
                        'pair': pair,
                        'correlation_data': correlation_data,
                        'opportunity_score': opportunity_score,
                        'independence_score': correlation_data.independence_score,
                        'is_independent': correlation_data.is_independent_move,
                        'primary_correlation': 'INDEPENDENT' if correlation_data.is_independent_move 
                                             else ('BTC' if correlation_data.btc_correlation > correlation_data.eth_correlation else 'ETH'),
                        'momentum_events': active_events,
                        'entry_urgency': self._calculate_entry_urgency_v2(correlation_data, active_events),
                        'expected_move_pct': self._estimate_expected_move_v2(correlation_data, active_events)
                    })
            
            # Rendezés: független mozgás prioritás
            opportunities.sort(
                key=lambda x: (x['is_independent'], x['independence_score'], x['opportunity_score']), 
                reverse=True
            )
            
            logger.info(f"🎯 Found {len(opportunities)} opportunities (independence focus)")
            
            return opportunities[:10]
            
        except Exception as e:
            logger.error(f"Independent opportunities search failed: {e}")
            return []

    def _calculate_opportunity_score_v2(self, correlation_data: CorrelationData, 
                                       active_events: List[MomentumEvent],
                                       priority_score: float) -> float:
        """Opportunity score V2 - FÜGGETLEN MOZGÁS fókusz"""
        try:
            score = 0.0
            
            # 1. Independence score (50% súly!)
            independence_component = correlation_data.independence_score * 0.5
            
            # 2. Traditional correlation (csak 20% súly)
            max_corr = max(abs(correlation_data.btc_correlation), 
                          abs(correlation_data.eth_correlation))
            correlation_component = max_corr * 0.2
            
            # 3. Momentum alignment (20% súly)
            momentum_component = 0
            if correlation_data.btc_momentum_aligned:
                momentum_component += 0.1
            if correlation_data.eth_momentum_aligned:
                momentum_component += 0.1
            
            # 4. Active events (10% súly)
            event_component = 0
            for event in active_events:
                if event.confidence > 0.7:
                    event_component += 0.05
            event_component = min(0.1, event_component)
            
            # Összegzés
            total_score = (independence_component + 
                          correlation_component + 
                          momentum_component + 
                          event_component)
            
            # Independence bonus
            if correlation_data.is_independent_move:
                total_score += 0.2
            
            # Delayed correlation penalty
            if correlation_data.delayed_correlation:
                total_score *= 0.7
            
            return min(1.0, total_score)
            
        except Exception as e:
            logger.error(f"Opportunity score v2 calculation failed: {e}")
            return 0.3

    def _calculate_entry_urgency_v2(self, correlation_data: CorrelationData, 
                                   active_events: List[MomentumEvent]) -> float:
        """Belépési sürgősség V2 - független mozgás prioritás"""
        urgency = 0.3  # Alacsonyabb base urgency
        
        # Független mozgás = magas sürgősség
        if correlation_data.is_independent_move:
            urgency += 0.5
        elif correlation_data.independence_score > 0.7:
            urgency += 0.3
        
        # Hagyományos korrelációs sürgősség (csökkentett)
        max_corr = max(abs(correlation_data.btc_correlation), 
                      abs(correlation_data.eth_correlation))
        if max_corr > 0.8:
            urgency += 0.2  # Csökkentett urgency
        
        # Active events (csak ha van korreláció)
        if not correlation_data.is_independent_move:
            for event in active_events:
                if event.confidence > 0.8:
                    urgency += 0.1  # Csökkentett event urgency
        
        # Optimal entry window
        if correlation_data.optimal_entry_window:
            urgency += 0.1
        
        return min(1.0, urgency)

    def _estimate_expected_move_v2(self, correlation_data: CorrelationData, 
                                  active_events: List[MomentumEvent]) -> float:
        """Várható ármozgás V2 - független mozgás alapján"""
        try:
            # Független mozgás = nagyobb várható mozgás
            if correlation_data.is_independent_move:
                base_move = 0.015  # 1.5% base move független mozgásnál
                independence_multiplier = correlation_data.independence_score * 2
            else:
                # Hagyományos momentum alapú
                base_move = max(abs(correlation_data.btc_momentum), 
                               abs(correlation_data.eth_momentum))
                independence_multiplier = 1.0
            
            # Correlation multiplier (kisebb hatás)
            max_corr = max(abs(correlation_data.btc_correlation), 
                          abs(correlation_data.eth_correlation))
            corr_multiplier = 0.5 + (max_corr * 0.5)  # 0.5-1.0 range
            
            # Event boost (csak korrelált mozgásoknál)
            event_boost = 0
            if not correlation_data.is_independent_move:
                for event in active_events:
                    event_boost = max(event_boost, event.magnitude * 0.001)
            
            expected_move = (base_move * independence_multiplier * corr_multiplier + event_boost) * 100
            return min(3.0, expected_move)  # Max 3%
            
        except Exception as e:
            logger.error(f"Expected move v2 estimation failed: {e}")
            return 0.8

    def detect_momentum_events(self) -> List[MomentumEvent]:
        """Momentum események detektálása - OPTIMALIZÁLT threshold"""
        try:
            current_time = time.time()
            events = []
            
            # 1. BTC momentum check
            btc_event = self._check_btc_momentum()
            if btc_event:
                events.append(btc_event)
            
            # 2. ETH momentum check  
            eth_event = self._check_eth_momentum()
            if eth_event:
                events.append(eth_event)
            
            # 3. Synchronized momentum check
            sync_event = self._check_synchronized_momentum()
            if sync_event:
                events.append(sync_event)
            
            # Clean old events
            self.momentum_events = [e for e in self.momentum_events 
                                  if current_time - e.timestamp < 3600]
            
            # Add new events
            for event in events:
                self.momentum_events.append(event)
                logger.info(
                    f"🚀 MOMENTUM EVENT: {event.source} {event.direction} "
                    f"magnitude: {event.magnitude:.3f}, confidence: {event.confidence:.3f}"
                )
            
            return events
            
        except Exception as e:
            logger.error(f"Momentum event detection failed: {e}")
            return []

    # BTC/ETH momentum check metódusok - OPTIMALIZÁLT threshold
    def _check_btc_momentum(self) -> Optional[MomentumEvent]:
        """BTC momentum - 0.03% threshold"""
        try:
            if len(self.btc_price_history) < 10:
                return None
            
            recent_prices = [p['price'] for p in self.btc_price_history[-5:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if abs(momentum) >= self.MOMENTUM_THRESHOLD:  # 0.0003 = 0.03%
                direction = "UP" if momentum > 0 else "DOWN"
                magnitude = min(1.0, abs(momentum) / 0.001)  # Normalizálás
                confidence = min(1.0, magnitude * 2)
                
                return MomentumEvent(
                    timestamp=time.time(),
                    source="BTC",
                    direction=direction,
                    magnitude=magnitude,
                    duration_estimate=300,
                    confidence=confidence,
                    triggered_pairs=[]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"BTC momentum check failed: {e}")
            return None

    def _check_eth_momentum(self) -> Optional[MomentumEvent]:
        """ETH momentum - 0.03% threshold"""
        try:
            if len(self.eth_price_history) < 10:
                return None
            
            recent_prices = [p['price'] for p in self.eth_price_history[-5:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if abs(momentum) >= self.MOMENTUM_THRESHOLD:  # 0.0003 = 0.03%
                direction = "UP" if momentum > 0 else "DOWN"
                magnitude = min(1.0, abs(momentum) / 0.001)
                confidence = min(1.0, magnitude * 2)
                
                return MomentumEvent(
                    timestamp=time.time(),
                    source="ETH",
                    direction=direction,
                    magnitude=magnitude,
                    duration_estimate=300,
                    confidence=confidence,
                    triggered_pairs=[]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"ETH momentum check failed: {e}")
            return None

    def _check_synchronized_momentum(self) -> Optional[MomentumEvent]:
        """Szinkronizált momentum - 0.03% threshold"""
        try:
            if len(self.btc_price_history) < 10 or len(self.eth_price_history) < 10:
                return None
            
            btc_recent = [p['price'] for p in self.btc_price_history[-5:]]
            btc_momentum = (btc_recent[-1] - btc_recent[0]) / btc_recent[0]
            
            eth_recent = [p['price'] for p in self.eth_price_history[-5:]]
            eth_momentum = (eth_recent[-1] - eth_recent[0]) / eth_recent[0]
            
            if (abs(btc_momentum) >= self.MOMENTUM_THRESHOLD and
                abs(eth_momentum) >= self.MOMENTUM_THRESHOLD and
                np.sign(btc_momentum) == np.sign(eth_momentum)):
                
                direction = "UP" if btc_momentum > 0 else "DOWN"
                magnitude = min(1.0, (abs(btc_momentum) + abs(eth_momentum)) / 0.002)
                confidence = min(1.0, magnitude * 1.5)
                
                return MomentumEvent(
                    timestamp=time.time(),
                    source="BTC_ETH_SYNC",
                    direction=direction,
                    magnitude=magnitude,
                    duration_estimate=600,
                    confidence=confidence,
                    triggered_pairs=[]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Synchronized momentum check failed: {e}")
            return None

    # Segéd metódusok...
    def _update_reference_data(self):
        """BTC/ETH referencia adatok - változatlan"""
        try:
            current_time = time.time()
            
            if not self.btc_price_history:
                base_btc = 50000
                for i in range(100):
                    price = base_btc * (0.98 + np.random.random() * 0.04)
                    self.btc_price_history.append({
                        'timestamp': current_time - (100-i) * 60,
                        'price': price
                    })
            
            if not self.eth_price_history:
                base_eth = 3000
                for i in range(100):
                    price = base_eth * (0.98 + np.random.random() * 0.04)
                    self.eth_price_history.append({
                        'timestamp': current_time - (100-i) * 60,
                        'price': price
                    })
            
            if current_time - self.last_update_time > 60:
                last_btc = self.btc_price_history[-1]['price']
                new_btc = last_btc * (0.999 + np.random.random() * 0.002)
                self.btc_price_history.append({
                    'timestamp': current_time,
                    'price': new_btc
                })
                
                last_eth = self.eth_price_history[-1]['price']
                new_eth = last_eth * (0.999 + np.random.random() * 0.002)
                self.eth_price_history.append({
                    'timestamp': current_time,
                    'price': new_eth
                })
                
                self.btc_price_history = self.btc_price_history[-500:]
                self.eth_price_history = self.eth_price_history[-500:]
                self.last_update_time = current_time
                
        except Exception as e:
            logger.error(f"Reference data update failed: {e}")

    def _calculate_multi_timeframe_correlations(self, pair: str, price_history: List[float]) -> Dict:
        """Multi-timeframe korreláció - változatlan algoritmus"""
        try:
            correlations = {
                'btc': {'1h': 0.5, '4h': 0.5, '24h': 0.5, 'weighted': 0.5},
                'eth': {'1h': 0.4, '4h': 0.4, '24h': 0.4, 'weighted': 0.4},
                'btc_eth': 0.7
            }
            
            if len(price_history) < 20 or len(self.btc_price_history) < 20:
                return correlations
            
            pair_prices = np.array(price_history[-100:])
            btc_prices = np.array([p['price'] for p in self.btc_price_history[-100:]])
            eth_prices = np.array([p['price'] for p in self.eth_price_history[-100:]])
            
            min_len = min(len(pair_prices), len(btc_prices), len(eth_prices))
            pair_prices = pair_prices[-min_len:]
            btc_prices = btc_prices[-min_len:]
            eth_prices = eth_prices[-min_len:]
            
            if min_len < 10:
                return correlations
            
            pair_returns = np.diff(pair_prices) / pair_prices[:-1]
            btc_returns = np.diff(btc_prices) / btc_prices[:-1]
            eth_returns = np.diff(eth_prices) / eth_prices[:-1]
            
            timeframes = {
                '1h': min(20, len(pair_returns)),
                '4h': min(60, len(pair_returns)),
                '24h': min(len(pair_returns), 480)
            }
            
            # BTC correlations
            for tf, window in timeframes.items():
                if window >= 5:
                    try:
                        corr, _ = pearsonr(pair_returns[-window:], btc_returns[-window:])
                        correlations['btc'][tf] = corr if not np.isnan(corr) else 0.5
                    except:
                        correlations['btc'][tf] = 0.5
            
            # ETH correlations  
            for tf, window in timeframes.items():
                if window >= 5:
                    try:
                        corr, _ = pearsonr(pair_returns[-window:], eth_returns[-window:])
                        correlations['eth'][tf] = corr if not np.isnan(corr) else 0.4
                    except:
                        correlations['eth'][tf] = 0.4
            
            # BTC-ETH correlation
            try:
                btc_eth_corr, _ = pearsonr(btc_returns[-60:], eth_returns[-60:])
                correlations['btc_eth'] = btc_eth_corr if not np.isnan(btc_eth_corr) else 0.7
            except:
                correlations['btc_eth'] = 0.7
            
            # Weighted correlations
            correlations['btc']['weighted'] = (
                correlations['btc']['1h'] * self.TIMEFRAME_WEIGHTS['1h'] +
                correlations['btc']['4h'] * self.TIMEFRAME_WEIGHTS['4h'] +
                correlations['btc']['24h'] * self.TIMEFRAME_WEIGHTS['24h']
            )
            
            correlations['eth']['weighted'] = (
                correlations['eth']['1h'] * self.TIMEFRAME_WEIGHTS['1h'] +
                correlations['eth']['4h'] * self.TIMEFRAME_WEIGHTS['4h'] +
                correlations['eth']['24h'] * self.TIMEFRAME_WEIGHTS['24h']
            )
            
            return correlations
            
        except Exception as e:
            logger.error(f"Multi-timeframe correlation calculation failed: {e}")
            return {
                'btc': {'1h': 0.5, '4h': 0.5, '24h': 0.5, 'weighted': 0.5},
                'eth': {'1h': 0.4, '4h': 0.4, '24h': 0.4, 'weighted': 0.4},
                'btc_eth': 0.7
            }

    def _analyze_momentum_alignment(self, pair: str, price_history: List[float]) -> Dict:
        """Momentum alignment - OPTIMALIZÁLT threshold"""
        try:
            if len(price_history) < 20 or len(self.btc_price_history) < 20:
                return {
                    'btc_momentum': 0.0, 'eth_momentum': 0.0, 'pair_momentum': 0.0,
                    'btc_aligned': False, 'eth_aligned': False, 'sync_score': 0.5
                }
            
            lookback = min(20, len(price_history))
            
            # Pair momentum
            pair_momentum = (price_history[-1] - price_history[-lookback]) / price_history[-lookback]
            
            # BTC momentum
            btc_current = self.btc_price_history[-1]['price']
            btc_past = self.btc_price_history[-lookback]['price']
            btc_momentum = (btc_current - btc_past) / btc_past
            
            # ETH momentum
            eth_current = self.eth_price_history[-1]['price']
            eth_past = self.eth_price_history[-lookback]['price']
            eth_momentum = (eth_current - eth_past) / eth_past
            
            # Alignment check - OPTIMALIZÁLT threshold (0.0003 = 0.03%)
            btc_aligned = (
                abs(btc_momentum) >= self.MOMENTUM_THRESHOLD and
                np.sign(pair_momentum) == np.sign(btc_momentum)
            )
            
            eth_aligned = (
                abs(eth_momentum) >= self.MOMENTUM_THRESHOLD and
                np.sign(pair_momentum) == np.sign(eth_momentum)
            )
            
            # Sync score
            btc_sync = abs(pair_momentum - btc_momentum) if btc_aligned else 1.0
            eth_sync = abs(pair_momentum - eth_momentum) if eth_aligned else 1.0
            sync_score = 1.0 - min(btc_sync, eth_sync, 1.0)
            
            return {
                'btc_momentum': btc_momentum,
                'eth_momentum': eth_momentum,
                'pair_momentum': pair_momentum,
                'btc_aligned': btc_aligned,
                'eth_aligned': eth_aligned,
                'sync_score': sync_score
            }
            
        except Exception as e:
            logger.error(f"Momentum alignment analysis failed: {e}")
            return {
                'btc_momentum': 0.0, 'eth_momentum': 0.0, 'pair_momentum': 0.0,
                'btc_aligned': False, 'eth_aligned': False, 'sync_score': 0.5
            }

    def _detect_correlation_trend(self, pair: str, correlations: Dict) -> str:
        """Korreláció trend - változatlan"""
        try:
            btc_short = correlations['btc']['1h']
            btc_long = correlations['btc']['24h']
            eth_short = correlations['eth']['1h']
            eth_long = correlations['eth']['24h']
            
            btc_trend = btc_short - btc_long
            eth_trend = eth_short - eth_long
            avg_trend = (btc_trend + eth_trend) / 2
            
            if avg_trend > 0.05:
                return "increasing"
            elif avg_trend < -0.05:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Correlation trend detection failed: {e}")
            return "stable"

    def _classify_correlation_strength(self, correlation: float) -> str:
        """Korreláció erősség - OPTIMALIZÁLT küszöbök"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.75:  # 75% (régen 90%)
            return "very_strong"
        elif abs_corr >= 0.6:  # 60% (régen 70%)
            return "strong"
        elif abs_corr >= 0.4:  # 40% (régen 50%)
            return "moderate"
        else:
            return "weak"

    def _create_default_correlation(self, pair: str, timeframe: str) -> CorrelationData:
        """Default korreláció - independence mezőkkel kiegészítve"""
        return CorrelationData(
            pair=pair, timeframe=timeframe,
            btc_correlation=0.5, btc_correlation_strength="moderate", btc_momentum_aligned=False,
            eth_correlation=0.4, eth_correlation_strength="moderate", eth_momentum_aligned=False,
            btc_eth_correlation=0.7, correlation_1h=0.5, correlation_4h=0.5, correlation_24h=0.5,
            correlation_trend="stable", btc_momentum=0.0, eth_momentum=0.0, pair_momentum=0.0,
            momentum_sync_score=0.5, 
            # ÚJ mezők
            independence_score=0.5, is_independent_move=False, delayed_correlation=False,
            correlation_signal="HOLD", confidence_score=0.5, optimal_entry_window=False
        )

    def _cache_correlation_data(self, pair: str, data: CorrelationData):
        """Korreláció cache - változatlan"""
        self.correlation_cache[pair] = {
            'data': data,
            'timestamp': time.time()
        }

    def _get_cached_correlation(self, pair: str) -> Optional[CorrelationData]:
        """Cache lekérés - változatlan"""
        if pair in self.correlation_cache:
            cached = self.correlation_cache[pair]
            if time.time() - cached['timestamp'] < 300:
                return cached['data']
        return None

    def get_analyzer_status(self) -> Dict:
        """Analyzer státusz - OPTIMALIZÁLT értékekkel"""
        return {
            'name': 'IndependentCorrelationAnalyzer',
            'momentum_threshold': f"{self.MOMENTUM_THRESHOLD*100:.2f}%",  # 0.03%
            'min_correlation': self.MIN_CORRELATION,  # 40%
            'independence_threshold': f"{self.INDEPENDENCE_THRESHOLD*100:.0f}%",  # 30%
            'strong_correlation': self.STRONG_CORRELATION,  # 65%
            'btc_data_points': len(self.btc_price_history),
            'eth_data_points': len(self.eth_price_history),
            'active_events': len([e for e in self.momentum_events if time.time() - e.timestamp < 300]),
            'cached_pairs': len(self.correlation_cache),
            'independent_moves_tracked': len(self.independent_moves),
            'success_rate': f"{self.successful_signals}/{self.total_signals}" if self.total_signals > 0 else "0/0",
            'focus': 'Independent movements prioritized over correlation following'
        }
