# strategy/correlation_analyzer.py

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
    btc_correlation_strength: str  # "weak", "moderate", "strong", "very_strong"
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
    correlation_trend: str  # "increasing", "decreasing", "stable"
    
    # Momentum analysis
    btc_momentum: float
    eth_momentum: float
    pair_momentum: float
    momentum_sync_score: float  # 0-1, how well synchronized
    
    # Trading signals
    correlation_signal: str  # "STRONG_BUY", "BUY", "HOLD", "AVOID"
    confidence_score: float
    optimal_entry_window: bool

@dataclass
class MomentumEvent:
    """BTC/ETH momentum esemény"""
    timestamp: float
    source: str  # "BTC" or "ETH"
    direction: str  # "UP" or "DOWN"
    magnitude: float  # 0-1
    duration_estimate: float  # seconds
    confidence: float  # 0-1
    triggered_pairs: List[str]

class CorrelationAnalyzer:
    """BTC/ETH korreláció és momentum elemzés"""
    
    def __init__(self):
        # 🎯 KORRELÁCIÓ BEÁLLÍTÁSOK
        self.MIN_CORRELATION = 0.8      # Min korreláció trading-hez
        self.STRONG_CORRELATION = 0.9   # Erős korreláció threshold
        self.MOMENTUM_THRESHOLD = 0.0006  # 0.06% min momentum (módosítva)
        self.SYNC_WINDOW = 300          # 5 perc szinkronizációs ablak
        
        # Timeframe weights
        self.TIMEFRAME_WEIGHTS = {
            '1h': 0.5,    # 50% weight for recent correlation
            '4h': 0.3,    # 30% weight for medium-term
            '24h': 0.2    # 20% weight for long-term
        }
        
        # Data storage
        self.btc_price_history = []
        self.eth_price_history = []
        self.correlation_cache = {}
        self.momentum_events = []
        
        # Performance tracking
        self.successful_signals = 0
        self.total_signals = 0
        self.last_update_time = 0
        
        logger.info("CorrelationAnalyzer initialized with 0.06% momentum threshold")

    def analyze_pair_correlation(self, pair: str, price_history: List[float], 
                                timeframe: str = "3m") -> CorrelationData:
        """
        Pair korreláció elemzés BTC/ETH-val
        """
        try:
            if len(price_history) < 20:
                logger.warning(f"Insufficient price history for {pair}")
                return self._create_default_correlation(pair, timeframe)
            
            # 1. BTC/ETH referencia adatok frissítése
            self._update_reference_data()
            
            # 2. Multi-timeframe korrelációk
            correlations = self._calculate_multi_timeframe_correlations(pair, price_history)
            
            # 3. Momentum elemzés
            momentum_data = self._analyze_momentum_alignment(pair, price_history)
            
            # 4. Dinamikus korreláció trend
            correlation_trend = self._detect_correlation_trend(pair, correlations)
            
            # 5. Trading signal generálás
            signal_data = self._generate_correlation_signal(correlations, momentum_data)
            
            # 6. Optimal entry window detection
            entry_window = self._detect_optimal_entry_window(momentum_data, correlations)
            
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
                
                correlation_signal=signal_data['signal'],
                confidence_score=signal_data['confidence'],
                optimal_entry_window=entry_window
            )
            
            # Cache eredmény
            self._cache_correlation_data(pair, correlation_data)
            
            logger.info(
                f"🔗 {pair}: BTC corr: {correlations['btc']['weighted']:.3f}, "
                f"ETH corr: {correlations['eth']['weighted']:.3f}, "
                f"Signal: {signal_data['signal']}"
            )
            
            return correlation_data
            
        except Exception as e:
            logger.error(f"Correlation analysis failed for {pair}: {e}")
            return self._create_default_correlation(pair, timeframe)

    def detect_momentum_events(self) -> List[MomentumEvent]:
        """
        BTC/ETH momentum események detektálása
        """
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
            
            # 4. Clean old events
            self.momentum_events = [e for e in self.momentum_events 
                                  if current_time - e.timestamp < 3600]  # Keep 1 hour
            
            # 5. Add new events
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

    def find_correlated_opportunities(self, available_pairs: List[str]) -> List[Dict]:
        """
        Korrelált kereskedési lehetőségek keresése
        """
        try:
            opportunities = []
            
            # Active momentum events check
            active_events = [e for e in self.momentum_events 
                           if time.time() - e.timestamp < 300]  # Last 5 minutes
            
            if not active_events:
                logger.info("No active momentum events for correlation trading")
                return []
            
            # Analyze each pair
            for pair in available_pairs:
                # Skip BTC and ETH themselves
                if pair in ['XBTUSD', 'BTCUSD', 'ETHUSD']:
                    continue
                
                # Get cached correlation data
                correlation_data = self._get_cached_correlation(pair)
                if not correlation_data:
                    continue
                
                # Check correlation thresholds
                max_correlation = max(correlation_data.btc_correlation, 
                                    correlation_data.eth_correlation)
                
                if max_correlation < self.MIN_CORRELATION:
                    continue
                
                # Calculate opportunity score
                opportunity_score = self._calculate_opportunity_score(
                    correlation_data, active_events
                )
                
                if opportunity_score > 0.6:  # Minimum opportunity score
                    opportunities.append({
                        'pair': pair,
                        'correlation_data': correlation_data,
                        'opportunity_score': opportunity_score,
                        'primary_correlation': 'BTC' if correlation_data.btc_correlation > correlation_data.eth_correlation else 'ETH',
                        'momentum_events': [e for e in active_events if pair in e.triggered_pairs or not e.triggered_pairs],
                        'entry_urgency': self._calculate_entry_urgency(correlation_data, active_events),
                        'expected_move_pct': self._estimate_expected_move(correlation_data, active_events)
                    })
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
            
            logger.info(f"🎯 Found {len(opportunities)} correlated opportunities")
            
            return opportunities[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"Correlated opportunities search failed: {e}")
            return []

    def _update_reference_data(self):
        """BTC/ETH referencia adatok frissítése"""
        try:
            current_time = time.time()
            
            # Simulate BTC/ETH price updates (replace with real data)
            # In real implementation, fetch from API
            
            if not self.btc_price_history:
                # Initialize with some data
                base_btc = 50000
                for i in range(100):
                    price = base_btc * (0.98 + np.random.random() * 0.04)
                    self.btc_price_history.append({
                        'timestamp': current_time - (100-i) * 60,
                        'price': price
                    })
            
            if not self.eth_price_history:
                # Initialize with some data
                base_eth = 3000
                for i in range(100):
                    price = base_eth * (0.98 + np.random.random() * 0.04)
                    self.eth_price_history.append({
                        'timestamp': current_time - (100-i) * 60,
                        'price': price
                    })
            
            # Add new data points (simulate real-time updates)
            if current_time - self.last_update_time > 60:  # Update every minute
                
                # BTC update
                last_btc = self.btc_price_history[-1]['price']
                new_btc = last_btc * (0.999 + np.random.random() * 0.002)
                self.btc_price_history.append({
                    'timestamp': current_time,
                    'price': new_btc
                })
                
                # ETH update
                last_eth = self.eth_price_history[-1]['price']
                new_eth = last_eth * (0.999 + np.random.random() * 0.002)
                self.eth_price_history.append({
                    'timestamp': current_time,
                    'price': new_eth
                })
                
                # Keep only last 500 data points
                self.btc_price_history = self.btc_price_history[-500:]
                self.eth_price_history = self.eth_price_history[-500:]
                
                self.last_update_time = current_time
                
        except Exception as e:
            logger.error(f"Reference data update failed: {e}")

    def _calculate_multi_timeframe_correlations(self, pair: str, price_history: List[float]) -> Dict:
        """Multi-timeframe korreláció számítás"""
        try:
            correlations = {
                'btc': {'1h': 0.7, '4h': 0.7, '24h': 0.7, 'weighted': 0.7},
                'eth': {'1h': 0.6, '4h': 0.6, '24h': 0.6, 'weighted': 0.6},
                'btc_eth': 0.8
            }
            
            if len(price_history) < 20 or len(self.btc_price_history) < 20:
                return correlations
            
            # Get price arrays
            pair_prices = np.array(price_history[-100:])  # Last 100 points
            btc_prices = np.array([p['price'] for p in self.btc_price_history[-100:]])
            eth_prices = np.array([p['price'] for p in self.eth_price_history[-100:]])
            
            # Ensure same length
            min_len = min(len(pair_prices), len(btc_prices), len(eth_prices))
            pair_prices = pair_prices[-min_len:]
            btc_prices = btc_prices[-min_len:]
            eth_prices = eth_prices[-min_len:]
            
            if min_len < 10:
                return correlations
            
            # Calculate returns
            pair_returns = np.diff(pair_prices) / pair_prices[:-1]
            btc_returns = np.diff(btc_prices) / btc_prices[:-1]
            eth_returns = np.diff(eth_prices) / eth_prices[:-1]
            
            # Different timeframe windows
            timeframes = {
                '1h': min(20, len(pair_returns)),    # Last 20 points (~1h for 3m data)
                '4h': min(60, len(pair_returns)),    # Last 60 points (~4h)
                '24h': min(len(pair_returns), 480)  # Up to 480 points (~24h)
            }
            
            # BTC correlations
            for tf, window in timeframes.items():
                if window >= 5:  # Minimum window
                    try:
                        corr, _ = pearsonr(pair_returns[-window:], btc_returns[-window:])
                        correlations['btc'][tf] = corr if not np.isnan(corr) else 0.7
                    except:
                        correlations['btc'][tf] = 0.7
            
            # ETH correlations
            for tf, window in timeframes.items():
                if window >= 5:
                    try:
                        corr, _ = pearsonr(pair_returns[-window:], eth_returns[-window:])
                        correlations['eth'][tf] = corr if not np.isnan(corr) else 0.6
                    except:
                        correlations['eth'][tf] = 0.6
            
            # BTC-ETH correlation
            try:
                btc_eth_corr, _ = pearsonr(btc_returns[-60:], eth_returns[-60:])
                correlations['btc_eth'] = btc_eth_corr if not np.isnan(btc_eth_corr) else 0.8
            except:
                correlations['btc_eth'] = 0.8
            
            # Weighted correlations (recent data more important)
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
                'btc': {'1h': 0.7, '4h': 0.7, '24h': 0.7, 'weighted': 0.7},
                'eth': {'1h': 0.6, '4h': 0.6, '24h': 0.6, 'weighted': 0.6},
                'btc_eth': 0.8
            }

    def _analyze_momentum_alignment(self, pair: str, price_history: List[float]) -> Dict:
        """Momentum alignment elemzés"""
        try:
            if len(price_history) < 20 or len(self.btc_price_history) < 20:
                return {
                    'btc_momentum': 0.0, 'eth_momentum': 0.0, 'pair_momentum': 0.0,
                    'btc_aligned': False, 'eth_aligned': False, 'sync_score': 0.5
                }
            
            # Calculate momentums (20-period change)
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
            
            # Alignment check (same direction and significant magnitude)
            btc_aligned = (
                abs(btc_momentum) >= self.MOMENTUM_THRESHOLD and
                np.sign(pair_momentum) == np.sign(btc_momentum)
            )
            
            eth_aligned = (
                abs(eth_momentum) >= self.MOMENTUM_THRESHOLD and
                np.sign(pair_momentum) == np.sign(eth_momentum)
            )
            
            # Synchronization score (0-1)
            btc_sync = abs(pair_momentum - btc_momentum) if btc_aligned else 1.0
            eth_sync = abs(pair_momentum - eth_momentum) if eth_aligned else 1.0
            
            # Overall sync score (lower is better, convert to 0-1 where 1 is best)
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
        """Korreláció trend detektálás"""
        try:
            # Compare short vs long term correlations
            btc_short = correlations['btc']['1h']
            btc_long = correlations['btc']['24h']
            
            eth_short = correlations['eth']['1h']
            eth_long = correlations['eth']['24h']
            
            # Average trend
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

    def _generate_correlation_signal(self, correlations: Dict, momentum_data: Dict) -> Dict:
        """Korreláció alapú trading signal"""
        try:
            # Base correlation scores
            btc_corr = correlations['btc']['weighted']
            eth_corr = correlations['eth']['weighted']
            max_corr = max(btc_corr, eth_corr)
            
            # Momentum alignment bonus
            momentum_bonus = 0
            if momentum_data['btc_aligned'] or momentum_data['eth_aligned']:
                momentum_bonus = 0.2
            
            if momentum_data['btc_aligned'] and momentum_data['eth_aligned']:
                momentum_bonus = 0.4  # Both aligned = strong signal
            
            # Sync score bonus
            sync_bonus = momentum_data['sync_score'] * 0.3
            
            # Calculate confidence
            confidence = max_corr + momentum_bonus + sync_bonus
            confidence = min(1.0, confidence)
            
            # Generate signal
            if (max_corr >= self.STRONG_CORRELATION and 
                (momentum_data['btc_aligned'] or momentum_data['eth_aligned']) and
                momentum_data['sync_score'] > 0.7):
                signal = "STRONG_BUY"
            elif (max_corr >= self.MIN_CORRELATION and
                  (momentum_data['btc_aligned'] or momentum_data['eth_aligned'])):
                signal = "BUY"
            elif max_corr >= 0.7:
                signal = "HOLD"
            else:
                signal = "AVOID"
            
            return {
                'signal': signal,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Correlation signal generation failed: {e}")
            return {'signal': 'HOLD', 'confidence': 0.5}

    def _detect_optimal_entry_window(self, momentum_data: Dict, correlations: Dict) -> bool:
        """Optimális belépési ablak detektálás"""
        try:
            # Check multiple conditions for optimal entry
            conditions = []
            
            # 1. Strong momentum alignment
            if momentum_data['btc_aligned'] and momentum_data['eth_aligned']:
                conditions.append(True)
            elif momentum_data['btc_aligned'] or momentum_data['eth_aligned']:
                conditions.append(0.5)  # Partial
            else:
                conditions.append(False)
            
            # 2. High synchronization
            conditions.append(momentum_data['sync_score'] > 0.6)
            
            # 3. Strong correlation
            max_corr = max(correlations['btc']['weighted'], correlations['eth']['weighted'])
            conditions.append(max_corr >= self.MIN_CORRELATION)
            
            # 4. Recent correlation strength (1h > 4h trend)
            btc_trend = correlations['btc']['1h'] > correlations['btc']['4h']
            eth_trend = correlations['eth']['1h'] > correlations['eth']['4h']
            conditions.append(btc_trend or eth_trend)
            
            # Evaluate conditions
            score = sum(1 for c in conditions if c is True)
            score += sum(0.5 for c in conditions if c == 0.5)
            
            return score >= 3.0  # Need at least 3 out of 4 conditions
            
        except Exception as e:
            logger.error(f"Optimal entry window detection failed: {e}")
            return False

    def _check_btc_momentum(self) -> Optional[MomentumEvent]:
        """BTC momentum esemény ellenőrzés"""
        try:
            if len(self.btc_price_history) < 20:
                return None
            
            # Recent price change (last 5 minutes)
            recent_prices = [p['price'] for p in self.btc_price_history[-5:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if abs(momentum) >= self.MOMENTUM_THRESHOLD:
                direction = "UP" if momentum > 0 else "DOWN"
                magnitude = min(1.0, abs(momentum) / 0.002)  # Normalize to 0-1
                confidence = min(1.0, magnitude * 2)
                
                return MomentumEvent(
                    timestamp=time.time(),
                    source="BTC",
                    direction=direction,
                    magnitude=magnitude,
                    duration_estimate=300,  # 5 minutes estimated
                    confidence=confidence,
                    triggered_pairs=[]  # Will be filled by caller
                )
            
            return None
            
        except Exception as e:
            logger.error(f"BTC momentum check failed: {e}")
            return None

    def _check_eth_momentum(self) -> Optional[MomentumEvent]:
        """ETH momentum esemény ellenőrzés"""
        try:
            if len(self.eth_price_history) < 20:
                return None
            
            # Recent price change (last 5 minutes)
            recent_prices = [p['price'] for p in self.eth_price_history[-5:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if abs(momentum) >= self.MOMENTUM_THRESHOLD:
                direction = "UP" if momentum > 0 else "DOWN"
                magnitude = min(1.0, abs(momentum) / 0.002)
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
        """Szinkronizált BTC/ETH momentum"""
        try:
            if len(self.btc_price_history) < 10 or len(self.eth_price_history) < 10:
                return None
            
            # BTC momentum
            btc_recent = [p['price'] for p in self.btc_price_history[-5:]]
            btc_momentum = (btc_recent[-1] - btc_recent[0]) / btc_recent[0]
            
            # ETH momentum
            eth_recent = [p['price'] for p in self.eth_price_history[-5:]]
            eth_momentum = (eth_recent[-1] - eth_recent[0]) / eth_recent[0]
            
            # Check if both have significant momentum in same direction
            if (abs(btc_momentum) >= self.MOMENTUM_THRESHOLD and
                abs(eth_momentum) >= self.MOMENTUM_THRESHOLD and
                np.sign(btc_momentum) == np.sign(eth_momentum)):
                
                direction = "UP" if btc_momentum > 0 else "DOWN"
                magnitude = min(1.0, (abs(btc_momentum) + abs(eth_momentum)) / 0.004)
                confidence = min(1.0, magnitude * 1.5)
                
                return MomentumEvent(
                    timestamp=time.time(),
                    source="BTC_ETH_SYNC",
                    direction=direction,
                    magnitude=magnitude,
                    duration_estimate=600,  # 10 minutes for sync events
                    confidence=confidence,
                    triggered_pairs=[]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Synchronized momentum check failed: {e}")
            return None

    def _calculate_opportunity_score(self, correlation_data: CorrelationData, 
                                   active_events: List[MomentumEvent]) -> float:
        """Lehetőség pontszám számítás"""
        try:
            score = 0.0
            
            # Base correlation score (40%)
            max_corr = max(correlation_data.btc_correlation, correlation_data.eth_correlation)
            correlation_score = max_corr * 0.4
            
            # Momentum alignment score (30%)
            momentum_score = 0
            if correlation_data.btc_momentum_aligned:
                momentum_score += 0.15
            if correlation_data.eth_momentum_aligned:
                momentum_score += 0.15
            
            # Sync score (20%)
            sync_score = correlation_data.momentum_sync_score * 0.2
            
            # Active events bonus (10%)
            event_score = 0
            for event in active_events:
                if event.confidence > 0.7:
                    event_score += 0.05
                if event.magnitude > 0.8:
                    event_score += 0.05
            event_score = min(0.1, event_score)
            
            total_score = correlation_score + momentum_score + sync_score + event_score
            return min(1.0, total_score)
            
        except Exception as e:
            logger.error(f"Opportunity score calculation failed: {e}")
            return 0.5

    def _calculate_entry_urgency(self, correlation_data: CorrelationData, 
                               active_events: List[MomentumEvent]) -> float:
        """Belépési sürgősség számítás"""
        urgency = 0.5  # Base urgency
        
        # High correlation = higher urgency
        max_corr = max(correlation_data.btc_correlation, correlation_data.eth_correlation)
        if max_corr > 0.95:
            urgency += 0.3
        elif max_corr > 0.9:
            urgency += 0.2
        
        # Active high-confidence events
        for event in active_events:
            if event.confidence > 0.8:
                urgency += 0.2
        
        # Optimal entry window
        if correlation_data.optimal_entry_window:
            urgency += 0.1
        
        return min(1.0, urgency)

    def _estimate_expected_move(self, correlation_data: CorrelationData, 
                              active_events: List[MomentumEvent]) -> float:
        """Várható ármozgás becslés"""
        try:
            # Base move from momentum
            base_move = max(abs(correlation_data.btc_momentum), abs(correlation_data.eth_momentum))
            
            # Correlation multiplier
            max_corr = max(correlation_data.btc_correlation, correlation_data.eth_correlation)
            corr_multiplier = max_corr
            
            # Event magnitude boost
            event_boost = 0
            for event in active_events:
                event_boost = max(event_boost, event.magnitude * 0.001)  # Convert to price %
            
            expected_move = (base_move * corr_multiplier + event_boost) * 100  # Convert to %
            return min(2.0, expected_move)  # Cap at 2%
            
        except Exception as e:
            logger.error(f"Expected move estimation failed: {e}")
            return 0.5

    # Helper methods
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Korreláció erősség osztályozás"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.9:
            return "very_strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        else:
            return "weak"

    def _create_default_correlation(self, pair: str, timeframe: str) -> CorrelationData:
        """Default korreláció adat létrehozás"""
        return CorrelationData(
            pair=pair, timeframe=timeframe,
            btc_correlation=0.7, btc_correlation_strength="moderate", btc_momentum_aligned=False,
            eth_correlation=0.6, eth_correlation_strength="moderate", eth_momentum_aligned=False,
            btc_eth_correlation=0.8, correlation_1h=0.7, correlation_4h=0.7, correlation_24h=0.7,
            correlation_trend="stable", btc_momentum=0.0, eth_momentum=0.0, pair_momentum=0.0,
            momentum_sync_score=0.5, correlation_signal="HOLD", confidence_score=0.5,
            optimal_entry_window=False
        )

    def _cache_correlation_data(self, pair: str, data: CorrelationData):
        """Korreláció adat cache-elés"""
        self.correlation_cache[pair] = {
            'data': data,
            'timestamp': time.time()
        }

    def _get_cached_correlation(self, pair: str) -> Optional[CorrelationData]:
        """Cache-elt korreláció adat lekérés"""
        if pair in self.correlation_cache:
            cached = self.correlation_cache[pair]
            if time.time() - cached['timestamp'] < 300:  # 5 min cache
                return cached['data']
        return None

    def get_analyzer_status(self) -> Dict:
        """Analyzer státusz"""
        return {
            'name': 'CorrelationAnalyzer',
            'momentum_threshold': f"{self.MOMENTUM_THRESHOLD*100:.2f}%",
            'min_correlation': self.MIN_CORRELATION,
            'btc_data_points': len(self.btc_price_history),
            'eth_data_points': len(self.eth_price_history),
            'active_events': len([e for e in self.momentum_events if time.time() - e.timestamp < 300]),
            'cached_pairs': len(self.correlation_cache),
            'success_rate': f"{self.successful_signals}/{self.total_signals}" if self.total_signals > 0 else "0/0"
        }
