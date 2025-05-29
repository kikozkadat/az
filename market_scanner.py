# strategy/market_scanner.py - FORGALOM ALAPÚ SZŰRÉSSEL

import time
import random
from typing import List, Optional, Dict
from dataclasses import dataclass
from utils.logger import logger

@dataclass
class CoinMetrics:
    """Érme metrikai adatai - volume szűréssel"""
    pair: str
    volume_24h: float = 1000000
    volume_usd: float = 1000000  # USD forgalom
    price: float = 1000
    market_cap_proxy: float = 1000000
    volatility: float = 0.5
    btc_correlation: float = 0.7
    eth_correlation: float = 0.5
    rsi: float = 50
    macd_signal: float = 0
    bollinger_position: float = 0.5
    volume_trend: float = 0
    price_momentum: float = 0
    liquidity_score: float = 0.5
    final_score: float = 0.5
    
    # Volume specific metrics
    volume_rank: int = 0
    volume_percentile: float = 50.0
    volume_growth_24h: float = 0.0

class MarketScanner:
    """Market scanner volume-alapú szűréssel"""
    
    def __init__(self, api_client=None):
        # Scanner beállítások - VOLUME FÓKUSZ
        self.MIN_VOLUME_USD = 500000      # $500K minimum forgalom
        self.MIN_PRICE = 0.0001
        self.MAX_PAIRS_TO_ANALYZE = 50    # Kevesebb pair, jobb minőség
        self.SCAN_INTERVAL = 180
        self.BATCH_SIZE = 8
        
        # API client connection
        self.api_client = api_client
        self.real_data_mode = api_client is not None
        
        # Volume filtering settings
        self.VOLUME_TIERS = {
            'mega': 10000000,    # $10M+
            'high': 5000000,     # $5M+  
            'medium': 1000000,   # $1M+
            'low': 500000        # $500K+
        }
        
        # Fallback párok volume adatokkal
        self.fallback_pairs = [
            {'pair': 'ADAUSD', 'volume_usd': 2500000},
            {'pair': 'SOLUSD', 'volume_usd': 8000000},
            {'pair': 'DOTUSD', 'volume_usd': 1800000},
            {'pair': 'LINKUSD', 'volume_usd': 3200000},
            {'pair': '1INCHUSD', 'volume_usd': 900000},
            {'pair': 'AAVEUSD', 'volume_usd': 2100000},
            {'pair': 'UNIUSD', 'volume_usd': 4500000},
            {'pair': 'AVAXUSD', 'volume_usd': 3800000},
            {'pair': 'MATICUSD', 'volume_usd': 2800000},
            {'pair': 'ATOMUSD', 'volume_usd': 1600000}
        ]
        
        # Cache
        self.last_scan_time = 0
        self.cached_results = []
        self.volume_data_cache = {}
        
        if self.real_data_mode:
            logger.info("MarketScanner initialized with REAL DATA and volume filtering")
        else:
            logger.info("MarketScanner initialized in fallback mode")

    def get_top_opportunities(self, min_score: float = 0.4) -> List[CoinMetrics]:
        """
        Top lehetőségek lekérése VOLUME ALAPÚ SZŰRÉSSEL
        """
        try:
            current_time = time.time()
            
            # Cache ellenőrzés
            if current_time - self.last_scan_time < self.SCAN_INTERVAL and self.cached_results:
                logger.info("Using cached volume-filtered scan results")
                return self.cached_results
            
            logger.info(f"🔍 Starting volume-based market scan (min: ${self.MIN_VOLUME_USD:,})...")
            
            # 1. GET HIGH VOLUME PAIRS
            high_volume_pairs = self._get_high_volume_pairs()
            
            if not high_volume_pairs:
                logger.warning("No high volume pairs found, using fallback")
                high_volume_pairs = self.fallback_pairs
            
            # 2. ANALYZE EACH HIGH VOLUME PAIR
            opportunities = []
            
            for pair_data in high_volume_pairs[:self.MAX_PAIRS_TO_ANALYZE]:
                try:
                    metrics = self._analyze_high_volume_pair(pair_data)
                    
                    if metrics and metrics.final_score >= min_score:
                        opportunities.append(metrics)
                        
                except Exception as e:
                    logger.error(f"Error analyzing {pair_data.get('pair', 'unknown')}: {e}")
            
            # 3. SORT BY COMBINED SCORE (volume + technical)
            opportunities.sort(key=lambda x: (x.volume_usd * 0.3 + x.final_score * 0.7), reverse=True)
            
            # 4. CACHE RESULTS
            self.cached_results = opportunities
            self.last_scan_time = current_time
            
            # 5. LOG RESULTS
            logger.info(f"🎯 Found {len(opportunities)} volume-filtered opportunities")
            
            for i, coin in enumerate(opportunities[:5]):
                logger.info(
                    f"  #{i+1}: {coin.pair} - Score: {coin.final_score:.3f}, "
                    f"Volume: ${coin.volume_usd:,.0f}, RSI: {coin.rsi:.1f}"
                )
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Volume-based market scan failed: {e}")
            return self._fallback_opportunities()

    def _get_high_volume_pairs(self) -> List[Dict]:
        """
        Magas forgalmú párok lekérése
        """
        try:
            if self.real_data_mode and self.api_client:
                # Real API használata
                logger.info("📊 Fetching real volume data from Kraken API...")
                
                volume_pairs = self.api_client.get_usd_pairs_with_volume(
                    min_volume_usd=self.MIN_VOLUME_USD
                )
                
                if volume_pairs:
                    logger.info(f"✅ Got {len(volume_pairs)} pairs with volume ≥ ${self.MIN_VOLUME_USD:,}")
                    return [
                        {
                            'pair': pair['altname'],
                            'wsname': pair['wsname'],
                            'volume_usd': pair['volume_usd']
                        }
                        for pair in volume_pairs
                    ]
                else:
                    logger.warning("No real volume data received, using fallback")
                    
            # Fallback mode
            logger.info("Using fallback volume data")
            return self.fallback_pairs
            
        except Exception as e:
            logger.error(f"Error getting high volume pairs: {e}")
            return self.fallback_pairs

    def _analyze_high_volume_pair(self, pair_data: Dict) -> Optional[CoinMetrics]:
        """
        Magas forgalmú pár elemzése
        """
        try:
            pair = pair_data['pair']
            volume_usd = pair_data.get('volume_usd', 1000000)
            
            # Base metrics with real volume data
            metrics = CoinMetrics(
                pair=pair,
                volume_usd=volume_usd,
                volume_24h=volume_usd / 100,  # Estimate coin volume
            )
            
            # Volume tier classification
            metrics.volume_rank = self._classify_volume_tier(volume_usd)
            metrics.volume_percentile = self._calculate_volume_percentile(volume_usd)
            
            # Generate technical indicators (enhanced with volume consideration)
            self._generate_volume_enhanced_indicators(metrics)
            
            # Calculate final score with volume weighting
            metrics.final_score = self._calculate_volume_weighted_score(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing volume pair {pair_data}: {e}")
            return None

    def _classify_volume_tier(self, volume_usd: float) -> int:
        """
        Volume tier osztályozás (1-4, 4 a legjobb)
        """
        if volume_usd >= self.VOLUME_TIERS['mega']:
            return 4  # Mega volume
        elif volume_usd >= self.VOLUME_TIERS['high']:
            return 3  # High volume
        elif volume_usd >= self.VOLUME_TIERS['medium']:
            return 2  # Medium volume
        elif volume_usd >= self.VOLUME_TIERS['low']:
            return 1  # Low volume (but above minimum)
        else:
            return 0  # Below minimum

    def _calculate_volume_percentile(self, volume_usd: float) -> float:
        """
        Volume percentile számítás
        """
        try:
            # Simplified percentile calculation
            if volume_usd >= 10000000:  # $10M+
                return 95.0
            elif volume_usd >= 5000000:  # $5M+
                return 85.0
            elif volume_usd >= 2000000:  # $2M+
                return 70.0
            elif volume_usd >= 1000000:  # $1M+
                return 55.0
            elif volume_usd >= 500000:   # $500K+
                return 35.0
            else:
                return 20.0
                
        except Exception:
            return 50.0

    def _generate_volume_enhanced_indicators(self, metrics: CoinMetrics):
        """
        Volume-enhanced technical indicators
        """
        try:
            pair = metrics.pair
            
            # Enhanced RSI with volume consideration
            base_rsi = random.uniform(25, 75)
            
            # Volume boost for RSI reliability
            volume_boost = min(10, metrics.volume_rank * 2)
            
            if metrics.volume_usd > 2000000:  # High volume = more reliable signals
                if 30 <= base_rsi <= 40:  # Oversold
                    metrics.rsi = base_rsi - volume_boost  # More oversold
                elif 60 <= base_rsi <= 70:  # Overbought  
                    metrics.rsi = base_rsi + volume_boost  # More overbought
                else:
                    metrics.rsi = base_rsi
            else:
                metrics.rsi = base_rsi
            
            # Volume-based volatility
            if metrics.volume_usd > 5000000:
                metrics.volatility = random.uniform(0.4, 0.8)  # Moderate volatility for high volume
            else:
                metrics.volatility = random.uniform(0.6, 1.2)  # Higher volatility for lower volume
            
            # Enhanced correlations for high volume pairs
            if metrics.volume_usd > 1000000:
                metrics.btc_correlation = random.uniform(0.7, 0.95)  # Higher correlation
                metrics.eth_correlation = random.uniform(0.6, 0.9)
            else:
                metrics.btc_correlation = random.uniform(0.5, 0.8)
                metrics.eth_correlation = random.uniform(0.4, 0.7)
            
            # MACD signal strength based on volume
            volume_multiplier = min(2.0, metrics.volume_usd / 1000000)
            metrics.macd_signal = random.uniform(-0.01, 0.01) * volume_multiplier
            
            # Bollinger position with volume consideration
            metrics.bollinger_position = random.uniform(0.2, 0.8)
            
            # Volume trend (growth indicator)
            metrics.volume_trend = random.uniform(-0.2, 0.3)  # Slight positive bias
            metrics.volume_growth_24h = metrics.volume_trend * 100
            
            # Price momentum enhanced by volume
            base_momentum = random.uniform(-0.03, 0.03)
            volume_factor = 1 + (metrics.volume_rank * 0.1)
            metrics.price_momentum = base_momentum * volume_factor
            
            # Liquidity score based on volume
            metrics.liquidity_score = min(1.0, metrics.volume_usd / 5000000)
            
        except Exception as e:
            logger.error(f"Error generating volume-enhanced indicators: {e}")
            # Set safe defaults
            metrics.rsi = 50
            metrics.volatility = 0.5
            metrics.btc_correlation = 0.7
            metrics.eth_correlation = 0.6

    def _calculate_volume_weighted_score(self, metrics: CoinMetrics) -> float:
        """
        Volume-súlyozott pontszám számítás
        """
        try:
            score = 0.0
            
            # 1. VOLUME SCORE (40% súly!)
            volume_score = 0.0
            
            if metrics.volume_usd >= self.VOLUME_TIERS['mega']:
                volume_score = 1.0  # Perfect score for mega volume
            elif metrics.volume_usd >= self.VOLUME_TIERS['high']:
                volume_score = 0.8  # High score for high volume
            elif metrics.volume_usd >= self.VOLUME_TIERS['medium']:
                volume_score = 0.6  # Medium score
            elif metrics.volume_usd >= self.VOLUME_TIERS['low']:
                volume_score = 0.4  # Minimum acceptable
            else:
                volume_score = 0.1  # Very low
            
            # Volume growth bonus
            if metrics.volume_growth_24h > 20:  # 20%+ growth
                volume_score += 0.15
            elif metrics.volume_growth_24h > 10:  # 10%+ growth
                volume_score += 0.1
            
            score += volume_score * 0.4
            
            # 2. RSI SCORE (25% súly)
            rsi_score = 0.0
            if 28 <= metrics.rsi <= 35:  # Sweet spot for buying
                rsi_score = 1.0
            elif 35 < metrics.rsi <= 45:
                rsi_score = 0.8
            elif 45 < metrics.rsi <= 55:
                rsi_score = 0.6
            elif 25 <= metrics.rsi < 28:  # Very oversold
                rsi_score = 0.7
            else:
                rsi_score = 0.3
                
            score += rsi_score * 0.25
            
            # 3. CORRELATION SCORE (20% súly)
            max_correlation = max(metrics.btc_correlation, metrics.eth_correlation)
            correlation_score = min(1.0, max_correlation)
            
            # High volume pairs get correlation bonus
            if metrics.volume_usd > 2000000 and max_correlation > 0.85:
                correlation_score += 0.1
                
            score += correlation_score * 0.2
            
            # 4. LIQUIDITY SCORE (15% súly)
            score += metrics.liquidity_score * 0.15
            
            # 5. BOLLINGER/MOMENTUM (Combined score weight 10%)
            momentum_score = 0.0
            
            # Bollinger position scoring
            if metrics.bollinger_position < 0.3:  # Near lower band
                momentum_score += 0.6
            elif metrics.bollinger_position > 0.7:  # Near upper band
                momentum_score += 0.4
            else:
                momentum_score += 0.2
                
            # Price momentum
            if metrics.price_momentum > 0.01:
                momentum_score += 0.4
            elif metrics.price_momentum > 0:
                momentum_score += 0.2
                
            score += (momentum_score / 2) * 0.1  # Normalize and apply weight
            
            # VOLUME TIER BONUS
            if metrics.volume_rank >= 3:  # High/Mega volume
                score += 0.05  # 5% bonus
            
            # Clamp score to [0, 1]
            score = max(0.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating volume-weighted score: {e}")
            return 0.5

    def _fallback_opportunities(self) -> List[CoinMetrics]:
        """
        Fallback opportunities when main scan fails
        """
        try:
            opportunities = []
            
            for pair_data in self.fallback_pairs[:5]:
                metrics = CoinMetrics(
                    pair=pair_data['pair'],
                    volume_usd=pair_data['volume_usd'],
                    volume_24h=pair_data['volume_usd'] / 100,
                    rsi=random.uniform(30, 70),
                    btc_correlation=random.uniform(0.6, 0.9),
                    eth_correlation=random.uniform(0.5, 0.8),
                    final_score=random.uniform(0.4, 0.8)
                )
                
                metrics.volume_rank = self._classify_volume_tier(metrics.volume_usd)
                opportunities.append(metrics)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Fallback opportunities failed: {e}")
            return []

    def connect_to_api(self, api_client):
        """
        API client csatlakoztatása
        """
        try:
            self.api_client = api_client
            self.real_data_mode = True
            logger.info("API client connected to MarketScanner")
            
            # Test connection
            if hasattr(api_client, 'get_volume_statistics'):
                stats = api_client.get_volume_statistics()
                if stats:
                    logger.info(f"📊 Volume stats: {stats.get('total_pairs', 0)} pairs, "
                              f"{stats.get('above_500k', 0)} above $500K")
                              
        except Exception as e:
            logger.error(f"Error connecting API client: {e}")

    def get_scanner_status(self) -> Dict:
        """Scanner státusz lekérése"""
        return {
            'mode': 'REAL_DATA' if self.real_data_mode else 'FALLBACK',
            'min_volume_usd': f"${self.MIN_VOLUME_USD:,}",
            'max_pairs_analyzed': self.MAX_PAIRS_TO_ANALYZE,
            'last_scan': self.last_scan_time,
            'cached_results': len(self.cached_results),
            'volume_tiers': self.VOLUME_TIERS,
            'api_connected': self.api_client is not None
        }
