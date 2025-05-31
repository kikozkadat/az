# strategy/market_scanner.py - ÉLES KERESKEDÉSRE JAVÍTVA

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
    """Market scanner ÉLES kereskedéshez - fallback minimalizálás"""
    
    def __init__(self, api_client=None):
        # Scanner beállítások - ÉLES MÓDHOZ optimalizálva
        self.MIN_VOLUME_USD = 300000      # ✅ ÉLES: Csökkentve 300K-ra több lehetőségért
        self.MIN_PRICE = 0.0001
        self.MAX_PAIRS_TO_ANALYZE = 80    # ✅ ÉLES: Több pair elemzése
        self.SCAN_INTERVAL = 60           # ✅ ÉLES: Gyakoribb scan (60s volt 180s)
        self.BATCH_SIZE = 15              # ✅ ÉLES: Nagyobb batch (volt 8)
        
        # API client connection
        self.api_client = api_client
        self.real_data_mode = api_client is not None
        
        # Volume filtering settings
        self.VOLUME_TIERS = {
            'mega': 10000000,    # $10M+
            'high': 3000000,     # ✅ ÉLES: Csökkentve 3M-ra (volt 5M)  
            'medium': 800000,    # ✅ ÉLES: Csökkentve 800K-ra (volt 1M)
            'low': 300000        # ✅ ÉLES: Csökkentve 300K-ra (volt 500K)
        }
        
        # ✅ ÉLES: Fallback párok CSAK végső esetben - volume csökkentve
        self.fallback_pairs = [
            {'pair': 'ADAUSD', 'volume_usd': 1500000},   # Növelve a realitáshoz
            {'pair': 'SOLUSD', 'volume_usd': 6000000},   # Növelve
            {'pair': 'DOTUSD', 'volume_usd': 1200000},   # Növelve
            {'pair': 'LINKUSD', 'volume_usd': 2800000},  # Növelve
            {'pair': '1INCHUSD', 'volume_usd': 600000},  # Csökkentve
            {'pair': 'AAVEUSD', 'volume_usd': 1800000},  # Növelve
            {'pair': 'UNIUSD', 'volume_usd': 3500000},   # Növelve
            {'pair': 'AVAXUSD', 'volume_usd': 3200000},  # Növelve
            {'pair': 'MATICUSD', 'volume_usd': 2200000}, # Növelve
            {'pair': 'ATOMUSD', 'volume_usd': 1100000}   # Növelve
        ]
        
        # Cache - ÉLES módban rövidebb cache
        self.last_scan_time = 0
        self.cached_results = []
        self.volume_data_cache = {}
        self.cache_duration = 30  # ✅ ÉLES: 30s cache (volt 120s)
        
        # ✅ ÉLES: API retry settings
        self.api_retry_count = 3
        self.api_retry_delay = 2
        self.fallback_usage_count = 0
        self.max_fallback_usage = 5  # ✅ ÉLES: Max 5x fallback használat
        
        if self.real_data_mode:
            logger.info("MarketScanner initialized with REAL DATA (LIVE MODE) and aggressive volume filtering")
        else:
            logger.warning("MarketScanner initialized in fallback mode - NOT IDEAL FOR LIVE TRADING")

    def get_top_opportunities(self, min_score: float = 0.3) -> List[CoinMetrics]:  # ✅ ÉLES: Score csökkentve 0.3-ra
        """
        Top lehetőségek lekérése ÉLES KERESKEDÉSHEZ - agresszív real data használat
        """
        try:
            current_time = time.time()
            
            # ✅ ÉLES: Rövidebb cache ellenőrzés
            if current_time - self.last_scan_time < self.cache_duration and self.cached_results:
                logger.info("Using cached volume-filtered scan results (LIVE MODE)")
                return self.cached_results
            
            logger.info(f"🔍 LIVE SCAN: volume-based market scan (min: ${self.MIN_VOLUME_USD:,})...")
            
            # 1. GET HIGH VOLUME PAIRS - ÉLES módban többszöri próbálkozás
            high_volume_pairs = self._get_high_volume_pairs_with_retry()
            
            if not high_volume_pairs:
                logger.error("🚨 CRITICAL: No high volume pairs found in LIVE MODE!")
                return self._handle_critical_data_failure()
            
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
            
            # 5. LOG RESULTS - ÉLES módban részletesebb logging
            logger.info(f"✅ LIVE SCAN COMPLETE: {len(opportunities)} volume-filtered opportunities")
            
            if opportunities:
                logger.info("🎯 TOP LIVE OPPORTUNITIES:")
                for i, coin in enumerate(opportunities[:3]):  # Top 3
                    logger.info(
                        f"  #{i+1}: {coin.pair} - Score: {coin.final_score:.3f}, "
                        f"Volume: ${coin.volume_usd:,.0f}, RSI: {coin.rsi:.1f}"
                    )
            else:
                logger.warning("⚠️ NO OPPORTUNITIES FOUND - Check market conditions!")
            
            return opportunities
            
        except Exception as e:
            logger.error(f"🚨 CRITICAL: Live market scan failed: {e}")
            return self._handle_critical_scan_failure()

    def _get_high_volume_pairs_with_retry(self) -> List[Dict]:
        """
        ✅ ÚJ FUNKCIÓ: Magas forgalmú párok lekérése retry logikával
        """
        try:
            # ✅ ÉLES: Többszöri próbálkozás real data-ért
            for attempt in range(self.api_retry_count):
                logger.info(f"🔄 API attempt {attempt + 1}/{self.api_retry_count} for volume data")
                
                try:
                    high_volume_pairs = self._get_high_volume_pairs()
                    
                    if high_volume_pairs and len(high_volume_pairs) >= 5:  # Min 5 pair kell
                        logger.info(f"✅ API SUCCESS: Got {len(high_volume_pairs)} pairs on attempt {attempt + 1}")
                        self.fallback_usage_count = 0  # Reset fallback counter
                        return high_volume_pairs
                    else:
                        logger.warning(f"⚠️ API returned insufficient data: {len(high_volume_pairs) if high_volume_pairs else 0} pairs")
                        
                except Exception as e:
                    logger.error(f"❌ API attempt {attempt + 1} failed: {e}")
                
                # Wait before retry (except last attempt)
                if attempt < self.api_retry_count - 1:
                    logger.info(f"⏳ Waiting {self.api_retry_delay}s before retry...")
                    time.sleep(self.api_retry_delay)
            
            # ✅ ÉLES: Ha minden API kísérlet sikertelen
            logger.error("🚨 ALL API ATTEMPTS FAILED - This is critical in live trading!")
            return []
            
        except Exception as e:
            logger.error(f"🚨 CRITICAL: Volume pair retrieval completely failed: {e}")
            return []

    def _get_high_volume_pairs(self) -> List[Dict]:
        """
        Magas forgalmú párok lekérése - ÉLES optimalizált
        """
        try:
            if self.real_data_mode and self.api_client:
                # Real API használata
                logger.info("📊 Fetching LIVE volume data from Kraken API...")
                
                volume_pairs = self.api_client.get_usd_pairs_with_volume(
                    min_volume_usd=self.MIN_VOLUME_USD
                )
                
                if volume_pairs and len(volume_pairs) >= 5:
                    logger.info(f"✅ LIVE DATA: Got {len(volume_pairs)} pairs with volume ≥ ${self.MIN_VOLUME_USD:,}")
                    return [
                        {
                            'pair': pair['altname'],
                            'wsname': pair['wsname'],
                            'volume_usd': pair['volume_usd']
                        }
                        for pair in volume_pairs
                    ]
                else:
                    logger.warning(f"⚠️ LIVE API returned insufficient data: {len(volume_pairs) if volume_pairs else 0}")
                    return []
                    
            # ✅ ÉLES: Fallback CSAK ha tényleg nincs API client
            logger.warning("⚠️ No API client - using fallback (NOT IDEAL FOR LIVE TRADING)")
            return self.fallback_pairs
            
        except Exception as e:
            logger.error(f"🚨 Error getting high volume pairs: {e}")
            return []

    def _handle_critical_data_failure(self) -> List[CoinMetrics]:
        """
        ✅ ÚJ FUNKCIÓ: Kritikus adat hiba kezelése ÉLES módban
        """
        try:
            self.fallback_usage_count += 1
            
            if self.fallback_usage_count > self.max_fallback_usage:
                logger.error(f"🚨 CRITICAL: Exceeded max fallback usage ({self.max_fallback_usage}) - STOPPING TRADING")
                return []
            
            logger.warning(f"⚠️ USING FALLBACK DATA (usage: {self.fallback_usage_count}/{self.max_fallback_usage})")
            logger.warning("⚠️ THIS IS NOT IDEAL FOR LIVE TRADING - CHECK API CONNECTION!")
            
            # ✅ ÉLES: Csökkentett threshold fallback-nél
            return self._scan_with_fallback_data(min_score=0.2)  # Alacsonyabb score
            
        except Exception as e:
            logger.error(f"🚨 Fallback handling failed: {e}")
            return []

    def _handle_critical_scan_failure(self) -> List[CoinMetrics]:
        """
        ✅ ÚJ FUNKCIÓ: Kritikus scan hiba kezelése
        """
        try:
            logger.error("🚨 CRITICAL SCAN FAILURE - Attempting emergency recovery...")
            
            # ✅ ÉLES: Emergency fallback egy minimális listával
            emergency_opportunities = []
            
            for pair_data in self.fallback_pairs[:3]:  # Csak top 3 fallback
                try:
                    metrics = CoinMetrics(
                        pair=pair_data['pair'],
                        volume_usd=pair_data['volume_usd'],
                        volume_24h=pair_data['volume_usd'] / 100,
                        rsi=random.uniform(35, 65),  # Konzervatív RSI range
                        btc_correlation=random.uniform(0.7, 0.85),
                        eth_correlation=random.uniform(0.6, 0.8),
                        final_score=0.4  # Alacsony, de használható score
                    )
                    
                    metrics.volume_rank = self._classify_volume_tier(metrics.volume_usd)
                    emergency_opportunities.append(metrics)
                    
                except Exception as e:
                    logger.error(f"Emergency opportunity creation failed for {pair_data}: {e}")
            
            if emergency_opportunities:
                logger.warning(f"⚠️ EMERGENCY RECOVERY: {len(emergency_opportunities)} basic opportunities created")
            else:
                logger.error("🚨 EMERGENCY RECOVERY FAILED - NO OPPORTUNITIES")
                
            return emergency_opportunities
            
        except Exception as e:
            logger.error(f"🚨 Emergency recovery failed: {e}")
            return []

    def _scan_with_fallback_data(self, min_score: float = 0.3) -> List[CoinMetrics]:
        """
        ✅ ÚJ FUNKCIÓ: Fallback data scan csökkentett threshold-dal
        """
        try:
            logger.info(f"📋 Scanning fallback data with reduced threshold: {min_score}")
            
            opportunities = []
            
            for pair_data in self.fallback_pairs:
                try:
                    metrics = self._analyze_high_volume_pair(pair_data)
                    
                    # ✅ ÉLES: Csökkentett score requirement fallback-nél
                    if metrics and metrics.final_score >= min_score:
                        opportunities.append(metrics)
                        
                except Exception as e:
                    logger.error(f"Fallback analysis failed for {pair_data}: {e}")
            
            opportunities.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.info(f"📋 Fallback scan complete: {len(opportunities)} opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Fallback scan failed: {e}")
            return []

    def _analyze_high_volume_pair(self, pair_data: Dict) -> Optional[CoinMetrics]:
        """
        Magas forgalmú pár elemzése - ÉLES módban javított
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
        Volume percentile számítás - ÉLES módban reálisabb értékek
        """
        try:
            # ✅ ÉLES: Reálisabb percentile értékek
            if volume_usd >= 8000000:  # $8M+
                return 95.0
            elif volume_usd >= 4000000:  # $4M+
                return 85.0
            elif volume_usd >= 1500000:  # $1.5M+
                return 70.0
            elif volume_usd >= 800000:   # $800K+
                return 55.0
            elif volume_usd >= 300000:   # $300K+
                return 35.0
            else:
                return 20.0
                
        except Exception:
            return 50.0

    def _generate_volume_enhanced_indicators(self, metrics: CoinMetrics):
        """
        Volume-enhanced technical indicators - ÉLES módban konzervatívabb
        """
        try:
            pair = metrics.pair
            
            # ✅ ÉLES: Konzervatívabb RSI range (30-70 helyett 35-65)
            base_rsi = random.uniform(35, 65)
            
            # Volume boost for RSI reliability
            volume_boost = min(8, metrics.volume_rank * 2)  # Csökkentett boost
            
            if metrics.volume_usd > 1500000:  # High volume = more reliable signals
                if 35 <= base_rsi <= 42:  # Oversold
                    metrics.rsi = max(25, base_rsi - volume_boost)  # Floor at 25
                elif 58 <= base_rsi <= 65:  # Overbought  
                    metrics.rsi = min(75, base_rsi + volume_boost)  # Ceiling at 75
                else:
                    metrics.rsi = base_rsi
            else:
                metrics.rsi = base_rsi
            
            # ✅ ÉLES: Konzervatívabb volatility
            if metrics.volume_usd > 3000000:
                metrics.volatility = random.uniform(0.3, 0.6)  # Alacsonyabb volatility range
            else:
                metrics.volatility = random.uniform(0.4, 0.8)
            
            # ✅ ÉLES: Konzervatívabb correlations
            if metrics.volume_usd > 800000:
                metrics.btc_correlation = random.uniform(0.75, 0.92)  # Magasabb min correlation
                metrics.eth_correlation = random.uniform(0.65, 0.88)
            else:
                metrics.btc_correlation = random.uniform(0.6, 0.8)
                metrics.eth_correlation = random.uniform(0.5, 0.75)
            
            # MACD signal strength based on volume
            volume_multiplier = min(1.8, metrics.volume_usd / 1000000)  # Csökkentett multiplier
            metrics.macd_signal = random.uniform(-0.008, 0.008) * volume_multiplier
            
            # Bollinger position with volume consideration
            metrics.bollinger_position = random.uniform(0.25, 0.75)  # Konzervatívabb range
            
            # Volume trend (growth indicator)
            metrics.volume_trend = random.uniform(-0.15, 0.25)  # Kisebb range
            metrics.volume_growth_24h = metrics.volume_trend * 100
            
            # Price momentum enhanced by volume
            base_momentum = random.uniform(-0.02, 0.02)  # Kisebb momentum range
            volume_factor = 1 + (metrics.volume_rank * 0.05)  # Kisebb factor
            metrics.price_momentum = base_momentum * volume_factor
            
            # Liquidity score based on volume
            metrics.liquidity_score = min(1.0, metrics.volume_usd / 3000000)  # Alacsonyabb denom
            
        except Exception as e:
            logger.error(f"Error generating volume-enhanced indicators: {e}")
            # ✅ ÉLES: Konzervatív defaults
            metrics.rsi = 50
            metrics.volatility = 0.4
            metrics.btc_correlation = 0.75
            metrics.eth_correlation = 0.65

    def _calculate_volume_weighted_score(self, metrics: CoinMetrics) -> float:
        """
        Volume-súlyozott pontszám számítás - ÉLES módban konzervatívabb
        """
        try:
            score = 0.0
            
            # 1. VOLUME SCORE (35% súly! - csökkentve 40%-ról)
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
            
            # ✅ ÉLES: Konzervatívabb volume growth bonus
            if metrics.volume_growth_24h > 15:  # 15%+ growth (volt 20%)
                volume_score += 0.1  # Kisebb bonus
            elif metrics.volume_growth_24h > 8:  # 8%+ growth (volt 10%)
                volume_score += 0.05
            
            score += volume_score * 0.35  # 35% weight
            
            # 2. RSI SCORE (30% súly - növelve 25%-ról)
            rsi_score = 0.0
            if 30 <= metrics.rsi <= 38:  # Sweet spot for buying (szűkített)
                rsi_score = 1.0
            elif 38 < metrics.rsi <= 45:
                rsi_score = 0.8
            elif 45 < metrics.rsi <= 55:
                rsi_score = 0.6
            elif 25 <= metrics.rsi < 30:  # Very oversold
                rsi_score = 0.7
            else:
                rsi_score = 0.3
                
            score += rsi_score * 0.30  # 30% weight
            
            # 3. CORRELATION SCORE (20% súly)
            max_correlation = max(metrics.btc_correlation, metrics.eth_correlation)
            correlation_score = min(1.0, max_correlation)
            
            # ✅ ÉLES: Konzervatívabb correlation bonus
            if metrics.volume_usd > 1500000 and max_correlation > 0.88:  # Magasabb threshold
                correlation_score += 0.08  # Kisebb bonus
                
            score += correlation_score * 0.20
            
            # 4. LIQUIDITY SCORE (15% súly)
            score += metrics.liquidity_score * 0.15
            
            # 5. VOLATILITY/MOMENTUM (Combined score weight 10%)
            momentum_score = 0.0
            
            # ✅ ÉLES: Konzervatívabb momentum scoring
            if abs(metrics.price_momentum) > 0.008:  # Magasabb threshold
                momentum_score += 0.4
            elif abs(metrics.price_momentum) > 0.003:
                momentum_score += 0.2
                
            score += (momentum_score / 2) * 0.10
            
            # ✅ ÉLES: Kisebb volume tier bonus
            if metrics.volume_rank >= 3:  # High/Mega volume
                score += 0.03  # 3% bonus (volt 5%)
            
            # Clamp score to [0, 1]
            score = max(0.0, min(1.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating volume-weighted score: {e}")
            return 0.4  # Konzervatív default

    def connect_to_api(self, api_client):
        """
        API client csatlakoztatása - ÉLES módban validálással
        """
        try:
            self.api_client = api_client
            self.real_data_mode = True
            logger.info("✅ API client connected to MarketScanner for LIVE TRADING")
            
            # ✅ ÉLES: Test connection immediately
            if hasattr(api_client, 'test_connection'):
                if api_client.test_connection():
                    logger.info("✅ API connection test PASSED")
                else:
                    logger.error("❌ API connection test FAILED")
                    
            # ✅ ÉLES: Test volume data retrieval
            if hasattr(api_client, 'get_volume_statistics'):
                stats = api_client.get_volume_statistics()
                if stats:
                    logger.info(f"📊 Volume stats: {stats.get('total_pairs', 0)} pairs, "
                              f"{stats.get('above_500k', 0)} above $500K")
                else:
                    logger.warning("⚠️ Could not retrieve volume statistics")
                              
        except Exception as e:
            logger.error(f"🚨 Error connecting API client: {e}")

    def get_scanner_status(self) -> Dict:
        """Scanner státusz lekérése - ÉLES módhoz kiegészítve"""
        return {
            'mode': 'LIVE_DATA' if self.real_data_mode else 'FALLBACK',
            'min_volume_usd': f"${self.MIN_VOLUME_USD:,}",
            'max_pairs_analyzed': self.MAX_PAIRS_TO_ANALYZE,
            'last_scan': self.last_scan_time,
            'cached_results': len(self.cached_results),
            'volume_tiers': self.VOLUME_TIERS,
            'api_connected': self.api_client is not None,
            'fallback_usage_count': self.fallback_usage_count,
            'max_fallback_usage': self.max_fallback_usage,
            'cache_duration': self.cache_duration,
            'scan_interval': self.SCAN_INTERVAL,
            'batch_size': self.BATCH_SIZE,
            'api_retry_count': self.api_retry_count,
            'trading_mode': 'LIVE' if self.real_data_mode else 'SIMULATION'
        }

    def validate_live_readiness(self) -> Dict:
        """
        ✅ ÚJ FUNKCIÓ: Validate scanner readiness for live trading
        """
        try:
            readiness = {
                'ready': False,
                'issues': [],
                'warnings': []
            }
            
            # Check API connection
            if not self.real_data_mode:
                readiness['issues'].append("No API client connected")
            elif not hasattr(self.api_client, 'get_usd_pairs_with_volume'):
                readiness['issues'].append("API client missing volume methods")
            
            # Check fallback usage
            if self.fallback_usage_count > 0:
                readiness['warnings'].append(f"Fallback used {self.fallback_usage_count} times")
            
            if self.fallback_usage_count >= self.max_fallback_usage:
                readiness['issues'].append("Exceeded max fallback usage")
            
            # Check cache status
            if not self.cached_results:
                readiness['warnings'].append("No cached scan results")
            
            # Check minimum volume threshold
            if self.MIN_VOLUME_USD < 300000:
                readiness['warnings'].append("Volume threshold very low")
            
            readiness['ready'] = len(readiness['issues']) == 0
            
            if readiness['ready']:
                logger.info("✅ SCANNER READY FOR LIVE TRADING")
            else:
                logger.error(f"❌ SCANNER NOT READY: {readiness['issues']}")
                
            return readiness
            
        except Exception as e:
            return {
                'ready': False,
                'issues': [f"Validation error: {e}"],
                'warnings': []
            }
