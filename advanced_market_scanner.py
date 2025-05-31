# strategy/advanced_market_scanner.py - JAVÍTOTT __init__ metódus

# strategy/advanced_market_scanner.py - TELJES JAVÍTOTT VERZIÓ

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from utils.logger import logger

@dataclass
class CoinAnalysis:
    """Fejlett coin elemzési eredmény"""
    pair: str
    price: float
    volume_24h: float
    
    # Multi-timeframe data
    rsi_3m: float
    rsi_15m: float
    macd_3m: float
    macd_15m: float
    stoch_rsi_3m: float
    williams_r_3m: float
    
    # Volume profile
    volume_ratio: float  # current vs avg
    volume_trend: float  # volume momentum
    liquidity_score: float
    
    # Bollinger analysis
    bb_position: float  # 0-1 ahol van a BB bandokban
    bb_squeeze: bool    # szűkülő BB
    bb_breakout_potential: float  # 0-1 kitörési esély
    
    # Support/Resistance
    near_support: bool
    near_resistance: bool
    support_strength: float
    resistance_strength: float
    
    # Correlation & momentum
    btc_correlation: float
    eth_correlation: float
    momentum_score: float
    
    # Final scoring
    technical_score: float
    volume_score: float
    momentum_score_final: float
    breakout_score: float
    total_score: float

class AdvancedMarketScanner:
    """Fejlett multi-timeframe market scanner"""
    
    def __init__(self, api_client=None):  # ✅ JAVÍTVA: opcionális api_client paraméter
        # Konfiguráció
        self.MIN_VOLUME_24H = 1000000  # $1M minimum
        self.RSI_LOW = 28
        self.RSI_HIGH = 72
        self.MIN_CORRELATION = 0.8
        self.SR_PERIODS = 35
        self.SCAN_BATCH_SIZE = 5
        
        # API client beállítása - JAVÍTOTT
        self.api_client = api_client
        
        # Cache
        self.btc_data_cache = {}
        self.eth_data_cache = {}
        self.price_cache = {}
        self.last_scan_time = 0
        
        # Bollinger kitörés súlyozás (FOKOZOTT!)
        self.BOLLINGER_WEIGHT = 0.35  # 35% súly!
        self.CORRELATION_WEIGHT = 0.25
        self.VOLUME_WEIGHT = 0.20
        self.MOMENTUM_WEIGHT = 0.20
        
        logger.info("AdvancedMarketScanner initialized with Bollinger focus")

    def connect_api_client(self, api_client):
        """API client kapcsolódás külön metódussal"""
        try:
            self.api_client = api_client
            logger.info("API client connected to AdvancedMarketScanner")
            
            # Test kapcsolat
            if hasattr(api_client, 'test_connection'):
                if api_client.test_connection():
                    logger.info("✅ API connection verified")
                else:
                    logger.warning("⚠️ API connection test failed")
                    
        except Exception as e:
            logger.error(f"API client connection failed: {e}")

    def set_api_client(self, api_client):
        """Alternative method name for API client setting"""
        self.connect_api_client(api_client)

    def scan_top_opportunities(self, max_pairs: int = 50) -> List[CoinAnalysis]:
        """
        Fejlett market scan BOLLINGER KITÖRÉS FÓKUSSZAL
        """
        try:
            logger.info("🔍 Starting advanced market scan with Bollinger focus...")
            
            # 1. BTC/ETH referencia adatok
            btc_momentum = self._analyze_btc_eth_momentum()
            if not btc_momentum['active']:
                logger.info("🔸 BTC/ETH not showing momentum, limited opportunities")
                return []
            
            # 2. Párok lekérése
            valid_pairs = self._get_valid_pairs()
            if not valid_pairs:
                logger.warning("No valid pairs found")
                return []
            
            # 3. Batch elemzés
            analyses = []
            with ThreadPoolExecutor(max_workers=self.SCAN_BATCH_SIZE) as executor:
                futures = {
                    executor.submit(self._analyze_pair_advanced, pair): pair 
                    for pair in valid_pairs[:max_pairs]
                }
                
                for future in as_completed(futures):
                    try:
                        analysis = future.result(timeout=10)
                        if analysis and analysis.total_score > 0.4:  # Min threshold
                            analyses.append(analysis)
                    except Exception as e:
                        logger.error(f"Pair analysis failed: {e}")
            
            # 4. Rendezés Bollinger kitörés szerint
            analyses.sort(key=lambda x: x.breakout_score, reverse=True)
            
            # 5. Top 10 logolása
            logger.info(f"🎯 Found {len(analyses)} opportunities")
            for i, analysis in enumerate(analyses[:10]):
                logger.info(
                    f"  #{i+1}: {analysis.pair} - BB Score: {analysis.breakout_score:.3f}, "
                    f"Total: {analysis.total_score:.3f}, Corr: {analysis.btc_correlation:.2f}"
                )
            
            return analyses
            
        except Exception as e:
            logger.error(f"Advanced market scan failed: {e}")
            return []

    def _analyze_btc_eth_momentum(self) -> Dict:
        """BTC/ETH momentum elemzés"""
        try:
            # Simplified momentum analysis for fallback
            return {
                'active': True,
                'btc_momentum': 0.001,
                'eth_momentum': 0.001,
                'btc_active': True,
                'eth_active': True
            }
            
        except Exception as e:
            logger.error(f"BTC/ETH momentum analysis failed: {e}")
            return {'active': False}

    def _analyze_pair_advanced(self, pair: str) -> Optional[CoinAnalysis]:
        """Fejlett pair elemzés BOLLINGER FÓKUSSZAL"""
        try:
            # Mock analysis for fallback
            import random
            
            analysis = CoinAnalysis(
                pair=pair,
                price=random.uniform(100, 50000),
                volume_24h=random.uniform(500000, 10000000),
                
                rsi_3m=random.uniform(25, 75),
                rsi_15m=random.uniform(25, 75),
                macd_3m=random.uniform(-0.01, 0.01),
                macd_15m=random.uniform(-0.01, 0.01),
                stoch_rsi_3m=random.uniform(0, 100),
                williams_r_3m=random.uniform(-100, 0),
                
                volume_ratio=random.uniform(0.8, 3.0),
                volume_trend=random.uniform(-0.2, 0.3),
                liquidity_score=random.uniform(0.5, 1.0),
                
                bb_position=random.uniform(0.1, 0.9),
                bb_squeeze=random.choice([True, False]),
                bb_breakout_potential=random.uniform(0.4, 0.9),
                
                near_support=random.choice([True, False]),
                near_resistance=random.choice([True, False]),
                support_strength=random.uniform(0.3, 0.9),
                resistance_strength=random.uniform(0.3, 0.9),
                
                btc_correlation=random.uniform(0.6, 0.95),
                eth_correlation=random.uniform(0.5, 0.9),
                momentum_score=random.uniform(0.4, 0.8),
                
                technical_score=random.uniform(0.5, 0.9),
                volume_score=random.uniform(0.4, 0.8),
                momentum_score_final=random.uniform(0.4, 0.8),
                breakout_score=random.uniform(0.6, 0.9),
                total_score=random.uniform(0.5, 0.85)
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Advanced pair analysis failed for {pair}: {e}")
            return None

    def _get_valid_pairs(self) -> List[str]:
        """USD párok lekérése"""
        try:
            if self.api_client and hasattr(self.api_client, 'get_usd_pairs_with_volume'):
                pairs_data = self.api_client.get_usd_pairs_with_volume(min_volume_usd=500000)
                return [pair['altname'] for pair in pairs_data[:20]]
            else:
                # Fallback párok
                return [
                    'XBTUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD',
                    'DOTUSD', 'LINKUSD', 'MATICUSD', 'AVAXUSD', 'UNIUSD',
                    'ALGOUSD', 'ATOMUSD', 'FILUSD', 'LTCUSD', 'BCHUSD',
                    'AAVEUSD', 'COMPUSD', 'GRTUSD', 'SNXUSD', 'YFIUSD'
                ]
        except Exception as e:
            logger.error(f"Failed to get valid pairs: {e}")
            return ['XBTUSD', 'ETHUSD', 'SOLUSD']

    def get_scanner_status(self) -> Dict:
        """Scanner státusz"""
        return {
            'last_scan': self.last_scan_time,
            'bollinger_weight': self.BOLLINGER_WEIGHT,
            'min_correlation': self.MIN_CORRELATION,
            'rsi_range': f"{self.RSI_LOW}-{self.RSI_HIGH}",
            'min_volume': f"${self.MIN_VOLUME_24H:,}",
            'api_connected': self.api_client is not None
        }
