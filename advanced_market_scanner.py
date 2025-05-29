# strategy/advanced_market_scanner.py

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
    
    def __init__(self):
        # Konfiguráció
        self.MIN_VOLUME_24H = 1000000  # $1M minimum
        self.RSI_LOW = 28
        self.RSI_HIGH = 72
        self.MIN_CORRELATION = 0.8
        self.SR_PERIODS = 35
        self.SCAN_BATCH_SIZE = 5
        
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
            btc_data = self._get_ohlc_data('XBTUSD', '3')
            eth_data = self._get_ohlc_data('ETHUSD', '3')
            
            if not btc_data or not eth_data:
                return {'active': False}
            
            # BTC momentum
            btc_df = pd.DataFrame(btc_data)
            btc_price_change = (btc_df['close'].iloc[-1] - btc_df['close'].iloc[-20]) / btc_df['close'].iloc[-20]
            
            # ETH momentum  
            eth_df = pd.DataFrame(eth_data)
            eth_price_change = (eth_df['close'].iloc[-1] - eth_df['close'].iloc[-20]) / eth_df['close'].iloc[-20]
            
            # Aktivitás check (min 0.06% mozgás)
            btc_active = abs(btc_price_change) >= 0.0006
            eth_active = abs(eth_price_change) >= 0.0006
            
            momentum_data = {
                'active': btc_active or eth_active,
                'btc_momentum': btc_price_change,
                'eth_momentum': eth_price_change,
                'btc_active': btc_active,
                'eth_active': eth_active
            }
            
            # Cache frissítés
            self.btc_data_cache = momentum_data
            self.eth_data_cache = momentum_data
            
            logger.info(f"BTC momentum: {btc_price_change:.4f}, ETH: {eth_price_change:.4f}")
            return momentum_data
            
        except Exception as e:
            logger.error(f"BTC/ETH momentum analysis failed: {e}")
            return {'active': False}

    def _analyze_pair_advanced(self, pair: str) -> Optional[CoinAnalysis]:
        """Fejlett pair elemzés BOLLINGER FÓKUSSZAL"""
        try:
            # Multi-timeframe adatok
            data_3m = self._get_ohlc_data(pair, '3')
            data_15m = self._get_ohlc_data(pair, '15')
            
            if not data_3m or not data_15m:
                return None
            
            df_3m = pd.DataFrame(data_3m)
            df_15m = pd.DataFrame(data_15m)
            
            # Alapadatok
            current_price = float(df_3m['close'].iloc[-1])
            volume_24h = self._get_24h_volume(pair)
            
            if volume_24h < self.MIN_VOLUME_24H:
                return None
            
            # 1. BOLLINGER ELEMZÉS (FOKOZOTT!)
            bb_analysis = self._analyze_bollinger_breakout(df_3m, df_15m)
            
            # 2. RSI multi-timeframe
            rsi_3m = self._calculate_rsi(df_3m['close'], 14)
            rsi_15m = self._calculate_rsi(df_15m['close'], 14)
            
            # RSI filter
            if not (self.RSI_LOW <= rsi_3m <= self.RSI_HIGH):
                return None
            
            # 3. MACD & momentum
            macd_3m = self._calculate_macd(df_3m['close'])
            macd_15m = self._calculate_macd(df_15m['close'])
            stoch_rsi = self._calculate_stoch_rsi(df_3m['close'])
            williams_r = self._calculate_williams_r(df_3m)
            
            # 4. Volume profil
            volume_analysis = self._analyze_volume_profile(df_3m)
            
            # 5. Support/Resistance
            sr_analysis = self._detect_support_resistance(df_3m, current_price)
            
            # 6. Korreláció számítás
            correlation = self._calculate_correlation(df_3m['close'])
            
            if correlation['btc'] < self.MIN_CORRELATION and correlation['eth'] < self.MIN_CORRELATION:
                return None
            
            # 7. SCORING (Bollinger fokozottan súlyozva)
            scores = self._calculate_advanced_scores(
                bb_analysis, volume_analysis, correlation, 
                rsi_3m, rsi_15m, macd_3m, macd_15m
            )
            
            # CoinAnalysis objektum
            analysis = CoinAnalysis(
                pair=pair,
                price=current_price,
                volume_24h=volume_24h,
                
                rsi_3m=rsi_3m,
                rsi_15m=rsi_15m,
                macd_3m=macd_3m,
                macd_15m=macd_15m,
                stoch_rsi_3m=stoch_rsi,
                williams_r_3m=williams_r,
                
                volume_ratio=volume_analysis['ratio'],
                volume_trend=volume_analysis['trend'],
                liquidity_score=volume_analysis['liquidity'],
                
                bb_position=bb_analysis['position'],
                bb_squeeze=bb_analysis['squeeze'],
                bb_breakout_potential=bb_analysis['breakout_potential'],
                
                near_support=sr_analysis['near_support'],
                near_resistance=sr_analysis['near_resistance'],
                support_strength=sr_analysis['support_strength'],
                resistance_strength=sr_analysis['resistance_strength'],
                
                btc_correlation=correlation['btc'],
                eth_correlation=correlation['eth'],
                momentum_score=scores['momentum'],
                
                technical_score=scores['technical'],
                volume_score=scores['volume'],
                momentum_score_final=scores['momentum'],
                breakout_score=scores['breakout'],  # FOKOZOTT!
                total_score=scores['total']
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Advanced pair analysis failed for {pair}: {e}")
            return None

    def _analyze_bollinger_breakout(self, df_3m: pd.DataFrame, df_15m: pd.DataFrame) -> Dict:
        """BOLLINGER KITÖRÉS ELEMZÉS - FOKOZOTT PONTSZÁM!"""
        try:
            # 3m Bollinger
            bb_3m = self._calculate_bollinger_bands(df_3m['close'], 20, 2)
            current_price = df_3m['close'].iloc[-1]
            
            # Bollinger pozíció (0=alsó, 1=felső)
            bb_width = bb_3m['upper'].iloc[-1] - bb_3m['lower'].iloc[-1]
            price_position = (current_price - bb_3m['lower'].iloc[-1]) / bb_width
            
            # Squeeze detection (szűkülő sávok)
            bb_width_avg = bb_width / df_3m['close'].iloc[-1]  # Relatív szélesség
            bb_width_history = []
            for i in range(-10, 0):  # Utolsó 10 gyertya
                try:
                    hist_width = (bb_3m['upper'].iloc[i] - bb_3m['lower'].iloc[i]) / df_3m['close'].iloc[i]
                    bb_width_history.append(hist_width)
                except:
                    pass
            
            squeeze = bb_width_avg < np.mean(bb_width_history) * 0.8 if bb_width_history else False
            
            # 🎯 KITÖRÉSI POTENCIÁL SZÁMÍTÁS (FOKOZOTT!)
            breakout_score = 0.0
            
            # 1. Felső sáv közelség (0.95+ = magas pontszám)
            if price_position >= 0.95:
                breakout_score += 0.4  # 40% már itt!
            elif price_position >= 0.90:
                breakout_score += 0.3
            elif price_position >= 0.85:
                breakout_score += 0.2
            
            # 2. Alsó sáv közelség (oversold bounce)
            elif price_position <= 0.05:
                breakout_score += 0.35
            elif price_position <= 0.10:
                breakout_score += 0.25
            
            # 3. Squeeze bónusz
            if squeeze:
                breakout_score += 0.2
            
            # 4. Volume konfirmáció a kitöréshez
            volume_spike = df_3m['volume'].iloc[-1] > df_3m['volume'].rolling(10).mean().iloc[-1] * 1.5
            if volume_spike and price_position >= 0.9:
                breakout_score += 0.15
            
            # 5. 15m timeframe konfirmáció
            bb_15m = self._calculate_bollinger_bands(df_15m['close'], 20, 2)
            price_15m_pos = (df_15m['close'].iloc[-1] - bb_15m['lower'].iloc[-1]) / (bb_15m['upper'].iloc[-1] - bb_15m['lower'].iloc[-1])
            
            if abs(price_position - price_15m_pos) < 0.1:  # Konzisztens pozíció
                breakout_score += 0.1
            
            breakout_score = min(1.0, breakout_score)  # Cap at 1.0
            
            return {
                'position': price_position,
                'squeeze': squeeze,
                'breakout_potential': breakout_score,
                'upper_band': bb_3m['upper'].iloc[-1],
                'lower_band': bb_3m['lower'].iloc[-1],
                'width_ratio': bb_width_avg
            }
            
        except Exception as e:
            logger.error(f"Bollinger analysis failed: {e}")
            return {'position': 0.5, 'squeeze': False, 'breakout_potential': 0.0}

    def _calculate_advanced_scores(self, bb_analysis: Dict, volume_analysis: Dict, 
                                 correlation: Dict, rsi_3m: float, rsi_15m: float,
                                 macd_3m: float, macd_15m: float) -> Dict:
        """Fejlett scoring BOLLINGER FÓKUSSZAL"""
        
        # 1. BOLLINGER SCORE (FOKOZOTT! 35%)
        breakout_score = bb_analysis['breakout_potential']
        
        # 2. CORRELATION SCORE (25%)
        correlation_score = max(correlation['btc'], correlation['eth'])
        if correlation['btc'] > 0.9 or correlation['eth'] > 0.9:
            correlation_score += 0.1  # Bónusz extra magas korrelációért
        
        # 3. VOLUME SCORE (20%)
        volume_score = min(1.0, volume_analysis['ratio'] / 3.0)  # 3x volume = max score
        if volume_analysis['trend'] > 0.1:  # Növekvő volume trend
            volume_score += 0.15
        
        # 4. MOMENTUM SCORE (20%)
        momentum_score = 0.0
        
        # RSI momentum
        if 35 <= rsi_3m <= 65:  # Sweet spot
            momentum_score += 0.3
        elif rsi_3m < 35:  # Oversold bounce potential
            momentum_score += 0.25
        
        # MACD konfirmáció
        if macd_3m > 0 and macd_15m > 0:  # Bullish mindkét timeframe
            momentum_score += 0.2
        elif macd_3m > macd_15m:  # Javuló momentum
            momentum_score += 0.1
        
        # TOTAL SCORE súlyozott átlag
        total_score = (
            breakout_score * self.BOLLINGER_WEIGHT +
            correlation_score * self.CORRELATION_WEIGHT +
            volume_score * self.VOLUME_WEIGHT +
            momentum_score * self.MOMENTUM_WEIGHT
        )
        
        return {
            'breakout': breakout_score,
            'correlation': correlation_score,
            'volume': volume_score,
            'momentum': momentum_score,
            'technical': (breakout_score + momentum_score) / 2,
            'total': total_score
        }

    # Segéd metódusok...
    def _get_valid_pairs(self) -> List[str]:
        """USD párok lekérése"""
        try:
            # Fallback párok ha API nem működik
            return [
                'XBTUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD',
                'DOTUSD', 'LINKUSD', 'MATICUSD', 'AVAXUSD', 'UNIUSD',
                'ALGOUSD', 'ATOMUSD', 'FILUSD', 'LTCUSD', 'BCHUSD',
                'AAVEUSD', 'COMPUSD', 'GRTUSD', 'SNXUSD', 'YFIUSD'
            ]
        except Exception as e:
            logger.error(f"Failed to get valid pairs: {e}")
            return ['XBTUSD', 'ETHUSD', 'SOLUSD']

    def _get_ohlc_data(self, pair: str, interval: str) -> Optional[List]:
        """OHLC adatok lekérése"""
        try:
            # Mock data - cseréld le valós API hívásra
            import random
            base_price = {'XBTUSD': 50000, 'ETHUSD': 3000, 'SOLUSD': 100}.get(pair, 1000)
            
            ohlc_data = []
            for i in range(100):
                price = base_price * (0.98 + random.random() * 0.04)
                volume = random.uniform(1000, 10000)
                ohlc_data.append({
                    'time': time.time() - (100-i) * 60,
                    'open': price * 0.999,
                    'high': price * 1.002,
                    'low': price * 0.998,
                    'close': price,
                    'volume': volume
                })
            return ohlc_data
        except Exception as e:
            logger.error(f"OHLC data fetch failed for {pair}: {e}")
            return None

    def _get_24h_volume(self, pair: str) -> float:
        """24h volume lekérése"""
        # Mock - cseréld le valós API hívásra
        import random
        return random.uniform(500000, 50000000)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI számítás"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0

    def _calculate_macd(self, prices: pd.Series) -> float:
        """MACD számítás"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return float(macd.iloc[-1]) if not macd.empty else 0.0

    def _calculate_stoch_rsi(self, prices: pd.Series) -> float:
        """Stochastic RSI"""
        rsi = self._calculate_rsi(prices)
        # Egyszerűsített - valós implementációhoz kell RSI history
        return min(100, max(0, rsi))

    def _calculate_williams_r(self, df: pd.DataFrame) -> float:
        """Williams %R"""
        try:
            high_14 = df['high'].rolling(14).max()
            low_14 = df['low'].rolling(14).min()
            current_close = df['close'].iloc[-1]
            williams_r = -100 * (high_14.iloc[-1] - current_close) / (high_14.iloc[-1] - low_14.iloc[-1])
            return float(williams_r)
        except:
            return -50.0

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict:
        """Bollinger Bands számítás"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Volume profil elemzés"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume trend
        recent_avg = df['volume'].tail(5).mean()
        older_avg = df['volume'].tail(20).head(15).mean()
        volume_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
        
        return {
            'ratio': volume_ratio,
            'trend': volume_trend,
            'liquidity': min(1.0, volume_ratio / 2.0)
        }

    def _detect_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Support/Resistance detection 35 gyertyával"""
        try:
            highs = df['high'].tail(self.SR_PERIODS)
            lows = df['low'].tail(self.SR_PERIODS)
            
            # Resistance szintek (local maxima)
            resistance_levels = []
            for i in range(2, len(highs) - 2):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    if highs.iloc[i] > highs.iloc[i-2] and highs.iloc[i] > highs.iloc[i+2]:
                        resistance_levels.append(highs.iloc[i])
            
            # Support szintek (local minima)
            support_levels = []
            for i in range(2, len(lows) - 2):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    if lows.iloc[i] < lows.iloc[i-2] and lows.iloc[i] < lows.iloc[i+2]:
                        support_levels.append(lows.iloc[i])
            
            # Közelség ellenőrzés (2% threshold)
            threshold = current_price * 0.02
            
            near_support = any(abs(current_price - level) <= threshold for level in support_levels)
            near_resistance = any(abs(current_price - level) <= threshold for level in resistance_levels)
            
            return {
                'near_support': near_support,
                'near_resistance': near_resistance,
                'support_strength': len(support_levels) / 10.0,
                'resistance_strength': len(resistance_levels) / 10.0
            }
            
        except Exception as e:
            logger.error(f"S/R detection failed: {e}")
            return {'near_support': False, 'near_resistance': False, 'support_strength': 0.0, 'resistance_strength': 0.0}

    def _calculate_correlation(self, prices: pd.Series) -> Dict:
        """BTC/ETH korreláció számítás"""
        try:
            # Mock correlation - cseréld le valós számításra
            import random
            btc_corr = random.uniform(0.6, 0.95)
            eth_corr = random.uniform(0.5, 0.9)
            
            return {
                'btc': btc_corr,
                'eth': eth_corr
            }
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return {'btc': 0.7, 'eth': 0.6}

    def get_scanner_status(self) -> Dict:
        """Scanner státusz"""
        return {
            'last_scan': self.last_scan_time,
            'bollinger_weight': self.BOLLINGER_WEIGHT,
            'min_correlation': self.MIN_CORRELATION,
            'rsi_range': f"{self.RSI_LOW}-{self.RSI_HIGH}",
            'min_volume': f"${self.MIN_VOLUME_24H:,}"
        }
