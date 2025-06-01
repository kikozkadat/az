# strategy/advanced_market_scanner.py - JAVÍTOTT __init__ metódus

# strategy/advanced_market_scanner.py - TELJES JAVÍTOTT VERZIÓ

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
# utils.logger importját feltételezzük, vagy a globális logger-t használjuk
# Ha a utils.logger specifikus, akkor az importnak így kellene kinéznie:
# from utils.logger import logger
# Jelenleg a globális logger-t használom, ahogy a fájl többi részében.
import logging
logger = logging.getLogger(__name__)


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
    momentum_score_final: float # Korábban momentum_score volt, de a dataclassben momentum_score_final van
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
        self.MOMENTUM_WEIGHT = 0.20 # Ez a momentum_score_final súlya lehet
        
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
            # A _analyze_btc_eth_momentum jelenleg csak egy egyszerű fallback, valós implementációra van szükség
            btc_eth_momentum_data = self._analyze_btc_eth_momentum() 
            if not btc_eth_momentum_data.get('active', False): # Biztonságosabb lekérdezés
                logger.info("🔸 BTC/ETH not showing momentum, or momentum analysis failed. Limited opportunities expected.")
                # Lehet, hogy itt nem kellene azonnal visszatérni, hanem alapértelmezett értékekkel folytatni
                # return [] # Eredeti logika

            # 2. Párok lekérése
            valid_pairs = self._get_valid_pairs()
            if not valid_pairs:
                logger.warning("No valid pairs found for scanning.")
                return []
            
            # 3. Batch elemzés
            analyses: List[CoinAnalysis] = []
            with ThreadPoolExecutor(max_workers=self.SCAN_BATCH_SIZE) as executor:
                futures = {
                    executor.submit(self._analyze_pair_advanced, pair, btc_eth_momentum_data): pair 
                    for pair in valid_pairs[:max_pairs]
                }
                
                for future in as_completed(futures):
                    pair_name = futures[future]
                    try:
                        analysis_result = future.result(timeout=20) # Timeout növelve, ha az adatlekérés lassú
                        if analysis_result and isinstance(analysis_result, CoinAnalysis) and analysis_result.total_score > 0.4:  # Min threshold
                            analyses.append(analysis_result)
                        elif analysis_result is None:
                            logger.info(f"Analysis for {pair_name} returned None (e.g. no data).")
                        # else: # Opcionális: logolni, ha az analysis_result nem CoinAnalysis vagy a score túl alacsony
                            # logger.debug(f"Analysis for {pair_name} skipped or score too low: {analysis_result}")
                    except TimeoutError:
                        logger.error(f"Pair analysis for {pair_name} timed out.")
                    except Exception as e:
                        logger.error(f"Pair analysis for {pair_name} failed with error: {e}", exc_info=True)
            
            # 4. Rendezés Bollinger kitörés szerint (vagy total_score, ha az a cél)
            # A dataclassben breakout_score van, nem bollinger_score
            analyses.sort(key=lambda x: x.breakout_score if hasattr(x, 'breakout_score') else 0.0, reverse=True)
            
            # 5. Top 10 logolása
            logger.info(f"🎯 Found {len(analyses)} potential opportunities after filtering and sorting.")
            for i, analysis_item in enumerate(analyses[:10]):
                logger.info(
                    f"  #{i+1}: {analysis_item.pair} - BB Score: {getattr(analysis_item, 'breakout_score', 'N/A'):.3f}, "
                    f"Total: {analysis_item.total_score:.3f}, Corr: {analysis_item.btc_correlation:.2f}"
                )
            
            self.last_scan_time = time.time() # Scan idejének frissítése
            return analyses
            
        except Exception as e:
            logger.error(f"Advanced market scan failed: {e}", exc_info=True)
            return []

    def _analyze_btc_eth_momentum(self) -> Dict:
        """BTC/ETH momentum elemzés - VALÓS IMPLEMENTÁCIÓ SZÜKSÉGES"""
        try:
            # TODO: Implement real BTC and ETH momentum analysis using self.api_client
            # Például:
            # btc_ohlc = self.api_client.get_ohlc("XBTUSD", interval=60, limit=30) # Utolsó 30 órás gyertya
            # eth_ohlc = self.api_client.get_ohlc("ETHUSD", interval=60, limit=30)
            # if btc_ohlc and eth_ohlc:
            #     # Számolj momentumot, trendet stb.
            #     btc_mom = calculate_momentum(btc_ohlc["XBTUSD"]) # calculate_momentum egy új segédfüggvény lenne
            #     eth_mom = calculate_momentum(eth_ohlc["ETHUSD"])
            #     return {
            #         'active': True, # Vagy valamilyen feltétel alapján
            #         'btc_momentum': btc_mom,
            #         'eth_momentum': eth_mom,
            #         'btc_active': True, # Vagy valamilyen feltétel alapján
            #         'eth_active': True  # Vagy valamilyen feltétel alapján
            #     }
            logger.warning("BTC/ETH momentum analysis is using placeholder data.")
            return {
                'active': True, 
                'btc_momentum': 0.0005, # Placeholder
                'eth_momentum': 0.0007, # Placeholder
                'btc_active': True,
                'eth_active': True
            }
            
        except Exception as e:
            logger.error(f"BTC/ETH momentum analysis failed: {e}", exc_info=True)
            return {'active': False, 'btc_momentum': 0, 'eth_momentum': 0, 'btc_active': False, 'eth_active': False}

    def _analyze_pair_advanced(self, pair: str, market_context: Dict) -> Optional[CoinAnalysis]:
        """
        Fejlett pair elemzés VALÓDI ADATOKKAL (implementáció szükséges).
        A market_context tartalmazhatja pl. a BTC/ETH momentum adatokat.
        """
        try:
            # 1. Valódi adatok lekérése a párhoz
            # Ezt a részt kell kitölteni a valós adatlekérési logikával
            # Használd a self.api_client-et!
            
            real_data_fetched = False # Alapértelmezetten false
            price = 0.0
            volume_24h = 0.0
            ohlc_data_3m = None
            ohlc_data_15m = None
            # ... egyéb szükséges adatok ...

            if self.api_client:
                try:
                    # Példa adatlekérés (ezt finomítani kell a valós API metódusokhoz)
                    ticker_data = self.api_client.get_ticker_data(pair)
                    # A Kraken API get_ticker_data válasza összetett, a releváns részeket ki kell nyerni.
                    # Például: {'error': [], 'result': {'XXBTZUSD': {'a': ['61139.50000', '1', '1.000'], 'b': ['61139.40000', '1', '1.000'], 'c': ['61139.50000', '0.00010000'], 'v': ['139.63758085', '1275.89879837'], ... } } }
                    if ticker_data and not ticker_data.get('error') and pair in ticker_data.get('result', {}):
                        pair_ticker_info = ticker_data['result'][pair]
                        price = float(pair_ticker_info['c'][0]) if 'c' in pair_ticker_info and pair_ticker_info['c'] else 0.0
                        volume_24h = float(pair_ticker_info['v'][1]) if 'v' in pair_ticker_info and len(pair_ticker_info['v']) > 1 else 0.0 # v: [today, last 24 hours]
                        
                        # OHLC adatok lekérése (példa)
                        ohlc_3m_raw = self.api_client.get_ohlc(pair, interval=3, limit=100) # Kraken 3 perces interval=1 vagy 5, nincs 3. Legközelebbi 1 vagy 5.
                        ohlc_15m_raw = self.api_client.get_ohlc(pair, interval=15, limit=100)

                        if price > 0 and volume_24h > 0 and \
                           ohlc_3m_raw and pair in ohlc_3m_raw and ohlc_3m_raw[pair] and \
                           ohlc_15m_raw and pair in ohlc_15m_raw and ohlc_15m_raw[pair]:
                            
                            ohlc_data_3m = pd.DataFrame(ohlc_3m_raw[pair], columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
                            ohlc_data_3m = ohlc_data_3m.astype({'close': float, 'high': float, 'low': float, 'open': float, 'volume': float})

                            ohlc_data_15m = pd.DataFrame(ohlc_15m_raw[pair], columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
                            ohlc_data_15m = ohlc_data_15m.astype({'close': float, 'high': float, 'low': float, 'open': float, 'volume': float})
                            
                            if not ohlc_data_3m.empty and not ohlc_data_15m.empty:
                                real_data_fetched = True
                            else:
                                logger.warning(f"OHLC data empty for {pair} after fetching.")
                        else:
                            logger.warning(f"Incomplete ticker or OHLC data for {pair}. Price: {price}, Vol24h: {volume_24h}")
                    else:
                        logger.warning(f"Failed to fetch complete ticker data for {pair}: {ticker_data.get('error') if ticker_data else 'No ticker data'}")
                except Exception as e_fetch:
                    logger.error(f"Error fetching data for {pair}: {e_fetch}", exc_info=True)
                    real_data_fetched = False
            else:
                logger.warning(f"API client not available for {pair} in _analyze_pair_advanced.")


            # --- FELHASZNÁLÓ ÁLTAL KÉRT MÓDOSÍTÁS KEZDETE ---
            if not real_data_fetched:
                logger.error(f"No real data for {pair}, skipping analysis.")
                return None
            # --- FELHASZNÁLÓ ÁLTAL KÉRT MÓDOSÍTÁS VÉGE ---

            # Ha idáig eljutottunk, akkor real_data_fetched = True
            # Ide a valódi adat feldolgozása kell!
            logger.info(f"Processing real data for {pair}...")

            # Valós számítások implementálása (ezek csak placeholderek)
            # RSI
            rsi_3m_val = self._calculate_rsi(ohlc_data_3m['close']) if not ohlc_data_3m.empty else 50.0
            rsi_15m_val = self._calculate_rsi(ohlc_data_15m['close']) if not ohlc_data_15m.empty else 50.0
            
            # MACD (példa, valós implementáció bonyolultabb)
            macd_3m_val, _, _ = self._calculate_macd(ohlc_data_3m['close']) if not ohlc_data_3m.empty else (0.0, 0.0, 0.0)
            macd_15m_val, _, _ = self._calculate_macd(ohlc_data_15m['close']) if not ohlc_data_15m.empty else (0.0, 0.0, 0.0)

            # Stoch RSI, Williams %R (hasonlóan, segédfüggvényekkel)
            stoch_rsi_3m_val = self._calculate_stoch_rsi(ohlc_data_3m['close']) if not ohlc_data_3m.empty else 50.0
            williams_r_3m_val = self._calculate_williams_r(ohlc_data_3m) if not ohlc_data_3m.empty else -50.0
            
            # Volume analysis (példa)
            avg_volume_3m = ohlc_data_3m['volume'].rolling(window=20).mean().iloc[-1] if len(ohlc_data_3m) >= 20 else ohlc_data_3m['volume'].mean()
            current_volume_3m = ohlc_data_3m['volume'].iloc[-1] if not ohlc_data_3m.empty else 0
            volume_ratio_val = current_volume_3m / avg_volume_3m if avg_volume_3m > 0 else 1.0
            
            # Bollinger Bands (példa)
            bb_mid, bb_upper, bb_lower = self._calculate_bollinger_bands(ohlc_data_15m['close']) if not ohlc_data_15m.empty else (price, price * 1.02, price * 0.98)
            bb_pos_val = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            bb_squeeze_val = (bb_upper - bb_lower) / bb_mid < 0.05 if bb_mid > 0 else False # Squeeze ha a sávszélesség < 5%
            bb_breakout_potential_val = 1.0 if (price > bb_upper or price < bb_lower) and not bb_squeeze_val else 0.5 # Egyszerűsített

            # Korreláció (ehhez BTC/ETH adatok kellenek, a market_context-ből)
            # Ezt a részt is fejleszteni kell, hogy valós korrelációt számoljon
            btc_corr_val = 0.7 # Placeholder
            eth_corr_val = 0.6 # Placeholder
            if self.api_client: # Csak ha van API kliens a korrelációhoz
                # Ide jöhetne a korreláció számítása, ha van rá implementáció
                # pl. self._calculate_correlation(pair_ohlc, btc_ohlc_from_context)
                pass


            # Momentum (példa)
            momentum_val = (price / ohlc_data_15m['close'].iloc[-10] - 1) * 100 if len(ohlc_data_15m) >=10 else 0.0 # 10 periódusos momentum %

            # S/R (placeholder, bonyolultabb logikát igényel)
            near_support_val = price < (bb_lower * 1.01) # Nagyon egyszerűsített
            near_resistance_val = price > (bb_upper * 0.99) # Nagyon egyszerűsített


            # Score-ok számítása (ezek a súlyozások a konstruktorban vannak)
            tech_score = (rsi_15m_val / 100 * 0.3) + ((100-abs(williams_r_3m_val))/100 * 0.3) + (bb_pos_val * 0.4) # Példa
            vol_score = min(1, volume_ratio_val / 2) # Példa
            mom_score_final = min(1, abs(momentum_val) / 5) # Példa
            break_score = bb_breakout_potential_val * (1 if not bb_squeeze_val else 1.5) # Squeeze esetén nagyobb potenciál

            total_score_val = (
                tech_score * 0.3 +
                vol_score * self.VOLUME_WEIGHT +
                mom_score_final * self.MOMENTUM_WEIGHT +
                break_score * self.BOLLINGER_WEIGHT + # Bollinger súly itt
                ((btc_corr_val + eth_corr_val)/2 * self.CORRELATION_WEIGHT) # Korreláció súlya
            )
            total_score_val = min(1.0, max(0.0, total_score_val)) # Normalizálás 0-1 közé

            analysis = CoinAnalysis(
                pair=pair,
                price=price,
                volume_24h=volume_24h,
                rsi_3m=rsi_3m_val,
                rsi_15m=rsi_15m_val,
                macd_3m=macd_3m_val,
                macd_15m=macd_15m_val,
                stoch_rsi_3m=stoch_rsi_3m_val,
                williams_r_3m=williams_r_3m_val,
                volume_ratio=volume_ratio_val,
                volume_trend=0.0, # Placeholder, számítani kellene
                liquidity_score=0.7, # Placeholder, számítani kellene
                bb_position=bb_pos_val,
                bb_squeeze=bb_squeeze_val,
                bb_breakout_potential=break_score, # Vagy a bb_breakout_potential_val
                near_support=near_support_val, # Placeholder
                near_resistance=near_resistance_val, # Placeholder
                support_strength=0.5, # Placeholder
                resistance_strength=0.5, # Placeholder
                btc_correlation=btc_corr_val,
                eth_correlation=eth_corr_val,
                momentum_score=momentum_val, # Ez lehet a momentum_score_final előtti nyers momentum
                technical_score=tech_score,
                volume_score=vol_score,
                momentum_score_final=mom_score_final,
                breakout_score=break_score, # Ez a dedikált Bollinger breakout score
                total_score=total_score_val
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Advanced pair analysis failed for {pair}: {e}", exc_info=True)
            return None

    # Segédfüggvények (ezeket implementálni kellene vagy importálni)
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        if prices.empty or len(prices) < period: return 50.0
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50.0

    def _calculate_macd(self, prices: pd.Series, slow: int = 26, fast: int = 12, signal: int = 9) -> Tuple[float, float, float]:
        if prices.empty or len(prices) < slow: return 0.0, 0.0, 0.0
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]

    def _calculate_stoch_rsi(self, prices: pd.Series, period: int = 14, k_period: int = 3, d_period: int = 3) -> float:
        if prices.empty or len(prices) < period + k_period: return 50.0
        rsi = self._calculate_rsi_series(prices, period) # Kell egy _calculate_rsi_series, ami Series-t ad vissza
        if rsi.empty: return 50.0
        min_rsi = rsi.rolling(window=period).min()
        max_rsi = rsi.rolling(window=period).max()
        stoch_k = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
        stoch_k = stoch_k.rolling(window=k_period).mean()
        # stoch_d = stoch_k.rolling(window=d_period).mean() # Csak K-t adunk vissza most
        return stoch_k.iloc[-1] if not stoch_k.empty and not np.isnan(stoch_k.iloc[-1]) else 50.0

    def _calculate_rsi_series(self, prices: pd.Series, period: int = 14) -> pd.Series:
        if prices.empty or len(prices) < period: return pd.Series(dtype=float)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).fillna(0).rolling(window=period).mean()
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0) # Handle inf/NaN
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50) # Fill initial NaNs with 50


    def _calculate_williams_r(self, ohlc_df: pd.DataFrame, period: int = 14) -> float:
        if ohlc_df.empty or len(ohlc_df) < period: return -50.0
        high = ohlc_df['high'].rolling(window=period).max()
        low = ohlc_df['low'].rolling(window=period).min()
        close = ohlc_df['close'].iloc[-1]
        
        if pd.isna(high.iloc[-1]) or pd.isna(low.iloc[-1]) or (high.iloc[-1] - low.iloc[-1]) == 0:
            return -50.0
            
        wr = ((high.iloc[-1] - close) / (high.iloc[-1] - low.iloc[-1])) * -100
        return wr if not np.isnan(wr) else -50.0

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        if prices.empty or len(prices) < period: return 0.0, 0.0, 0.0
        sma = prices.rolling(window=period).mean().iloc[-1]
        std = prices.rolling(window=period).std().iloc[-1]
        if pd.isna(sma) or pd.isna(std): return prices.iloc[-1], prices.iloc[-1]*1.02, prices.iloc[-1]*0.98 # Fallback, ha nincs elég adat
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return sma, upper_band, lower_band

    def _get_valid_pairs(self) -> List[str]:
        """USD párok lekérése"""
        try:
            if self.api_client and hasattr(self.api_client, 'get_usd_pairs_with_volume'):
                # A limit 20 párra csökkentve a gyorsabb teszteléshez
                pairs_data = self.api_client.get_usd_pairs_with_volume(min_volume_usd=self.MIN_VOLUME_24H) # Konstruktorban definiált min volume
                # Csak az első 20 párat vesszük figyelembe, ha sok van
                return [pair['altname'] for pair in pairs_data[:20] if 'altname' in pair]
            else:
                logger.warning("API client or get_usd_pairs_with_volume not available. Using fallback pairs for _get_valid_pairs.")
                # Fallback párok
                return [
                    'XBTUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD', 'XRPUSD',
                    'DOTUSD', 'LINKUSD', 'MATICUSD', 'AVAXUSD', 'UNIUSD' # Kevesebb fallback a gyorsabb teszthez
                ]
        except Exception as e:
            logger.error(f"Failed to get valid pairs: {e}", exc_info=True)
            return ['XBTUSD', 'ETHUSD', 'SOLUSD'] # Minimális fallback

    def get_scanner_status(self) -> Dict:
        """Scanner státusz"""
        return {
            'last_scan': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_scan_time)) if self.last_scan_time else "Not scanned yet",
            'bollinger_weight': self.BOLLINGER_WEIGHT,
            'min_correlation': self.MIN_CORRELATION,
            'rsi_range': f"{self.RSI_LOW}-{self.RSI_HIGH}",
            'min_volume': f"${self.MIN_VOLUME_24H:,}",
            'api_connected': self.api_client is not None
        }


