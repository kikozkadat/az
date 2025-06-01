# strategy/indicator_engine.py - FIXED VERSION WITH TYPE SAFETY

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class IndicatorEngine:
    """Technikai indikátor motor - javított típuskezeléssel"""
    
    def __init__(self):
        """IndicatorEngine inicializálás paraméter nélkül"""
        # Alapértelmezett beállítások
        self.default_rsi_period = 14
        self.default_ema_fast = 9
        self.default_ema_slow = 21
        self.default_bb_period = 20
        self.default_bb_std = 2
        self.default_atr_period = 14
        
        # Számítási cache
        self.calculation_cache = {}
        
    def set_api_client(self, api_client):
        """API client beállítása (opcionális)"""
        self.api_client = api_client
        
    def connect_api_client(self, api_client):
        """API client kapcsolat (opcionális)"""
        self.api_client = api_client

    @staticmethod
    def _ensure_numeric_series(series: pd.Series, default_value: float = 0.0) -> pd.Series:
        """Biztosítja, hogy a Series numerikus legyen"""
        try:
            # Konvertálás numeric típusra
            numeric_series = pd.to_numeric(series, errors='coerce')
            # NaN értékek helyettesítése
            numeric_series = numeric_series.fillna(default_value)
            return numeric_series
        except Exception as e:
            logger.error(f"Error converting series to numeric: {e}")
            return pd.Series([default_value] * len(series), index=series.index)

    @staticmethod
    def _ensure_numeric_dataframe(data: pd.DataFrame) -> pd.DataFrame:
        """Biztosítja, hogy a DataFrame összes numerikus oszlopa float típusú legyen"""
        try:
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = IndicatorEngine._ensure_numeric_series(data[col])
            return data
        except Exception as e:
            logger.error(f"Error ensuring numeric dataframe: {e}")
            return data

    @staticmethod
    def compute_rsi(data, period=14):
        """RSI számítás javított hibakezeléssel és típus biztonsággal"""
        try:
            # Ensure numeric data
            data = IndicatorEngine._ensure_numeric_dataframe(data)
            
            if len(data) < period + 1:
                return pd.Series([50] * len(data), index=data.index)
                
            close_prices = data['close']
            delta = close_prices.diff()
            
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            # Avoid division by zero
            loss = loss.replace(0, 0.0001)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN values
            rsi = rsi.fillna(50)
            return rsi
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return pd.Series([50] * len(data), index=data.index)

    @staticmethod
    def compute_stoch_rsi(data, period=14):
        """Stochastic RSI számítás típus biztonsággal"""
        try:
            rsi = IndicatorEngine.compute_rsi(data, period)
            min_val = rsi.rolling(window=period).min()
            max_val = rsi.rolling(window=period).max()
            
            # Avoid division by zero
            range_val = max_val - min_val
            range_val = range_val.replace(0, 0.0001)
            
            stoch_rsi = (rsi - min_val) / range_val
            return stoch_rsi.fillna(0.5)
            
        except Exception as e:
            logger.error(f"Stoch RSI calculation failed: {e}")
            return pd.Series([0.5] * len(data), index=data.index)

    @staticmethod
    def compute_ema(data, span):
        """EMA számítás típus biztonsággal"""
        try:
            # Ensure numeric data
            data = IndicatorEngine._ensure_numeric_dataframe(data)
            
            if len(data) < span:
                return pd.Series(data['close'].mean(), index=data.index)
                
            close_prices = data['close']
            return close_prices.ewm(span=span, adjust=False).mean()
            
        except Exception as e:
            logger.error(f"EMA calculation failed: {e}")
            return pd.Series(data['close'].mean() if 'close' in data else 1000, index=data.index)

    @staticmethod
    def compute_ema_crossover(data, fast=9, slow=21):
        """EMA crossover típus biztonsággal"""
        try:
            ema_fast = IndicatorEngine.compute_ema(data, fast)
            ema_slow = IndicatorEngine.compute_ema(data, slow)
            return ema_fast, ema_slow
        except Exception as e:
            logger.error(f"EMA crossover calculation failed: {e}")
            default_val = data['close'].mean() if 'close' in data and not data.empty else 1000
            return (pd.Series([default_val] * len(data), index=data.index),
                    pd.Series([default_val] * len(data), index=data.index))

    @staticmethod
    def compute_bollinger(data, period=20, std_factor=2):
        """Bollinger Bands számítás típus biztonsággal"""
        try:
            # Ensure numeric data
            data = IndicatorEngine._ensure_numeric_dataframe(data)
            
            if len(data) < period:
                # Not enough data, return price-based fallback
                price = float(data['close'].iloc[-1]) if not data.empty else 1000.0
                fallback_series = pd.Series([price] * len(data), index=data.index)
                upper = fallback_series * 1.02  # +2%
                lower = fallback_series * 0.98  # -2%
                return upper, fallback_series, lower
                
            close_prices = data['close']
            ma = close_prices.rolling(window=period).mean()
            std = close_prices.rolling(window=period).std()
            
            # Fill NaN values with price data
            ma = ma.fillna(close_prices)
            std = std.fillna(close_prices.std() if len(data) > 1 else 0.02 * close_prices.mean())
            
            upper_band = ma + std_factor * std
            lower_band = ma - std_factor * std
            
            return upper_band, ma, lower_band
            
        except Exception as e:
            logger.error(f"Bollinger calculation failed: {e}")
            # Emergency fallback
            price = float(data['close'].iloc[-1]) if 'close' in data and not data.empty else 1000.0
            fallback_series = pd.Series([price] * len(data), index=data.index)
            upper = fallback_series * 1.02
            lower = fallback_series * 0.98
            return upper, fallback_series, lower

    @staticmethod
    def compute_macd(data, fast=12, slow=26, signal=9):
        """MACD számítás típus biztonsággal"""
        try:
            # Ensure numeric data
            data = IndicatorEngine._ensure_numeric_dataframe(data)
            
            if len(data) < slow + signal:
                return (pd.Series([0] * len(data), index=data.index),
                        pd.Series([0] * len(data), index=data.index),
                        pd.Series([0] * len(data), index=data.index))
                        
            close_prices = data['close']
            ema_fast = close_prices.ewm(span=fast).mean()
            ema_slow = close_prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)
            
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return (pd.Series([0] * len(data), index=data.index),
                    pd.Series([0] * len(data), index=data.index),
                    pd.Series([0] * len(data), index=data.index))

    @staticmethod
    def compute_atr(data, period=14):
        """Average True Range számítás típus biztonsággal"""
        try:
            # Ensure numeric data
            data = IndicatorEngine._ensure_numeric_dataframe(data)
            
            if len(data) < period + 1:
                return pd.Series([0.01] * len(data), index=data.index)
                
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr.fillna(0.01)
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return pd.Series([0.01] * len(data), index=data.index)

    @staticmethod
    def compute_williams_r(data, period=14):
        """Williams %R számítás típus biztonsággal"""
        try:
            # Ensure numeric data
            data = IndicatorEngine._ensure_numeric_dataframe(data)
            
            if len(data) < period:
                return pd.Series([-50] * len(data), index=data.index)
                
            high = data['high']
            low = data['low']
            close = data['close']
            
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            # Avoid division by zero
            range_val = highest_high - lowest_low
            range_val = range_val.replace(0, 0.0001)
            
            williams_r = -100 * (highest_high - close) / range_val
            return williams_r.fillna(-50)
            
        except Exception as e:
            logger.error(f"Williams %R calculation failed: {e}")
            return pd.Series([-50] * len(data), index=data.index)

    @staticmethod
    def compute_volume_sma(data, period=20):
        """Volume Simple Moving Average típus biztonsággal"""
        try:
            if 'volume' not in data.columns:
                return pd.Series([1000] * len(data), index=data.index)
                
            # Ensure numeric volume
            volume = pd.to_numeric(data['volume'], errors='coerce').fillna(1000)
            return volume.rolling(window=period).mean().fillna(1000)
            
        except Exception as e:
            logger.error(f"Volume SMA calculation failed: {e}")
            return pd.Series([1000] * len(data), index=data.index)

    def compute_all_indicators(self, data):
        """Összes indikátor számítása egyszerre"""
        try:
            # Ensure numeric data first
            data = self._ensure_numeric_dataframe(data)
            
            indicators = {}
            
            # Basic indicators
            indicators['rsi'] = self.compute_rsi(data)
            indicators['ema_fast'], indicators['ema_slow'] = self.compute_ema_crossover(data)
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self.compute_bollinger(data)
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.compute_macd(data)
            indicators['atr'] = self.compute_atr(data)
            indicators['williams_r'] = self.compute_williams_r(data)
            indicators['stoch_rsi'] = self.compute_stoch_rsi(data)
            indicators['volume_sma'] = self.compute_volume_sma(data)
            
            return indicators
            
        except Exception as e:
            logger.error(f"All indicators calculation failed: {e}")
            return {}

    def compute_scalping_signals(self, data):
        """Scalping jelek számítása - JAVÍTOTT TÍPUSKEZELÉSSEL"""
        try:
            # KRITIKUS: Ensure all data is numeric before processing
            data = self._ensure_numeric_dataframe(data)
            
            signals = {}
            
            # RSI jelzések
            rsi = self.compute_rsi(data, period=14)
            if not rsi.empty:
                current_rsi = float(rsi.iloc[-1])  # Explicit float conversion
                signals['rsi_oversold'] = current_rsi < 30
                signals['rsi_overbought'] = current_rsi > 70
                signals['rsi_neutral'] = 30 <= current_rsi <= 70
                signals['rsi_value'] = current_rsi
            
            # Bollinger Bands jelzések
            bb_upper, bb_middle, bb_lower = self.compute_bollinger(data)
            if not bb_upper.empty and not data.empty:
                # Ensure all values are float
                current_price = float(data['close'].iloc[-1])
                upper_price = float(bb_upper.iloc[-1])
                lower_price = float(bb_lower.iloc[-1])
                middle_price = float(bb_middle.iloc[-1])
                
                signals['bb_breakout_up'] = current_price > upper_price
                signals['bb_breakout_down'] = current_price < lower_price
                signals['bb_squeeze'] = (upper_price - lower_price) / middle_price < 0.04  # 4% width
                
                # Safe division
                band_width = upper_price - lower_price
                if band_width > 0:
                    signals['bb_position'] = (current_price - lower_price) / band_width
                else:
                    signals['bb_position'] = 0.5
            
            # EMA jelzések
            ema_fast, ema_slow = self.compute_ema_crossover(data)
            if not ema_fast.empty and not ema_slow.empty:
                # Ensure float conversion
                fast_current = float(ema_fast.iloc[-1])
                slow_current = float(ema_slow.iloc[-1])
                
                signals['ema_bullish'] = fast_current > slow_current
                signals['ema_bearish'] = fast_current < slow_current
                
                # EMA crossover detection
                if len(ema_fast) > 1:
                    prev_fast = float(ema_fast.iloc[-2])
                    prev_slow = float(ema_slow.iloc[-2])
                    
                    signals['ema_golden_cross'] = (prev_fast <= prev_slow) and (fast_current > slow_current)
                    signals['ema_death_cross'] = (prev_fast >= prev_slow) and (fast_current < slow_current)
            
            # MACD jelzések
            macd, macd_signal, macd_hist = self.compute_macd(data)
            if not macd.empty:
                # Ensure float conversion
                macd_current = float(macd.iloc[-1])
                signal_current = float(macd_signal.iloc[-1])
                hist_current = float(macd_hist.iloc[-1])
                
                signals['macd_bullish'] = macd_current > signal_current
                signals['macd_bearish'] = macd_current < signal_current
                signals['macd_histogram_positive'] = hist_current > 0
                
                # MACD crossover
                if len(macd) > 1:
                    macd_prev = float(macd.iloc[-2])
                    signal_prev = float(macd_signal.iloc[-2])
                    
                    signals['macd_bullish_crossover'] = (macd_prev <= signal_prev) and (macd_current > signal_current)
                    signals['macd_bearish_crossover'] = (macd_prev >= signal_prev) and (macd_current < signal_current)
            
            # Volume jelzések
            if 'volume' in data.columns:
                volume_sma = self.compute_volume_sma(data, period=20)
                if not volume_sma.empty and not data['volume'].empty:
                    # Ensure numeric conversion
                    current_volume = float(pd.to_numeric(data['volume'].iloc[-1], errors='coerce'))
                    avg_volume = float(volume_sma.iloc[-1])
                    
                    if avg_volume > 0:
                        signals['volume_spike'] = current_volume > avg_volume * 1.5
                        signals['high_volume'] = current_volume > avg_volume * 2.0
                        signals['low_volume'] = current_volume < avg_volume * 0.5
                    else:
                        signals['volume_spike'] = False
                        signals['high_volume'] = False
                        signals['low_volume'] = True
            
            return signals
            
        except Exception as e:
            logger.error(f"Scalping signals calculation failed: {e}", exc_info=True)
            return {}

    def get_trading_recommendation(self, data):
        """Trading ajánlás generálás"""
        try:
            signals = self.compute_scalping_signals(data)
            
            if not signals:
                return {'action': 'HOLD', 'confidence': 0.5, 'reason': 'No signals available'}
            
            # Scoring system
            bullish_score = 0
            bearish_score = 0
            
            # RSI scoring
            if signals.get('rsi_oversold', False):
                bullish_score += 2
            elif signals.get('rsi_overbought', False):
                bearish_score += 2
            elif signals.get('rsi_neutral', False):
                bullish_score += 0.5
            
            # Bollinger scoring
            if signals.get('bb_breakout_up', False):
                bullish_score += 2
            elif signals.get('bb_breakout_down', False):
                bearish_score += 2
            
            # EMA scoring
            if signals.get('ema_golden_cross', False):
                bullish_score += 3
            elif signals.get('ema_death_cross', False):
                bearish_score += 3
            elif signals.get('ema_bullish', False):
                bullish_score += 1
            elif signals.get('ema_bearish', False):
                bearish_score += 1
            
            # MACD scoring
            if signals.get('macd_bullish_crossover', False):
                bullish_score += 2
            elif signals.get('macd_bearish_crossover', False):
                bearish_score += 2
            elif signals.get('macd_bullish', False):
                bullish_score += 1
            elif signals.get('macd_bearish', False):
                bearish_score += 1
            
            # Volume confirmation
            if signals.get('volume_spike', False):
                if bullish_score > bearish_score:
                    bullish_score += 1
                else:
                    bearish_score += 1
            
            # Decision logic
            total_score = max(bullish_score, bearish_score)
            confidence = min(1.0, total_score / 8.0)  # Normalize to 0-1
            
            if bullish_score > bearish_score and bullish_score >= 3:
                action = 'BUY'
                reason = f"Bullish signals: {bullish_score} vs bearish: {bearish_score}"
            elif bearish_score > bullish_score and bearish_score >= 3:
                action = 'SELL'
                reason = f"Bearish signals: {bearish_score} vs bullish: {bullish_score}"
            else:
                action = 'HOLD'
                reason = f"Insufficient signals (Bull: {bullish_score}, Bear: {bearish_score})"
            
            return {
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Trading recommendation failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {e}'}

    def clear_cache(self):
        """Cache törlése"""
        self.calculation_cache.clear()

    def get_cache_size(self):
        """Cache méret lekérdezése"""
        return len(self.calculation_cache)

    def get_status(self):
        """IndicatorEngine státusz"""
        return {
            'name': 'IndicatorEngine',
            'cache_size': self.get_cache_size(),
            'available_indicators': [
                'RSI', 'EMA', 'Bollinger Bands', 'MACD', 'ATR', 
                'Williams %R', 'Stochastic RSI', 'Volume SMA'
            ],
            'micro_trading_optimized': True,
            'scalping_signals': True,
            'type_safe': True  # Új flag a típusbiztonságról
        }

    # Additional helper methods for type safety
    @staticmethod
    def safe_float_conversion(value: Union[str, float, int, None], default: float = 0.0) -> float:
        """Biztonságos float konverzió"""
        try:
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to float, using default {default}")
            return default

    @staticmethod
    def prepare_dataframe_for_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame előkészítése indikátor számításokhoz"""
        try:
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            for col in required_columns:
                if col not in df.columns:
                    # Add missing column with default values
                    if col == 'volume':
                        df[col] = 1000
                    else:
                        # For price columns, use close price or a default
                        df[col] = df.get('close', 100)
            
            # Convert all numeric columns to float
            for col in required_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].mean() if df[col].notna().any() else 100)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing dataframe: {e}")
            return df
