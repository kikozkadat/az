# strategy/indicator_engine.py - Javított verzió

import pandas as pd
import numpy as np

class IndicatorEngine:
    @staticmethod
    def compute_rsi(data, period=14):
        """RSI számítás javított hibakezeléssel"""
        try:
            if len(data) < period + 1:
                return pd.Series([50] * len(data), index=data.index)
                
            delta = data['close'].diff()
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
            print(f"[ERROR] RSI calculation failed: {e}")
            return pd.Series([50] * len(data), index=data.index)

    @staticmethod
    def compute_stoch_rsi(data, period=14):
        """Stochastic RSI számítás"""
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
            print(f"[ERROR] Stoch RSI calculation failed: {e}")
            return pd.Series([0.5] * len(data), index=data.index)

    @staticmethod
    def compute_ema(data, span):
        """EMA számítás javított hibakezeléssel"""
        try:
            if len(data) < span:
                return pd.Series(data['close'].mean(), index=data.index)
            return data['close'].ewm(span=span, adjust=False).mean()
        except Exception as e:
            print(f"[ERROR] EMA calculation failed: {e}")
            return pd.Series(data['close'].mean(), index=data.index)

    @staticmethod
    def compute_ema_crossover(data, fast=9, slow=21):
        """EMA crossover javított hibakezeléssel"""
        try:
            ema_fast = IndicatorEngine.compute_ema(data, fast)
            ema_slow = IndicatorEngine.compute_ema(data, slow)
            return ema_fast, ema_slow
        except Exception as e:
            print(f"[ERROR] EMA crossover calculation failed: {e}")
            default_val = data['close'].mean() if not data.empty else 1000
            return (pd.Series([default_val] * len(data), index=data.index),
                    pd.Series([default_val] * len(data), index=data.index))

    @staticmethod
    def compute_bollinger(data, period=20, std_factor=2):
        """
        Bollinger Bands számítás - JAVÍTOTT!
        Returns: (upper_band, middle_band, lower_band)
        """
        try:
            if len(data) < period:
                # Not enough data, return price-based fallback
                price = data['close'].iloc[-1] if not data.empty else 1000
                fallback_series = pd.Series([price] * len(data), index=data.index)
                upper = fallback_series * 1.02  # +2%
                lower = fallback_series * 0.98  # -2%
                return upper, fallback_series, lower
                
            ma = data['close'].rolling(window=period).mean()
            std = data['close'].rolling(window=period).std()
            
            # Fill NaN values with price data
            ma = ma.fillna(data['close'])
            std = std.fillna(data['close'].std() if len(data) > 1 else 0.02 * data['close'].mean())
            
            upper_band = ma + std_factor * std
            lower_band = ma - std_factor * std
            
            return upper_band, ma, lower_band
            
        except Exception as e:
            print(f"[ERROR] Bollinger calculation failed: {e}")
            # Emergency fallback
            price = data['close'].iloc[-1] if not data.empty else 1000
            fallback_series = pd.Series([price] * len(data), index=data.index)
            upper = fallback_series * 1.02
            lower = fallback_series * 0.98
            return upper, fallback_series, lower

    @staticmethod
    def compute_macd(data, fast=12, slow=26, signal=9):
        """MACD számítás"""
        try:
            if len(data) < slow + signal:
                return (pd.Series([0] * len(data), index=data.index),
                        pd.Series([0] * len(data), index=data.index),
                        pd.Series([0] * len(data), index=data.index))
                        
            ema_fast = data['close'].ewm(span=fast).mean()
            ema_slow = data['close'].ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)
            
        except Exception as e:
            print(f"[ERROR] MACD calculation failed: {e}")
            return (pd.Series([0] * len(data), index=data.index),
                    pd.Series([0] * len(data), index=data.index),
                    pd.Series([0] * len(data), index=data.index))

    @staticmethod
    def compute_atr(data, period=14):
        """Average True Range számítás"""
        try:
            if len(data) < period + 1:
                return pd.Series([0.01] * len(data), index=data.index)
                
            high = pd.to_numeric(data['high'], errors='coerce')
            low = pd.to_numeric(data['low'], errors='coerce')
            close = pd.to_numeric(data['close'], errors='coerce')
            
            # True Range calculation
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr.fillna(0.01)
            
        except Exception as e:
            print(f"[ERROR] ATR calculation failed: {e}")
            return pd.Series([0.01] * len(data), index=data.index)

    @staticmethod
    def compute_williams_r(data, period=14):
        """Williams %R számítás"""
        try:
            if len(data) < period:
                return pd.Series([-50] * len(data), index=data.index)
                
            high = pd.to_numeric(data['high'], errors='coerce')
            low = pd.to_numeric(data['low'], errors='coerce')
            close = pd.to_numeric(data['close'], errors='coerce')
            
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            # Avoid division by zero
            range_val = highest_high - lowest_low
            range_val = range_val.replace(0, 0.0001)
            
            williams_r = -100 * (highest_high - close) / range_val
            return williams_r.fillna(-50)
            
        except Exception as e:
            print(f"[ERROR] Williams %R calculation failed: {e}")
            return pd.Series([-50] * len(data), index=data.index)

    @staticmethod
    def compute_volume_sma(data, period=20):
        """Volume Simple Moving Average"""
        try:
            if 'volume' not in data.columns:
                return pd.Series([1000] * len(data), index=data.index)
                
            volume = pd.to_numeric(data['volume'], errors='coerce').fillna(1000)
            return volume.rolling(window=period).mean().fillna(1000)
            
        except Exception as e:
            print(f"[ERROR] Volume SMA calculation failed: {e}")
            return pd.Series([1000] * len(data), index=data.index)

    @staticmethod
    def compute_all_indicators(data):
        """Összes indikátor számítása egyszerre"""
        try:
            indicators = {}
            
            # Basic indicators
            indicators['rsi'] = IndicatorEngine.compute_rsi(data)
            indicators['ema_fast'], indicators['ema_slow'] = IndicatorEngine.compute_ema_crossover(data)
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = IndicatorEngine.compute_bollinger(data)
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = IndicatorEngine.compute_macd(data)
            indicators['atr'] = IndicatorEngine.compute_atr(data)
            indicators['williams_r'] = IndicatorEngine.compute_williams_r(data)
            indicators['stoch_rsi'] = IndicatorEngine.compute_stoch_rsi(data)
            indicators['volume_sma'] = IndicatorEngine.compute_volume_sma(data)
            
            return indicators
            
        except Exception as e:
            print(f"[ERROR] All indicators calculation failed: {e}")
            return {}
