# strategy/indicator_engine.py - Javított verzió

import pandas as pd
import numpy as np

class IndicatorEngine:
    """Technikai indikátor motor - paraméter nélküli inicializálás"""
    
    def __init__(self):  # ✅ JAVÍTVA: nincs paraméter
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

    def compute_all_indicators(self, data):
        """Összes indikátor számítása egyszerre"""
        try:
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
            print(f"[ERROR] All indicators calculation failed: {e}")
            return {}

    def compute_micro_trading_indicators(self, data):
        """Mikro-trading specifikus indikátorok"""
        try:
            indicators = {}
            
            # Gyors RSI (5 periódus)
            indicators['rsi_fast'] = self.compute_rsi(data, period=5)
            
            # Bollinger Bands (10 periódus, szűkebb sávok)
            indicators['bb_upper_fast'], indicators['bb_middle_fast'], indicators['bb_lower_fast'] = self.compute_bollinger(data, period=10, std_factor=1.5)
            
            # Gyors EMA-k
            indicators['ema_3'], indicators['ema_8'] = self.compute_ema_crossover(data, fast=3, slow=8)
            
            # MACD gyors beállítás
            indicators['macd_fast'], indicators['macd_signal_fast'], indicators['macd_hist_fast'] = self.compute_macd(data, fast=5, slow=13, signal=5)
            
            # ATR 7 periódus
            indicators['atr_fast'] = self.compute_atr(data, period=7)
            
            # Williams %R gyors
            indicators['williams_r_fast'] = self.compute_williams_r(data, period=7)
            
            return indicators
            
        except Exception as e:
            print(f"[ERROR] Micro trading indicators failed: {e}")
            return {}

    def compute_scalping_signals(self, data):
        """Scalping jelek számítása"""
        try:
            signals = {}
            
            # RSI jelzések
            rsi = self.compute_rsi(data, period=14)
            if not rsi.empty:
                current_rsi = rsi.iloc[-1]
                signals['rsi_oversold'] = current_rsi < 30
                signals['rsi_overbought'] = current_rsi > 70
                signals['rsi_neutral'] = 30 <= current_rsi <= 70
                signals['rsi_value'] = current_rsi
            
            # Bollinger Bands jelzések
            bb_upper, bb_middle, bb_lower = self.compute_bollinger(data)
            if not bb_upper.empty and not data.empty:
                current_price = data['close'].iloc[-1]
                upper_price = bb_upper.iloc[-1]
                lower_price = bb_lower.iloc[-1]
                middle_price = bb_middle.iloc[-1]
                
                signals['bb_breakout_up'] = current_price > upper_price
                signals['bb_breakout_down'] = current_price < lower_price
                signals['bb_squeeze'] = (upper_price - lower_price) / middle_price < 0.04  # 4% width
                signals['bb_position'] = (current_price - lower_price) / (upper_price - lower_price)
            
            # EMA jelzések
            ema_fast, ema_slow = self.compute_ema_crossover(data)
            if not ema_fast.empty and not ema_slow.empty:
                signals['ema_bullish'] = ema_fast.iloc[-1] > ema_slow.iloc[-1]
                signals['ema_bearish'] = ema_fast.iloc[-1] < ema_slow.iloc[-1]
                
                # EMA crossover detection
                if len(ema_fast) > 1:
                    prev_fast = ema_fast.iloc[-2]
                    prev_slow = ema_slow.iloc[-2]
                    curr_fast = ema_fast.iloc[-1]
                    curr_slow = ema_slow.iloc[-1]
                    
                    signals['ema_golden_cross'] = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
                    signals['ema_death_cross'] = (prev_fast >= prev_slow) and (curr_fast < curr_slow)
            
            # MACD jelzések
            macd, macd_signal, macd_hist = self.compute_macd(data)
            if not macd.empty:
                signals['macd_bullish'] = macd.iloc[-1] > macd_signal.iloc[-1]
                signals['macd_bearish'] = macd.iloc[-1] < macd_signal.iloc[-1]
                signals['macd_histogram_positive'] = macd_hist.iloc[-1] > 0
                
                # MACD crossover
                if len(macd) > 1:
                    signals['macd_bullish_crossover'] = (macd.iloc[-2] <= macd_signal.iloc[-2]) and (macd.iloc[-1] > macd_signal.iloc[-1])
                    signals['macd_bearish_crossover'] = (macd.iloc[-2] >= macd_signal.iloc[-2]) and (macd.iloc[-1] < macd_signal.iloc[-1])
            
            # Volume jelzések
            if 'volume' in data.columns:
                volume_sma = self.compute_volume_sma(data, period=20)
                if not volume_sma.empty:
                    current_volume = data['volume'].iloc[-1]
                    avg_volume = volume_sma.iloc[-1]
                    signals['volume_spike'] = current_volume > avg_volume * 1.5
                    signals['high_volume'] = current_volume > avg_volume * 2.0
                    signals['low_volume'] = current_volume < avg_volume * 0.5
            
            return signals
            
        except Exception as e:
            print(f"[ERROR] Scalping signals calculation failed: {e}")
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
            print(f"[ERROR] Trading recommendation failed: {e}")
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
            'scalping_signals': True
        }
