# strategy/support_resistance_detector.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import signal
from utils.logger import logger

@dataclass
class SupportResistanceLevel:
    """Support/Resistance szint"""
    price: float
    level_type: str  # "support" vagy "resistance"
    strength: float  # 0-1, szint erőssége
    touch_count: int  # hányszor érintette az ár
    last_touch_time: float
    zone_width: float  # ±zóna szélessége
    confidence: float  # 0-1, megbízhatóság
    volume_confirmation: bool  # volume megerősítés
    breakout_probability: float  # 0-1, kitörés valószínűség

@dataclass
class MarketStructure:
    """Piaci struktúra elemzés"""
    pair: str
    current_price: float
    
    # Support/Resistance szintek
    support_levels: List[SupportResistanceLevel]
    resistance_levels: List[SupportResistanceLevel]
    
    # Aktuális pozíció
    nearest_support: Optional[SupportResistanceLevel]
    nearest_resistance: Optional[SupportResistanceLevel]
    support_distance_pct: float
    resistance_distance_pct: float
    
    # Trend analysis
    trend_direction: str  # "UP", "DOWN", "SIDEWAYS"
    trend_strength: float  # 0-1
    trend_age: int  # gyertyák száma
    
    # Key levels
    key_support: float
    key_resistance: float
    range_high: float
    range_low: float
    range_midpoint: float
    
    # Breakout analysis
    breakout_imminent: bool
    breakout_direction: str  # "UP", "DOWN", "UNCERTAIN"
    breakout_target: float
    breakout_confidence: float

class SupportResistanceDetector:
    """35 gyertyás Support/Resistance detection"""
    
    def __init__(self):
        # 🎯 DETECTION SETTINGS (35 gyertya)
        self.LOOKBACK_PERIOD = 35
        self.MIN_TOUCHES = 2
        self.LEVEL_TOLERANCE = 0.005  # 0.5% tolerance
        self.VOLUME_CONFIRMATION_THRESHOLD = 1.2  # 20% above average
        
        # Strength calculation weights
        self.TOUCH_COUNT_WEIGHT = 0.4
        self.VOLUME_WEIGHT = 0.3
        self.AGE_WEIGHT = 0.2
        self.REJECTION_WEIGHT = 0.1
        
        # Trend detection
        self.TREND_MIN_PERIODS = 10
        self.TREND_STRENGTH_THRESHOLD = 0.6
        
        # Breakout detection
        self.BREAKOUT_VOLUME_MULTIPLIER = 1.5
        self.BREAKOUT_CONFIRMATION_CANDLES = 3
        
        # Caching
        self.level_cache = {}
        self.structure_cache = {}
        
        logger.info("SupportResistanceDetector initialized with 35-candle analysis")

    def detect_market_structure(self, pair: str, ohlcv_data: List[Dict]) -> MarketStructure:
        """
        Teljes piaci struktúra elemzés 35 gyertyával
        """
        try:
            if len(ohlcv_data) < self.LOOKBACK_PERIOD:
                logger.warning(f"Insufficient data for {pair}: {len(ohlcv_data)} < {self.LOOKBACK_PERIOD}")
                return self._create_default_structure(pair, ohlcv_data)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data[-self.LOOKBACK_PERIOD:])
            current_price = float(df['close'].iloc[-1])
            
            # 1. DETECT SUPPORT/RESISTANCE LEVELS
            support_levels = self._detect_support_levels(df)
            resistance_levels = self._detect_resistance_levels(df)
            
            # 2. VALIDATE LEVELS WITH VOLUME
            support_levels = self._validate_levels_with_volume(support_levels, df, "support")
            resistance_levels = self._validate_levels_with_volume(resistance_levels, df, "resistance")
            
            # 3. FIND NEAREST LEVELS
            nearest_support = self._find_nearest_level(current_price, support_levels, "below")
            nearest_resistance = self._find_nearest_level(current_price, resistance_levels, "above")
            
            # 4. CALCULATE DISTANCES
            support_distance = self._calculate_distance_pct(current_price, nearest_support.price) if nearest_support else 10.0
            resistance_distance = self._calculate_distance_pct(nearest_resistance.price, current_price) if nearest_resistance else 10.0
            
            # 5. TREND ANALYSIS
            trend_data = self._analyze_trend(df)
            
            # 6. KEY LEVELS IDENTIFICATION
            key_levels = self._identify_key_levels(df, support_levels, resistance_levels)
            
            # 7. BREAKOUT ANALYSIS  
            breakout_data = self._analyze_breakout_potential(df, current_price, nearest_support, nearest_resistance)
            
            structure = MarketStructure(
                pair=pair,
                current_price=current_price,
                
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                support_distance_pct=support_distance,
                resistance_distance_pct=resistance_distance,
                
                trend_direction=trend_data['direction'],
                trend_strength=trend_data['strength'],
                trend_age=trend_data['age'],
                
                key_support=key_levels['support'],
                key_resistance=key_levels['resistance'],
                range_high=key_levels['range_high'],
                range_low=key_levels['range_low'],
                range_midpoint=key_levels['midpoint'],
                
                breakout_imminent=breakout_data['imminent'],
                breakout_direction=breakout_data['direction'],
                breakout_target=breakout_data['target'],
                breakout_confidence=breakout_data['confidence']
            )
            
            # Cache result
            self._cache_structure(pair, structure)
            
            logger.info(
                f"📊 {pair} Structure: Trend: {trend_data['direction']}, "
                f"S/R: {len(support_levels)}/{len(resistance_levels)}, "
                f"Breakout: {breakout_data['direction']} ({breakout_data['confidence']:.2f})"
            )
            
            return structure
            
        except Exception as e:
            logger.error(f"Market structure detection failed for {pair}: {e}")
            return self._create_default_structure(pair, ohlcv_data)

    def _detect_support_levels(self, df: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Support szintek detektálás"""
        try:
            support_levels = []
            lows = df['low'].values
            volumes = df['volume'].values
            timestamps = df.index.values
            
            # Find local minima using scipy
            # Use more sensitive parameters for 35 candles
            local_min_indices = signal.argrelextrema(lows, np.less, order=2)[0]
            
            # Filter for significant lows
            significant_lows = []
            for idx in local_min_indices:
                if idx >= 2 and idx <= len(lows) - 3:  # Ensure we have context
                    low_price = lows[idx]
                    
                    # Check if it's a significant low (not just noise)
                    context_range = max(lows[max(0, idx-5):idx+6]) - min(lows[max(0, idx-5):idx+6])
                    if context_range > low_price * 0.01:  # At least 1% range
                        significant_lows.append({
                            'index': idx,
                            'price': low_price,
                            'volume': volumes[idx],
                            'timestamp': timestamps[idx]
                        })
            
            # Group nearby levels
            grouped_levels = self._group_nearby_levels(significant_lows, "support")
            
            # Calculate level strength and properties
            for level_group in grouped_levels:
                avg_price = np.mean([l['price'] for l in level_group])
                touch_count = len(level_group)
                
                # Calculate touches within tolerance
                touches = self._count_level_touches(df, avg_price, "support")
                
                # Volume confirmation
                level_volumes = [l['volume'] for l in level_group]
                avg_volume = np.mean(volumes)
                volume_confirmation = np.mean(level_volumes) > avg_volume * self.VOLUME_CONFIRMATION_THRESHOLD
                
                # Calculate strength
                strength = self._calculate_level_strength(touches, level_volumes, avg_volume, len(df))
                
                # Last touch time
                last_touch = max(l['timestamp'] for l in level_group)
                
                # Zone width (price tolerance)
                price_range = max(l['price'] for l in level_group) - min(l['price'] for l in level_group)
                zone_width = max(price_range, avg_price * self.LEVEL_TOLERANCE)
                
                # Confidence based on multiple factors
                confidence = min(1.0, strength * (1.2 if volume_confirmation else 0.8))
                
                # Breakout probability (how likely to break)
                breakout_prob = self._calculate_breakout_probability(df, avg_price, "support")
                
                if strength > 0.3:  # Minimum strength threshold
                    support_level = SupportResistanceLevel(
                        price=avg_price,
                        level_type="support",
                        strength=strength,
                        touch_count=touches,
                        last_touch_time=last_touch,
                        zone_width=zone_width,
                        confidence=confidence,
                        volume_confirmation=volume_confirmation,
                        breakout_probability=breakout_prob
                    )
                    support_levels.append(support_level)
            
            # Sort by strength
            support_levels.sort(key=lambda x: x.strength, reverse=True)
            
            return support_levels[:5]  # Top 5 support levels
            
        except Exception as e:
            logger.error(f"Support level detection failed: {e}")
            return []

    def _detect_resistance_levels(self, df: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Resistance szintek detektálás"""
        try:
            resistance_levels = []
            highs = df['high'].values
            volumes = df['volume'].values
            timestamps = df.index.values
            
            # Find local maxima
            local_max_indices = signal.argrelextrema(highs, np.greater, order=2)[0]
            
            # Filter for significant highs
            significant_highs = []
            for idx in local_max_indices:
                if idx >= 2 and idx <= len(highs) - 3:
                    high_price = highs[idx]
                    
                    # Check significance
                    context_range = max(highs[max(0, idx-5):idx+6]) - min(highs[max(0, idx-5):idx+6])
                    if context_range > high_price * 0.01:
                        significant_highs.append({
                            'index': idx,
                            'price': high_price,
                            'volume': volumes[idx],
                            'timestamp': timestamps[idx]
                        })
            
            # Group nearby levels
            grouped_levels = self._group_nearby_levels(significant_highs, "resistance")
            
            # Calculate level properties
            for level_group in grouped_levels:
                avg_price = np.mean([l['price'] for l in level_group])
                touch_count = len(level_group)
                
                # Calculate touches
                touches = self._count_level_touches(df, avg_price, "resistance")
                
                # Volume confirmation
                level_volumes = [l['volume'] for l in level_group]
                avg_volume = np.mean(volumes)
                volume_confirmation = np.mean(level_volumes) > avg_volume * self.VOLUME_CONFIRMATION_THRESHOLD
                
                # Strength
                strength = self._calculate_level_strength(touches, level_volumes, avg_volume, len(df))
                
                # Other properties
                last_touch = max(l['timestamp'] for l in level_group)
                price_range = max(l['price'] for l in level_group) - min(l['price'] for l in level_group)
                zone_width = max(price_range, avg_price * self.LEVEL_TOLERANCE)
                confidence = min(1.0, strength * (1.2 if volume_confirmation else 0.8))
                breakout_prob = self._calculate_breakout_probability(df, avg_price, "resistance")
                
                if strength > 0.3:
                    resistance_level = SupportResistanceLevel(
                        price=avg_price,
                        level_type="resistance",
                        strength=strength,
                        touch_count=touches,
                        last_touch_time=last_touch,
                        zone_width=zone_width,
                        confidence=confidence,
                        volume_confirmation=volume_confirmation,
                        breakout_probability=breakout_prob
                    )
                    resistance_levels.append(resistance_level)
            
            resistance_levels.sort(key=lambda x: x.strength, reverse=True)
            return resistance_levels[:5]  # Top 5 resistance levels
            
        except Exception as e:
            logger.error(f"Resistance level detection failed: {e}")
            return []

    def _group_nearby_levels(self, levels: List[Dict], level_type: str) -> List[List[Dict]]:
        """Közeli szintek csoportosítása"""
        try:
            if not levels:
                return []
            
            # Sort by price
            sorted_levels = sorted(levels, key=lambda x: x['price'])
            groups = []
            current_group = [sorted_levels[0]]
            
            for i in range(1, len(sorted_levels)):
                current_price = sorted_levels[i]['price']
                group_avg = np.mean([l['price'] for l in current_group])
                
                # Check if within tolerance
                if abs(current_price - group_avg) / group_avg <= self.LEVEL_TOLERANCE:
                    current_group.append(sorted_levels[i])
                else:
                    groups.append(current_group)
                    current_group = [sorted_levels[i]]
            
            groups.append(current_group)
            
            # Filter groups with minimum touches
            valid_groups = [g for g in groups if len(g) >= self.MIN_TOUCHES]
            
            return valid_groups
            
        except Exception as e:
            logger.error(f"Level grouping failed: {e}")
            return []

    def _count_level_touches(self, df: pd.DataFrame, level_price: float, level_type: str) -> int:
        """Szint érintések számolása"""
        try:
            touches = 0
            tolerance = level_price * self.LEVEL_TOLERANCE
            
            if level_type == "support":
                # Count how many times price came near this support level
                for idx, row in df.iterrows():
                    if (row['low'] <= level_price + tolerance and 
                        row['low'] >= level_price - tolerance):
                        touches += 1
            else:  # resistance
                # Count how many times price came near this resistance level
                for idx, row in df.iterrows():
                    if (row['high'] >= level_price - tolerance and 
                        row['high'] <= level_price + tolerance):
                        touches += 1
            
            return touches
            
        except Exception as e:
            logger.error(f"Touch counting failed: {e}")
            return 1

    def _calculate_level_strength(self, touches: int, level_volumes: List[float], 
                                avg_volume: float, total_periods: int) -> float:
        """Szint erősség számítás"""
        try:
            # Touch count component (40%)
            touch_score = min(1.0, touches / 5.0) * self.TOUCH_COUNT_WEIGHT
            
            # Volume component (30%)
            if level_volumes and avg_volume > 0:
                volume_ratio = np.mean(level_volumes) / avg_volume
                volume_score = min(1.0, volume_ratio) * self.VOLUME_WEIGHT
            else:
                volume_score = 0
            
            # Age component (20%) - newer levels are stronger
            age_score = 0.8 * self.AGE_WEIGHT  # Assume medium age
            
            # Rejection component (10%) - how well price respected the level
            rejection_score = min(1.0, touches / 3.0) * self.REJECTION_WEIGHT
            
            total_strength = touch_score + volume_score + age_score + rejection_score
            return min(1.0, total_strength)
            
        except Exception as e:
            logger.error(f"Strength calculation failed: {e}")
            return 0.5

    def _calculate_breakout_probability(self, df: pd.DataFrame, level_price: float, 
                                      level_type: str) -> float:
        """Kitörés valószínűség számítás"""
        try:
            recent_data = df.tail(10)  # Last 10 candles
            
            # Volume trend near the level
            volume_trend = 0
            for idx, row in recent_data.iterrows():
                distance = abs(row['close'] - level_price) / level_price
                if distance < 0.02:  # Within 2% of level
                    volume_trend += row['volume']
            
            avg_volume = df['volume'].mean()
            volume_factor = (volume_trend / len(recent_data)) / avg_volume if avg_volume > 0 else 1
            
            # Price compression near level
            recent_volatility = recent_data['high'].std() / recent_data['close'].mean()
            historical_volatility = df['high'].std() / df['close'].mean()
            compression_factor = historical_volatility / recent_volatility if recent_volatility > 0 else 1
            
            # Number of recent touches
            recent_touches = 0
            tolerance = level_price * self.LEVEL_TOLERANCE
            
            for idx, row in recent_data.iterrows():
                if level_type == "support":
                    if row['low'] <= level_price + tolerance:
                        recent_touches += 1
                else:
                    if row['high'] >= level_price - tolerance:
                        recent_touches += 1
            
            touch_factor = min(1.0, recent_touches / 3.0)
            
            # Combine factors
            breakout_prob = (volume_factor * 0.4 + compression_factor * 0.3 + touch_factor * 0.3)
            return min(1.0, max(0.0, breakout_prob))
            
        except Exception as e:
            logger.error(f"Breakout probability calculation failed: {e}")
            return 0.5

    def _validate_levels_with_volume(self, levels: List[SupportResistanceLevel], 
                                   df: pd.DataFrame, level_type: str) -> List[SupportResistanceLevel]:
        """Volume alapú szint validálás"""
        try:
            validated_levels = []
            avg_volume = df['volume'].mean()
            
            for level in levels:
                # Check volume at level touches
                volume_confirmation = level.volume_confirmation
                
                # Additional volume analysis
                level_volumes = []
                tolerance = level.price * self.LEVEL_TOLERANCE
                
                for idx, row in df.iterrows():
                    near_level = False
                    if level_type == "support":
                        near_level = (row['low'] <= level.price + tolerance and 
                                    row['low'] >= level.price - tolerance)
                    else:
                        near_level = (row['high'] >= level.price - tolerance and 
                                    row['high'] <= level.price + tolerance)
                    
                    if near_level:
                        level_volumes.append(row['volume'])
                
                # Enhanced volume confirmation
                if level_volumes:
                    avg_level_volume = np.mean(level_volumes)
                    if avg_level_volume > avg_volume * 0.8:  # At least 80% of average
                        level.volume_confirmation = True
                        validated_levels.append(level)
                else:
                    if level.strength > 0.6:  # Keep strong levels even without volume
                        validated_levels.append(level)
            
            return validated_levels
            
        except Exception as e:
            logger.error(f"Volume validation failed: {e}")
            return levels

    def _find_nearest_level(self, current_price: float, levels: List[SupportResistanceLevel], 
                          direction: str) -> Optional[SupportResistanceLevel]:
        """Legközelebbi szint keresése"""
        try:
            if not levels:
                return None
            
            if direction == "below":
                # Find highest support below current price
                below_levels = [l for l in levels if l.price < current_price]
                if below_levels:
                    return max(below_levels, key=lambda x: x.price)
            else:  # "above"
                # Find lowest resistance above current price
                above_levels = [l for l in levels if l.price > current_price]
                if above_levels:
                    return min(above_levels, key=lambda x: x.price)
            
            return None
            
        except Exception as e:
            logger.error(f"Nearest level search failed: {e}")
            return None

    def _calculate_distance_pct(self, price1: float, price2: float) -> float:
        """Százalékos távolság számítás"""
        try:
            return abs(price1 - price2) / min(price1, price2) * 100
        except:
            return 0.0

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Trend elemzés"""
        try:
            closes = df['close'].values
            
            # Simple linear regression for trend
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            
            # Normalize slope
            avg_price = np.mean(closes)
            trend_strength = abs(slope) / avg_price * len(closes)
            trend_strength = min(1.0, trend_strength * 10)  # Scale
            
            # Determine direction
            if slope > avg_price * 0.001:  # 0.1% per candle
                direction = "UP"
            elif slope < -avg_price * 0.001:
                direction = "DOWN" 
            else:
                direction = "SIDEWAYS"
            
            # Trend age (how long has this trend been active)
            age = self._calculate_trend_age(closes, direction)
            
            return {
                'direction': direction,
                'strength': trend_strength,
                'age': age,
                'slope': slope
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {'direction': 'SIDEWAYS', 'strength': 0.5, 'age': 10, 'slope': 0}

    def _calculate_trend_age(self, closes: np.ndarray, direction: str) -> int:
        """Trend életkor számítás"""
        try:
            # Simplified - count consecutive moves in trend direction
            age = 0
            if direction == "UP":
                for i in range(len(closes)-1, 0, -1):
                    if closes[i] > closes[i-1]:
                        age += 1
                    else:
                        break
            elif direction == "DOWN":
                for i in range(len(closes)-1, 0, -1):
                    if closes[i] < closes[i-1]:
                        age += 1
                    else:
                        break
            
            return age
            
        except Exception as e:
            logger.error(f"Trend age calculation failed: {e}")
            return 0

    def _identify_key_levels(self, df: pd.DataFrame, support_levels: List[SupportResistanceLevel], 
                           resistance_levels: List[SupportResistanceLevel]) -> Dict:
        """Kulcs szintek azonosítása"""
        try:
            # Range calculation
            range_high = df['high'].max()
            range_low = df['low'].min()
            range_midpoint = (range_high + range_low) / 2
            
            # Key support (strongest support)
            key_support = support_levels[0].price if support_levels else range_low
            
            # Key resistance (strongest resistance)
            key_resistance = resistance_levels[0].price if resistance_levels else range_high
            
            return {
                'support': key_support,
                'resistance': key_resistance,
                'range_high': range_high,
                'range_low': range_low,
                'midpoint': range_midpoint
            }
            
        except Exception as e:
            logger.error(f"Key levels identification failed: {e}")
            return {
                'support': df['low'].min(),
                'resistance': df['high'].max(),
                'range_high': df['high'].max(),
                'range_low': df['low'].min(),
                'midpoint': (df['high'].max() + df['low'].min()) / 2
            }

    def _analyze_breakout_potential(self, df: pd.DataFrame, current_price: float,
                                  nearest_support: Optional[SupportResistanceLevel],
                                  nearest_resistance: Optional[SupportResistanceLevel]) -> Dict:
        """Kitörés potenciál elemzés"""
        try:
            # Default values
            imminent = False
            direction = "UNCERTAIN"
            target = current_price
            confidence = 0.5
            
            # Check if near key levels
            if nearest_resistance and nearest_support:
                resistance_distance = (nearest_resistance.price - current_price) / current_price
                support_distance = (current_price - nearest_support.price) / current_price
                
                # Near resistance
                if resistance_distance < 0.02:  # Within 2%
                    if nearest_resistance.breakout_probability > 0.7:
                        imminent = True
                        direction = "UP"
                        target = nearest_resistance.price * 1.02  # 2% above resistance
                        confidence = nearest_resistance.breakout_probability
                
                # Near support
                elif support_distance < 0.02:  # Within 2%
                    if nearest_support.breakout_probability > 0.7:
                        imminent = True
                        direction = "DOWN"
                        target = nearest_support.price * 0.98  # 2% below support
                        confidence = nearest_support.breakout_probability
            
            # Volume confirmation
            recent_volume = df['volume'].tail(3).mean()
            avg_volume = df['volume'].mean()
            
            if recent_volume > avg_volume * self.BREAKOUT_VOLUME_MULTIPLIER:
                confidence *= 1.2  # Boost confidence with volume
                confidence = min(1.0, confidence)
            
            return {
                'imminent': imminent,
                'direction': direction,
                'target': target,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Breakout analysis failed: {e}")
            return {
                'imminent': False,
                'direction': "UNCERTAIN",
                'target': current_price,
                'confidence': 0.5
            }

    def _create_default_structure(self, pair: str, ohlcv_data: List[Dict]) -> MarketStructure:
        """Default market structure létrehozás"""
        try:
            if ohlcv_data:
                current_price = float(ohlcv_data[-1]['close'])
                df = pd.DataFrame(ohlcv_data)
                range_high = df['high'].max()
                range_low = df['low'].min()
            else:
                current_price = 1000.0
                range_high = 1050.0
                range_low = 950.0
            
            return MarketStructure(
                pair=pair,
                current_price=current_price,
                support_levels=[],
                resistance_levels=[],
                nearest_support=None,
                nearest_resistance=None,
                support_distance_pct=5.0,
                resistance_distance_pct=5.0,
                trend_direction="SIDEWAYS",
                trend_strength=0.5,
                trend_age=10,
                key_support=range_low,
                key_resistance=range_high,
                range_high=range_high,
                range_low=range_low,
                range_midpoint=(range_high + range_low) / 2,
                breakout_imminent=False,
                breakout_direction="UNCERTAIN",
                breakout_target=current_price,
                breakout_confidence=0.5
            )
            
        except Exception as e:
            logger.error(f"Default structure creation failed: {e}")
            return MarketStructure(
                pair=pair, current_price=1000.0, support_levels=[], resistance_levels=[],
                nearest_support=None, nearest_resistance=None, support_distance_pct=5.0,
                resistance_distance_pct=5.0, trend_direction="SIDEWAYS", trend_strength=0.5,
                trend_age=10, key_support=950.0, key_resistance=1050.0, range_high=1050.0,
                range_low=950.0, range_midpoint=1000.0, breakout_imminent=False,
                breakout_direction="UNCERTAIN", breakout_target=1000.0, breakout_confidence=0.5
            )

    def _cache_structure(self, pair: str, structure: MarketStructure):
        """Structure cache-elése"""
        self.structure_cache[pair] = {
            'data': structure,
            'timestamp': time.time()
        }

    def get_cached_structure(self, pair: str) -> Optional[MarketStructure]:
        """Cache-elt structure lekérés"""
        if pair in self.structure_cache:
            cached = self.structure_cache[pair]
            if time.time() - cached['timestamp'] < 300:  # 5 min cache
                return cached['data']
        return None

    def get_detector_status(self) -> Dict:
        """Detector státusz"""
        return {
            'name': 'SupportResistanceDetector',
            'lookback_period': self.LOOKBACK_PERIOD,
            'min_touches': self.MIN_TOUCHES,
            'level_tolerance': f"{self.LEVEL_TOLERANCE*100:.1f}%",
            'cached_structures': len(self.structure_cache),
            'volume_confirmation_threshold': f"{self.VOLUME_CONFIRMATION_THRESHOLD:.1f}x",
            'breakout_volume_multiplier': f"{self.BREAKOUT_VOLUME_MULTIPLIER:.1f}x"
        }
