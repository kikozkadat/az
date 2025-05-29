# strategy/ml_scoring_engine.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import pickle
import os
import time
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.logger import logger

@dataclass
class MLFeatures:
    """ML input features"""
    # Technical indicators
    rsi_3m: float
    rsi_15m: float
    macd_3m: float
    macd_15m: float
    stoch_rsi: float
    williams_r: float
    
    # Bollinger features (FOKOZOTT!)
    bb_position: float
    bb_squeeze: bool
    bb_breakout_potential: float
    bb_width_ratio: float
    
    # Volume features
    volume_ratio: float
    volume_trend: float
    volume_spike: bool
    
    # Correlation (KRITIKUS!)
    btc_correlation: float
    eth_correlation: float
    btc_momentum: float
    eth_momentum: float
    
    # Market structure
    near_support: bool
    near_resistance: bool
    support_strength: float
    price_momentum_5m: float
    
    # Volatility clustering
    atr_ratio: float
    volatility_spike: bool
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML"""
        return np.array([
            self.rsi_3m, self.rsi_15m, self.macd_3m, self.macd_15m,
            self.stoch_rsi, self.williams_r, self.bb_position,
            float(self.bb_squeeze), self.bb_breakout_potential, self.bb_width_ratio,
            self.volume_ratio, self.volume_trend, float(self.volume_spike),
            self.btc_correlation, self.eth_correlation, self.btc_momentum, self.eth_momentum,
            float(self.near_support), float(self.near_resistance), self.support_strength,
            self.price_momentum_5m, self.atr_ratio, float(self.volatility_spike)
        ])

@dataclass
class MLPrediction:
    """ML prediction eredmény"""
    pair: str
    probability: float  # 0-1 siker valószínűség
    confidence: float   # Model confidence
    feature_importance: Dict[str, float]
    risk_score: float   # Kockázati pontszám
    expected_return: float  # Várható hozam
    bollinger_signal_strength: float  # BOLLINGER specifikus jel erősség

class MLScoringEngine:
    """Gépi tanulás alapú scoring rendszer"""
    
    def __init__(self):
        self.model_path = "models/micro_trading_model.pkl"
        self.scaler_path = "models/feature_scaler.pkl"
        self.history_path = "data/ml_training_data.csv"
        
        # Models
        self.rf_model = None
        self.dt_model = None
        self.scaler = StandardScaler()
        
        # Feature names
        self.feature_names = [
            'rsi_3m', 'rsi_15m', 'macd_3m', 'macd_15m', 'stoch_rsi', 'williams_r',
            'bb_position', 'bb_squeeze', 'bb_breakout_potential', 'bb_width_ratio',
            'volume_ratio', 'volume_trend', 'volume_spike',
            'btc_correlation', 'eth_correlation', 'btc_momentum', 'eth_momentum',
            'near_support', 'near_resistance', 'support_strength',
            'price_momentum_5m', 'atr_ratio', 'volatility_spike'
        ]
        
        # BOLLINGER súlyozás ML-ben is (FOKOZOTT!)
        self.bollinger_features = ['bb_position', 'bb_squeeze', 'bb_breakout_potential', 'bb_width_ratio']
        self.correlation_features = ['btc_correlation', 'eth_correlation', 'btc_momentum', 'eth_momentum']
        
        # Training data storage
        self.training_data = []
        self.last_training_time = 0
        self.min_training_samples = 100
        
        # Performance tracking
        self.prediction_history = []
        self.accuracy_metrics = {}
        
        # Load existing model
        self._load_models()
        logger.info("MLScoringEngine initialized with Bollinger focus")

    def predict_trade_success(self, features: MLFeatures) -> MLPrediction:
        """
        Fő ML predikció BOLLINGER FÓKUSSZAL
        """
        try:
            # Feature array
            feature_array = features.to_array().reshape(1, -1)
            
            # Scale features
            if hasattr(self.scaler, 'mean_'):
                feature_array_scaled = self.scaler.transform(feature_array)
            else:
                feature_array_scaled = feature_array
            
            # Predictions
            rf_prob = 0.5
            dt_prob = 0.5
            confidence = 0.5
            
            if self.rf_model:
                rf_prob = self.rf_model.predict_proba(feature_array_scaled)[0][1]
                confidence = max(self.rf_model.predict_proba(feature_array_scaled)[0])
            
            if self.dt_model:
                dt_prob = self.dt_model.predict_proba(feature_array_scaled)[0][1]
            
            # Ensemble prediction
            ensemble_prob = (rf_prob * 0.7 + dt_prob * 0.3)
            
            # 🎯 BOLLINGER BOOSTING (FOKOZOTT SÚLY!)
            bollinger_boost = self._calculate_bollinger_boost(features)
            ensemble_prob = min(1.0, ensemble_prob + bollinger_boost)
            
            # Feature importance
            importance = self._get_feature_importance(features)
            
            # Risk scoring
            risk_score = self._calculate_risk_score(features)
            
            # Expected return kalkuláció
            expected_return = self._calculate_expected_return(features, ensemble_prob)
            
            # Bollinger signal strength
            bollinger_strength = self._calculate_bollinger_signal_strength(features)
            
            prediction = MLPrediction(
                pair=getattr(features, 'pair', 'UNKNOWN'),
                probability=ensemble_prob,
                confidence=confidence,
                feature_importance=importance,
                risk_score=risk_score,
                expected_return=expected_return,
                bollinger_signal_strength=bollinger_strength
            )
            
            # Store for learning
            self.prediction_history.append({
                'timestamp': time.time(),
                'features': feature_array[0],
                'prediction': ensemble_prob,
                'bollinger_boost': bollinger_boost
            })
            
            return prediction
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            # Fallback prediction
            return MLPrediction(
                pair=getattr(features, 'pair', 'UNKNOWN'),
                probability=0.5,
                confidence=0.3,
                feature_importance={},
                risk_score=0.5,
                expected_return=0.0,
                bollinger_signal_strength=features.bb_breakout_potential
            )

    def _calculate_bollinger_boost(self, features: MLFeatures) -> float:
        """
        BOLLINGER KITÖRÉS BOOST SZÁMÍTÁS (FOKOZOTT!)
        """
        boost = 0.0
        
        # 1. Alapvető Bollinger pozíció boost
        if features.bb_breakout_potential > 0.8:
            boost += 0.15  # +15% boost magas kitörési potenciálnál
        elif features.bb_breakout_potential > 0.6:
            boost += 0.10
        elif features.bb_breakout_potential > 0.4:
            boost += 0.05
        
        # 2. Squeeze + magas korreláció kombinációja
        if features.bb_squeeze and (features.btc_correlation > 0.85 or features.eth_correlation > 0.85):
            boost += 0.12
        
        # 3. Bollinger position extremes
        if features.bb_position > 0.95:  # Felső sáv közelében
            boost += 0.08
        elif features.bb_position < 0.05:  # Alsó sáv közelében (oversold bounce)
            boost += 0.10
        
        # 4. Volume confirmation
        if features.volume_spike and features.bb_position > 0.9:
            boost += 0.05
        
        # 5. Correlation momentum alignment
        if (features.btc_momentum > 0.001 or features.eth_momentum > 0.001) and features.bb_position > 0.85:
            boost += 0.06
        
        return min(0.25, boost)  # Max 25% boost

    def _calculate_bollinger_signal_strength(self, features: MLFeatures) -> float:
        """Bollinger jel erősség kalkuláció"""
        strength = features.bb_breakout_potential
        
        # Modifiers
        if features.bb_squeeze:
            strength += 0.2
        
        if features.volume_ratio > 2.0:
            strength += 0.15
        
        if features.btc_correlation > 0.9:
            strength += 0.1
        
        return min(1.0, strength)

    def _get_feature_importance(self, features: MLFeatures) -> Dict[str, float]:
        """Feature importance kalkuláció"""
        importance = {}
        
        if self.rf_model and hasattr(self.rf_model, 'feature_importances_'):
            for i, name in enumerate(self.feature_names):
                importance[name] = float(self.rf_model.feature_importances_[i])
        else:
            # Fallback manual importance
            importance = {
                'bb_breakout_potential': 0.25,  # HIGHEST!
                'btc_correlation': 0.15,
                'eth_correlation': 0.12,
                'bb_position': 0.10,
                'volume_ratio': 0.08,
                'rsi_3m': 0.06,
                'rsi_15m': 0.05,
                'macd_3m': 0.04,
                'bb_squeeze': 0.03,
                'volume_trend': 0.03,
                'others': 0.09
            }
        
        return importance

    def _calculate_risk_score(self, features: MLFeatures) -> float:
        """Kockázati pontszám (0=alacsony, 1=magas)"""
        risk = 0.0
        
        # RSI extremes
        if features.rsi_3m > 75 or features.rsi_3m < 25:
            risk += 0.2
        
        # Volatility
        if features.volatility_spike:
            risk += 0.15
        
        # Low correlation during momentum
        if (features.btc_momentum > 0.002 or features.eth_momentum > 0.002):
            if features.btc_correlation < 0.7 and features.eth_correlation < 0.7:
                risk += 0.25
        
        # Volume anomalies
        if features.volume_ratio > 5.0:  # Extreme volume
            risk += 0.1
        elif features.volume_ratio < 0.5:  # Too low volume
            risk += 0.15
        
        # Near resistance without breakout signal
        if features.near_resistance and features.bb_breakout_potential < 0.3:
            risk += 0.2
        
        return min(1.0, risk)

    def _calculate_expected_return(self, features: MLFeatures, probability: float) -> float:
        """Várható hozam kalkuláció ($0.20 target alapján)"""
        base_target = 0.20  # $0.20 base target
        
        # Bollinger multiplier
        bb_multiplier = 1.0 + (features.bb_breakout_potential * 0.5)
        
        # Correlation multiplier
        corr_multiplier = 1.0 + (max(features.btc_correlation, features.eth_correlation) - 0.8) * 2
        
        # Volume multiplier
        vol_multiplier = 1.0 + min(0.3, (features.volume_ratio - 1.0) * 0.1)
        
        # Risk adjustment
        risk_adjustment = 1.0 - (self._calculate_risk_score(features) * 0.3)
        
        expected = base_target * bb_multiplier * corr_multiplier * vol_multiplier * risk_adjustment * probability
        
        return round(expected, 3)

    def train_model(self, force_retrain: bool = False) -> bool:
        """Model training with emphasis on Bollinger patterns"""
        try:
            # Check if we need to train
            if not force_retrain and len(self.training_data) < self.min_training_samples:
                logger.info(f"Not enough training data: {len(self.training_data)}/{self.min_training_samples}")
                return False
            
            if not force_retrain and (time.time() - self.last_training_time) < 3600:  # 1 hour
                return False
            
            logger.info("🧠 Starting ML model training with Bollinger focus...")
            
            # Prepare training data
            X, y = self._prepare_training_data()
            if len(X) < 50:
                logger.warning("Insufficient training data")
                return False
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest (primary model)
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',  # Handle imbalanced data
                random_state=42
            )
            self.rf_model.fit(X_train_scaled, y_train)
            
            # Train Decision Tree (secondary)
            self.dt_model = DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                class_weight='balanced',
                random_state=42
            )
            self.dt_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            rf_score = self.rf_model.score(X_test_scaled, y_test)
            dt_score = self.dt_model.score(X_test_scaled, y_test)
            
            logger.info(f"Model training completed - RF: {rf_score:.3f}, DT: {dt_score:.3f}")
            
            # Save models
            self._save_models()
            self.last_training_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Training data előkészítése"""
        X = []
        y = []
        
        for sample in self.training_data:
            X.append(sample['features'])
            y.append(sample['success'])  # 1 if profitable, 0 if not
        
        return np.array(X), np.array(y)

    def add_training_sample(self, features: MLFeatures, success: bool, profit: float):
        """Training sample hozzáadása"""
        sample = {
            'timestamp': time.time(),
            'features': features.to_array(),
            'success': success,
            'profit': profit,
            'bollinger_strength': features.bb_breakout_potential
        }
        
        self.training_data.append(sample)
        
        # Limit training data size
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-1000:]
        
        # Auto-retrain periodically
        if len(self.training_data) % 50 == 0:
            self.train_model()

    def _save_models(self):
        """Model mentése"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'rf_model': self.rf_model,
                    'dt_model': self.dt_model,
                    'training_data': self.training_data[-500:],  # Last 500 samples
                    'feature_names': self.feature_names
                }, f)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")

    def _load_models(self):
        """Model betöltése"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.rf_model = data.get('rf_model')
                    self.dt_model = data.get('dt_model')
                    self.training_data = data.get('training_data', [])
                    
                logger.info(f"Models loaded with {len(self.training_data)} training samples")
            
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Model loading failed: {e}")

    def get_model_performance(self) -> Dict:
        """Model teljesítmény metrikák"""
        return {
            'training_samples': len(self.training_data),
            'last_training': self.last_training_time,
            'prediction_count': len(self.prediction_history),
            'model_loaded': self.rf_model is not None,
            'bollinger_focus': True,
            'feature_count': len(self.feature_names)
        }

    def generate_sentiment_score(self, pair: str) -> float:
        """Mock sentiment score (Twitter/Reddit placeholder)"""
        # TODO: Implement real sentiment analysis
        import random
        base_sentiment = random.uniform(0.3, 0.7)
        
        # Boost sentiment if BTC/ETH correlation is high
        # This simulates that correlated coins tend to have aligned sentiment
        return base_sentiment

    def calculate_volatility_cluster(self, price_history: List[float]) -> Dict:
        """Volatility clustering analysis"""
        if len(price_history) < 20:
            return {'cluster_strength': 0.5, 'expected_volatility': 0.02}
        
        # Calculate returns
        returns = np.diff(price_history) / price_history[:-1]
        
        # Volatility (rolling std)
        volatility = pd.Series(returns).rolling(10).std()
        
        # Cluster detection (high vol followed by high vol)
        current_vol = volatility.iloc[-1] if not volatility.empty else 0.02
        avg_vol = volatility.mean() if not volatility.empty else 0.02
        
        cluster_strength = min(1.0, current_vol / avg_vol) if avg_vol > 0 else 0.5
        
        return {
            'cluster_strength': cluster_strength,
            'expected_volatility': current_vol,
            'vol_regime': 'high' if current_vol > avg_vol * 1.5 else 'normal'
        }
