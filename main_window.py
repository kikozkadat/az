# gui/main_window.py - FIXED VERSION with proper method definitions

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QListWidget, QPushButton, QTabWidget, QScrollArea, QSizePolicy, QGroupBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import sys
import pandas as pd
import time
import random
from typing import Optional, Dict, List, Tuple # Tuple hozzáadva

# Logging beállítása
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MODIFIED IMPORT BLOCK STARTS
try:
    # Advanced scanner használata, ha elérhető
    from strategy.advanced_market_scanner import AdvancedMarketScanner, CoinAnalysis
    from strategy.ml_scoring_engine import MLScoringEngine, MLFeatures, MLPrediction # Add hozzá ezt!
    from strategy.market_scanner import MarketScanner, CoinMetrics
    from strategy.scorer import CoinScorer
    from strategy.indicator_engine import IndicatorEngine
    from strategy.correlation_analyzer import CorrelationAnalyzer
    from strategy.support_resistance_detector import SupportResistanceDetector
    from strategy.intelligent_trader import IntelligentTrader
    # Új importok a felhasználó által megadott kód alapján (ha szükségesek lennének, de a metódusban közvetlenül nem használja őket)
    from strategy.micro_strategy_engine import MicroStrategyEngine # Feltételezve, hogy létezik ilyen modul
    # from strategy.dynamic_risk_manager import DynamicRiskManager # KIKOMMENTELVE a felhasználó kérésére
    ADVANCED_MODULES_AVAILABLE = True
    logger.info("✅ Advanced trading modules loaded successfully")
except ImportError as e:
    pass  # Nincs szükség warning-ra a célom az hogy ne legyen bacis mode
    # logger.warning(f"⚠️ Advanced modules not available: {e}") # EREDETI
    ADVANCED_MODULES_AVAILABLE = False # Ezt itt hagytam, hogy a logika továbbra is működjön, ha tényleg hiányzik valami más

    # Fallback osztályok
    class CoinAnalysis:
        def __init__(self, pair='UNKNOWN', price=0, volume_24h=0, **kwargs):
            self.pair = pair
            self.price = price
            self.volume_24h = volume_24h
            self.total_score = kwargs.get('total_score', 0)
            self.rsi_3m = kwargs.get('rsi_3m', 50)
            self.btc_correlation = kwargs.get('btc_correlation', 0.5)
            self.eth_correlation = kwargs.get('eth_correlation', 0.5)

    class AdvancedMarketScanner:
        def __init__(self, api_client=None): # Megtartjuk az api_client paramétert a kompatibilitás miatt, ha máshol használva van
            self.api_client = api_client
        def scan_top_opportunities(self, max_pairs=20):
            return []

    class MLFeatures:
        def __init__(self, **kwargs):
            self.rsi_3m = kwargs.get('rsi_3m', 50)
            self.rsi_15m = kwargs.get('rsi_15m', 50)
            self.bb_position = kwargs.get('bb_position', 0.5)
            self.volume_ratio = kwargs.get('volume_ratio', 1.0)
            self.btc_correlation = kwargs.get('btc_correlation', 0.5)

    class MLPrediction:
        def __init__(self, **kwargs):
            self.probability = kwargs.get('probability', 0.5)
            self.confidence = kwargs.get('confidence', 0.5)
            self.expected_return = kwargs.get('expected_return', 0.15)

    class MLScoringEngine:
        def __init__(self):
            pass
        def predict_trade_success(self, features):
            return MLPrediction()

    class MicroStrategyEngine: # Fallback
        def __init__(self):
            logger.info("Fallback MicroStrategyEngine initialized")
        # Add dummy methods if needed

    class DynamicRiskManager: # Fallback - Ezt itt hagyom, hátha máshol még hivatkozás van rá, de nem lesz példányosítva
        def __init__(self):
            logger.info("Fallback DynamicRiskManager initialized (DE NEM LESZ HASZNÁLVA)")
        # Add dummy methods if needed

    class CorrelationAnalyzer: # Fallback
        def __init__(self): # Nincs api_client paraméter a fallbackben sem, a felhasználói kérésnek megfelelően
            logger.info("Fallback CorrelationAnalyzer initialized (no api_client)")
        def analyze_pair_correlation(self, pair, price_history):
            # Dummy implementáció
            class DummyCorrResult:
                confidence_score = 0.5
            return DummyCorrResult()


    class SupportResistanceDetector: # Fallback
        def __init__(self, api_client=None):
            self.api_client = api_client
            logger.info("Fallback SupportResistanceDetector initialized")
        # Add dummy methods if needed

# MODIFIED IMPORT BLOCK ENDS


# Standard imports with fallbacks
try:
    from core.trailing_stop import TrailingStopManager
except ImportError:
    logger.warning("⚠️ TrailingStopManager not found. Using fallback.")
    class TrailingStopManager:
        def __init__(self, target_profit=0.005): self.target_profit = target_profit
        def link(self, pair, position_manager, trader): pass

try:
    from data.kraken_api_client import KrakenAPIClient
except ImportError:
    logger.warning("⚠️ KrakenAPIClient not found. Using fallback.")
    class KrakenAPIClient:
        def __init__(self): self.ws_client = None
        def test_connection(self): return True
        def get_valid_usd_pairs(self): return [{"altname": "XBTUSD", "wsname": "XBT/USD", "base": "XBT", "quote": "USD"}]
        def get_ohlc(self, pair, interval=1, limit=None): return {pair: []} if limit else {pair: [[time.time(),1,1,1,1,1,1,1]]}
        def get_fallback_pairs(self): return [{"altname": "XBTUSD", "wsname": "XBT/USD", "base": "XBT", "quote": "USD"}]
        def initialize_websocket(self, pair_names): return False
        def get_current_price(self, pair): return random.uniform(100,1000) if pair == "XBTUSD" else random.uniform(1,10)
        def get_ticker_data(self, pair): return {'price': self.get_current_price(pair), 'bid': self.get_current_price(pair)*0.99, 'ask': self.get_current_price(pair)*1.01, 'volume_24h': random.uniform(100,1000), 'high_24h': self.get_current_price(pair)*1.05, 'low_24h': self.get_current_price(pair)*0.95}
        def cleanup(self): pass
        ws_data_available = False
        def get_usd_pairs_with_volume(self, min_volume_usd=500000): return [{"altname": "XBTUSD", "wsname": "XBT/USD", "base":"XBT", "quote":"USD", "volume_usd": min_volume_usd + 1000}]
        def get_market_data(self): return [{"symbol": "XBTUSD", "score": 0.75, "price": self.get_current_price("XBTUSD"), "volume_usd": 5000000, "volume_24h": 5000000/self.get_current_price("XBTUSD")}]


# Import GUI components with fallbacks
try:
    from gui.dashboard_panel import DashboardPanel
    from gui.settings_panel import SettingsPanel
    from gui.position_list_widget import PositionListWidget
    from gui.scoring_panel import ScoringPanel
    from gui.stats_panel import StatsPanel
    from gui.advanced_control_panel import AdvancedControlPanel
    from gui.live_trading_panel import LiveTradingPanel
except ImportError as e:
    logger.warning(f"GUI components import error: {e}. Using fallbacks.")
    class DashboardPanel(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.layout = QVBoxLayout(self)
            self.fields = {}
        def update_field(self, name, value):
            if name not in self.fields:
                label = QLabel(f"{name}: {value}")
                self.layout.addWidget(label)
                self.fields[name] = label
            else:
                self.fields[name].setText(f"{name}: {value}")

    class SettingsPanel(QWidget):
        def __init__(self, parent=None): super().__init__(parent)
        def get_settings(self): return {}

    class PositionListWidget(QWidget): # Fallback
        def __init__(self, position_manager, parent_window=None): # Módosítva position_manager-re
            super().__init__(parent_window)
            self.position_manager = position_manager # Módosítva position_manager-re
            self.setLayout(QVBoxLayout())
            self.list_widget = QListWidget()
            self.layout().addWidget(self.list_widget)
            logger.info("Fallback PositionListWidget initialized.")

        def update_history(self):
            # Dummy implementáció, vagy a self.position_manager.get_history() alapján
            if hasattr(self, 'list_widget') and hasattr(self.position_manager, 'get_history'):
                 history = self.position_manager.get_history() # Feltételezve, hogy van ilyen metódus
                 # Itt frissíthetnél egy másik listát a history-val
                 pass

        def update_positions(self, positions):
            if hasattr(self, 'list_widget'):
                self.list_widget.clear()
                if positions and isinstance(positions, dict):
                    if not positions:
                        self.list_widget.addItem("No open positions.")
                        return
                    for pair, data in positions.items():
                        entry_price = data.get('entry_price', 0)
                        volume = data.get('volume', 0)
                        status = data.get('status', 'N/A')
                        side = data.get('side', 'N/A')
                        pnl_text = ""
                        # Egyszerűsített PnL kijelzés a fallbackhez
                        if status == 'open' and entry_price > 0:
                            # Itt nem tudjuk a valós aktuális árat, így a PnL-t nem mutatjuk
                            pnl_text = " (PnL: N/A)"

                        self.list_widget.addItem(f"{pair} ({side.upper()}) | Entry: ${entry_price:.4f} | Vol: {volume:.4f} | {status.upper()}{pnl_text}")
                else:
                    self.list_widget.addItem("No open positions or invalid data.")
            else:
                logger.warning("Fallback PositionListWidget has no list_widget attribute for update_positions.")


    class ScoringPanel(QWidget):
        def __init__(self, scorer, data_func, parent=None):
            super().__init__(parent)
            self.scorer = scorer
            self.data_func = data_func
            self.setLayout(QVBoxLayout())
        def refresh_panel(self): pass
        def start_auto_refresh(self): pass

    class StatsPanel(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setLayout(QVBoxLayout())
        def update_stats(self): pass

    class AdvancedControlPanel(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setLayout(QVBoxLayout())

    class LiveTradingPanel(QWidget):
        start_live_trading = pyqtSignal()
        stop_live_trading = pyqtSignal()
        emergency_stop = pyqtSignal()
        settings_changed = pyqtSignal(dict)
        def __init__(self, api_client=None, parent_window=None):
            super().__init__(parent_window)
            self.api_client = api_client
            self.parent_window = parent_window
            self.setLayout(QVBoxLayout())
            self.mode_combo = QListWidget()
            self.mode_combo.addItem("Bollinger Breakout")
            self.layout().addWidget(self.mode_combo)
        def is_trading_active(self): return False
        def get_current_settings(self): return {'auto_trading': False, 'max_positions': 1}
        def update_opportunities(self, opps): pass
        def update_trade_stats(self, trade_data): pass
        def get_session_stats(self): return {}
        def set_trading_active_status(self, status): pass

# Import core components
from core.trade_manager import TradeManager # TradeManager importálása
from core.position_manager import PositionManager # Ezt használjuk a self.position_manager inicializálásához
from utils.trade_logger import TradeLogger
from core.wallet_display import WalletManager
from utils.history_analyzer import HistoryAnalyzer
# from strategy.decision_ai import DecisionEngine # Ezt a felhasználó nem kérte visszaállítani
# from strategy.intelligent_trader import IntelligentTrader # Már fentebb importálva
# from strategy.market_scanner import MarketScanner # Már fentebb importálva
# from strategy.indicator_engine import IndicatorEngine # Már fentebb importálva
# from strategy.scorer import CoinScorer # Már fentebb importálva

class LiveTradingThread(QThread):
    opportunity_found = pyqtSignal(dict)
    trade_executed = pyqtSignal(dict)
    position_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    scan_progress = pyqtSignal(str)
    pump_detected = pyqtSignal(dict)
    ai_decision = pyqtSignal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.running = False
        self.scan_interval = 180
        self.min_volume_usd = 500000
        self.min_score_threshold = 0.5
        self.pump_threshold = 0.1
        self.final_candidates = []

    def start_trading(self):
        self.running = True
        self.start()

    def stop_trading(self):
        self.running = False

    def run(self):
        logger.info("🚀 Live trading thread started")
        while self.running:
            try:
                self.execute_trading_cycle()
                for _ in range(self.scan_interval):
                    if not self.running:
                        break
                    self.msleep(1000)
            except Exception as e:
                error_msg = f"Trading cycle error: {e}"
                logger.error(error_msg, exc_info=True)
                self.error_occurred.emit(error_msg)
        logger.info("🔴 Live trading thread stopped")

    def execute_trading_cycle(self):
        if not self.running: return
        try:
            self.scan_progress.emit("Analyzing market data...")

            # VALÓDI ADVANCED SCANNER HASZNÁLATA (a meglévő modulokkal)
            if hasattr(self.main_window, 'market_scanner') and self.main_window.market_scanner and self.main_window.ai_mode_active:
                self.scan_progress.emit("🤖 Running Market Scanner (AI Mode)...")

                # 1. Market scan a MarketScannerrel
                opportunities = [] # Lista a CoinAnalysis/CoinMetrics objektumoknak
                if hasattr(self.main_window.market_scanner, 'get_top_opportunities'):
                    # A get_top_opportunities valószínűleg egy score-t is vár, vagy belsőleg számolja
                    # és CoinMetrics vagy hasonló objektumok listáját adja vissza.
                    opportunities = self.main_window.market_scanner.get_top_opportunities(min_score=0.5) # Példa min_score
                elif hasattr(self.main_window.market_scanner, 'scan_top_opportunities'): # Fallback a régebbi névre
                     opportunities = self.main_window.market_scanner.scan_top_opportunities(min_volume_usd=self.min_volume_usd, max_pairs=20)


                if opportunities:
                    self.scan_progress.emit(f"MarketScanner found {len(opportunities)} initial opportunities.")
                    # Használjuk a CoinScorer-t a legjobb kiválasztásához
                    best_metrics = None
                    if hasattr(self.main_window, 'coin_scorer') and self.main_window.coin_scorer and hasattr(self.main_window.coin_scorer, 'select_best_coin'):
                        # Az opportunities elemei legyenek dict-ek vagy CoinMetrics objektumok
                        # Ha CoinMetrics, akkor a __dict__ jó, ha már dict, akkor önmaga
                        opportunities_as_dicts = []
                        for opp_obj in opportunities:
                            if hasattr(opp_obj, '__dict__'):
                                opportunities_as_dicts.append(opp_obj.__dict__)
                            elif isinstance(opp_obj, dict):
                                opportunities_as_dicts.append(opp_obj)
                            else:
                                logger.warning(f"Opportunity is not a dict or has no __dict__: {type(opp_obj)}")

                        if opportunities_as_dicts:
                             best_metrics = self.main_window.coin_scorer.select_best_coin(
                                opportunities_as_dicts,
                                min_score=self.min_score_threshold
                            )

                    if best_metrics and isinstance(best_metrics, dict): # Biztosítjuk, hogy best_metrics dict
                        pair_for_correlation = best_metrics.get('pair', best_metrics.get('symbol'))
                        if not pair_for_correlation:
                            logger.warning("Best metrics does not contain 'pair' or 'symbol'. Cannot proceed with correlation.")
                            return # Vagy continue a ciklusban, ha több opportunity-t dolgoznánk fel

                        self.scan_progress.emit(f"Best scored: {pair_for_correlation} (Score: {best_metrics.get('score',0):.2f})")

                        # Correlation check
                        correlation_confidence = 0.0 # Alapértelmezett
                        if hasattr(self.main_window, 'correlation_analyzer') and self.main_window.correlation_analyzer and hasattr(self.main_window.correlation_analyzer, 'analyze_pair_correlation'):
                            # Price history lekérése a korrelációhoz (pl. utolsó 100 gyertya)
                            price_history_data = []
                            if self.main_window.api:
                                ohlc_corr = self.main_window.api.get_ohlc(pair_for_correlation, interval=15, limit=100) # 15 perces gyertyák
                                if ohlc_corr and pair_for_correlation in ohlc_corr:
                                    price_history_data = [c[4] for c in ohlc_corr[pair_for_correlation]] # Záróárak

                            corr_data_obj = self.main_window.correlation_analyzer.analyze_pair_correlation(
                                pair_for_correlation,
                                price_history_data
                            )
                            correlation_confidence = getattr(corr_data_obj, 'confidence_score', 0.0)
                            logger.info(f"Correlation for {pair_for_correlation}: Confidence {correlation_confidence:.2f}")
                        else:
                            logger.warning("CorrelationAnalyzer not available. Skipping correlation check (defaulting to low confidence).")
                            correlation_confidence = 0.7 # Magasabb alapértelmezett, ha nincs analizátor, hogy ne akadjon el a flow

                        # Döntés a score és korreláció alapján
                        # A min_score_threshold-t már a select_best_coin-ban figyelembe vettük (vagy itt kellene)
                        # A korrelációt is súlyozhatnánk a döntésben
                        if best_metrics.get('score', 0) >= self.min_score_threshold and correlation_confidence > 0.6: # Példa korrelációs küszöb
                            current_price_final = self.main_window.get_current_price_for_pair(pair_for_correlation)
                            if not current_price_final or current_price_final <= 0:
                                current_price_final = best_metrics.get('price', best_metrics.get('close')) # Fallback
                                if not current_price_final or current_price_final <= 0:
                                    logger.warning(f"Still no valid price for {pair_for_correlation} after fallback. Skipping.")
                                    return

                            opportunity_data = {
                                'pair': pair_for_correlation,
                                'score': best_metrics.get('score', 0.5),
                                'reason': f"AI Score: {best_metrics.get('score', 0):.2f}, Corr: {correlation_confidence:.2f}",
                                'entry_price': current_price_final,
                                'volume': 50.0 / current_price_final if current_price_final > 0 else 0, # $50 pozíció
                                'signal': 'buy', # Alapértelmezett jelzés, ezt finomítani kellene
                                'total_score': best_metrics.get('score', 0.5), # Vagy egy kombinált score
                                'stop_loss': current_price_final * 0.96 if current_price_final > 0 else 0, # 4% SL
                                'take_profit': current_price_final * 1.008 if current_price_final > 0 else 0, # 0.8% TP
                                'position_size_usd': 50.0,
                                # További adatok, ha a best_metrics tartalmazza őket
                                'bollinger_score': best_metrics.get('bollinger_score', 0.0),
                                'correlation_score': correlation_confidence, # A számított korreláció
                                'volume_score': best_metrics.get('volume_score', 0.0),
                                'ml_confidence': best_metrics.get('ml_confidence', 0.0), # Ha van ilyen a CoinScorer kimenetében
                                'expected_return': best_metrics.get('expected_return', 0.0)
                            }

                            self.opportunity_found.emit(opportunity_data)
                            self.ai_decision.emit({
                                'decision': f"Opportunity: {opportunity_data['pair']}",
                                'pair': opportunity_data['pair'],
                                'score': opportunity_data['score']
                            })
                            self.scan_progress.emit(f"✅ Top candidate: {opportunity_data['pair']} Score: {opportunity_data['score']:.2f}")
                        else:
                             self.scan_progress.emit(f"📉 Candidate {pair_for_correlation} skipped (Score: {best_metrics.get('score',0):.2f} / Corr: {correlation_confidence:.2f})")
                    else: # if not best_metrics
                        self.scan_progress.emit("📊 CoinScorer found no best coin or result is not a dict.")
                        self.final_candidates = []
                else: # if not opportunities
                    self.scan_progress.emit("📊 MarketScanner found no opportunities.")
                    self.final_candidates = []

            else: # Fallback, ha nincs market_scanner vagy ai_mode_active false
                logger.info("Using basic indicator scan (fallback)...")
                self.scan_progress.emit("📊 Running Basic Indicator Scanner...")

                if hasattr(self.main_window, 'api') and self.main_window.api and \
                   hasattr(self.main_window, 'indicator_engine') and self.main_window.indicator_engine:

                    pairs_to_scan_basic = self.main_window.api.get_usd_pairs_with_volume(min_volume_usd=self.min_volume_usd)[:10] # Max 10 pár
                    basic_scan_results = []

                    for pair_data_basic in pairs_to_scan_basic:
                        pair_name_basic = pair_data_basic.get('altname')
                        if not pair_name_basic: continue

                        ohlc_basic = self.main_window.api.get_ohlc(pair_name_basic, interval=1, limit=50)
                        if ohlc_basic and pair_name_basic in ohlc_basic and ohlc_basic[pair_name_basic]:
                            df_basic = pd.DataFrame(ohlc_basic[pair_name_basic],
                                              columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
                            df_basic = df_basic.astype({"close": float}) # Csak a close kell float-ként

                            if not df_basic.empty:
                                indicators_basic = self.main_window.indicator_engine.compute_all_indicators(df_basic.copy()) # Másolaton dolgozunk
                                signals_basic = self.main_window.indicator_engine.compute_scalping_signals(df_basic.copy())
                                score_basic = self._calculate_basic_score(indicators_basic, signals_basic)

                                if score_basic >= self.min_score_threshold:
                                    basic_scan_results.append({
                                        'pair': pair_name_basic,
                                        'score': score_basic,
                                        'volume_usd': pair_data_basic.get('volume_usd', 0),
                                        'price': float(df_basic['close'].iloc[-1]),
                                        # 'indicators': indicators_basic, # Lehet túl sok adat a jelzéshez
                                        # 'signals': signals_basic
                                    })

                    if basic_scan_results:
                        basic_scan_results.sort(key=lambda x: x['score'], reverse=True)
                        best_basic_opp = basic_scan_results[0]

                        self.final_candidates = [{
                            'altname': res['pair'], 'final_score': res['score'],
                            'volume_usd': res['volume_usd'], 'pump_detected': False
                        } for res in basic_scan_results[:5]]

                        opportunity_data_basic = {
                            'pair': best_basic_opp['pair'],
                            'score': best_basic_opp['score'],
                            'reason': f"Basic Indicator Score: {best_basic_opp['score']:.3f}",
                            'entry_price': best_basic_opp['price'],
                            'volume': 50.0 / best_basic_opp['price'] if best_basic_opp['price'] > 0 else 0,
                            'stop_loss': best_basic_opp['price'] * 0.96 if best_basic_opp['price'] > 0 else 0,
                            'take_profit': best_basic_opp['price'] * 1.008 if best_basic_opp['price'] > 0 else 0,
                            'signal': 'buy',
                            'total_score': best_basic_opp['score'],
                            'position_size_usd': 50.0
                        }
                        self.opportunity_found.emit(opportunity_data_basic)
                        self.ai_decision.emit({
                            'decision': f"Basic scan found: {best_basic_opp['pair']}",
                            'pair': best_basic_opp['pair'],
                            'score': best_basic_opp['score']
                        })
                        self.scan_progress.emit(f"✅ Basic scan completed: {len(basic_scan_results)} candidates")
                    else:
                        self.scan_progress.emit("📊 Basic scan found no opportunities.")
                        self.final_candidates = []
                else:
                    logger.warning("API or IndicatorEngine not available for basic scan.")
                    self.scan_progress.emit("⚠️ Basic scan modules missing.")
                    self.final_candidates = []


            self.scan_progress.emit(f"✅ Scan cycle completed at {time.strftime('%H:%M:%S')}")

        except Exception as e:
            error_msg = f"Trading cycle execution error: {e}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            self.scan_progress.emit(f"⚠️ Scan failed: {str(e)[:50]}")

    def _create_ml_features_from_analysis(self, coin_analysis: CoinAnalysis) -> Optional[MLFeatures]:
        """CoinAnalysis objektumból MLFeatures létrehozása"""
        try:
            # Az MLFeatures importja itt történik, ahogy a felhasználó kérte
            # from strategy.ml_scoring_engine import MLFeatures # Ez már fentebb van az ADVANCED_MODULES_IMPORT blokkban
            # De ha az ADVANCED_MODULES_AVAILABLE False, akkor a fallbackot használjuk

            if not isinstance(coin_analysis, CoinAnalysis): # Típusellenőrzés
                logger.warning(f"Invalid type for coin_analysis: {type(coin_analysis)}. Expected CoinAnalysis.")
                return None

            # Összegyűjtjük azokat az attribútumokat, amiket az MLFeatures vár,
            # getattr-ral, hogy ne legyen hiba, ha egy attribútum hiányzik a CoinAnalysis-ból.
            # Az MLFeatures konstruktorának paramétereit kell itt használni.
            features_dict = {
                'rsi_3m': getattr(coin_analysis, 'rsi_3m', 50.0),
                'rsi_15m': getattr(coin_analysis, 'rsi_15m', 50.0),
                'macd_3m': getattr(coin_analysis, 'macd_3m', 0.0),
                'macd_15m': getattr(coin_analysis, 'macd_15m', 0.0),
                'stoch_rsi': getattr(coin_analysis, 'stoch_rsi_3m', 50.0),
                'williams_r': getattr(coin_analysis, 'williams_r_3m', -50.0),

                'bb_position': getattr(coin_analysis, 'bb_position', 0.5),
                'bb_squeeze': getattr(coin_analysis, 'bb_squeeze', False),
                'bb_breakout_potential': getattr(coin_analysis, 'bb_breakout_potential', 0.0),
                'bb_width_ratio': getattr(coin_analysis, 'bb_width_ratio', 0.5),

                'volume_ratio': getattr(coin_analysis, 'volume_ratio', 1.0),
                'volume_trend': getattr(coin_analysis, 'volume_trend', 0.0),
                'volume_spike': getattr(coin_analysis, 'volume_ratio', 1.0) > 2.0,

                'btc_correlation': getattr(coin_analysis, 'btc_correlation', 0.0),
                'eth_correlation': getattr(coin_analysis, 'eth_correlation', 0.0),
                'btc_momentum': getattr(coin_analysis, 'btc_momentum', 0.001),
                'eth_momentum': getattr(coin_analysis, 'eth_momentum', 0.001),

                'near_support': getattr(coin_analysis, 'near_support', False),
                'near_resistance': getattr(coin_analysis, 'near_resistance', False),
                'support_strength': getattr(coin_analysis, 'support_strength', 0.0),
                'price_momentum_5m': getattr(coin_analysis, 'momentum_score', 0.0),

                'atr_ratio': getattr(coin_analysis, 'atr_ratio', 1.0),
                'volatility_spike': getattr(coin_analysis, 'volatility_spike', False),

                'current_price': getattr(coin_analysis, 'price', 0.0),
                'volume_24h_usd': getattr(coin_analysis, 'volume_24h_usd', getattr(coin_analysis, 'volume_24h', 0.0)),
                'price_change_pct_1h': getattr(coin_analysis, 'price_change_pct_1h', 0.0),
                'volatility_15m': getattr(coin_analysis, 'atr_15m_pct', 0.01),
                'market_trend_btc': getattr(coin_analysis, 'btc_trend_短期', 'NEUTRAL'),
                'bb_status_15m': getattr(coin_analysis, 'bb_status_15m', 'MIDDLE'),
                'trend_strength_1h': getattr(coin_analysis, 'adx_1h', 20),
                'sr_level_proximity': getattr(coin_analysis, 'sr_proximity', 0.5)
            }

            # MLFeatures példányosítása a gyűjtött attribútumokkal
            # Ha az MLFeatures osztály importja sikertelen volt, a fallback osztályt használjuk.
            # A fallback MLFeatures konstruktora fogadja a **kwargs-ot.
            # Itt most már az import blokk kezeli az MLFeatures fallback-et,
            # így közvetlenül hívhatjuk.
            # Ensure MLFeatures is available (either real or fallback from the top import block)
            if 'MLFeatures' in globals() or 'MLFeatures' in locals():
                return MLFeatures(**features_dict)
            else:
                # This case should ideally not be reached if the top import block is correct
                logger.error("MLFeatures class is not defined (neither real nor fallback). Cannot create features.")
                return None

        except Exception as e:
            logger.error(f"ML features creation from CoinAnalysis failed for {getattr(coin_analysis, 'pair', 'UnknownPair')}: {e}", exc_info=True)
            return None

    def _calculate_basic_score(self, indicators: dict, signals: dict) -> float:
        """Alap score számítás a basic scanner számára"""
        score = 0.0

        # RSI score
        if signals.get('rsi_oversold'):
            score += 0.2
        elif signals.get('rsi_neutral'):
            score += 0.1

        # Bollinger score
        if signals.get('bb_breakout_up'):
            score += 0.3
        elif indicators.get('bb_position', 0.5) > 0.8:
            score += 0.15
        elif indicators.get('bb_position', 0.5) < 0.2:
            score += 0.15


        # EMA score
        if signals.get('ema_golden_cross'):
            score += 0.3
        elif signals.get('ema_bullish'):
            score += 0.1

        # Volume score
        if signals.get('volume_spike'):
            score += 0.1

        return min(1.0, max(0.0, score))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎯 Advanced Live Trading Bot - $50 Bollinger Breakout Edition")
        self.setMinimumSize(1600, 900)
        self.resize(1800, 1100)

        self.api = KrakenAPIClient()
        try:
            # Próbáljuk inicializálni a PositionManager-t az API klienssel
            self.position_manager = PositionManager(self.api)
            logger.info("✅ PositionManager initialized successfully with API client.")
        except TypeError: # Elkapja, ha a PositionManager konstruktora nem fogad argumentumot
            try:
                self.position_manager = PositionManager()
                logger.info("✅ PositionManager initialized successfully (no API client passed, or constructor takes no args).")
            except Exception as e_pm_no_arg:
                logger.error(f"PositionManager initialization (no args) failed: {e_pm_no_arg}", exc_info=True)
                self._initialize_fallback_position_manager()
        except Exception as e_pm: # Más PositionManager inicializálási hiba
            logger.error(f"PositionManager initialization failed: {e_pm}", exc_info=True)
            self._initialize_fallback_position_manager()


        # self.ai_mode_active = ADVANCED_MODULES_AVAILABLE # Ezt az init_ai_components-ben állítjuk be

        self.live_trading_thread = None
        self.update_counter = 0

        self.balance_label = None
        self.indicator_label = None
        self.live_status_label = None
        self.ai_status_label = None
        self.ai_decision_label = None
        self.scan_progress_label = None
        self.ai_toggle_btn = None
        self.ai_force_scan_btn = None
        self.pair_list = None
        self.manual_trade_button = None
        self.close_button = None
        self.chart_widget = None
        self.ai_decision_display = None
        self.ai_opportunities_display = None
        self.position_info_label = None
        self.ai_stats_label = None
        self.right_tabs = None
        self.live_panel = None
        self.ai_control_panel = None
        self.dashboard = None
        self.position_list_widget = None # Ezt a setup_ui-ban hozzuk létre
        self.scoring_panel = None
        self.stats_panel = None
        self.settings_panel = None
        self.ai_performance_labels = {}
        self.ai_decision_log = None
        self.min_volume_display_label = None
        self.min_score_display_label = None
        self.pump_thresh_display_label = None
        self.ai_mode_label = None

        self.current_chart_pair = None
        self.live_trading_active = False

        self.ai_decisions = []
        self.ai_performance = {
            'total_decisions': 0,
            'successful_trades': 0,
            'ai_profit': 0.0,
            'ai_accuracy': 0.0
        }
        self.last_trade_pnl = 0.0

        self.init_ai_components() # AI komponensek inicializálása
        self.test_api_connection()
        self.setup_ui() # Ez hozza létre a GUI elemeket, beleértve a PositionListWidget-et
        self.init_standard_components() # Standard komponensek, pl. TradeManager összekötése a PositionManagerrel
        self.setup_timers()
        self.setup_live_trading_connections()

        self.initialize_live_trading_system()

        self.update_chart()
        self.refresh_balance()
        self.refresh_open_trades()
        self.refresh_stats()
        self.start_score_updater()

        logger.info("✅ Enhanced Main window initialized successfully")

    def _initialize_fallback_position_manager(self):
        """Fallback PositionManager inicializálása, ha a valós nem sikerül."""
        logger.warning("⚠️ Initializing fallback PositionManager due to an error with the primary one.")
        class FallbackPositionManager:
            def __init__(self):
                self.positions: Dict[str, dict] = {}
                self.position_history: List[dict] = []
                self.max_positions = 5
                logger.info("FallbackPositionManager instance created.")
            def get_all_positions(self) -> Dict[str, dict]: return self.positions.copy()
            def get_position(self, pair: str) -> Optional[dict]: return self.positions.get(pair)
            def open_position(self, pair: str, side: str, entry_price: float, volume: float, **kwargs) -> bool:
                if len(self.positions) >= self.max_positions:
                    logger.warning(f"Fallback PM: Max positions ({self.max_positions}) reached. Cannot open {pair}.")
                    return False
                logger.info(f"Fallback PM: Opening {side.upper()} {pair} @ {entry_price:.4f}, Vol: {volume:.4f}")
                self.positions[pair] = {
                    'pair': pair, 'side': side, 'entry_price': entry_price, 'volume': volume,
                    'status': 'open', 'open_time': time.time(), **kwargs
                }
                return True
            def close_position(self, pair: str, exit_price: Optional[float], reason: str = "") -> Optional[dict]:
                if pair not in self.positions:
                    logger.warning(f"Fallback PM: No open position for {pair} to close.")
                    return None
                
                pos_data = self.positions.pop(pair)
                pnl = 0.0
                if exit_price is not None and pos_data.get('entry_price') is not None and pos_data.get('volume') is not None:
                    entry = pos_data['entry_price']
                    vol = pos_data['volume']
                    if pos_data['side'] == 'buy':
                        pnl = (exit_price - entry) * vol
                    else: # sell
                        pnl = (entry - exit_price) * vol
                
                closed_info = {
                    **pos_data, 'status': 'closed', 'exit_price': exit_price,
                    'close_time': time.time(), 'reason': reason, 'pnl': pnl
                }
                self.position_history.append(closed_info)
                logger.info(f"Fallback PM: Closed {pair} @ {exit_price} P&L: ${pnl:.2f} Reason: {reason}")
                return {'pnl': pnl, **closed_info} # Visszaadja a PnL-t és a zárt pozíció adatait
            def get_statistics(self) -> dict:
                total_trades = len(self.position_history)
                wins = sum(1 for trade in self.position_history if trade.get('pnl', 0) > 0)
                win_rate = (wins / total_trades) if total_trades > 0 else 0.0
                total_profit = sum(trade.get('pnl', 0) for trade in self.position_history)
                return {'total_trades': total_trades, 'win_rate': win_rate, 'total_profit': total_profit}
            def close_all_positions_market(self, current_prices: Optional[Dict[str, float]] = None) -> list:
                closed_trades_info = []
                for pair in list(self.positions.keys()): # list() másolatot készít, hogy iterálás közben törölhessünk
                    # A fallbackben nem kérünk le valós árat, egyszerűen szimuláljuk a zárást
                    # Ha lenne current_prices, azt használhatnánk, de itt most None-t feltételezünk
                    exit_price_sim = self.positions[pair]['entry_price'] * (1.001 if self.positions[pair]['side'] == 'buy' else 0.999) # Kis szimulált mozgás
                    closed_info = self.close_position(pair, exit_price_sim, reason="FALLBACK_CLOSE_ALL_MARKET")
                    if closed_info:
                        closed_trades_info.append(closed_info)
                logger.info(f"Fallback PM: Closed all ({len(closed_trades_info)}) positions market (simulated).")
                return closed_trades_info
            def get_history(self): # Hozzáadott metódus a fallbackhez
                return self.position_history[:]


        self.position_manager = FallbackPositionManager()
        logger.info("✅ Fallback PositionManager initialized and assigned to self.position_manager.")


    def init_ai_components(self):
        """AI komponensek inicializálása"""
        try:
            if ADVANCED_MODULES_AVAILABLE:
                try:
                    self.ai_scanner = AdvancedMarketScanner()
                    self.ml_engine = MLScoringEngine()
                    self.micro_strategy = MicroStrategyEngine()
                    # self.risk_manager = DynamicRiskManager() # KIKOMMENTELVE a felhasználó kérésére

                    # FIX: CorrelationAnalyzer nem fogad el api_client paramétert
                    self.correlation_analyzer = CorrelationAnalyzer()  # Paraméter nélkül hívjuk

                    self.sr_detector = SupportResistanceDetector()
                    self.ai_mode_active = True
                    logger.info("🚀 Advanced AI trading components initialized")
                    self._setup_ai_connections()  # AI kapcsolatok beállítása
                except Exception as e:
                    logger.error(f"⚠️ AI modules initialization failed: {e}")
                    self.ai_mode_active = False
                    self._init_fallback_components()
            else:
                self.ai_mode_active = False
                # A felhasználó kérte, hogy itt ne legyen warning, ha az ADVANCED_MODULES_AVAILABLE False
                # logger.warning("⚠️ Advanced AI modules not available, using basic mode. Initializing fallback components.")
                if not ADVANCED_MODULES_AVAILABLE: # Csak akkor logolunk, ha tényleg ez volt az ok
                     logger.info("ℹ️ Advanced AI modules are not available (ADVANCED_MODULES_AVAILABLE=False). Initializing fallback components.")
                self._init_fallback_components()  # Fallback AI komponensek
        except Exception as e:
            logger.error(f"Critical error in init_ai_components: {e}")
            self.ai_mode_active = False
            self._init_fallback_components()

    def _setup_ai_connections(self):
        """AI komponensek közötti kapcsolatok beállítása (pl. API kliens átadása)"""
        try:
            if self.api: # API kliens átadása, ha létezik
                components_to_setup = []
                if hasattr(self, 'ai_scanner'):
                    components_to_setup.append(self.ai_scanner)
                if hasattr(self, 'sr_detector'):
                     components_to_setup.append(self.sr_detector)
                if hasattr(self, 'micro_strategy'):
                    components_to_setup.append(self.micro_strategy)
                # if hasattr(self, 'risk_manager'): # KIKOMMENTELVE
                #     components_to_setup.append(self.risk_manager)

                for component in components_to_setup:
                    if component and self.api:
                        if hasattr(component, 'api_client') and getattr(component, 'api_client') is None:
                            try:
                                component.api_client = self.api
                                logger.info(f"API client set for {type(component).__name__}")
                            except AttributeError: pass
                        elif hasattr(component, 'connect_api_client') and callable(getattr(component, 'connect_api_client')):
                            component.connect_api_client(self.api)
                            logger.info(f"API client connected for {type(component).__name__} via connect_api_client")
                        elif hasattr(component, 'set_api_client') and callable(getattr(component, 'set_api_client')):
                            component.set_api_client(self.api)
                            logger.info(f"API client set for {type(component).__name__} via set_api_client")
                logger.info("✅ AI component API connections (re)established/verified for new components.")
        except Exception as e:
            logger.error(f"AI connections setup failed: {e}", exc_info=True)

    def _init_fallback_components(self):
        """Fallback AI komponensek inicializálása, ha a fejlett modulok nem elérhetők vagy hibásak."""
        logger.info("🔄 Initializing fallback AI components (if not already present)...")
        if not hasattr(self, 'market_scanner'):
            self.market_scanner = MarketScanner(self.api) if hasattr(self, 'api') else MarketScanner()
            logger.info("Fallback MarketScanner initialized.")
        if not hasattr(self, 'coin_scorer'):
            self.coin_scorer = CoinScorer()
            logger.info("Fallback CoinScorer initialized.")
        if not hasattr(self, 'indicator_engine'):
            self.indicator_engine = IndicatorEngine()
            logger.info("Fallback IndicatorEngine initialized.")
        if not hasattr(self, 'intelligent_trader'):
            self.intelligent_trader = IntelligentTrader(main_window=self)
            logger.info("Fallback IntelligentTrader initialized.")
        if not hasattr(self, 'ai_scanner'):
            self.ai_scanner = AdvancedMarketScanner()
            logger.info("Fallback AdvancedMarketScanner (self.ai_scanner) initialized.")
        if not hasattr(self, 'ml_engine'):
            self.ml_engine = MLScoringEngine()
            logger.info("Fallback MLScoringEngine (self.ml_engine) initialized.")
        if not hasattr(self, 'micro_strategy'):
            self.micro_strategy = MicroStrategyEngine()
            logger.info("Fallback MicroStrategyEngine (self.micro_strategy) initialized.")
        # self.risk_manager nem lesz inicializálva itt sem, a felhasználó kérése szerint
        if not hasattr(self, 'correlation_analyzer'):
            self.correlation_analyzer = CorrelationAnalyzer()
            logger.info("Fallback CorrelationAnalyzer (self.correlation_analyzer) initialized.")
        if not hasattr(self, 'sr_detector'):
            self.sr_detector = SupportResistanceDetector()
            logger.info("Fallback SupportResistanceDetector (self.sr_detector) initialized.")

        self.ai_mode_active = False
        logger.info("✅ Fallback AI components verified/initialized. AI mode set to BASIC.")
        self._setup_fallback_api_connections()


    def _setup_fallback_api_connections(self):
        """API kliens átadása a fallback komponenseknek, ha szükséges."""
        try:
            if self.api:
                fallback_components_needing_api = []
                if hasattr(self, 'market_scanner') and hasattr(self.market_scanner, 'api_client'):
                    fallback_components_needing_api.append(self.market_scanner)
                if hasattr(self, 'ai_scanner') and hasattr(self.ai_scanner, 'api_client'):
                     fallback_components_needing_api.append(self.ai_scanner)
                if hasattr(self, 'sr_detector') and hasattr(self.sr_detector, 'api_client'):
                     fallback_components_needing_api.append(self.sr_detector)

                for component in fallback_components_needing_api:
                    if component and self.api:
                        if getattr(component, 'api_client', True) is None:
                            try:
                                component.api_client = self.api
                                logger.info(f"API client set for FALLBACK component {type(component).__name__}")
                            except AttributeError: pass
                        elif hasattr(component, 'connect_api_client') and callable(getattr(component, 'connect_api_client')):
                            component.connect_api_client(self.api)
                            logger.info(f"API client connected for FALLBACK {type(component).__name__} via connect_api_client")
                        elif hasattr(component, 'set_api_client') and callable(getattr(component, 'set_api_client')):
                            component.set_api_client(self.api)
                            logger.info(f"API client set for FALLBACK {type(component).__name__} via set_api_client")
                logger.info("API connections for fallback components checked/attempted.")
        except Exception as e:
            logger.error(f"Error setting API client for fallback components: {e}", exc_info=True)


    def init_standard_components(self):
        """Standard (nem-AI specifikus) komponensek inicializálása."""
        # TradeManager inicializálása az api_client-tel és a position_manager-rel
        if hasattr(self, 'api') and hasattr(self, 'position_manager') and self.api and self.position_manager:
            self.trader = TradeManager(self.api, self.position_manager) # MÓDOSÍTOTT PÉLDÁNYOSÍTÁS
            logger.info("✅ TradeManager initialized with self.api and self.position_manager.")
        else:
            # Fallback, ha valamiért hiányozna az api vagy a position_manager
            # Ez elvileg nem fordulhat elő, ha a __init__ helyesen lefutott
            missing_comps = []
            if not hasattr(self, 'api') or not self.api : missing_comps.append("self.api")
            if not hasattr(self, 'position_manager') or not self.position_manager: missing_comps.append("self.position_manager")
            logger.error(f" CRITICAL: {', '.join(missing_comps)} not available for TradeManager initialization. Using None for missing parts.")
            # Próbáljuk meg legalább azzal inicializálni, ami van
            api_to_pass = self.api if hasattr(self, 'api') and self.api else None
            pm_to_pass = self.position_manager if hasattr(self, 'position_manager') and self.position_manager else None
            self.trader = TradeManager(api_to_pass, pm_to_pass)

        # Ellenőrizzük, hogy a trader.position_manager is be lett-e állítva (a konstruktorban kellett volna)
        # és hogy az megegyezik-e a MainWindow self.position_manager-ével.
        if hasattr(self.trader, 'position_manager') and self.trader.position_manager is self.position_manager:
            logger.info("🔗 TradeManager's position_manager confirmed to be self.position_manager.")
        elif hasattr(self, 'position_manager') and self.position_manager:
             # Ha a trader konstruktora nem állította be, vagy felülírta, itt megpróbáljuk újra
            if hasattr(self.trader, 'position_manager'): # Ha van ilyen attribútuma a tradernek
                self.trader.position_manager = self.position_manager
                logger.info("🔗 TradeManager.position_manager explicitly (re)set to self.position_manager.")
            else:
                logger.warning("⚠️ TradeManager does not have 'position_manager' attribute to link after new instantiation.")
        elif not hasattr(self, 'position_manager') or not self.position_manager:
             logger.error(" CRITICAL: self.position_manager was not available to link with TradeManager.")


        self.trade_logger = TradeLogger()
        self.wallet_display = WalletManager()
        self.analyzer = HistoryAnalyzer()
        self.trailing_stop_manager = TrailingStopManager()
        self.initialize_websocket_data_feed()
        logger.info("✅ Standard components initialized.")


    def refresh_balance(self):
        """Pénztárca egyenlegének frissítése a GUI-n."""
        try:
            if hasattr(self, 'wallet_display') and self.wallet_display and hasattr(self.wallet_display, 'get_usd_balance'):
                balance = self.wallet_display.get_usd_balance()
                if balance is not None:
                    if self.balance_label:
                        self.balance_label.setText(f"Balance: ${balance:.2f}")
                    if self.dashboard:
                        self.dashboard.update_field("Account Balance", f"${balance:.2f}")
                else:
                    if self.balance_label:
                        self.balance_label.setText("Balance: Error fetching")
            else:
                fallback_balance = 66.0
                if self.balance_label:
                    self.balance_label.setText(f"Balance: ${fallback_balance:.2f} (Sim)")
                if self.dashboard:
                    self.dashboard.update_field("Account Balance", f"${fallback_balance:.2f} (Simulation)")
                logger.warning("wallet_display or get_usd_balance not available, using fallback balance.")
        except Exception as e:
            logger.error(f"[WALLET] Balance refresh failed: {e}", exc_info=True)
            if self.balance_label:
                self.balance_label.setText("Balance: Error")

    def refresh_open_trades(self):
        """Nyitott kereskedések listájának frissítése a GUI-n."""
        try:
            if not hasattr(self, 'position_manager') or not self.position_manager:
                if self.position_info_label:
                    self.position_info_label.setText("Positions:\n(Position manager not available)")
                logger.warning("Position manager not available for refreshing open trades.")
                return

            open_positions = self.position_manager.get_all_positions() if hasattr(self.position_manager, 'get_all_positions') else {}

            if not open_positions:
                if self.position_info_label:
                    self.position_info_label.setText("Positions:\n(none)")
            else:
                msg = "Open Positions:\n"
                for pair, data in open_positions.items():
                    side = data.get('side', 'N/A')
                    entry_price = data.get('entry_price', 0)
                    volume = data.get('volume', 0)
                    current_price = self._get_real_current_price(pair, entry_price)


                    pnl_text = "N/A"
                    pnl_color_char = ""

                    if current_price is not None and entry_price > 0 and volume > 0:
                        pnl = (current_price - entry_price) * volume if side.lower() == 'buy' else (entry_price - current_price) * volume
                        pnl_text = f"${pnl:.2f}"
                        pnl_color_char = "🟢" if pnl >= 0 else "🔴"

                    msg += f"{pnl_color_char} {pair} ({side.upper()}): Entry ${entry_price:.4f}, Vol {volume:.6f} - P&L: {pnl_text}\n"

                if self.position_info_label:
                    self.position_info_label.setText(msg.strip())
            
            # Frissítjük a PositionListWidget-et is, ha létezik
            if self.position_list_widget and hasattr(self.position_list_widget, 'update_positions'):
                self.position_list_widget.update_positions(open_positions)


        except Exception as e:
            logger.error(f"[UI] Position display refresh failed: {e}", exc_info=True)
            if self.position_info_label:
                self.position_info_label.setText("Positions:\nError refreshing")

    def refresh_stats(self):
        """Kereskedési statisztikák frissítése a GUI-n."""
        try:
            if self.ai_stats_label:
                current_time = pd.Timestamp.now().strftime('%H:%M:%S')
                total_trades = 0
                win_rate = 0.0
                total_profit = 0.0

                if hasattr(self, 'trade_logger') and self.trade_logger and hasattr(self.trade_logger, 'get_statistics'):
                    stats_trade_logger = self.trade_logger.get_statistics()
                    # Használjuk a TradeLogger statisztikáit, ha elérhetőek és megbízhatóbbak
                    total_trades = stats_trade_logger.get('total_trades', 0)
                    win_rate = stats_trade_logger.get('win_rate', 0.0) * 100
                    total_profit = stats_trade_logger.get('total_profit', 0.0)
                elif hasattr(self, 'position_manager') and self.position_manager and hasattr(self.position_manager, 'get_statistics'):
                    # Fallback a PositionManager statisztikáira
                    stats_pm = self.position_manager.get_statistics()
                    total_trades = stats_pm.get('total_trades', 0)
                    win_rate = stats_pm.get('win_rate', 0.0) * 100
                    total_profit = stats_pm.get('total_profit', 0.0)


                stats_text = f"Trading Stats ({current_time}):\n"
                stats_text += f"Total Trades: {total_trades}\n"
                stats_text += f"Win Rate: {win_rate:.1f}%\n"
                stats_text += f"Total P&L: ${total_profit:.2f}"

                self.ai_stats_label.setText(stats_text)

            if self.stats_panel and hasattr(self.stats_panel, 'update_stats'):
                self.stats_panel.update_stats() # Ez valószínűleg a PositionManager adatait használja majd

        except Exception as e:
            logger.error(f"[STATS] Stats refresh failed: {e}", exc_info=True)
            if self.ai_stats_label:
                self.ai_stats_label.setText("Trading Stats:\nError updating")

    def start_score_updater(self):
        """Coin pontozó rendszer frissítőjének indítása."""
        try:
            logger.info("📊 Starting score updater...")
            if self.scoring_panel and hasattr(self.scoring_panel, 'start_auto_refresh'):
                self.scoring_panel.start_auto_refresh()
                logger.info("✅ Score updater delegated to scoring panel's auto-refresh.")
            elif self.scoring_panel and hasattr(self.scoring_panel, 'refresh_panel'):
                 self.scoring_panel.refresh_panel()
                 logger.info("✅ Initial score panel refresh triggered.")
            else:
                logger.warning("⚠️ Scoring panel not available or missing refresh method for score updates.")

        except Exception as e:
            logger.error(f"[SCORER] Score updater initialization failed: {e}", exc_info=True)


    def initialize_websocket_data_feed(self):
        """WebSocket adatfolyam inicializálása a kötési párokhoz."""
        try:
            logger.info("🚀 Initializing WebSocket real-time data feed...")
            if not (hasattr(self, 'api') and self.api and hasattr(self.api, 'get_valid_usd_pairs')):
                logger.error("API client not initialized or get_valid_usd_pairs method missing.")
                if self.dashboard:
                    self.dashboard.update_field("Data Feed Status", "🔴 API ERROR")
                return

            initial_pairs_for_ws = []
            if self.pair_list and self.pair_list.count() > 0:
                initial_pairs_for_ws = [self.pair_list.item(i).text().replace("/", "") for i in range(min(self.pair_list.count(), 5))]
            else:
                valid_pairs_from_api = self.api.get_valid_usd_pairs()
                if valid_pairs_from_api:
                    initial_pairs_for_ws = [p['altname'] for p in valid_pairs_from_api[:2] if 'altname' in p]
                else:
                    initial_pairs_for_ws = ["XBTUSD", "ETHUSD"]

            if not initial_pairs_for_ws:
                logger.warning("No pairs selected for initial WebSocket subscription.")
                if self.dashboard: self.dashboard.update_field("Data Feed Status", "⚪ NO PAIRS FOR WS")
                return

            logger.info(f"📡 Subscribing to {len(initial_pairs_for_ws)} pairs via WebSocket: {initial_pairs_for_ws}")
            success = self.api.initialize_websocket(initial_pairs_for_ws) if hasattr(self.api, 'initialize_websocket') else False

            if success:
                logger.info("✅ WebSocket real-time data feed potentially active.")
                if self.dashboard:
                    self.dashboard.update_field("Data Feed Status", "🟢 WebSocket Attempted")
                    self.dashboard.update_field("WS Monitored Pairs", str(len(initial_pairs_for_ws)))
            else:
                logger.warning("⚠️ WebSocket initialization call returned false. Using REST API fallback.")
                if self.dashboard:
                    self.dashboard.update_field("Data Feed Status", "🟡 REST Fallback")

        except Exception as e:
            logger.error(f"❌ WebSocket initialization failed: {e}", exc_info=True)
            if self.dashboard:
                self.dashboard.update_field("Data Feed Status", "🔴 WS INIT ERROR")

    def setup_ui(self):
        """Felhasználói felület összerakása."""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        left_widget = self.create_enhanced_left_panel()
        left_widget.setMaximumWidth(220)
        main_layout.addWidget(left_widget)

        middle_widget = self.create_ai_enhanced_middle_panel()
        main_layout.addWidget(middle_widget, 1)

        right_widget = self.create_enhanced_right_panel()
        right_widget.setMaximumWidth(450)
        main_layout.addWidget(right_widget)

    def create_enhanced_left_panel(self):
        """Bal oldali vezérlőpanel létrehozása, AI funkciókkal bővítve."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        ai_group = QGroupBox("🤖 Live Trading Status")
        ai_group.setStyleSheet("QGroupBox { font-weight: bold; color: white; }")
        ai_layout = QVBoxLayout()

        self.ai_status_label = QLabel("Live Trading: INITIALIZING")
        self.ai_status_label.setStyleSheet("font-weight: bold; color: #00AAFF; font-size: 12px;")
        ai_layout.addWidget(self.ai_status_label)

        self.ai_decision_label = QLabel("Last Scan: None")
        self.ai_decision_label.setStyleSheet("color: #CCCCCC; font-size: 10px;")
        ai_layout.addWidget(self.ai_decision_label)

        self.scan_progress_label = QLabel("Scan Progress: Idle")
        self.scan_progress_label.setStyleSheet("color: #CCCCCC; font-size: 10px;")
        ai_layout.addWidget(self.scan_progress_label)

        self.ai_toggle_btn = QPushButton("🚀 Start Live Trading")
        self.ai_toggle_btn.clicked.connect(self.toggle_live_trading)
        self.ai_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #2ECC71; }
        """)
        ai_layout.addWidget(self.ai_toggle_btn)

        self.ai_force_scan_btn = QPushButton("🔄 Force Rescan")
        self.ai_force_scan_btn.clicked.connect(self.force_live_scan)
        ai_layout.addWidget(self.ai_force_scan_btn)

        test_pos_button = QPushButton("🧪 Open Test Position")
        test_pos_button.clicked.connect(self.open_test_position)
        test_pos_button.setStyleSheet("QPushButton { background-color: #3498DB; color: white; }")
        ai_layout.addWidget(test_pos_button)

        manual_close_button = QPushButton("🚪 Manual Close Selected")
        manual_close_button.clicked.connect(self.manual_close_current_position)
        manual_close_button.setStyleSheet("QPushButton { background-color: #E67E22; color: white; }")
        ai_layout.addWidget(manual_close_button)

        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)

        try:
            self.settings_panel = SettingsPanel()
            layout.addWidget(self.settings_panel)
        except Exception as e:
            logger.warning(f"Could not create SettingsPanel: {e}")
            layout.addWidget(QLabel("Settings Panel (Error)"))


        pairs_label = QLabel("💰 High-Volume Pairs (Chart):")
        pairs_label.setStyleSheet("color: white; font-weight: bold; font-size: 11px;")
        layout.addWidget(pairs_label)

        self.pair_list = QListWidget()
        self.load_trading_pairs()
        self.pair_list.setMaximumHeight(150)
        self.pair_list.currentTextChanged.connect(self.on_pair_changed)
        layout.addWidget(self.pair_list)

        self.manual_trade_button = QPushButton("👤 Manual Trade (Selected)")
        self.manual_trade_button.clicked.connect(self.execute_manual_trade)
        self.manual_trade_button.setStyleSheet("QPushButton { background-color: #9B59B6; color: white; }")
        layout.addWidget(self.manual_trade_button)

        self.close_button = QPushButton("❌ Close All Positions")
        self.close_button.clicked.connect(self.manual_close_all_positions)
        self.close_button.setStyleSheet("QPushButton { background-color: #E74C3C; color: white; font-weight: bold; }")
        layout.addWidget(self.close_button)

        layout.addStretch()
        return widget

    def create_ai_enhanced_middle_panel(self):
        """Középső panel létrehozása, AI információkkal és jobb charttal."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)

        status_top_layout = QHBoxLayout()
        self.balance_label = QLabel("Balance: Loading...")
        self.balance_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #39ff14;")
        status_top_layout.addWidget(self.balance_label)

        self.ai_mode_label = QLabel(f"🤖 AI Modules: {'ADVANCED' if self.ai_mode_active else 'BASIC'}")
        self.ai_mode_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #8E44AD;")
        status_top_layout.addWidget(self.ai_mode_label, 0, Qt.AlignRight)
        layout.addLayout(status_top_layout)

        status_mid_layout = QHBoxLayout()
        self.indicator_label = QLabel("🎯 Chart Status: Select a pair!")
        self.indicator_label.setStyleSheet("font-size: 11px; color: white; padding: 2px;")
        status_mid_layout.addWidget(self.indicator_label, 2)

        self.live_status_label = QLabel("⚪ OFFLINE")
        self.live_status_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #FF4444; padding: 2px;")
        status_mid_layout.addWidget(self.live_status_label, 1, Qt.AlignRight)
        layout.addLayout(status_mid_layout)


        self.ai_decision_display = QLabel("🧠 Live Scanner: Initializing...")
        self.ai_decision_display.setStyleSheet("font-size: 11px; color: white; font-weight: bold; padding: 3px; background-color: #2C3E50; border-radius: 3px;")
        self.ai_decision_display.setWordWrap(True)
        self.ai_decision_display.setFixedHeight(40)
        layout.addWidget(self.ai_decision_display)

        self.setup_ai_enhanced_chart()
        layout.addWidget(self.chart_widget, 2)

        self.ai_opportunities_display = QTextEdit()
        self.ai_opportunities_display.setMaximumHeight(120)
        self.ai_opportunities_display.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #00AAFF;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                border: 1px solid #555;
                border-radius: 5px;
            }
        """)
        self.ai_opportunities_display.setPlaceholderText("🚀 Top Scan Candidates will appear here...")
        layout.addWidget(self.ai_opportunities_display)

        info_layout = QHBoxLayout()
        self.position_info_label = QLabel("Positions: (none)")
        self.position_info_label.setStyleSheet("background:#191919; color:white; border:1px solid #222; padding:6px; font-size: 9px;")
        self.position_info_label.setWordWrap(True)
        self.position_info_label.setMinimumHeight(60)
        info_layout.addWidget(self.position_info_label)

        self.ai_stats_label = QLabel("Trading Stats: Starting...")
        self.ai_stats_label.setStyleSheet("background:#191919; color:#8E44AD; border:1px solid #222; padding:6px; font-size: 9px;")
        self.ai_stats_label.setWordWrap(True)
        self.ai_stats_label.setMinimumHeight(60)
        info_layout.addWidget(self.ai_stats_label)

        layout.addLayout(info_layout)
        return widget

    def setup_ai_enhanced_chart(self):
        """Fejlettebb diagram beállítása."""
        self.chart_widget = pg.PlotWidget()
        self.chart_widget.showGrid(x=True, y=True, alpha=0.3)
        self.chart_widget.setBackground((20, 20, 20))
        self.chart_widget.getAxis('left').setPen(pg.mkPen(color=(150, 150, 150)))
        self.chart_widget.getAxis('bottom').setPen(pg.mkPen(color=(150, 150, 150)))
        self.chart_widget.setLabel('left', 'Price ($)', color='lightgray', size='10pt')
        self.chart_widget.setLabel('bottom', 'Time', color='lightgray', size='10pt')
        self.chart_widget.setMinimumHeight(300)

    def create_enhanced_right_panel(self):
        """Jobb oldali panel létrehozása fülekkel, AI vezérlőkkel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)

        self.right_tabs = QTabWidget()
        self.right_tabs.setTabPosition(QTabWidget.North)

        try:
            self.live_panel = LiveTradingPanel(api_client=self.api, parent_window=self)
        except Exception as e:
            logger.error(f"Failed to create LiveTradingPanel: {e}", exc_info=True)
            self.live_panel = QWidget()
            self.live_panel.setLayout(QVBoxLayout())
            self.live_panel.layout().addWidget(QLabel("Live Trading Panel (Error)"))

        live_scroll = QScrollArea()
        live_scroll.setWidget(self.live_panel)
        live_scroll.setWidgetResizable(True)
        live_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        live_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_tabs.addTab(live_scroll, "🚀 Live Control")

        try:
            if 'AdvancedControlPanel' in globals() or 'AdvancedControlPanel' in locals():
                 self.ai_control_panel = AdvancedControlPanel()
            else:
                logger.warning("AdvancedControlPanel not found, creating default AI control tab content.")
                self.ai_control_panel = self.create_ai_control_tab_content()
        except Exception as e:
            logger.error(f"Failed to create AI Control Panel content: {e}", exc_info=True)
            self.ai_control_panel = QWidget()
            self.ai_control_panel.setLayout(QVBoxLayout())
            self.ai_control_panel.layout().addWidget(QLabel("AI Control Panel (Error)"))

        ai_scroll = QScrollArea()
        ai_scroll.setWidget(self.ai_control_panel)
        ai_scroll.setWidgetResizable(True)
        ai_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ai_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_tabs.addTab(ai_scroll, "🧠 AI Console")

        dashboard_content_widget = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_content_widget)
        dashboard_layout.setSpacing(5)

        try:
            self.dashboard = DashboardPanel()
            dashboard_layout.addWidget(self.dashboard)
        except Exception as e:
            logger.warning(f"Could not create DashboardPanel: {e}")
            dashboard_layout.addWidget(QLabel("Dashboard Panel (Error)"))

        try:
            # Itt használjuk a self.position_manager-t
            if hasattr(self, 'position_manager') and self.position_manager:
                self.position_list_widget = PositionListWidget(self.position_manager, parent_window=self)
                dashboard_layout.addWidget(self.position_list_widget)
            else:
                logger.error("self.position_manager not available for PositionListWidget creation.")
                dashboard_layout.addWidget(QLabel("Position List (Error - No PositionManager)"))

        except Exception as e:
            logger.warning(f"Could not create PositionListWidget: {e}", exc_info=True)
            dashboard_layout.addWidget(QLabel(f"Position List (Error: {str(e)[:30]})"))


        dashboard_scroll = QScrollArea()
        dashboard_scroll.setWidget(dashboard_content_widget)
        dashboard_scroll.setWidgetResizable(True)
        dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_tabs.addTab(dashboard_scroll, "📊 Data & Positions")

        analysis_content_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_content_widget)
        analysis_layout.setSpacing(5)

        try:
            if hasattr(self, 'get_all_coin_data'):
                scorer_instance = getattr(self, 'coin_scorer', None)
                if scorer_instance:
                    self.scoring_panel = ScoringPanel(scorer_instance, self.get_all_coin_data)
                    analysis_layout.addWidget(self.scoring_panel)
                else:
                    logger.warning("self.coin_scorer not available for ScoringPanel.")
                    analysis_layout.addWidget(QLabel("Scoring Panel (Scorer Error)"))
            else:
                logger.warning("get_all_coin_data method not found, ScoringPanel cannot be created.")
                analysis_layout.addWidget(QLabel("Scoring Panel (Data Error)"))
        except Exception as e:
            logger.warning(f"Could not create ScoringPanel: {e}")
            analysis_layout.addWidget(QLabel("Scoring Panel (Error)"))

        try:
            self.stats_panel = StatsPanel() # Ennek is szüksége lehet a position_manager-re, ha statisztikákat jelenít meg
            if hasattr(self.stats_panel, 'set_position_manager') and hasattr(self, 'position_manager'): # Opcionális
                self.stats_panel.set_position_manager(self.position_manager)
            analysis_layout.addWidget(self.stats_panel)
        except Exception as e:
            logger.warning(f"Could not create StatsPanel: {e}")
            analysis_layout.addWidget(QLabel("Stats Panel (Error)"))

        analysis_scroll = QScrollArea()
        analysis_scroll.setWidget(analysis_content_widget)
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_tabs.addTab(analysis_scroll, "📈 Analysis & Stats")

        layout.addWidget(self.right_tabs)
        return widget

    def create_ai_control_tab_content(self):
        """Az 'AI Console' fül tartalmának létrehozása."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)

        ai_status_group = QGroupBox("📈 Overall AI Performance")
        ai_status_layout = QGridLayout()
        self.ai_performance_labels = {}
        metrics = ["Bot Accuracy", "Total Trades", "Successful Trades", "Total P&L", "Last Trade P&L"]
        for i, metric in enumerate(metrics):
            label = QLabel(f"{metric}:")
            label.setStyleSheet("color: #CCCCCC; font-size: 11px;")
            value = QLabel("N/A")
            value.setStyleSheet("color: #00AAFF; font-weight: bold; font-size: 11px;")
            ai_status_layout.addWidget(label, i, 0)
            ai_status_layout.addWidget(value, i, 1)
            self.ai_performance_labels[metric] = value
        ai_status_group.setLayout(ai_status_layout)
        layout.addWidget(ai_status_group)

        scanner_params_group = QGroupBox("🎯 Live Scanner Parameters")
        scanner_params_layout = QVBoxLayout()

        self.min_volume_display_label = QLabel("Min Volume USD: N/A")
        scanner_params_layout.addWidget(self.min_volume_display_label)
        self.min_score_display_label = QLabel("Min Score Candidate: N/A")
        scanner_params_layout.addWidget(self.min_score_display_label)
        self.pump_thresh_display_label = QLabel("Pump Threshold %: N/A")
        scanner_params_layout.addWidget(self.pump_thresh_display_label)

        scanner_params_group.setLayout(scanner_params_layout)
        layout.addWidget(scanner_params_group)


        log_group = QGroupBox("📝 AI Event Log")
        log_layout = QVBoxLayout()
        self.ai_decision_log = QTextEdit()
        self.ai_decision_log.setMaximumHeight(250)
        self.ai_decision_log.setStyleSheet("""
            QTextEdit {
                background-color: #1A1A1A;
                color: #9F54FF;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9px;
                border: 1px solid #444;
                border-radius: 3px;
            }
        """)
        self.ai_decision_log.setReadOnly(True)
        log_layout.addWidget(self.ai_decision_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        return content


    def create_advanced_tab(self):
        """Az 'Advanced' fül tartalmának létrehozása (ha ADVANCED_MODULES_AVAILABLE)."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)

        if self.ai_mode_active:
            header = QLabel("⚙️ Advanced AI Module Controls")
            header.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF6B35; padding: 5px;")
            layout.addWidget(header)

            if hasattr(self, 'ai_scanner') and self.ai_scanner:
                scanner_group = QGroupBox("Advanced Market Scanner Controls")
                scanner_layout = QVBoxLayout(scanner_group)
                if ADVANCED_MODULES_AVAILABLE and isinstance(self.ai_scanner, AdvancedMarketScanner):
                    scanner_layout.addWidget(QLabel("Real AdvancedMarketScanner settings..."))
                else:
                    scanner_layout.addWidget(QLabel("Advanced Market Scanner (Fallback or basic) settings..."))
                layout.addWidget(scanner_group)

            if hasattr(self, 'ml_engine') and self.ml_engine:
                ml_group = QGroupBox("ML Scoring Engine Controls")
                ml_layout = QVBoxLayout(ml_group)
                if ADVANCED_MODULES_AVAILABLE and isinstance(self.ml_engine, MLScoringEngine):
                     ml_layout.addWidget(QLabel("Real MLScoringEngine settings..."))
                else:
                    ml_layout.addWidget(QLabel("ML Scoring Engine (Fallback or basic) settings..."))
                layout.addWidget(ml_group)

            if hasattr(self, 'micro_strategy') and self.micro_strategy:
                ms_group = QGroupBox("Micro Strategy Engine Controls")
                ms_layout = QVBoxLayout(ms_group)
                if ADVANCED_MODULES_AVAILABLE and isinstance(self.micro_strategy, MicroStrategyEngine):
                     ms_layout.addWidget(QLabel("Real MicroStrategyEngine settings..."))
                else:
                    ms_layout.addWidget(QLabel("Micro Strategy Engine (Fallback or basic) settings..."))
                layout.addWidget(ms_group)

            # if hasattr(self, 'risk_manager') and self.risk_manager: # KIKOMMENTELVE
            #     # ...
            #     layout.addWidget(rm_group)

            if hasattr(self, 'coin_scorer') and self.coin_scorer:
                scorer_group = QGroupBox("Coin Scorer (Legacy) Controls")
                scorer_layout = QVBoxLayout(scorer_group)
                scorer_layout.addWidget(QLabel("Coin Scorer specific settings placeholder..."))
                layout.addWidget(scorer_group)
        else:
            layout.addWidget(QLabel("Advanced AI modules are not active (using fallbacks)."))

        layout.addStretch()
        return content


    def setup_timers(self):
        """Időzítők beállítása a rendszeres frissítésekhez."""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.enhanced_update_cycle)
        self.timer.start(30000)

        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self.monitor_positions_with_ai)
        self.position_timer.start(20000)

    def setup_live_trading_connections(self):
        """Live trading panel és a főablak közötti jelzések összekötése."""
        try:
            if self.live_panel:
                if hasattr(self.live_panel, 'start_live_trading'):
                    self.live_panel.start_live_trading.connect(self.start_live_trading_action)
                if hasattr(self.live_panel, 'stop_live_trading'):
                    self.live_panel.stop_live_trading.connect(self.stop_live_trading_action)
                if hasattr(self.live_panel, 'emergency_stop'):
                    self.live_panel.emergency_stop.connect(self.emergency_stop_all)
                if hasattr(self.live_panel, 'settings_changed'):
                    self.live_panel.settings_changed.connect(self.update_trading_settings)
                logger.info("✅ Live trading UI connections established.")
            else:
                logger.error("⚠️ self.live_panel is not initialized. Cannot set up live trading connections.")

        except Exception as e:
            logger.error(f"⚠️ Live trading connections setup failed: {e}", exc_info=True)

    def test_api_connection(self):
        """API kapcsolat tesztelése és eredmények logolása."""
        try:
            logger.info("🔗 Testing Kraken API connection...")
            if self.api.test_connection():
                logger.info("[STARTUP] ✅ API connection successful")
                if self.dashboard: self.dashboard.update_field("API Status", "🟢 CONNECTED")

                try:
                    volume_pairs = self.api.get_usd_pairs_with_volume(min_volume_usd=100000)
                    if volume_pairs:
                        logger.info(f"📊 Found {len(volume_pairs)} high-volume pairs (test). First: {volume_pairs[0].get('wsname', 'N/A') if volume_pairs else 'N/A'}")
                        if self.dashboard: self.dashboard.update_field("Data Mode", "REAL DATA")
                    else:
                        logger.warning("⚠️ No high-volume pairs found in test, API might be limited or market quiet.")
                        if self.dashboard: self.dashboard.update_field("Data Mode", "LIMITED DATA?")
                except Exception as e_vol:
                    logger.warning(f"⚠️ Volume data test failed: {e_vol}")
                    if self.dashboard: self.dashboard.update_field("Data Mode", "VOLUME API ERROR")

            else:
                logger.error("[STARTUP] ❌ API connection failed - using fallback data")
                if self.dashboard:
                    self.dashboard.update_field("API Status", "🔴 FAILED")
                    self.dashboard.update_field("Data Mode", "FALLBACK")

            if self.ai_mode_active:
                logger.info("[STARTUP] 🚀 Advanced trading modules are potentially active.")
                self._test_advanced_components()
            else:
                logger.info("[STARTUP] ⚠️ Basic trading mode (Advanced modules are not available, using fallbacks).")


        except Exception as e:
            logger.error(f"[STARTUP] ❌ API connection test error: {e}", exc_info=True)
            if self.dashboard:
                self.dashboard.update_field("API Status", f"🔴 ERROR: {str(e)[:20]}")

    def _test_advanced_components(self):
        """Fejlett AI komponensek tesztelése (ha elérhetők)."""
        if not self.ai_mode_active:
            logger.info("Skipping advanced component test as ai_mode_active is False.")
            return

        logger.info("🧪 Testing initialized ADVANCED trading components' status methods (if available)...")
        components_to_test = {
            "Advanced Market Scanner (New)": getattr(self, 'ai_scanner', None),
            "ML Scoring Engine (New)": getattr(self, 'ml_engine', None),
            "Micro Strategy Engine (New)": getattr(self, 'micro_strategy', None),
            # "Dynamic Risk Manager (New)": getattr(self, 'risk_manager', None), # KIKOMMENTELVE
            "Correlation Analyzer (New)": getattr(self, 'correlation_analyzer', None),
            "Support/Resistance Detector (New)": getattr(self, 'sr_detector', None),
        }
        for name, component in components_to_test.items():
            if component:
                status_method_name = None
                if hasattr(component, 'get_status') and callable(getattr(component, 'get_status')):
                    status_method_name = 'get_status'
                is_real_module = ADVANCED_MODULES_AVAILABLE
                if status_method_name:
                    try:
                        status = getattr(component, status_method_name)()
                        logger.info(f"  Component '{name}': Status - {status} {'(Real Module)' if is_real_module else '(Fallback/Basic)'}")
                    except Exception as e_status:
                        logger.warning(f"  Component '{name}': Error getting status - {e_status} {'(Real Module)' if is_real_module else '(Fallback/Basic)'}")
                else:
                    logger.info(f"  Component '{name}': No standard status method found, but component is initialized. {'(Real Module)' if is_real_module else '(Fallback/Basic)'}")
            else:
                logger.warning(f"  Component '{name}': Not initialized (this shouldn't happen if ai_mode_active is True).")


    def load_trading_pairs(self):
        """Kereskedési párok betöltése a GUI listába, volumen alapján szűrve."""
        try:
            logger.info("[PAIRS_UI] Loading high-volume pairs for chart dropdown...")
            if not (hasattr(self, 'api') and self.api and hasattr(self.api, 'get_usd_pairs_with_volume')):
                logger.warning("API client or get_usd_pairs_with_volume method not available. Using fallback pairs.")
                self._load_fallback_pairs_to_ui()
                return

            volume_pairs_ui = self.api.get_usd_pairs_with_volume(min_volume_usd=100000)

            if volume_pairs_ui and isinstance(volume_pairs_ui, list):
                pair_names_ui = []
                for pair_info in volume_pairs_ui[:25]:
                    altname = pair_info.get("altname")
                    wsname = pair_info.get("wsname", altname)
                    if altname and altname.upper() not in ['USDTZUSD', 'USDCUSD', 'DAIUSD', 'EURZEUR', 'GBPZGBP']:
                        display_name = wsname if wsname else altname
                        pair_names_ui.append(display_name)

                if pair_names_ui:
                    self.pair_list.clear()
                    self.pair_list.addItems(pair_names_ui)
                    if self.pair_list.count() > 0:
                        self.pair_list.setCurrentRow(0)
                        self.on_pair_changed(self.pair_list.item(0).text())
                    logger.info(f"[PAIRS_UI] ✅ Loaded {len(pair_names_ui)} pairs for UI dropdown. First: {pair_names_ui[0] if pair_names_ui else 'N/A'}")
                    if self.dashboard: self.dashboard.update_field("UI Pair List", f"{len(pair_names_ui)} pairs")
                    return

            logger.warning("[PAIRS_UI] No high-volume pairs returned from API or list is empty. Using fallback.")
            self._load_fallback_pairs_to_ui()

        except Exception as e:
            logger.error(f"[PAIRS_UI] ❌ Error loading pairs for UI: {e}", exc_info=True)
            self._load_fallback_pairs_to_ui()

    def _load_fallback_pairs_to_ui(self):
        """Fallback párok betöltése a UI listába."""
        fallback_display_pairs = ["XBT/USD", "ETH/USD", "ADA/USD", "SOL/USD", "DOT/USD", "LINK/USD"]
        self.pair_list.clear()
        self.pair_list.addItems(fallback_display_pairs)
        if self.pair_list.count() > 0:
            self.pair_list.setCurrentRow(0)
            self.on_pair_changed(self.pair_list.item(0).text())
        logger.info(f"[PAIRS_UI] Loaded fallback pairs: {fallback_display_pairs}")
        if self.dashboard: self.dashboard.update_field("UI Pair List", f"{len(fallback_display_pairs)} (fallback)")


    def on_pair_changed(self, pair_display_name: str):
        """Kiválasztott kereskedési pár változásának kezelése."""
        if pair_display_name:
            self.current_chart_pair = pair_display_name.replace("/", "")

            logger.info(f"[UI_CHART] Selected pair for chart: {pair_display_name} (Using altname: {self.current_chart_pair})")
            self.update_chart()
            if self.dashboard: self.dashboard.update_field("Selected Pair", pair_display_name)


    def get_current_price_for_pair(self, pair_altname: str) -> float or None:
        """Aktuális ár lekérdezése egy adott kereskedési párhoz (altname formátumban)."""
        try:
            if not self.api: return None

            if hasattr(self.api, 'ws_client') and self.api.ws_client and \
               hasattr(self.api.ws_client, 'get_current_price') and \
               hasattr(self.api.ws_client, 'is_pair_subscribed') and self.api.ws_client.is_pair_subscribed(pair_altname):
                price_ws = self.api.ws_client.get_current_price(pair_altname)
                if price_ws is not None and price_ws > 0:
                    return float(price_ws)

            price_rest = self.api.get_current_price(pair_altname)
            if price_rest is not None and price_rest > 0:
                return float(price_rest)

            return None

        except Exception as e:
            logger.error(f"❌ Price fetch error for {pair_altname}: {e}", exc_info=True)
            return None

    def _get_real_current_price(self, pair: str, entry_price: float = None) -> float:
        """Get real current price from API"""
        try:
            current_price = self.get_current_price_for_pair(pair)
            if current_price and current_price > 0:
                return current_price
            logger.warning(f"API price fetch failed for {pair} or returned invalid price. Using fallback.")
            return entry_price if entry_price else 1000.0
        except Exception as e:
            logger.error(f"Real price fetch failed for {pair}: {e}", exc_info=True)
            return entry_price if entry_price else 1000.0


    def update_chart(self):
        """Árfolyamdiagram frissítése a kiválasztott párhoz."""
        pair_to_chart_altname = getattr(self, 'current_chart_pair', None)

        if not pair_to_chart_altname:
            if self.indicator_label:
                self.indicator_label.setText("🎯 Chart Status: Select a pair!")
            if self.chart_widget:
                self.chart_widget.clear()
                self.chart_widget.setTitle("Select a pair from the list", color='gray')
            return

        pair_display_name = pair_to_chart_altname
        if self.pair_list and self.pair_list.currentItem():
            pair_display_name = self.pair_list.currentItem().text()


        current_time_str = pd.Timestamp.now().strftime('%H:%M:%S')
        websocket_active_for_pair = False
        if hasattr(self.api, 'ws_client') and self.api.ws_client and \
           hasattr(self.api.ws_client, 'is_pair_subscribed') and self.api.ws_client.is_pair_subscribed(pair_to_chart_altname):
            websocket_active_for_pair = True

        live_trading_panel_active = False
        if self.live_panel and hasattr(self.live_panel, 'is_trading_active'):
            live_trading_panel_active = self.live_panel.is_trading_active()

        label_style_live_ws = "font-size: 11px; font-weight: bold; color: #00FF00;"
        label_style_ws_ready = "font-size: 11px; font-weight: bold; color: #00AAFF;"
        label_style_live_rest = "font-size: 11px; font-weight: bold; color: #27AE60;"
        label_style_offline_rest = "font-size: 11px; font-weight: bold; color: #FFAA00;"
        label_style_no_data = "font-size: 11px; font-weight: bold; color: #FF4444;"

        current_indicator_text = f"🎯 {pair_display_name} - {current_time_str}"
        current_live_status_text = "N/A"
        current_live_status_style = label_style_no_data


        if websocket_active_for_pair:
            if live_trading_panel_active:
                current_live_status_text = "🟢 LIVE+WS"
                current_live_status_style = label_style_live_ws
            else:
                current_live_status_text = "📡 WS-READY"
                current_live_status_style = label_style_ws_ready
        else:
            if live_trading_panel_active:
                current_live_status_text = "🟠 LIVE (REST)"
                current_live_status_style = label_style_live_rest
            else:
                current_live_status_text = "⚪ OFFLINE (REST)"
                current_live_status_style = label_style_offline_rest


        if self.indicator_label:
            self.indicator_label.setText(current_indicator_text)
        if self.live_status_label:
            self.live_status_label.setText(current_live_status_text)
            self.live_status_label.setStyleSheet(current_live_status_style)

        data_source_for_title = "WS" if websocket_active_for_pair else "REST"
        if self.chart_widget:
            self.chart_widget.setTitle(f"{pair_display_name} - 1 Min Chart ({data_source_for_title}) - Loading...", color='gray')


        try:
            raw_ohlc_data = self.api.get_ohlc(pair_to_chart_altname, interval=1, limit=120)

            if not raw_ohlc_data or not isinstance(raw_ohlc_data, dict):
                logger.warning(f"[UI_CHART] No OHLC data received for {pair_to_chart_altname}")
                self._show_chart_error(f"No data for {pair_display_name}")
                return

            pair_data_key = pair_to_chart_altname
            if pair_to_chart_altname not in raw_ohlc_data and len(raw_ohlc_data) == 1:
                 pair_data_key = next(iter(raw_ohlc_data))
            elif pair_to_chart_altname not in raw_ohlc_data and 'last' in raw_ohlc_data :
                 pass # Kraken API néha 'last' kulcsot ad ahelyett, hogy a pár nevét adná vissza, ha csak egy pár van
            elif pair_to_chart_altname not in raw_ohlc_data:
                logger.warning(f"[UI_CHART] OHLC data received, but key '{pair_to_chart_altname}' not found. Keys: {list(raw_ohlc_data.keys())}")
                self._show_chart_error(f"Data key error for {pair_display_name}")
                return


            ohlc_values = raw_ohlc_data.get(pair_data_key)

            if not ohlc_values or not isinstance(ohlc_values, list) or len(ohlc_values) == 0:
                logger.warning(f"[UI_CHART] Empty or invalid OHLC values for {pair_display_name} (key: {pair_data_key})")
                self._show_chart_error(f"Empty data for {pair_display_name}")
                return

            try:
                df = pd.DataFrame(ohlc_values, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
                df = df.astype({"time": int, "open": float, "high": float, "low": float, "close": float})
                df['time_dt'] = pd.to_datetime(df['time'], unit='s')
                df_display = df
            except Exception as e_df:
                logger.error(f"[UI_CHART] DataFrame processing failed for {pair_display_name}: {e_df}", exc_info=True)
                self._show_chart_error(f"Data processing error for {pair_display_name}")
                return

            if self.chart_widget:
                self.chart_widget.clear()
                self.chart_widget.setTitle(f"{pair_display_name} - 1 Min Chart ({data_source_for_title}) - {current_time_str}", color='white')

                time_values_ts = df_display['time'].values
                close_array = df_display['close'].values

                self.chart_widget.plot(time_values_ts, close_array, pen=pg.mkPen(color='#00AAFF', width=2), name='Price')

                if len(df_display) >= 20:
                    try:
                        sma_20 = df_display['close'].rolling(20, min_periods=1).mean()
                        self.chart_widget.plot(time_values_ts, sma_20.values, pen=pg.mkPen(color='#FFDB58', width=1, style=Qt.DotLine), name='SMA20')

                        bb_std = df_display['close'].rolling(20, min_periods=1).std()
                        bb_upper = sma_20 + (bb_std * 2)
                        bb_lower = sma_20 - (bb_std * 2)

                        fill_upper_item = pg.PlotDataItem(time_values_ts, bb_upper.values, pen=pg.mkPen(color=(255,107,53,80)))
                        fill_lower_item = pg.PlotDataItem(time_values_ts, bb_lower.values, pen=pg.mkPen(color=(255,107,53,80)))
                        fill = pg.FillBetweenItem(fill_lower_item, fill_upper_item, brush=pg.mkBrush(255,107,53,30))
                        self.chart_widget.addItem(fill)

                        self.chart_widget.plot(time_values_ts, bb_upper.values, pen=pg.mkPen(color=(255,107,53,150), style=Qt.DashLine), name='BB Upper')
                        self.chart_widget.plot(time_values_ts, bb_lower.values, pen=pg.mkPen(color=(255,107,53,150), style=Qt.DashLine), name='BB Lower')
                    except Exception as e_ta:
                        logger.warning(f"[UI_CHART] Technical indicators calculation failed for {pair_display_name}: {e_ta}")

                self._format_chart_time_axis(df_display, time_values_ts)
                self._add_position_markers_to_chart(pair_to_chart_altname, time_values_ts)
                self._update_price_display_on_dashboard(pair_display_name, df_display, websocket_active_for_pair)


        except Exception as e:
            logger.error(f"[UI_CHART] Chart update failed for {pair_display_name}: {e}", exc_info=True)
            self._show_chart_error(f"Chart error for {pair_display_name}: {str(e)[:50]}")


    def _show_chart_error(self, error_message: str):
        """Hibaüzenet megjelenítése a diagramon."""
        if self.chart_widget:
            self.chart_widget.clear()
            self.chart_widget.setTitle(error_message, color='red', size='10pt')

    def _format_chart_time_axis(self, df_with_datetime: pd.DataFrame, time_values_timestamps: list):
        """Időtengely formázása a diagramon, hogy olvashatóbb legyen."""
        try:
            if self.chart_widget and not df_with_datetime.empty and len(time_values_timestamps) > 0:
                axis = self.chart_widget.getAxis('bottom')

                if len(df_with_datetime) < 2 or len(time_values_timestamps) < 2:
                    axis.setTicks(None)
                    return

                num_candles = len(df_with_datetime)
                tick_spacing = 1
                if num_candles > 100: tick_spacing = 30
                elif num_candles > 50: tick_spacing = 15
                elif num_candles > 20: tick_spacing = 5

                tick_indices = list(range(0, num_candles, tick_spacing))
                if num_candles -1 not in tick_indices and num_candles > 1 :
                    tick_indices.append(num_candles-1)

                tick_strings = [df_with_datetime['time_dt'].iloc[idx].strftime('%H:%M') for idx in tick_indices if idx < num_candles]
                tick_values_ts = [time_values_timestamps[idx] for idx in tick_indices if idx < len(time_values_timestamps)]

                if tick_values_ts and tick_strings:
                    ticks = [list(zip(tick_values_ts, tick_strings))]
                    axis.setTicks(ticks)
                else:
                    axis.setTicks(None)
        except Exception as e:
            logger.warning(f"Time axis formatting failed: {e}", exc_info=True)
            if hasattr(self, 'chart_widget') and self.chart_widget:
                 self.chart_widget.getAxis('bottom').setTicks(None)


    def _add_position_markers_to_chart(self, pair_altname: str, time_values_ts: list):
        """Pozíció nyitási és zárási pontok, valamint SL/TP szintek hozzáadása a diagramhoz."""
        try:
            if not self.chart_widget or not hasattr(self, 'position_manager') or not self.position_manager: return

            current_pos_data = self.position_manager.get_position(pair_altname)
            if current_pos_data and current_pos_data.get('entry_price'):
                entry_price = current_pos_data['entry_price']
                side = current_pos_data.get('side', 'buy')
                entry_time_unix = current_pos_data.get('entry_time_unix', current_pos_data.get('open_time'))

                marker_color = QColor(0, 255, 0, 180) if side == 'buy' else QColor(255, 0, 0, 180)
                sl = current_pos_data.get('stop_loss')
                tp = current_pos_data.get('take_profit')

                entry_line = pg.InfiniteLine(pos=entry_price, angle=0,
                                             pen=pg.mkPen(marker_color, width=2, style=Qt.SolidLine),
                                             label=f'Entry: {entry_price:.4f}',
                                             labelOpts={'color': marker_color, 'movable': True, 'position': 0.95})
                self.chart_widget.addItem(entry_line)

                if entry_time_unix and time_values_ts is not None and len(time_values_ts) > 0 and entry_time_unix >= time_values_ts[0] and entry_time_unix <= time_values_ts[-1]:
                    arrow_symbol = 's'
                    arrow_brush = pg.mkBrush(marker_color)
                    entry_scatter = pg.ScatterPlotItem(x=[entry_time_unix], y=[entry_price],
                                                       symbol=arrow_symbol, size=12, pen=pg.mkPen(None), brush=arrow_brush)
                    self.chart_widget.addItem(entry_scatter)

                if sl:
                    sl_line = pg.InfiniteLine(pos=sl, angle=0,
                                              pen=pg.mkPen(QColor(255,165,0,180), width=1, style=Qt.DashLine),
                                              label=f'SL: {sl:.4f}',
                                              labelOpts={'color': QColor(255,165,0), 'movable': True, 'position': 0.9})
                    self.chart_widget.addItem(sl_line)
                if tp:
                    tp_line = pg.InfiniteLine(pos=tp, angle=0,
                                              pen=pg.mkPen(QColor(60,179,113,180), width=1, style=Qt.DashLine),
                                              label=f'TP: {tp:.4f}',
                                              labelOpts={'color': QColor(60,179,113), 'movable': True, 'position': 0.85})
                    self.chart_widget.addItem(tp_line)
        except Exception as e:
            logger.warning(f"Position markers drawing failed for {pair_altname}: {e}", exc_info=True)

    def _update_price_display_on_dashboard(self, pair_display_name: str, df_display: pd.DataFrame, websocket_active_for_pair: bool):
        """Árfolyamadatok frissítése a dashboardon."""
        try:
            if not df_display.empty and self.dashboard:
                pair_altname_for_price = self.current_chart_pair
                current_price_ws = None
                if websocket_active_for_pair:
                    current_price_ws = self.get_current_price_for_pair(pair_altname_for_price)

                current_price_chart = df_display['close'].iloc[-1]
                display_price = current_price_ws if current_price_ws is not None and current_price_ws > 0 else current_price_chart

                self.dashboard.update_field("Current Price", f"${display_price:.6f}")
                data_source_text = "WebSocket Live" if websocket_active_for_pair and current_price_ws is not None else "REST API (Chart)"
                self.dashboard.update_field("Price Data Source", data_source_text)
        except Exception as e:
            logger.warning(f"Dashboard price display update failed for {pair_display_name}: {e}", exc_info=True)


    def enhanced_update_cycle(self):
        """Fő frissítési ciklus a GUI elemekhez és adatokhoz."""
        self.update_counter += 1
        try:
            self.update_chart()
            self.refresh_balance()
            self.refresh_open_trades()

            if self.update_counter % 2 == 0:
                self.update_ai_performance_display()
                if self.stats_panel and hasattr(self.stats_panel, 'update_stats'):
                    self.stats_panel.update_stats()

            if self.update_counter % 3 == 0:
                self.refresh_stats()
                if self.position_list_widget and hasattr(self.position_list_widget, 'update_history'):
                    self.position_list_widget.update_history()

            if self.live_trading_thread and self.live_trading_thread.isRunning():
                if self.min_volume_display_label and hasattr(self.live_trading_thread, 'min_volume_usd'):
                    self.min_volume_display_label.setText(f"Min Volume USD: {self.live_trading_thread.min_volume_usd:,.0f}")
                if self.min_score_display_label and hasattr(self.live_trading_thread, 'min_score_threshold'):
                    self.min_score_display_label.setText(f"Min Score Candidate: {self.live_trading_thread.min_score_threshold:.2f}")
                if self.pump_thresh_display_label and hasattr(self.live_trading_thread, 'pump_threshold'):
                    pump_val = self.live_trading_thread.pump_threshold
                    self.pump_thresh_display_label.setText(f"Pump Threshold %: {float(pump_val)*100 if isinstance(pump_val, (int,float)) else 'N/A'}")
        except Exception as e:
            logger.error(f"[ERROR] Enhanced update cycle failed: {e}", exc_info=True)


    def get_all_coin_data(self) -> list:
        """Coin adatok lekérdezése a scoring panel számára, volumen alapján szűrve."""
        try:
            if not (hasattr(self, 'api') and self.api and hasattr(self.api, 'get_market_data')):
                logger.warning("API client or get_market_data method not available. Using fallback coin data.")
                return self._get_fallback_coin_data()

            logger.info("📊 Getting real coin data with market data for scoring panel...")
            market_data_list = self.api.get_market_data()

            if not market_data_list:
                logger.warning("⚠️ No market data received from API. Using fallback coin data.")
                return self._get_fallback_coin_data()

            ui_coin_data = []
            for coin_info in market_data_list[:30]:
                if not all(k in coin_info for k in ['symbol', 'price', 'volume_usd']):
                    logger.warning(f"Skipping coin due to missing data: {coin_info.get('symbol', 'UNKNOWN')}")
                    continue

                coin_entry = {
                    'symbol': coin_info.get('symbol'),
                    'pair': coin_info.get('symbol'),
                    'close': coin_info.get('price'),
                    'volume_usd': coin_info.get('volume_usd'),
                    'score_from_scanner': coin_info.get('score'),
                    'rsi_15m': coin_info.get('rsi_15m', random.uniform(30, 70)),
                    'correl_btc': coin_info.get('correl_btc', random.uniform(0.5, 0.95)),
                }
                ui_coin_data.append({k: v for k, v in coin_entry.items() if v is not None})

            if ui_coin_data:
                logger.info(f"✅ Prepared real coin data for scoring panel: {len(ui_coin_data)} pairs.")
                return ui_coin_data
            else:
                logger.warning("⚠️ No valid coin data created from API response. Using fallback.")
                return self._get_fallback_coin_data()
        except Exception as e:
            logger.error(f"❌ get_all_coin_data for UI error: {e}", exc_info=True)
            return self._get_fallback_coin_data()


    def _get_fallback_coin_data(self) -> list:
        """Fallback coin adatok szolgáltatása, ha az API nem elérhető."""
        fallback_data = [
            {'symbol': 'XBTUSD', 'pair': 'XBTUSD', 'close': 60000, 'volume_usd': 50000000, 'score_from_scanner': 0.8, 'rsi_15m': 55.0, 'correl_btc': 1.0},
            {'symbol': 'ETHUSD', 'pair': 'ETHUSD', 'close': 3000, 'volume_usd': 25000000, 'score_from_scanner': 0.7, 'rsi_15m': 48.2, 'correl_btc': 0.85},
            {'symbol': 'ADAUSD', 'pair': 'ADAUSD', 'close': 0.45, 'volume_usd': 2500000, 'score_from_scanner': 0.6, 'rsi_15m': 38.5, 'correl_btc': 0.75},
            {'symbol': 'SOLUSD', 'pair': 'SOLUSD', 'close': 150.0, 'volume_usd': 8000000, 'score_from_scanner': 0.75, 'rsi_15m': 65.2, 'correl_btc': 0.80},
        ]
        logger.info(f"🔄 Using UI fallback data for {len(fallback_data)} pairs")
        return fallback_data

    def get_live_trading_settings(self) -> dict:
        """Aktuális live trading beállítások lekérdezése a LiveTradingPanel-ből."""
        if self.live_panel and hasattr(self.live_panel, 'get_current_settings'):
            settings = self.live_panel.get_current_settings()
            if settings and isinstance(settings, dict):
                return settings
            else:
                logger.warning("LiveTradingPanel.get_current_settings did not return a valid dict. Using default.")
        else:
            logger.warning("LiveTradingPanel not available or no get_current_settings method. Using default settings.")

        return {
            'auto_trading_enabled': False, 'position_size_usd': 50.0, 'max_active_trades': 1,
            'stop_loss_pct': 4.0, 'take_profit_target_pct': 0.8, 'min_score_for_auto_trade': 0.7,
            'scanner_min_volume_usd': 500000, 'scanner_min_score_candidate': 0.5,
            'scanner_pump_threshold_pct': 10.0, 'manual_position_size': 50.0,
            'manual_sl_pct': 4.0, 'manual_tp_pct': 0.8, 'scan_interval_sec': 180,
        }

    def initialize_live_trading_system(self):
        """Live trading rendszer kezdeti állapotának beállítása."""
        try:
            logger.info("🚀 Initializing Live Trading System UI state...")
            self.live_trading_active = False
            if self.ai_status_label:
                self.ai_status_label.setText(f"Live Trading: READY ({'Adv.' if self.ai_mode_active else 'Basic'})")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: #00AAFF;")
            if self.ai_toggle_btn:
                self.ai_toggle_btn.setText("🚀 Start Live Trading")
                self.ai_toggle_btn.setEnabled(True)
            logger.info("✅ Live Trading System UI initialized, ready to be started.")
        except Exception as e:
            logger.error(f"❌ Live trading system UI initialization failed: {e}", exc_info=True)
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: INIT ERROR")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: red;")

    def toggle_live_trading(self):
        """Live trading indítása/leállítása a UI gombról."""
        try:
            if not self.live_trading_active:
                self.start_live_trading_action()
            else:
                self.stop_live_trading_action()
        except Exception as e:
            logger.error(f"❌ Live trading toggle failed: {e}", exc_info=True)
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: TOGGLE ERROR")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: red;")


    def start_live_trading_action(self):
        """Live trading indítása és a kapcsolódó UI elemek frissítése."""
        if self.live_trading_active:
            logger.info("Live trading is already active.")
            return
        try:
            logger.info("🚀 Attempting to start Live Trading System...")
            if not self.live_trading_thread or not self.live_trading_thread.isRunning():
                self.live_trading_thread = LiveTradingThread(self)
                self.live_trading_thread.opportunity_found.connect(self.handle_live_opportunity)
                self.live_trading_thread.trade_executed.connect(self.handle_live_trade)
                self.live_trading_thread.ai_decision.connect(self.handle_ai_decision)
                self.live_trading_thread.error_occurred.connect(self.handle_thread_error)
                self.live_trading_thread.scan_progress.connect(self.handle_scan_progress)
                self.live_trading_thread.pump_detected.connect(self.handle_pump_detection)
            else:
                logger.info("LiveTradingThread already exists and might be running. Will try to use existing.")

            current_ui_settings = self.get_live_trading_settings()
            if self.live_trading_thread:
                if 'scanner_min_volume_usd' in current_ui_settings:
                    self.live_trading_thread.min_volume_usd = float(current_ui_settings['scanner_min_volume_usd'])
                if 'scanner_min_score_candidate' in current_ui_settings:
                    self.live_trading_thread.min_score_threshold = float(current_ui_settings['scanner_min_score_candidate'])
                if 'scanner_pump_threshold_pct' in current_ui_settings:
                    self.live_trading_thread.pump_threshold = float(current_ui_settings['scanner_pump_threshold_pct']) / 100.0
                if 'scan_interval_sec' in current_ui_settings:
                    self.live_trading_thread.scan_interval = int(current_ui_settings['scan_interval_sec'])

                if self.min_volume_display_label: self.min_volume_display_label.setText(f"Min Volume USD: {self.live_trading_thread.min_volume_usd:,.0f}")
                if self.min_score_display_label: self.min_score_display_label.setText(f"Min Score Candidate: {self.live_trading_thread.min_score_threshold:.2f}")
                if self.pump_thresh_display_label: self.pump_thresh_display_label.setText(f"Pump Threshold %: {self.live_trading_thread.pump_threshold*100:.1f}")

            self.live_trading_thread.start_trading()
            self.live_trading_active = True

            if self.ai_toggle_btn: self.ai_toggle_btn.setText("🛑 Stop Live Trading")
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: ACTIVE")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: #27AE60;")
            if self.live_panel and hasattr(self.live_panel, 'set_trading_active_status'):
                self.live_panel.set_trading_active_status(True)
            logger.info("✅ Live Trading System started with current settings.")
            self.ai_decision_log_append("📈 Live Trading System Started.")
        except Exception as e:
            logger.error(f"❌ Failed to start Live Trading System: {e}", exc_info=True)
            self.live_trading_active = False
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: START ERROR")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: red;")
            self.ai_decision_log_append(f"❌ Live Trading Start Error: {str(e)[:50]}")


    def stop_live_trading_action(self):
        """Live trading leállítása és a kapcsolódó UI elemek frissítése."""
        if not self.live_trading_active:
            logger.info("Live trading is not active.")
            return
        try:
            logger.info("🔴 Attempting to stop Live Trading System...")
            if self.live_trading_thread and self.live_trading_thread.isRunning():
                self.live_trading_thread.stop_trading()
            self.live_trading_active = False

            if self.ai_toggle_btn: self.ai_toggle_btn.setText("🚀 Start Live Trading")
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: STOPPED")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: #E74C3C;")
            if self.scan_progress_label: self.scan_progress_label.setText("Scan Progress: Idle")
            if self.live_panel and hasattr(self.live_panel, 'set_trading_active_status'):
                self.live_panel.set_trading_active_status(False)
            logger.info("✅ Live Trading System stopped.")
            self.ai_decision_log_append("📉 Live Trading System Stopped.")
        except Exception as e:
            logger.error(f"❌ Failed to stop Live Trading System: {e}", exc_info=True)
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: STOP ERROR")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: red;")
            self.ai_decision_log_append(f"❌ Live Trading Stop Error: {str(e)[:50]}")

    def emergency_stop_all(self):
        """Vészleállítás: minden aktivitás azonnali beszüntetése."""
        try:
            logger.critical("🚨 EMERGENCY STOP ACTIVATED!")
            if self.live_trading_active:
                self.stop_live_trading_action()

            if self.position_manager and hasattr(self.position_manager, 'close_all_positions_market'):
                closed_info = self.position_manager.close_all_positions_market()
                logger.info(f"🚨 Emergency close all positions: {closed_info}")
                self.ai_decision_log_append(f"🚨 EMERGENCY CLOSED POSITIONS: {len(closed_info)} closed.")
            elif self.position_manager and hasattr(self.position_manager, 'get_all_positions'):
                all_pos = self.position_manager.get_all_positions()
                for pair_alt in list(all_pos.keys()):
                    price = self._get_real_current_price(pair_alt)
                    if price:
                        self.position_manager.close_position(pair_alt, price, reason="EMERGENCY_STOP")
                        logger.info(f"🚨 Emergency closed {pair_alt} at {price}")
                        self.ai_decision_log_append(f"🚨 EMERGENCY CLOSED {pair_alt}")
                    else:
                        logger.warning(f"🚨 Could not get price for {pair_alt} during emergency stop.")
            self.refresh_open_trades()
            self.refresh_balance()

            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: EMERGENCY STOP")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: red; background-color: yellow;")
            if self.ai_decision_display:
                self.ai_decision_display.setText("🚨 EMERGENCY STOP - All systems halted!")
            if self.ai_toggle_btn:
                self.ai_toggle_btn.setEnabled(False)
            logger.info("🚨 Emergency stop procedure completed.")
            self.ai_decision_log_append("🚨 EMERGENCY STOP ACTIVATED - ALL SYSTEMS HALTED.")
        except Exception as e:
            logger.error(f"❌ Emergency stop procedure failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ EMERGENCY STOP FAILED: {str(e)[:50]}")


    def update_trading_settings(self, settings_dict: dict):
        """Kereskedési beállítások frissítése a LiveTradingPanel-ből érkező jelzés alapján."""
        try:
            logger.info(f"⚙️ MainWindow received settings update from UI: {settings_dict}")

            if self.live_trading_thread and self.live_trading_thread.isRunning():
                if 'scanner_min_volume_usd' in settings_dict and hasattr(self.live_trading_thread, 'min_volume_usd'):
                    self.live_trading_thread.min_volume_usd = float(settings_dict['scanner_min_volume_usd'])
                    logger.info(f"   LTT min_volume_usd updated to: {self.live_trading_thread.min_volume_usd}")
                if 'scanner_min_score_candidate' in settings_dict and hasattr(self.live_trading_thread, 'min_score_threshold'):
                    self.live_trading_thread.min_score_threshold = float(settings_dict['scanner_min_score_candidate'])
                    logger.info(f"   LTT min_score_threshold updated to: {self.live_trading_thread.min_score_threshold}")
                if 'scanner_pump_threshold_pct' in settings_dict and hasattr(self.live_trading_thread, 'pump_threshold'):
                    self.live_trading_thread.pump_threshold = float(settings_dict['scanner_pump_threshold_pct']) / 100.0
                    logger.info(f"   LTT pump_threshold updated to: {self.live_trading_thread.pump_threshold}")
                if 'scan_interval_sec' in settings_dict and hasattr(self.live_trading_thread, 'scan_interval'):
                    self.live_trading_thread.scan_interval = int(settings_dict['scan_interval_sec'])
                    logger.info(f"   LTT scan_interval updated to: {self.live_trading_thread.scan_interval}s")

            if self.min_volume_display_label:
                 min_vol_disp = settings_dict.get('scanner_min_volume_usd', (self.live_trading_thread.min_volume_usd if self.live_trading_thread and hasattr(self.live_trading_thread, 'min_volume_usd') else "N/A"))
                 self.min_volume_display_label.setText(f"Min Volume USD: {min_vol_disp:,.0f}" if isinstance(min_vol_disp, (int,float)) else f"Min Volume USD: {min_vol_disp}")
            if self.min_score_display_label:
                 min_score_disp = settings_dict.get('scanner_min_score_candidate', (self.live_trading_thread.min_score_threshold if self.live_trading_thread and hasattr(self.live_trading_thread, 'min_score_threshold') else "N/A"))
                 self.min_score_display_label.setText(f"Min Score Candidate: {min_score_disp:.2f}" if isinstance(min_score_disp, float) else f"Min Score Candidate: {min_score_disp}")
            if self.pump_thresh_display_label:
                 pump_thresh_val_from_settings = settings_dict.get('scanner_pump_threshold_pct')
                 if pump_thresh_val_from_settings is not None:
                     pump_thresh_disp = pump_thresh_val_from_settings
                 elif self.live_trading_thread and hasattr(self.live_trading_thread, 'pump_threshold'):
                     pump_thresh_disp = float(self.live_trading_thread.pump_threshold) * 100
                 else:
                     pump_thresh_disp = "N/A"
                 self.pump_thresh_display_label.setText(f"Pump Threshold %: {pump_thresh_disp:.1f}" if isinstance(pump_thresh_disp, float) else f"Pump Threshold %: {pump_thresh_disp}")

            self.ai_decision_log_append(f"⚙️ Settings Updated: {list(settings_dict.keys())}")
            logger.info("Trading settings updated successfully.")
        except Exception as e:
            logger.error(f"❌ Error updating trading settings: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ Settings Update Error: {str(e)[:50]}")

    def force_live_scan(self):
        """Azonnali piaci újra-szkennelés kérése a LiveTradingThread-től."""
        try:
            logger.info("🔄 Force rescan initiated by user...")
            if self.ai_decision_display: self.ai_decision_display.setText("🔄 Forcing market rescan by Live Scanner...")

            if self.live_trading_thread and self.live_trading_active and self.live_trading_thread.isRunning():
                if hasattr(self.live_trading_thread, 'force_scan_now') and callable(self.live_trading_thread.force_scan_now):
                    self.live_trading_thread.force_scan_now()
                    logger.info("🔄 Force scan signal sent to LiveTradingThread.")
                else:
                    self.live_trading_thread.scan_progress.emit("Force scan requested (manual trigger).")
                    logger.info("🔄 Force scan requested. LiveTradingThread will execute on its next cycle.")
                self.ai_decision_log_append("🔄 Force Rescan Requested.")
            elif not self.live_trading_active:
                 if self.ai_decision_display: self.ai_decision_display.setText("🔄 Start Live Trading to enable scans.")
                 logger.info("🔄 Live trading not active, cannot force scan.")
                 self.ai_decision_log_append("🔄 Cannot Force Scan: Live Trading Inactive.")
            else:
                if self.ai_decision_display: self.ai_decision_display.setText("🔄 Live Scanner not ready for forced scan.")
                logger.warning("🔄 Live trading thread not available or not running for forced scan.")
                self.ai_decision_log_append("🔄 Cannot Force Scan: Thread Not Ready.")
        except Exception as e:
            logger.error(f"❌ Force rescan failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ Force Rescan Error: {str(e)[:50]}")


    def handle_live_opportunity(self, opportunity_data: dict):
        """A LiveTradingThread által talált kereskedési lehetőség kezelése."""
        try:
            pair = opportunity_data.get('pair', 'Unknown')
            score = opportunity_data.get('score', 0.0)
            reason = opportunity_data.get('reason', 'N/A')
            logger.info(f"🚀 Live Opportunity Received by MainWindow: {pair} (Score: {score:.3f}) Reason: {reason}")

            if self.ai_decision_display:
                self.ai_decision_display.setText(f"🎯 Candidate: {pair} Score: {score:.2f}")

            if self.live_trading_thread and hasattr(self.live_trading_thread, 'final_candidates') and self.live_trading_thread.final_candidates:
                opp_text = f"🏆 TOP SCAN CANDIDATES - {time.strftime('%H:%M:%S')} 🏆\n"
                opp_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                for i, cand in enumerate(self.live_trading_thread.final_candidates[:3]):
                    pump_ind = "🔥" if cand.get('pump_detected') else "📊"
                    opp_text += (f"{i+1}. {pump_ind} {cand.get('altname','N/A')}: Score {cand.get('final_score',0):.3f}, "
                                 f"Vol ${cand.get('volume_usd',0):,.0f}\n")
                if self.ai_opportunities_display: self.ai_opportunities_display.setPlainText(opp_text)
            elif self.ai_opportunities_display:
                self.ai_opportunities_display.setPlainText("Waiting for next scan results from LiveTradingThread...")

            self.ai_decision_log_append(f"🎯 Scan Found: {pair} Score: {score:.3f} ({reason[:30]})")

            current_settings = self.get_live_trading_settings()
            if current_settings.get('auto_trading_enabled', False) and \
               score >= current_settings.get('min_score_for_auto_trade', 0.7):
                logger.info(f"🤖 Auto-trading criteria met for {pair}. Proceeding with execution logic.")
                self.execute_ai_opportunity_from_live_thread(opportunity_data)
            else:
                logger.info(f"📉 Auto-trading criteria NOT met for {pair} (Score: {score:.3f} vs Threshold: {current_settings.get('min_score_for_auto_trade', 0.7)}) or auto-trade disabled.")
                self.ai_decision_log_append(f"📉 Skipped {pair}: Score {score:.3f} (Thr: {current_settings.get('min_score_for_auto_trade', 0.7)})")
        except Exception as e:
            logger.error(f"❌ Live opportunity handling in MainWindow failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ Opp. Handle Error: {str(e)[:40]}")

    def execute_ai_opportunity_from_live_thread(self, trade_setup_data: dict):
        """Kereskedés végrehajtása a LiveTradingThread-től kapott adatok alapján."""
        try:
            pair_altname = trade_setup_data.get('pair')
            if not pair_altname:
                logger.error("❌ Trade execution failed: Pair (altname) information missing from trade_setup_data.")
                self.ai_decision_log_append("❌ Exec Fail: No Pair Info")
                return

            logger.info(f"🤖 MainWindow attempting to execute opportunity from LiveThread: {pair_altname}")

            current_settings = self.get_live_trading_settings()
            max_active_trades = current_settings.get('max_active_trades', 1)

            if len(self.position_manager.get_all_positions()) >= max_active_trades:
                logger.warning(f"⚠️ Max active positions ({max_active_trades}) reached. Cannot execute new trade for {pair_altname}.")
                self.ai_decision_log_append(f"⚠️ Max Pos Skip: {pair_altname}")
                return

            signal_value = trade_setup_data.get('signal', 'buy')
            side = str(signal_value).lower()
            if side not in ['buy', 'sell']:
                logger.error(f"❌ Invalid trade side '{side}' for {pair_altname}. Skipping.")
                self.ai_decision_log_append(f"❌ Invalid Side: {pair_altname} ({side})")
                return

            entry_price = trade_setup_data.get('entry_price')
            volume = trade_setup_data.get('volume')
            stop_loss = trade_setup_data.get('stop_loss')
            take_profit = trade_setup_data.get('take_profit')
            position_size_usd = trade_setup_data.get('position_size_usd', current_settings.get('position_size_usd', 50.0))

            if entry_price is None or entry_price <= 0:
                logger.error(f"❌ Invalid or missing entry price for {pair_altname}. Cannot execute.")
                self.ai_decision_log_append(f"❌ Exec Fail: No Price {pair_altname}")
                return

            if volume is None or volume <= 0:
                if position_size_usd > 0 and entry_price > 0:
                    volume = position_size_usd / entry_price
                else:
                    logger.error(f"❌ Invalid or missing volume/position_size_usd for {pair_altname}. Cannot execute.")
                    self.ai_decision_log_append(f"❌ Exec Fail: No Volume {pair_altname}")
                    return

            if stop_loss is None:
                sl_pct = current_settings.get('stop_loss_pct', 4.0) / 100.0
                stop_loss = entry_price * (1 - sl_pct) if side == 'buy' else entry_price * (1 + sl_pct)
            if take_profit is None:
                tp_pct = current_settings.get('take_profit_target_pct', 0.8) / 100.0
                take_profit = entry_price * (1 + tp_pct) if side == 'buy' else entry_price * (1 - tp_pct)

            success = self.position_manager.open_position(
                pair=pair_altname, side=side, entry_price=entry_price, volume=volume,
                stop_loss=stop_loss, take_profit=take_profit, entry_time_unix=time.time()
            )

            if success:
                reason = f"AI_LIVE_SCAN_SCORE_{trade_setup_data.get('total_score', 0.0):.3f}"
                self.trade_logger.log(
                    pair=pair_altname, side=side, entry_price=entry_price,
                    exit_price=None, volume=volume, pnl=0.0, reason=reason
                )
                self.ai_performance['total_decisions'] += 1
                if self.ai_decision_display:
                    self.ai_decision_display.setText(f"✅ EXECUTED: {side.upper()} {pair_altname} ${position_size_usd:.0f}")
                self.ai_decision_log_append(f"✅ EXECUTED: {side.upper()} {pair_altname} Score: {trade_setup_data.get('total_score',0.0):.3f}")
                self.update_ai_performance_display()
                self.refresh_open_trades()
                self.refresh_balance()
                logger.info(f"✅ AI Live Scan trade executed: {side.upper()} {pair_altname} for ${position_size_usd:.0f}")
            else:
                logger.error(f"❌ AI Live Scan trade execution failed (PositionManager) for: {pair_altname}")
                if self.ai_decision_display: self.ai_decision_display.setText(f"❌ EXECUTION FAILED: {pair_altname}")
                self.ai_decision_log_append(f"❌ EXEC FAIL (PM): {pair_altname}")
        except Exception as e:
            logger.error(f"❌ MainWindow execution of AI opportunity failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ EXEC ERROR: {str(e)[:30]}")


    def handle_live_trade(self, trade_data: dict):
        """A LiveTradingThread által végrehajtott vagy frissített kereskedés kezelése."""
        try:
            logger.info(f"📈 Trade Update Received by MainWindow: {trade_data}")
            pair = trade_data.get('pair', 'N/A')
            action = trade_data.get('action', 'N/A')
            status = trade_data.get('status', 'updated')
            pnl = trade_data.get('pnl', 0.0)
            self.last_trade_pnl = pnl

            if status == 'closed':
                if pnl > 0:
                    self.ai_performance['successful_trades'] += 1
                self.ai_performance['ai_profit'] += pnl

            self.update_ai_performance_display()
            self.refresh_open_trades()
            self.refresh_balance()
            log_entry = f"TRADE EVENT: {pair} - {action} - Status: {status}, P&L: ${pnl:.2f}"
            self.ai_decision_log_append(log_entry)
            if self.ai_decision_label:
                self.ai_decision_label.setText(f"Last Trade: {pair} {action} ${pnl:.2f}")
        except Exception as e:
            logger.error(f"❌ Live trade handling in MainWindow failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ Trade Handle Error: {str(e)[:30]}")

    def handle_ai_decision(self, decision_data: dict):
        """A LiveTradingThread által küldött általános AI döntési esemény kezelése."""
        try:
            self.ai_decisions.append(decision_data)
            if len(self.ai_decisions) > 100:
                self.ai_decisions = self.ai_decisions[-100:]
            decision_text = decision_data.get('decision', 'No decision text')
            pair = decision_data.get('pair', 'N/A')
            score = decision_data.get('score', 0.0)
            if self.ai_decision_label:
                self.ai_decision_label.setText(f"Last Event: {decision_text[:35]}")
            log_entry = f"SCANNER EVENT: {decision_text} (Pair: {pair}, Score: {score:.2f})"
            self.ai_decision_log_append(log_entry)
        except Exception as e:
            logger.error(f"❌ AI decision handling in MainWindow failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ AI Dec. Handle Error: {str(e)[:30]}")


    def handle_thread_error(self, error_message: str):
        """A LiveTradingThread által küldött hibaüzenet kezelése."""
        logger.error(f"❌ LiveTradingThread Error Reported to MainWindow: {error_message}")
        if self.ai_status_label:
            self.ai_status_label.setText("Live Trading: THREAD ERROR")
            self.ai_status_label.setStyleSheet("font-weight: bold; color: red;")
        if self.ai_decision_display:
            self.ai_decision_display.setText(f"❌ Scanner Error: {error_message[:100]}")
        self.ai_decision_log_append(f"THREAD ERROR: {error_message}")


    def handle_scan_progress(self, progress_message: str):
        """A LiveTradingThread által küldött szkennelési folyamat üzenetének kezelése."""
        if self.scan_progress_label:
            self.scan_progress_label.setText(f"Scan: {progress_message[:40]}")
        if "✅ Scan completed" in progress_message or "⚠️ Scan took" in progress_message or "Forcing" in progress_message or "✅ AI Scan completed" in progress_message or "✅ Basic scan completed" in progress_message:
             if self.ai_decision_display: self.ai_decision_display.setText(progress_message)

    def handle_pump_detection(self, pump_data: dict):
        """A LiveTradingThread által észlelt pump esemény kezelése."""
        try:
            pair = pump_data.get('pair', 'Unknown')
            pump_pct = pump_data.get('pump_pct', 0.0)
            volume_usd = pump_data.get('volume_usd', 0)
            logger.info(f"🔥 PUMP DETECTED (Signal to MainWindow): {pair} {pump_pct:.2f}% Volume: ${volume_usd:,.0f}")
            if self.ai_decision_display:
                self.ai_decision_display.setText(f"🔥 PUMP ALERT: {pair} {pump_pct:.1f}%")
            log_entry = f"PUMP ALERT: {pair} {pump_pct:.2f}% (Vol: ${volume_usd:,.0f})"
            self.ai_decision_log_append(log_entry)
        except Exception as e:
            logger.error(f"❌ Pump detection handling in MainWindow failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ Pump Handle Error: {str(e)[:30]}")

    def monitor_positions_with_ai(self):
        """Nyitott pozíciók figyelése és kezelése (SL/TP, esetleg AI alapú zárás)."""
        try:
            if not (hasattr(self, 'position_manager') and self.position_manager and hasattr(self.position_manager, 'get_all_positions')):
                return
            open_positions = self.position_manager.get_all_positions()
            if not open_positions: return

            for pair_altname, position_data in list(open_positions.items()):
                current_price = self._get_real_current_price(pair_altname, position_data.get('entry_price'))
                if not current_price:
                    logger.warning(f"⚠️ Cannot monitor position for {pair_altname}, current price unavailable.")
                    continue
                self.monitor_single_position_standard_sl_tp(pair_altname, position_data, current_price)
        except Exception as e:
            logger.error(f"❌ Position monitoring cycle failed: {e}", exc_info=True)

    def monitor_single_position_standard_sl_tp(self, pair_altname: str, position_data: dict, current_price: float):
        """Egyetlen pozíció figyelése standard SL/TP alapján."""
        try:
            entry_price = position_data.get('entry_price')
            stop_loss = position_data.get('stop_loss')
            take_profit = position_data.get('take_profit')
            side = position_data.get('side', 'buy').lower()
            if entry_price is None: return

            should_close = False
            close_reason = ""
            if side == 'buy':
                if stop_loss and current_price <= stop_loss: should_close = True; close_reason = "STOP_LOSS_HIT"
                elif take_profit and current_price >= take_profit: should_close = True; close_reason = "TAKE_PROFIT_HIT"
            elif side == 'sell':
                if stop_loss and current_price >= stop_loss: should_close = True; close_reason = "STOP_LOSS_HIT"
                elif take_profit and current_price <= take_profit: should_close = True; close_reason = "TAKE_PROFIT_HIT"

            if should_close:
                logger.info(f"🛡️ Standard monitor triggered close for {pair_altname}: {close_reason} at price {current_price:.4f}")
                self.close_position_programmatically(pair_altname, current_price, close_reason)
        except Exception as e:
            logger.error(f"❌ Standard SL/TP monitoring failed for {pair_altname}: {e}", exc_info=True)


    def close_position_programmatically(self, pair_altname: str, exit_price: float, reason: str):
        """Pozíció zárása programatikusan (pl. SL/TP, AI döntés alapján)."""
        try:
            if not hasattr(self, 'position_manager') or not self.position_manager:
                logger.error("Cannot close position: self.position_manager is not initialized.")
                return

            position_data_before_close = self.position_manager.get_position(pair_altname)
            if not position_data_before_close:
                logger.warning(f"⚠️ Attempted to close non-existent or already closed position: {pair_altname}")
                return

            closed_position_info = self.position_manager.close_position(pair_altname, exit_price, reason=reason)

            if closed_position_info:
                pnl = closed_position_info.get('pnl', 0.0)
                self.last_trade_pnl = pnl
                side = closed_position_info.get('side', position_data_before_close.get('side', 'N/A'))
                entry_price_log = closed_position_info.get('entry_price', position_data_before_close.get('entry_price', 0.0))
                volume_log = closed_position_info.get('volume', position_data_before_close.get('volume', 0.0))

                self.trade_logger.log(
                    pair=pair_altname, side=side, entry_price=entry_price_log,
                    exit_price=exit_price, volume=volume_log, pnl=pnl, reason=reason
                )
                if closed_position_info.get('status', 'closed') == 'closed':
                    if pnl > 0: self.ai_performance['successful_trades'] += 1
                    self.ai_performance['ai_profit'] += pnl

                if self.ai_decision_display:
                    self.ai_decision_display.setText(f"🚪 CLOSED: {pair_altname} P&L: ${pnl:.2f} ({reason})")
                self.ai_decision_log_append(f"🚪 CLOSED: {pair_altname} P&L ${pnl:.2f} ({reason})")
                self.update_ai_performance_display()
                self.refresh_open_trades()
                self.refresh_balance()
                logger.info(f"🚪 Position for {pair_altname} closed by logic '{reason}'. P&L: ${pnl:.2f}")
            else:
                logger.warning(f"⚠️ Failed to close position for {pair_altname} via PositionManager (returned None or False).")
                self.ai_decision_log_append(f"⚠️ Close Fail (PM): {pair_altname} ({reason})")
        except Exception as e:
            logger.error(f"❌ Programmatic position close for {pair_altname} failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ Close Error: {pair_altname} {str(e)[:30]}")

    def execute_manual_trade(self):
        """Manuális kereskedés végrehajtása a kiválasztott páron."""
        try:
            if not self.pair_list or not self.pair_list.currentItem():
                logger.warning("⚠️ No pair selected for manual trade.")
                self.ai_decision_log_append("Manual Trade: No pair selected.")
                return

            pair_display_name = self.pair_list.currentItem().text()
            pair_altname = pair_display_name.replace("/", "")
            current_price = self._get_real_current_price(pair_altname)

            if not current_price or current_price <= 0:
                logger.error(f"❌ Could not get valid price for manual trade on {pair_altname}.")
                self.ai_decision_log_append(f"Manual Trade: No valid price for {pair_altname}.")
                return

            trade_settings = self.get_live_trading_settings()
            position_size_usd = trade_settings.get('manual_position_size', 50.0)
            stop_loss_pct = trade_settings.get('manual_sl_pct', 4.0) / 100.0
            take_profit_pct = trade_settings.get('manual_tp_pct', 0.8) / 100.0
            side = "buy"
            volume = position_size_usd / current_price
            stop_loss = current_price * (1 - stop_loss_pct) if side == 'buy' else current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct) if side == 'buy' else current_price * (1 - take_profit_pct)

            if not hasattr(self, 'position_manager') or not self.position_manager:
                logger.error("Cannot execute manual trade: self.position_manager is not initialized.")
                self.ai_decision_log_append("Manual Trade: PositionManager missing.")
                return

            success = self.position_manager.open_position(
                pair=pair_altname, side=side, entry_price=current_price,
                volume=volume, stop_loss=stop_loss, take_profit=take_profit,
                entry_time_unix=time.time()
            )

            if success:
                self.trade_logger.log(
                    pair=pair_altname, side=side, entry_price=current_price,
                    exit_price=None, volume=volume, pnl=0.0, reason="MANUAL_TRADE"
                )
                logger.info(f"✅ Manual trade executed: {side.upper()} {pair_display_name} @ ${current_price:.4f}")
                self.ai_decision_log_append(f"👤 MANUAL TRADE: {side.upper()} {pair_display_name} @ ${current_price:.4f}")
                self.refresh_open_trades()
                self.refresh_balance()
                self.ai_performance['total_decisions'] +=1
                self.update_ai_performance_display()
            else:
                logger.error(f"❌ Manual trade failed for: {pair_display_name} (PositionManager denied).")
                self.ai_decision_log_append(f"Manual Trade Fail (PM): {pair_display_name}")
        except Exception as e:
            logger.error(f"❌ Manual trade execution failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"Manual Trade Error: {str(e)[:30]}")

    def open_test_position(self):
        """Teszt pozíció nyitása $50 értékben a kiválasztott páron."""
        try:
            if not self.pair_list or not self.pair_list.currentItem():
                logger.warning("⚠️ No pair selected for test position.")
                self.ai_decision_log_append("Test Pos: No pair selected.")
                return

            pair_display_name = self.pair_list.currentItem().text()
            pair_altname = pair_display_name.replace("/", "")
            logger.info(f"🧪 Opening test position for {pair_display_name} (alt: {pair_altname})")
            current_price = self._get_real_current_price(pair_altname)

            if not current_price or current_price <= 0:
                logger.error(f"❌ Could not get valid price for test position on {pair_altname}.")
                self.ai_decision_log_append(f"Test Pos: No price for {pair_altname}.")
                return

            position_size_usd = 50.0
            volume = position_size_usd / current_price
            sl_pct = 0.04; tp_pct = 0.008; side = "buy"
            stop_loss = current_price * (1 - sl_pct)
            take_profit = current_price * (1 + tp_pct)

            if not hasattr(self, 'position_manager') or not self.position_manager:
                logger.error("Cannot open test position: self.position_manager is not initialized.")
                self.ai_decision_log_append("Test Pos: PositionManager missing.")
                return

            success = self.position_manager.open_position(
                pair=pair_altname, side=side, entry_price=current_price,
                volume=volume, stop_loss=stop_loss, take_profit=take_profit,
                entry_time_unix=time.time()
            )

            if success:
                self.trade_logger.log(
                    pair=pair_altname, side=side, entry_price=current_price,
                    exit_price=None, volume=volume, pnl=0.0, reason="TEST_POSITION"
                )
                logger.info(f"✅ Test position opened: {side.upper()} {pair_display_name} ${position_size_usd:.0f} @ {current_price:.6f}")
                self.ai_decision_log_append(f"🧪 TEST POS OPENED: {side.upper()} {pair_display_name}")
                self.refresh_open_trades()
                self.refresh_balance()
                if self.dashboard: self.dashboard.update_field("Last Action", f"TEST BUY {pair_display_name}")
                self.ai_performance['total_decisions'] +=1
                self.update_ai_performance_display()
            else:
                logger.error(f"❌ Failed to open test position for {pair_display_name} (PositionManager denied).")
                self.ai_decision_log_append(f"Test Pos Fail (PM): {pair_display_name}")
        except Exception as e:
            logger.error(f"Test position opening failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"Test Pos Error: {str(e)[:30]}")


    def manual_close_current_position(self):
        """A listában kiválasztott aktuális nyitott pozíció manuális zárása."""
        try:
            if not self.pair_list or not self.pair_list.currentItem():
                logger.warning("⚠️ No pair selected for manual close.")
                self.ai_decision_log_append("Manual Close: No pair selected.")
                return

            pair_display_name = self.pair_list.currentItem().text()
            pair_altname = pair_display_name.replace("/", "")

            if not hasattr(self, 'position_manager') or not self.position_manager:
                logger.error("Cannot manually close: self.position_manager is not initialized.")
                self.ai_decision_log_append("Manual Close: PositionManager missing.")
                return

            position_data = self.position_manager.get_position(pair_altname)
            if not position_data:
                logger.warning(f"⚠️ No open position found for {pair_display_name} (alt: {pair_altname}) to close manually.")
                self.ai_decision_log_append(f"Manual Close: No open pos for {pair_display_name}.")
                return

            current_price = self._get_real_current_price(pair_altname, position_data.get('entry_price'))
            if not current_price or current_price <= 0:
                logger.error(f"❌ Could not get valid price for manually closing {pair_altname}.")
                self.ai_decision_log_append(f"Manual Close: No price for {pair_altname}.")
                return

            logger.info(f"🚪 Attempting to manually close position for {pair_display_name} at price {current_price:.4f}")
            self.close_position_programmatically(pair_altname, current_price, "MANUAL_CLOSE_SELECTED")
            if self.dashboard: self.dashboard.update_field("Last Action", f"MANUAL CLOSE {pair_display_name}")
        except Exception as e:
            logger.error(f"Manual close selected position failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"Manual Close Error: {str(e)[:30]}")


    def manual_close_all_positions(self):
        """Minden nyitott pozíció manuális zárása."""
        try:
            logger.info("🚪 Attempting to manually close ALL open positions...")
            if not (hasattr(self, 'position_manager') and self.position_manager and hasattr(self.position_manager, 'get_all_positions')):
                logger.warning("Position manager not available, cannot close all positions.")
                self.ai_decision_log_append("Close All: No Pos. Manager.")
                return

            open_positions = self.position_manager.get_all_positions()
            if not open_positions:
                logger.info("ℹ️ No open positions to close.")
                self.ai_decision_log_append("Close All: No open positions.")
                return

            closed_count = 0
            for pair_altname in list(open_positions.keys()):
                current_price = self._get_real_current_price(pair_altname, open_positions[pair_altname].get('entry_price'))
                if current_price and current_price > 0:
                    self.close_position_programmatically(pair_altname, current_price, "MANUAL_CLOSE_ALL")
                    closed_count += 1
                else:
                    logger.error(f"❌ Could not get price for {pair_altname} during CLOSE ALL. Skipping.")
                    self.ai_decision_log_append(f"Close All: No price for {pair_altname}.")

            logger.info(f"📴 Manual CLOSE ALL action: Attempted to close {len(open_positions)} positions. Successfully initiated close for: {closed_count}.")
            self.ai_decision_log_append(f"🚪 CLOSE ALL: {closed_count} positions closed.")
            if self.dashboard: self.dashboard.update_field("Last Action", f"CLOSED ALL ({closed_count})")
        except Exception as e:
            logger.error(f"Manual close all positions failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"Close All Error: {str(e)[:30]}")

    def update_ai_performance_display(self):
        """AI kereskedési teljesítményének frissítése a GUI-n."""
        try:
            if not hasattr(self, 'ai_performance_labels') or not self.ai_performance_labels: return

            total_decisions = self.ai_performance.get('total_decisions', 0)
            successful_trades = self.ai_performance.get('successful_trades', 0)
            total_pnl = self.ai_performance.get('ai_profit', 0.0)
            accuracy = (successful_trades / total_decisions * 100) if total_decisions > 0 else 0.0
            self.ai_performance['ai_accuracy'] = accuracy

            if 'Bot Accuracy' in self.ai_performance_labels: self.ai_performance_labels['Bot Accuracy'].setText(f"{accuracy:.1f}%")
            if 'Total Trades' in self.ai_performance_labels: self.ai_performance_labels['Total Trades'].setText(str(total_decisions))
            if 'Successful Trades' in self.ai_performance_labels: self.ai_performance_labels['Successful Trades'].setText(str(successful_trades))
            if 'Total P&L' in self.ai_performance_labels: self.ai_performance_labels['Total P&L'].setText(f"${total_pnl:.2f}")
            if 'Last Trade P&L' in self.ai_performance_labels: self.ai_performance_labels['Last Trade P&L'].setText(f"${getattr(self, 'last_trade_pnl', 0.0):.2f}")

            if self.ai_stats_label:
                self.ai_stats_label.setText(
                    f"Overall AI Stats: {successful_trades}/{total_decisions} wins | "
                    f"{accuracy:.1f}% acc | ${total_pnl:.2f} P&L"
                )
        except Exception as e:
            logger.error(f"❌ AI Performance display update failed: {e}", exc_info=True)

    def ai_decision_log_append(self, message: str):
        """Üzenet hozzáfűzése az AI döntési naplóhoz a GUI-n, időbélyeggel."""
        if self.ai_decision_log:
            timestamp = time.strftime('%H:%M:%S')
            self.ai_decision_log.append(f"[{timestamp}] {message}")
            self.limit_ai_log_lines()

    def limit_ai_log_lines(self, max_lines=100):
        """Az AI döntési napló sorainak limitálása a GUI-n."""
        try:
            if not (hasattr(self, 'ai_decision_log') and self.ai_decision_log): return
            current_text = self.ai_decision_log.toPlainText()
            lines = current_text.split('\n')
            if len(lines) > max_lines:
                limited_text = '\n'.join(lines[-max_lines:])
                self.ai_decision_log.setPlainText(limited_text)
                cursor = self.ai_decision_log.textCursor()
                cursor.movePosition(cursor.End)
                self.ai_decision_log.setTextCursor(cursor)
        except Exception as e:
            logger.error(f"❌ AI Log limit failed: {e}")


    def run_advanced_trading_cycle(self):
        """Fejlett AI kereskedési ciklus futtatása (ha az AI mód aktív)."""
        try:
            if not self.ai_mode_active:
                logger.info("Advanced trading cycle skipped: AI mode not active.")
                return
            logger.info("🤖 Running advanced AI trading cycle (called from MainWindow context)...")
            opportunities = []
            if hasattr(self, 'ai_scanner') and self.ai_scanner and hasattr(self.ai_scanner, 'scan_top_opportunities'):
                opportunities = self.ai_scanner.scan_top_opportunities(max_pairs=10)
                logger.info(f"Advanced Scan (MainWindow): Found {len(opportunities)} potential opportunities.")
                self.ai_decision_log_append(f"Adv. Scan (MW): {len(opportunities)} opps.")

            for opp_analysis_obj in opportunities:
                if not isinstance(opp_analysis_obj, CoinAnalysis):
                    logger.warning(f"Skipping invalid opportunity object: {type(opp_analysis_obj)}")
                    continue
                opp_pair = getattr(opp_analysis_obj, 'pair', 'UnknownPair')
                logger.info(f"Analyzing opportunity (MainWindow): {opp_pair}")
                ml_features_instance = self._create_ml_features_from_analysis(opp_analysis_obj)
                if not ml_features_instance:
                    logger.warning(f"Could not create ML features for {opp_pair} (MainWindow). Skipping.")
                    continue
                prediction_obj = None
                if hasattr(self, 'ml_engine') and self.ml_engine and hasattr(self.ml_engine, 'predict_trade_success'):
                    prediction_obj = self.ml_engine.predict_trade_success(ml_features_instance)
                    prob = getattr(prediction_obj, 'probability', 0.0)
                    conf = getattr(prediction_obj, 'confidence', 0.0)
                    logger.info(f"  ML Prediction for {opp_pair} (MainWindow): Prob={prob:.3f}, Conf={conf:.3f}")
                    self.ai_decision_log_append(f"ML (MW) {opp_pair}: P={prob:.2f} C={conf:.2f}")
                else:
                    logger.debug(f"ML Engine not available for {opp_pair} (MainWindow)")
                trade_setup_obj = None
                if trade_setup_obj:
                    logger.info(f"Opportunity for {opp_pair} processed (MainWindow). Further execution logic would follow.")
            logger.info("🤖 Advanced AI trading cycle (MainWindow context) finished.")
        except Exception as e:
            logger.error(f"Advanced AI trading cycle (MainWindow context) failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ Adv. Cycle (MW) Error: {str(e)[:30]}")


    def _create_ml_features(self, opportunity_analysis: CoinAnalysis) -> Optional[MLFeatures]:
        logger.warning("_create_ml_features (MainWindow) called, redirecting to _create_ml_features_from_analysis. This might indicate a structural issue.")
        return self._create_ml_features_from_analysis(opportunity_analysis)


    def closeEvent(self, event):
        """Alkalmazás bezárásakor lefutó tisztítási folyamatok."""
        try:
            logger.info("🔄 Shutting down AI trading bot...")
            if self.live_trading_active and self.live_trading_thread and self.live_trading_thread.isRunning():
                logger.info("Stopping LiveTradingThread...")
                self.live_trading_thread.stop_trading()
                if not self.live_trading_thread.wait(5000):
                    logger.warning("⚠️ LiveTradingThread did not stop gracefully, attempting to terminate.")
                    self.live_trading_thread.terminate()
                    self.live_trading_thread.wait()
                else: logger.info("✅ LiveTradingThread stopped.")
            elif self.live_trading_thread and not self.live_trading_thread.isRunning():
                 logger.info("LiveTradingThread was already stopped or not started.")

            if hasattr(self.api, 'cleanup') and callable(self.api.cleanup):
                logger.info("Cleaning up API client...")
                self.api.cleanup()
                logger.info("✅ API client cleaned up.")

            timers_to_stop = ['timer', 'position_timer']
            for timer_name in timers_to_stop:
                if hasattr(self, timer_name):
                    timer_instance = getattr(self, timer_name)
                    if timer_instance and timer_instance.isActive():
                        timer_instance.stop()
                        logger.info(f"Timer '{timer_name}' stopped.")

            if hasattr(self, 'ai_performance'):
                try:
                    import json
                    os.makedirs('logs', exist_ok=True)
                    perf_file = os.path.join('logs', 'bot_session_performance.json')
                    with open(perf_file, 'w') as f:
                        json.dump(self.ai_performance, f, indent=2)
                    logger.info(f"✅ Bot session performance data saved to {perf_file}")
                except Exception as e_save:
                    logger.error(f"⚠️ Performance data save failed: {e_save}", exc_info=True)

            logger.info("✅ Trading bot shutdown completed successfully.")
            event.accept()
        except Exception as e:
            logger.error(f"❌ Critical error during shutdown: {e}", exc_info=True)
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

