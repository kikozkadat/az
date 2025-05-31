# gui/main_window.py - FIXED VERSION with proper method definitions

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QListWidget, QPushButton, QTabWidget, QScrollArea, QSizePolicy, QGroupBox, QTextEdit) # QGroupBox, QTextEdit hozzáadva
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QColor # QColor importálása a QGroupBox stílusához és a chart markerekhez
import pyqtgraph as pg
import sys
import pandas as pd
import time
import random

# Logging beállítása - Hozzáadva a jobb hibakeresés érdekében
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 🚀 ADVANCED MODULES IMPORT
try:
    from strategy.advanced_market_scanner import AdvancedMarketScanner, CoinAnalysis
    from strategy.ml_scoring_engine import MLScoringEngine, MLFeatures
    from strategy.micro_strategy_engine import MicroStrategyEngine, MicroTradeSetup
    from core.dynamic_risk_manager import DynamicRiskManager
    from strategy.correlation_analyzer import CorrelationAnalyzer
    from strategy.support_resistance_detector import SupportResistanceDetector
    ADVANCED_MODULES_AVAILABLE = True
    logger.info("✅ Advanced trading modules loaded successfully") # Log üzenet hozzáadva
except ImportError as e:
    logger.warning(f"⚠️ Advanced modules not available: {e}") # Log üzenet hozzáadva
    ADVANCED_MODULES_AVAILABLE = False
    # Fallback osztályok az advanced modulokhoz, ha nem érhetők el
    class AdvancedMarketScanner: pass
    class MLScoringEngine: pass
    class MicroStrategyEngine: pass
    class DynamicRiskManager: pass
    class CorrelationAnalyzer: pass
    class SupportResistanceDetector: pass
    class CoinAnalysis: pass # Hozzáadva, hogy ne legyen hiba, ha az ADVANCED_MODULES_AVAILABLE False
    class MLFeatures: pass # Hozzáadva
    class MicroTradeSetup: pass # Hozzáadva


# Standard imports with fallbacks
try:
    from core.trailing_stop import TrailingStopManager
except ImportError:
    logger.warning("⚠️ TrailingStopManager not found. Using fallback.") # Log üzenet
    class TrailingStopManager:
        def __init__(self, target_profit=0.005): self.target_profit = target_profit
        def link(self, pair, position_manager, trader): pass

try:
    from data.kraken_api_client import KrakenAPIClient
except ImportError:
    logger.warning("⚠️ KrakenAPIClient not found. Using fallback.") # Log üzenet
    class KrakenAPIClient:
        def __init__(self): self.ws_client = None # Hozzáadva a ws_client inicializálása
        def test_connection(self): return True
        def get_valid_usd_pairs(self): return [{"altname": "XBTUSD", "wsname": "XBT/USD", "base": "XBT", "quote": "USD"}] # Kiegészítve base és quote mezőkkel
        def get_ohlc(self, pair, interval=1, limit=None): return {pair: []} if limit else {pair: [[time.time(),1,1,1,1,1,1,1]]} # Kiegészítve limit paraméterrel és dummy adattal
        def get_fallback_pairs(self): return [{"altname": "XBTUSD", "wsname": "XBT/USD", "base": "XBT", "quote": "USD"}] # Kiegészítve base és quote mezőkkel
        def initialize_websocket(self, pair_names): return False
        def get_current_price(self, pair): return random.uniform(100,1000) if pair == "XBTUSD" else random.uniform(1,10) # Dummy ár generálás
        def get_ticker_data(self, pair): return {'price': self.get_current_price(pair), 'bid': self.get_current_price(pair)*0.99, 'ask': self.get_current_price(pair)*1.01, 'volume_24h': random.uniform(100,1000), 'high_24h': self.get_current_price(pair)*1.05, 'low_24h': self.get_current_price(pair)*0.95} # Dummy ticker adatok
        def cleanup(self): pass
        ws_data_available = False
        def get_usd_pairs_with_volume(self, min_volume_usd=500000): return [{"altname": "XBTUSD", "wsname": "XBT/USD", "base":"XBT", "quote":"USD", "volume_usd": min_volume_usd + 1000}] # Dummy volume adatok
        def get_market_data(self): return [{"symbol": "XBTUSD", "score": 0.75, "price": self.get_current_price("XBTUSD"), "volume_usd": 5000000, "volume_24h": 5000000/self.get_current_price("XBTUSD")}] # Dummy market adatok


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
    logger.warning(f"GUI components import error: {e}. Using fallbacks.") # Log üzenet
    # Fallback classes
    class DashboardPanel(QWidget):
        def __init__(self, parent=None): # parent hozzáadva
            super().__init__(parent) # parent átadása
            self.layout = QVBoxLayout(self) # self átadása a layoutnak
            self.fields = {}
        def update_field(self, name, value):
            if name not in self.fields:
                label = QLabel(f"{name}: {value}")
                self.layout.addWidget(label)
                self.fields[name] = label
            else:
                self.fields[name].setText(f"{name}: {value}")

    class SettingsPanel(QWidget):
        def __init__(self, parent=None): super().__init__(parent) # parent hozzáadva és átadva
        def get_settings(self): return {} # Dummy implementáció

    class PositionListWidget(QWidget): # QListWidget helyett QWidget, hogy ne legyen hiba
        def __init__(self, pos_manager, parent_window=None):
            super().__init__(parent_window) # parent_window átadása
            self.pos_manager = pos_manager # pos_manager tárolása
            self.setLayout(QVBoxLayout()) # Layout hozzáadása
        def update_history(self): pass
        def update_positions(self, positions): pass # Hozzáadva a metódus

    class ScoringPanel(QWidget):
        def __init__(self, scorer, data_func, parent=None): # parent hozzáadva
            super().__init__(parent) # parent átadása
            self.scorer = scorer # scorer tárolása
            self.data_func = data_func # data_func tárolása
            self.setLayout(QVBoxLayout()) # Layout hozzáadása
        def refresh_panel(self): pass # Dummy implementáció
        def start_auto_refresh(self): pass # Dummy implementáció

    class StatsPanel(QWidget):
        def __init__(self, parent=None): # parent hozzáadva
            super().__init__(parent) # parent átadása
            self.setLayout(QVBoxLayout()) # Layout hozzáadása
        def update_stats(self): pass

    class AdvancedControlPanel(QWidget):
        def __init__(self, parent=None): # parent hozzáadva
            super().__init__(parent) # parent átadása
            self.setLayout(QVBoxLayout()) # Layout hozzáadása

    class LiveTradingPanel(QWidget):
        start_live_trading = pyqtSignal()
        stop_live_trading = pyqtSignal()
        emergency_stop = pyqtSignal()
        settings_changed = pyqtSignal(dict)
        def __init__(self, api_client=None, parent_window=None): # api_client és parent_window hozzáadva
            super().__init__(parent_window) # parent_window átadása
            self.api_client = api_client # api_client tárolása
            self.parent_window = parent_window # parent_window tárolása
            self.setLayout(QVBoxLayout())
            self.mode_combo = QListWidget() # QComboBox helyett QListWidget a konzisztencia miatt
            self.mode_combo.addItem("Bollinger Breakout") # Példa elem
            self.layout().addWidget(self.mode_combo) # Hozzáadás a layout-hoz
        def is_trading_active(self): return False
        def get_current_settings(self): return {'auto_trading': False, 'max_positions': 1} # Dummy beállítások
        def update_opportunities(self, opps): pass
        def update_trade_stats(self, trade_data): pass
        def get_session_stats(self): return {}
        def set_trading_active_status(self, status): pass # Dummy implementáció

# Import core components
from core.trade_manager import TradeManager
from core.position_manager import PositionManager
from utils.trade_logger import TradeLogger # TradeLogger importálása
from core.wallet_display import WalletManager
from utils.history_analyzer import HistoryAnalyzer
from strategy.decision_ai import DecisionEngine
from strategy.intelligent_trader import IntelligentTrader
from strategy.market_scanner import MarketScanner
from strategy.indicator_engine import IndicatorEngine
from strategy.scorer import CoinScorer

class LiveTradingThread(QThread): # KrakenLiveTradingThread helyett általánosabb név
    opportunity_found = pyqtSignal(dict)
    trade_executed = pyqtSignal(dict)
    position_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    scan_progress = pyqtSignal(str) # Hozzáadva a scan_progress jelzés
    pump_detected = pyqtSignal(dict) # Hozzáadva a pump_detected jelzés
    ai_decision = pyqtSignal(dict) # Hozzáadva az ai_decision jelzés

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.running = False
        self.scan_interval = 180 # Alapértelmezett scan interval
        self.min_volume_usd = 500000 # Alapértelmezett min volume
        self.min_score_threshold = 0.5 # Alapértelmezett min score
        self.pump_threshold = 0.1 # Alapértelmezett pump threshold (10%)
        self.final_candidates = [] # Hozzáadva a final_candidates lista

    def start_trading(self):
        self.running = True
        self.start()

    def stop_trading(self):
        self.running = False
        # self.quit() # Nem szükséges, a while ciklusból kilép
        # self.wait() # Nem szükséges itt, a closeEvent-ben kezeljük

    def run(self):
        logger.info("🚀 Live trading thread started") # Log üzenet
        while self.running:
            try:
                self.scan_progress.emit("Scanning for opportunities...") # Scan progress jelzés
                self.execute_trading_cycle()
                # Dinamikus várakozás a scan_interval alapján
                for _ in range(self.scan_interval):
                    if not self.running:
                        break
                    self.msleep(1000) # 1 másodpercenként ellenőrizzük a futást
            except Exception as e:
                error_msg = f"Trading cycle error: {e}"
                logger.error(error_msg, exc_info=True) # Részletes hiba logolása
                self.error_occurred.emit(error_msg)
        logger.info("🔴 Live trading thread stopped") # Log üzenet

    def execute_trading_cycle(self):
        if not self.running: return
        try:
            # Itt történne a tényleges piac szkennelés és döntéshozatal
            # Ez a rész most csak egy dummy implementáció
            self.scan_progress.emit("Analyzing market data...")

            # Példa: API hívás a piac szkenneléséhez (main_window.api használatával)
            if hasattr(self.main_window, 'api') and self.main_window.api:
                # Itt lehetne hívni pl. self.main_window.api.get_market_data()
                # és feldolgozni az eredményeket
                pass

            # Dummy pump detection
            if random.random() < 0.05: # 5% esély egy pump észlelésére
                dummy_pump_data = {
                    'pair': random.choice(['XBTUSD', 'ETHUSD', 'SOLUSD']),
                    'pump_pct': random.uniform(5, 15),
                    'volume_usd': random.uniform(100000, 1000000)
                }
                self.pump_detected.emit(dummy_pump_data)
                self.ai_decision.emit({'decision': f"Pump detected: {dummy_pump_data['pair']}", 'pair': dummy_pump_data['pair'], 'score': 0.0})


            # Dummy opportunity
            if random.random() < 0.1: # 10% esély egy lehetőség észlelésére
                opportunity_pair_altname = random.choice(['XBTUSD', 'ETHUSD', 'SOLUSD', 'ADAUSD'])
                entry_price_for_opp = self.main_window.get_current_price_for_pair(opportunity_pair_altname) or random.uniform(100,1000)
                opportunity = {
                    'pair': opportunity_pair_altname,
                    'score': random.uniform(self.min_score_threshold, 0.95), # min_score_threshold használata
                    'reason': 'Mock opportunity based on dummy scan',
                    'entry_price': entry_price_for_opp,
                    'volume': 0.001, # Dummy volume
                    'stop_loss': entry_price_for_opp * 0.96,
                    'take_profit': entry_price_for_opp * 1.008,
                    'signal': 'buy',
                    'total_score': random.uniform(self.min_score_threshold, 0.95),
                    'position_size_usd': 50.0
                }
                self.opportunity_found.emit(opportunity)
                self.ai_decision.emit({'decision': f"Opportunity found: {opportunity['pair']}", 'pair': opportunity['pair'], 'score': opportunity['score']})

                # Dummy trade execution
                if opportunity['score'] > (self.min_score_threshold + 0.1): # Csak magasabb score esetén
                    self.trade_executed.emit({
                        'pair': opportunity['pair'],
                        'action': 'OPEN',
                        'status': 'executed',
                        'pnl': 0.0
                    })

            self.final_candidates = [ # Dummy final candidates
                {'altname': 'XBTUSD', 'final_score': 0.85, 'volume_usd': 5000000, 'pump_detected': False},
                {'altname': 'ETHUSD', 'final_score': 0.78, 'volume_usd': 3000000, 'pump_detected': True},
            ]

            self.scan_progress.emit(f"✅ Scan completed at {time.strftime('%H:%M:%S')}")
        except Exception as e:
            error_msg = f"Trading cycle execution error: {e}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
            self.scan_progress.emit(f"⚠️ Scan failed: {str(e)[:50]}")

    def force_scan_now(self): # Hozzáadva a force_scan_now metódus
        logger.info("🔄 Force scan requested in LiveTradingThread.")
        self.scan_progress.emit("Forcing rescan...")
        # Itt lehetne megszakítani a várakozást és azonnal futtatni a ciklust,
        # de egy egyszerűbb megoldás, ha csak a következő ciklus hamarabb lefut.
        # Vagy egy dedikált eseményt küldhetnénk, amit a run() ciklus figyel.
        # Most csak egy log üzenetet és progress jelzést küldünk.
        # A tényleges azonnali scanhez a run() ciklus logikáját kellene módosítani.
        # Egyelőre ez a szál nem futtat azonnali scant, csak jelzi a kérést.


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎯 Advanced Live Trading Bot - $50 Bollinger Breakout Edition")
        self.setMinimumSize(1600, 900) # Méret növelve
        self.resize(1800, 1100) # Méret növelve

        self.api = KrakenAPIClient() # API kliens inicializálása korábban
        self.ai_mode_active = ADVANCED_MODULES_AVAILABLE # ai_mode_active beállítása

        self.init_ai_components() # AI komponensek inicializálása

        self.live_trading_thread = None
        self.update_counter = 0
        # self.last_advanced_scan = 0 # Eltávolítva, nincs használatban
        # self.active_trade_setup = None # Eltávolítva, nincs használatban

        # GUI elemek attribútumként való deklarálása, hogy később elérhetők legyenek
        self.balance_label = None
        self.indicator_label = None
        self.live_status_label = None
        self.ai_status_label = None
        self.ai_decision_label = None # Ez a bal oldali panelen lévő "Last Scan"
        self.scan_progress_label = None
        self.ai_toggle_btn = None
        self.ai_force_scan_btn = None
        self.pair_list = None
        self.manual_trade_button = None
        self.close_button = None # Ez a "Close All" gomb lesz
        self.chart_widget = None
        self.ai_decision_display = None # Ez a középső panelen lévő "Live Scanner" kijelző
        self.ai_opportunities_display = None
        self.position_info_label = None
        self.ai_stats_label = None # Ez a középső panelen lévő "Trading Stats"
        self.right_tabs = None
        self.live_panel = None
        self.ai_control_panel = None # Korábban ai_control_panel_tab
        self.dashboard = None
        self.position_list_widget = None # Korábban position_list
        self.scoring_panel = None
        self.stats_panel = None
        self.settings_panel = None # Hozzáadva a settings_panel
        self.ai_performance_labels = {} # AI teljesítmény címkék tárolása
        self.ai_decision_log = None # AI döntési napló
        self.min_volume_display_label = None # Új címkék a paraméterek kijelzéséhez
        self.min_score_display_label = None
        self.pump_thresh_display_label = None
        self.ai_mode_label = None # Hozzáadva az AI mód kijelzéséhez

        self.current_chart_pair = None # Aktuálisan megjelenített pár (altname formátumban)
        self.live_trading_active = False # Jelzi, hogy a live trading aktív-e

        self.ai_decisions = [] # AI döntések tárolása
        self.ai_performance = { # AI teljesítmény adatok
            'total_decisions': 0,
            'successful_trades': 0,
            'ai_profit': 0.0,
            'ai_accuracy': 0.0
        }
        self.last_trade_pnl = 0.0 # Utolsó trade P&L tárolása


        self.test_api_connection()
        self.setup_ui() # UI beállítása az attribútumok deklarálása után
        self.init_standard_components() # Standard komponensek inicializálása IDE lett ÁTHELYEZVE
        self.setup_timers()
        self.setup_live_trading_connections()

        self.initialize_live_trading_system() # Live trading rendszer inicializálása

        # Kezdeti frissítések
        self.update_chart()
        self.refresh_balance()
        self.refresh_open_trades()
        self.refresh_stats()
        self.start_score_updater()

        logger.info("✅ Enhanced Main window initialized successfully") # Log üzenet

    def init_ai_components(self):
        """AI komponensek inicializálása"""
        if ADVANCED_MODULES_AVAILABLE:
            self.ai_scanner = AdvancedMarketScanner()
            self.ml_engine = MLScoringEngine()
            self.micro_strategy = MicroStrategyEngine()
            self.risk_manager = DynamicRiskManager()
            self.correlation_analyzer = CorrelationAnalyzer()
            self.sr_detector = SupportResistanceDetector()
            self.ai_mode_active = True
            logger.info("🚀 Advanced AI trading components initialized")
            self._setup_ai_connections() # AI kapcsolatok beállítása
        else:
            self.ai_mode_active = False
            logger.warning("⚠️ Advanced AI modules not available, using basic mode. Initializing fallback components.")
            self._init_fallback_components() # Fallback AI komponensek

    def _setup_ai_connections(self):
        """AI komponensek közötti kapcsolatok beállítása (pl. API kliens átadása)"""
        try:
            if self.ai_mode_active and self.api:
                components_to_setup = [
                    getattr(self, 'ai_scanner', None),
                    getattr(self, 'ml_engine', None),
                    getattr(self, 'micro_strategy', None),
                    getattr(self, 'risk_manager', None),
                    getattr(self, 'correlation_analyzer', None),
                    getattr(self, 'sr_detector', None)
                ]
                for component in components_to_setup:
                    if component:
                        if hasattr(component, 'connect_api_client'):
                            component.connect_api_client(self.api)
                        elif hasattr(component, 'set_api_client'):
                            component.set_api_client(self.api)
                        elif hasattr(component, 'api_client'): # Ha direktben beállítható
                            component.api_client = self.api
                logger.info("✅ AI component API connections established/verified.")
        except Exception as e:
            logger.error(f"AI connections setup failed: {e}", exc_info=True)

    def _init_fallback_components(self):
        """Fallback AI komponensek inicializálása, ha a fejlett modulok nem érhetők el."""
        logger.info("🔄 Initializing fallback AI components...")
        class DummyAIScanner:
            api_client = None
            def scan_top_opportunities(self, max_pairs=20): return []
            def get_scanner_status(self): return {'mode': 'FALLBACK', 'status': 'Basic mode', 'name':'DummyScanner'}
            def connect_api_client(self, api_client): self.api_client = api_client
            def set_api_client(self, api_client): self.api_client = api_client
        class DummyMLEngine:
            api_client = None
            def predict_trade_success(self, features): return type('MLPrediction', (), {'probability': 0.5, 'confidence': 0.5, 'expected_return': 0.15})()
            def set_api_client(self, api_client): self.api_client = api_client
            def get_model_performance(self): return {'training_samples': 0, 'accuracy': 0.0, 'name': 'DummyML'}
        class DummyMicroStrategy:
            api_client = None
            def analyze_micro_opportunity(self, coin_analysis, ml_prediction): return None
            def get_strategy_status(self): return {'name': 'DummyMicroStrategy', 'status': 'Fallback mode'}
            def set_api_client(self, api_client): self.api_client = api_client
        class DummyRiskManager:
            api_client = None
            def calculate_position_size(self, *args, **kwargs): return {'position_size_usd': 50.0, 'volume': 0.001}
            def get_risk_status(self): return {'name': 'DummyRiskManager', 'status': 'Fallback mode'}
            def set_api_client(self, api_client): self.api_client = api_client
            def calculate_trade_parameters(self, *args, **kwargs): # Hozzáadva a hiányzó metódus
                 return {
                    'position_size_usd': 50.0,
                    'volume': 0.001,
                    'entry_price': kwargs.get('entry_price', 100),
                    'stop_loss_price': kwargs.get('stop_loss_price', 90),
                    'take_profit_price': kwargs.get('take_profit_price', 110)
                }
        # ... (többi dummy osztály)
        self.ai_scanner = DummyAIScanner()
        self.ml_engine = DummyMLEngine()
        self.micro_strategy = DummyMicroStrategy()
        self.risk_manager = DummyRiskManager()
        # ...
        logger.info("✅ Fallback AI components initialized.")


    def init_standard_components(self):
        """Standard (nem-AI specifikus) komponensek inicializálása."""
        self.scorer = CoinScorer()
        # self.api már inicializálva van
        self.indicators = IndicatorEngine()
        self.trader = TradeManager() # TradeManager példányosítása
        self.position = PositionManager()
        self.trade_logger = TradeLogger() # TradeLogger példányosítása, korábban self.logger
        self.wallet_display = WalletManager()
        self.analyzer = HistoryAnalyzer()
        self.decision_engine = DecisionEngine() # DecisionEngine példányosítása
        self.intelligent_trader = IntelligentTrader(main_window=self) # IntelligentTrader példányosítása
        self.market_scanner = MarketScanner(self.api) # MarketScanner példányosítása API klienssel
        self.trailing_stop_manager = TrailingStopManager() # TrailingStopManager példányosítása, korábban self.trailing
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
                # Fallback, ha a wallet_display nem elérhető vagy nincs meg a metódus
                fallback_balance = 66.0  # Szimulált egyenleg
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
            if not hasattr(self, 'position') or not self.position:
                if self.position_info_label:
                    self.position_info_label.setText("Positions:\n(Position manager not available)")
                logger.warning("Position manager not available for refreshing open trades.")
                return

            open_positions = self.position.get_all_positions() if hasattr(self.position, 'get_all_positions') else {}

            if not open_positions:
                if self.position_info_label:
                    self.position_info_label.setText("Positions:\n(none)")
            else:
                msg = "Open Positions:\n"
                for pair, data in open_positions.items():
                    side = data.get('side', 'N/A')
                    entry_price = data.get('entry_price', 0)
                    volume = data.get('volume', 0)
                    # Itt a _get_real_current_price-t hívjuk, ami a valós árat adja vissza
                    current_price = self._get_real_current_price(pair, entry_price)


                    pnl_text = "N/A"
                    pnl_color_char = "" # Karakter a szín jelzésére

                    if current_price is not None and entry_price > 0 and volume > 0:
                        pnl = (current_price - entry_price) * volume if side.lower() == 'buy' else (entry_price - current_price) * volume
                        pnl_text = f"${pnl:.2f}"
                        pnl_color_char = "🟢" if pnl >= 0 else "🔴" # Zöld profit, piros veszteség

                    msg += f"{pnl_color_char} {pair} ({side.upper()}): Entry ${entry_price:.4f}, Vol {volume:.6f} - P&L: {pnl_text}\n"

                if self.position_info_label:
                    self.position_info_label.setText(msg.strip())
                if self.position_list_widget and hasattr(self.position_list_widget, 'update_positions'): # Feltételezve, hogy van ilyen metódus
                    self.position_list_widget.update_positions(open_positions)


        except Exception as e:
            logger.error(f"[UI] Position display refresh failed: {e}", exc_info=True)
            if self.position_info_label:
                self.position_info_label.setText("Positions:\nError refreshing")

    def refresh_stats(self):
        """Kereskedési statisztikák frissítése a GUI-n."""
        try:
            if self.ai_stats_label: # Korábban stats_label, most a középső panelen lévő
                current_time = pd.Timestamp.now().strftime('%H:%M:%S')
                total_trades = 0
                win_rate = 0.0 # Legyen float
                total_profit = 0.0 # Legyen float

                if hasattr(self, 'trade_logger') and self.trade_logger and hasattr(self.trade_logger, 'get_statistics'):
                    stats = self.trade_logger.get_statistics() # TradeLogger-ből vesszük a statisztikát
                    total_trades = stats.get('total_trades', 0)
                    win_rate = stats.get('win_rate', 0.0) * 100 # Százalékban
                    total_profit = stats.get('total_profit', 0.0)
                elif hasattr(self, 'position') and self.position and hasattr(self.position, 'get_statistics'): # Fallback a PositionManager-re
                    stats = self.position.get_statistics()
                    total_trades = stats.get('total_trades', 0)
                    win_rate = stats.get('win_rate', 0.0) * 100 # Százalékban
                    total_profit = stats.get('total_profit', 0.0)


                stats_text = f"Trading Stats ({current_time}):\n"
                stats_text += f"Total Trades: {total_trades}\n"
                stats_text += f"Win Rate: {win_rate:.1f}%\n"
                stats_text += f"Total P&L: ${total_profit:.2f}"

                self.ai_stats_label.setText(stats_text)

            if self.stats_panel and hasattr(self.stats_panel, 'update_stats'):
                self.stats_panel.update_stats() # A dedikált StatsPanel frissítése

        except Exception as e:
            logger.error(f"[STATS] Stats refresh failed: {e}", exc_info=True)
            if self.ai_stats_label:
                self.ai_stats_label.setText("Trading Stats:\nError updating")

    def start_score_updater(self):
        """Coin pontozó rendszer frissítőjének indítása."""
        try:
            logger.info("📊 Starting score updater...")
            if self.scoring_panel and hasattr(self.scoring_panel, 'start_auto_refresh'):
                self.scoring_panel.start_auto_refresh() # Feltételezve, hogy van ilyen metódus
                logger.info("✅ Score updater delegated to scoring panel's auto-refresh.")
            elif self.scoring_panel and hasattr(self.scoring_panel, 'refresh_panel'):
                 # Manuális frissítés, ha nincs auto-refresh
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

            # Kezdeti párok a WebSocket-hez (pl. a pair_list első néhány eleme)
            initial_pairs_for_ws = []
            if self.pair_list and self.pair_list.count() > 0:
                # Az altname formátumot használjuk (pl. XBTUSD)
                initial_pairs_for_ws = [self.pair_list.item(i).text().replace("/", "") for i in range(min(self.pair_list.count(), 5))] # Max 5 pár
            else:
                # Fallback, ha a pair_list üres
                valid_pairs_from_api = self.api.get_valid_usd_pairs()
                if valid_pairs_from_api:
                    initial_pairs_for_ws = [p['altname'] for p in valid_pairs_from_api[:2] if 'altname' in p] # Max 2 pár API-ból
                else:
                    initial_pairs_for_ws = ["XBTUSD", "ETHUSD"] # Végső fallback

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
        main_layout = QHBoxLayout(self) # self átadása a fő layoutnak
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        # self.setLayout(main_layout) # Már megtörtént a QHBoxLayout(self) miatt

        left_widget = self.create_enhanced_left_panel() # Bal oldali panel új neve
        left_widget.setMaximumWidth(220) # Kicsit szélesebb
        main_layout.addWidget(left_widget)

        middle_widget = self.create_ai_enhanced_middle_panel() # Középső panel új neve
        main_layout.addWidget(middle_widget, 1) # Súlyozás, hogy ez töltse ki a helyet

        right_widget = self.create_enhanced_right_panel()
        right_widget.setMaximumWidth(450) # Kicsit szélesebb
        main_layout.addWidget(right_widget)

    def create_enhanced_left_panel(self): # Korábban create_left_panel
        """Bal oldali vezérlőpanel létrehozása, AI funkciókkal bővítve."""
        widget = QWidget()
        layout = QVBoxLayout(widget) # widget átadása a layoutnak
        layout.setSpacing(8) # Kicsit nagyobb térköz

        # AI Live Trading Status GroupBox
        ai_group = QGroupBox("🤖 Live Trading Status")
        ai_group.setStyleSheet("QGroupBox { font-weight: bold; color: white; }") # Stílus
        ai_layout = QVBoxLayout() # Belső layout

        self.ai_status_label = QLabel("Live Trading: INITIALIZING")
        self.ai_status_label.setStyleSheet("font-weight: bold; color: #00AAFF; font-size: 12px;")
        ai_layout.addWidget(self.ai_status_label)

        self.ai_decision_label = QLabel("Last Scan: None") # Ez a bal oldali panelen lévő
        self.ai_decision_label.setStyleSheet("color: #CCCCCC; font-size: 10px;")
        ai_layout.addWidget(self.ai_decision_label)

        self.scan_progress_label = QLabel("Scan Progress: Idle")
        self.scan_progress_label.setStyleSheet("color: #CCCCCC; font-size: 10px;")
        ai_layout.addWidget(self.scan_progress_label)

        self.ai_toggle_btn = QPushButton("🚀 Start Live Trading")
        self.ai_toggle_btn.clicked.connect(self.toggle_live_trading) # Metódus csatlakoztatása
        self.ai_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white; /* Betűszín hozzáadva */
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #2ECC71; }
        """)
        ai_layout.addWidget(self.ai_toggle_btn)

        self.ai_force_scan_btn = QPushButton("🔄 Force Rescan")
        self.ai_force_scan_btn.clicked.connect(self.force_live_scan) # Metódus csatlakoztatása
        ai_layout.addWidget(self.ai_force_scan_btn)

        test_pos_button = QPushButton("🧪 Open Test Position") # Korábban self.test_button
        test_pos_button.clicked.connect(self.open_test_position)
        test_pos_button.setStyleSheet("QPushButton { background-color: #3498DB; color: white; }") # Stílus
        ai_layout.addWidget(test_pos_button)

        manual_close_button = QPushButton("🚪 Manual Close Selected") # Korábban self.close_button (de az más volt)
        manual_close_button.clicked.connect(self.manual_close_current_position)
        manual_close_button.setStyleSheet("QPushButton { background-color: #E67E22; color: white; }") # Stílus
        ai_layout.addWidget(manual_close_button)

        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)

        # Settings Panel (ha van)
        try:
            self.settings_panel = SettingsPanel() # Korábban self.settings
            layout.addWidget(self.settings_panel)
        except Exception as e:
            logger.warning(f"Could not create SettingsPanel: {e}")
            layout.addWidget(QLabel("Settings Panel (Error)"))


        pairs_label = QLabel("💰 High-Volume Pairs (Chart):")
        pairs_label.setStyleSheet("color: white; font-weight: bold; font-size: 11px;")
        layout.addWidget(pairs_label)

        self.pair_list = QListWidget()
        self.load_trading_pairs() # Metódus hívása a párok betöltésére
        self.pair_list.setMaximumHeight(150) # Kicsit magasabb
        self.pair_list.currentTextChanged.connect(self.on_pair_changed)
        layout.addWidget(self.pair_list)

        self.manual_trade_button = QPushButton("👤 Manual Trade (Selected)")
        self.manual_trade_button.clicked.connect(self.execute_manual_trade) # Metódus csatlakoztatása
        self.manual_trade_button.setStyleSheet("QPushButton { background-color: #9B59B6; color: white; }") # Stílus
        layout.addWidget(self.manual_trade_button)

        self.close_button = QPushButton("❌ Close All Positions") # Ez a "Close All" gomb
        self.close_button.clicked.connect(self.manual_close_all_positions) # Metódus csatlakoztatása
        self.close_button.setStyleSheet("QPushButton { background-color: #E74C3C; color: white; font-weight: bold; }") # Stílus
        layout.addWidget(self.close_button)

        layout.addStretch()
        # widget.setLayout(layout) # Már megtörtént a QVBoxLayout(widget) miatt
        return widget

    def create_ai_enhanced_middle_panel(self): # Korábban create_middle_panel
        """Középső panel létrehozása, AI információkkal és jobb charttal."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)

        # Status sor (Balance, AI Mode)
        status_top_layout = QHBoxLayout() # Nincs szülője, majd hozzáadjuk a fő layout-hoz
        self.balance_label = QLabel("Balance: Loading...")
        self.balance_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #39ff14;")
        status_top_layout.addWidget(self.balance_label)

        self.ai_mode_label = QLabel(f"🤖 AI Modules: {'ADVANCED' if self.ai_mode_active else 'BASIC'}")
        self.ai_mode_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #8E44AD;")
        status_top_layout.addWidget(self.ai_mode_label, 0, Qt.AlignRight) # Jobbra igazítás
        layout.addLayout(status_top_layout)

        # Status sor (Chart Info, Live Status)
        status_mid_layout = QHBoxLayout()
        self.indicator_label = QLabel("🎯 Chart Status: Select a pair!") # Korábban "Live Trading System: Initializing..."
        self.indicator_label.setStyleSheet("font-size: 11px; color: white; padding: 2px;")
        status_mid_layout.addWidget(self.indicator_label, 2) # Súlyozás

        self.live_status_label = QLabel("⚪ OFFLINE") # Korábban piros volt
        self.live_status_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #FF4444; padding: 2px;") # Kezdetben offline
        status_mid_layout.addWidget(self.live_status_label, 1, Qt.AlignRight) # Jobbra igazítás
        layout.addLayout(status_mid_layout)


        # AI Decision Display (Új elem) - Ez a középső panelen lévő
        self.ai_decision_display = QLabel("🧠 Live Scanner: Initializing...")
        self.ai_decision_display.setStyleSheet("font-size: 11px; color: white; font-weight: bold; padding: 3px; background-color: #2C3E50; border-radius: 3px;")
        self.ai_decision_display.setWordWrap(True)
        self.ai_decision_display.setFixedHeight(40) # Fix magasság
        layout.addWidget(self.ai_decision_display)

        # Chart
        self.setup_ai_enhanced_chart() # Chart beállító metódus hívása
        layout.addWidget(self.chart_widget, 2) # Súlyozás, hogy ez töltse ki a legtöbb helyet

        # AI Opportunities Display (Új elem)
        self.ai_opportunities_display = QTextEdit()
        self.ai_opportunities_display.setMaximumHeight(120) # Magasság korlátozása
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

        # Info Layout (Positions, AI Stats)
        info_layout = QHBoxLayout()
        self.position_info_label = QLabel("Positions: (none)") # Korábban open_trades_label
        self.position_info_label.setStyleSheet("background:#191919; color:white; border:1px solid #222; padding:6px; font-size: 9px;")
        self.position_info_label.setWordWrap(True)
        self.position_info_label.setMinimumHeight(60) # Minimum magasság
        info_layout.addWidget(self.position_info_label)

        self.ai_stats_label = QLabel("Trading Stats: Starting...") # Korábban stats_label, most a középső panelen
        self.ai_stats_label.setStyleSheet("background:#191919; color:#8E44AD; border:1px solid #222; padding:6px; font-size: 9px;") # Más szín
        self.ai_stats_label.setWordWrap(True)
        self.ai_stats_label.setMinimumHeight(60) # Minimum magasság
        info_layout.addWidget(self.ai_stats_label)

        layout.addLayout(info_layout)
        return widget

    def setup_ai_enhanced_chart(self):
        """Fejlettebb diagram beállítása."""
        self.chart_widget = pg.PlotWidget()
        self.chart_widget.showGrid(x=True, y=True, alpha=0.3) # Finomabb rács
        self.chart_widget.setBackground((20, 20, 20)) # Sötétebb háttér
        self.chart_widget.getAxis('left').setPen(pg.mkPen(color=(150, 150, 150))) # Világosabb tengely
        self.chart_widget.getAxis('bottom').setPen(pg.mkPen(color=(150, 150, 150)))
        self.chart_widget.setLabel('left', 'Price ($)', color='lightgray', size='10pt') # Nagyobb betűméret
        self.chart_widget.setLabel('bottom', 'Time', color='lightgray', size='10pt')
        self.chart_widget.setMinimumHeight(300) # Nagyobb minimum magasság
        # self.ai_buy_markers = [] # Eltávolítva, a charton helyezzük el őket
        # self.ai_sell_markers = [] # Eltávolítva


    def create_enhanced_right_panel(self):
        """Jobb oldali panel létrehozása fülekkel, AI vezérlőkkel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)

        self.right_tabs = QTabWidget()
        self.right_tabs.setTabPosition(QTabWidget.North) # Fülek fent

        # Live Trading Panel Tab
        try:
            self.live_panel = LiveTradingPanel(api_client=self.api, parent_window=self)
        except Exception as e:
            logger.error(f"Failed to create LiveTradingPanel: {e}", exc_info=True)
            self.live_panel = QWidget() # Fallback
            self.live_panel.setLayout(QVBoxLayout())
            self.live_panel.layout().addWidget(QLabel("Live Trading Panel (Error)"))

        live_scroll = QScrollArea()
        live_scroll.setWidget(self.live_panel)
        live_scroll.setWidgetResizable(True)
        live_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        live_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_tabs.addTab(live_scroll, "🚀 Live Control") # Átnevezve

        # AI Control Tab (Új fül)
        try:
            # Az AdvancedControlPanel-t használjuk itt, ha van, vagy egy egyszerűbb panelt
            if ADVANCED_MODULES_AVAILABLE and hasattr(sys.modules[__name__], 'AdvancedControlPanel'): # Ellenőrizzük, hogy létezik-e az osztály
                 self.ai_control_panel = AdvancedControlPanel() # Korábban self.ai_control_panel_tab
            else: # Fallback, ha nincs AdvancedControlPanel
                self.ai_control_panel = self.create_ai_control_tab_content() # Külön metódus a tartalomhoz
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
        self.right_tabs.addTab(ai_scroll, "🧠 AI Console") # Átnevezve

        # Dashboard Tab (Adatok fül)
        dashboard_content_widget = QWidget() # Konténer widget a scroll area-hoz
        dashboard_layout = QVBoxLayout(dashboard_content_widget)
        dashboard_layout.setSpacing(5)

        try:
            self.dashboard = DashboardPanel()
            dashboard_layout.addWidget(self.dashboard)
        except Exception as e:
            logger.warning(f"Could not create DashboardPanel: {e}")
            dashboard_layout.addWidget(QLabel("Dashboard Panel (Error)"))

        try:
            self.position_list_widget = PositionListWidget(self.position, parent_window=self) # Korábban self.position_list
            dashboard_layout.addWidget(self.position_list_widget)
        except Exception as e:
            logger.warning(f"Could not create PositionListWidget: {e}")
            dashboard_layout.addWidget(QLabel("Position List (Error)"))

        dashboard_scroll = QScrollArea()
        dashboard_scroll.setWidget(dashboard_content_widget) # Konténer widget beállítása
        dashboard_scroll.setWidgetResizable(True)
        dashboard_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_tabs.addTab(dashboard_scroll, "📊 Data & Positions") # Átnevezve

        # Analysis Tab (Statisztikák fül)
        analysis_content_widget = QWidget() # Konténer widget
        analysis_layout = QVBoxLayout(analysis_content_widget)
        analysis_layout.setSpacing(5)

        try:
            if hasattr(self, 'get_all_coin_data'): # Ellenőrizzük, hogy létezik-e a metódus
                self.scoring_panel = ScoringPanel(self.scorer, self.get_all_coin_data)
                analysis_layout.addWidget(self.scoring_panel)
            else:
                logger.warning("get_all_coin_data method not found, ScoringPanel cannot be created.")
                analysis_layout.addWidget(QLabel("Scoring Panel (Data Error)"))
        except Exception as e:
            logger.warning(f"Could not create ScoringPanel: {e}")
            analysis_layout.addWidget(QLabel("Scoring Panel (Error)"))

        try:
            self.stats_panel = StatsPanel()
            analysis_layout.addWidget(self.stats_panel)
        except Exception as e:
            logger.warning(f"Could not create StatsPanel: {e}")
            analysis_layout.addWidget(QLabel("Stats Panel (Error)"))

        analysis_scroll = QScrollArea()
        analysis_scroll.setWidget(analysis_content_widget) # Konténer widget beállítása
        analysis_scroll.setWidgetResizable(True)
        analysis_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.right_tabs.addTab(analysis_scroll, "📈 Analysis & Stats") # Átnevezve

        layout.addWidget(self.right_tabs)
        return widget

    def create_ai_control_tab_content(self): # Korábban create_advanced_tab, de most AI vezérlő
        """Az 'AI Console' fül tartalmának létrehozása."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)

        # AI Performance GroupBox
        ai_status_group = QGroupBox("📈 Overall AI Performance")
        ai_status_layout = QGridLayout() # GridLayout a jobb elrendezésért
        self.ai_performance_labels = {} # Címkék tárolása a frissítéshez
        metrics = ["Bot Accuracy", "Total Trades", "Successful Trades", "Total P&L", "Last Trade P&L"]
        for i, metric in enumerate(metrics):
            label = QLabel(f"{metric}:")
            label.setStyleSheet("color: #CCCCCC; font-size: 11px;")
            value = QLabel("N/A") # Kezdeti érték
            value.setStyleSheet("color: #00AAFF; font-weight: bold; font-size: 11px;")
            ai_status_layout.addWidget(label, i, 0)
            ai_status_layout.addWidget(value, i, 1)
            self.ai_performance_labels[metric] = value
        ai_status_group.setLayout(ai_status_layout)
        layout.addWidget(ai_status_group)

        # Scanner Parameters GroupBox (LiveTradingThread paramétereinek kijelzése)
        scanner_params_group = QGroupBox("🎯 Live Scanner Parameters")
        scanner_params_layout = QVBoxLayout()

        # Ezeket a címkéket frissíteni kell, amikor a live_trading_thread elindul vagy a beállítások változnak
        self.min_volume_display_label = QLabel("Min Volume USD: N/A")
        scanner_params_layout.addWidget(self.min_volume_display_label)
        self.min_score_display_label = QLabel("Min Score Candidate: N/A")
        scanner_params_layout.addWidget(self.min_score_display_label)
        self.pump_thresh_display_label = QLabel("Pump Threshold %: N/A")
        scanner_params_layout.addWidget(self.pump_thresh_display_label)

        scanner_params_group.setLayout(scanner_params_layout)
        layout.addWidget(scanner_params_group)


        # AI Decision Log GroupBox
        log_group = QGroupBox("📝 AI Event Log")
        log_layout = QVBoxLayout()
        self.ai_decision_log = QTextEdit()
        self.ai_decision_log.setMaximumHeight(250) # Kicsit magasabb
        self.ai_decision_log.setStyleSheet("""
            QTextEdit {
                background-color: #1A1A1A; /* Sötétebb */
                color: #9F54FF; /* Lila-szerűbb */
                font-family: 'Consolas', 'Courier New', monospace; /* Jobb monospaced fontok */
                font-size: 9px;
                border: 1px solid #444;
                border-radius: 3px;
            }
        """)
        self.ai_decision_log.setReadOnly(True) # Legyen csak olvasható
        log_layout.addWidget(self.ai_decision_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        return content


    def create_advanced_tab(self):
        """Az 'Advanced' fül tartalmának létrehozása (ha ADVANCED_MODULES_AVAILABLE)."""
        # Ez a metódus most már csak akkor releváns, ha tényleg vannak külön
        # vezérlők a fejlett AI modulokhoz, amelyeket nem a LiveTradingPanel kezel.
        # Ha nincsenek, akkor ez a fül elhagyható, vagy csak státuszinformációt mutathat.
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)

        if self.ai_mode_active:
            header = QLabel("⚙️ Advanced AI Module Controls")
            header.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF6B35; padding: 5px;")
            layout.addWidget(header)

            # Itt lehetne hozzáadni specifikus vezérlőket az ai_scanner, ml_engine stb. számára,
            # ha ezeket a felhasználó dinamikusan szeretné állítani a GUI-ról.
            # Példa:
            if hasattr(self, 'ai_scanner') and self.ai_scanner:
                scanner_group = QGroupBox("Advanced Market Scanner")
                scanner_layout = QVBoxLayout(scanner_group)
                # ... vezérlők a scannerhez ...
                scanner_layout.addWidget(QLabel("Scanner specific settings placeholder..."))
                layout.addWidget(scanner_group)

            if hasattr(self, 'ml_engine') and self.ml_engine:
                ml_group = QGroupBox("ML Scoring Engine")
                ml_layout = QVBoxLayout(ml_group)
                # ... vezérlők az ML motorhoz ...
                ml_layout.addWidget(QLabel("ML Engine specific settings placeholder..."))
                layout.addWidget(ml_group)
        else:
            layout.addWidget(QLabel("Advanced AI modules are not active."))


        layout.addStretch()
        return content


    def setup_timers(self):
        """Időzítők beállítása a rendszeres frissítésekhez."""
        self.timer = QTimer(self) # Fő UI frissítő timer
        self.timer.timeout.connect(self.enhanced_update_cycle) # Korábban main_update_cycle
        self.timer.start(15000) # 15 másodpercenként (ritkábban, mint korábban)

        # A scan_timer és position_timer mostantól a LiveTradingThread-ben vagy
        # a PositionManager-ben lehetnek, vagy itt maradnak, ha a főszálon kell futniuk.
        # A LiveTradingThread saját scan_interval-lal rendelkezik.
        # A pozíciófigyelés lehet egy külön timer itt, vagy a PositionManager része.

        self.position_timer = QTimer(self) # Pozíciófigyelő timer
        self.position_timer.timeout.connect(self.monitor_positions_with_ai) # Korábban monitor_positions
        self.position_timer.start(10000) # 10 másodpercenként

    def setup_live_trading_connections(self):
        """Live trading panel és a főablak közötti jelzések összekötése."""
        try:
            if self.live_panel: # Ellenőrizzük, hogy a live_panel létezik-e
                if hasattr(self.live_panel, 'start_live_trading'):
                    self.live_panel.start_live_trading.connect(self.start_live_trading_action) # _action használata
                if hasattr(self.live_panel, 'stop_live_trading'):
                    self.live_panel.stop_live_trading.connect(self.stop_live_trading_action) # _action használata
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
            logger.info("🔗 Testing Kraken API connection...") # Log üzenet
            if self.api.test_connection():
                logger.info("[STARTUP] ✅ API connection successful")
                if self.dashboard: self.dashboard.update_field("API Status", "🟢 CONNECTED")

                # Próbáljunk meg volume adatokat lekérni
                try:
                    volume_pairs = self.api.get_usd_pairs_with_volume(min_volume_usd=100000) # Kisebb limit a teszthez
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

            # AI modulok státuszának ellenőrzése
            if self.ai_mode_active:
                logger.info("[STARTUP] 🚀 Advanced AI trading modules are active.")
                self._test_advanced_components() # Speciális AI komponensek tesztelése
            else:
                logger.info("[STARTUP] ⚠️ Basic trading mode (Advanced AI modules not available).")


        except Exception as e:
            logger.error(f"[STARTUP] ❌ API connection test error: {e}", exc_info=True)
            if self.dashboard:
                self.dashboard.update_field("API Status", f"🔴 ERROR: {str(e)[:20]}")

    def _test_advanced_components(self):
        """Fejlett AI komponensek tesztelése (ha elérhetők)."""
        if not self.ai_mode_active: return
        logger.info("🧪 Testing advanced AI components' status methods...")
        components_to_test = {
            "AI Scanner": getattr(self, 'ai_scanner', None),
            "ML Engine": getattr(self, 'ml_engine', None),
            "Micro Strategy": getattr(self, 'micro_strategy', None),
            "Risk Manager": getattr(self, 'risk_manager', None),
            "Correlation Analyzer": getattr(self, 'correlation_analyzer', None),
            "S/R Detector": getattr(self, 'sr_detector', None),
        }
        for name, component in components_to_test.items():
            if component:
                status_method_name = None
                if hasattr(component, 'get_status'): status_method_name = 'get_status'
                elif hasattr(component, 'get_scanner_status'): status_method_name = 'get_scanner_status'
                elif hasattr(component, 'get_model_performance'): status_method_name = 'get_model_performance'
                elif hasattr(component, 'get_strategy_status'): status_method_name = 'get_strategy_status'
                elif hasattr(component, 'get_risk_status'): status_method_name = 'get_risk_status'
                elif hasattr(component, 'get_analyzer_status'): status_method_name = 'get_analyzer_status'
                elif hasattr(component, 'get_detector_status'): status_method_name = 'get_detector_status'

                if status_method_name and hasattr(component, status_method_name) and callable(getattr(component, status_method_name)):
                    try:
                        status = getattr(component, status_method_name)()
                        logger.info(f"  Component '{name}': Status - {status}")
                    except Exception as e_status:
                        logger.warning(f"  Component '{name}': Error getting status - {e_status}")
                else:
                    logger.info(f"  Component '{name}': No standard status method found.")
            else:
                logger.info(f"  Component '{name}': Not initialized.")


    def load_trading_pairs(self):
        """Kereskedési párok betöltése a GUI listába, volumen alapján szűrve."""
        try:
            logger.info("[PAIRS_UI] Loading high-volume pairs for chart dropdown...")
            if not (hasattr(self, 'api') and self.api and hasattr(self.api, 'get_usd_pairs_with_volume')):
                logger.warning("API client or get_usd_pairs_with_volume method not available. Using fallback pairs.")
                self._load_fallback_pairs_to_ui()
                return

            volume_pairs_ui = self.api.get_usd_pairs_with_volume(min_volume_usd=1000000) # Magasabb limit a UI-hoz

            if volume_pairs_ui and isinstance(volume_pairs_ui, list):
                pair_names_ui = []
                for pair_info in volume_pairs_ui[:25]: # Max 25 pár a listába
                    altname = pair_info.get("altname")
                    wsname = pair_info.get("wsname", altname) # wsname preferálása a megjelenítéshez
                    if altname and altname.upper() not in ['USDTZUSD', 'USDCUSD', 'DAIUSD', 'EURZEUR', 'GBPZGBP']: # Stablcoinok és nem USD párok szűrése
                        display_name = wsname if wsname else altname # Pl. "XBT/USD"
                        pair_names_ui.append(display_name)

                if pair_names_ui:
                    self.pair_list.clear()
                    self.pair_list.addItems(pair_names_ui)
                    if self.pair_list.count() > 0:
                        self.pair_list.setCurrentRow(0)
                        # Az on_pair_changed implicit módon meghívódik a setCurrentRow után, ha a text változik,
                        # de explicit hívás biztosabb, ha az első elem ugyanaz maradna.
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


    def on_pair_changed(self, pair_display_name: str): # pair_display_name pl. "XBT/USD"
        """Kiválasztott kereskedési pár változásának kezelése."""
        if pair_display_name:
            # Az `altname` formátumot (pl. XBTUSD) használjuk a `current_chart_pair` tárolására
            # és az API hívásokhoz. A `pair_display_name` (pl. XBT/USD) a UI-n jelenik meg.
            self.current_chart_pair = pair_display_name.replace("/", "")

            logger.info(f"[UI_CHART] Selected pair for chart: {pair_display_name} (Using altname: {self.current_chart_pair})")
            self.update_chart() # Ez most már a self.current_chart_pair-t fogja használni implicit módon
            if self.dashboard: self.dashboard.update_field("Selected Pair", pair_display_name)


    def get_current_price_for_pair(self, pair_altname: str) -> float or None: # pair_altname pl. "XBTUSD"
        """Aktuális ár lekérdezése egy adott kereskedési párhoz (altname formátumban)."""
        try:
            if not self.api: return None

            # Próbálkozás a WebSocket adatokkal, ha elérhető és friss
            if hasattr(self.api, 'ws_client') and self.api.ws_client and \
               hasattr(self.api.ws_client, 'get_current_price') and \
               hasattr(self.api.ws_client, 'is_pair_subscribed') and self.api.ws_client.is_pair_subscribed(pair_altname):
                price_ws = self.api.ws_client.get_current_price(pair_altname)
                if price_ws is not None and price_ws > 0:
                    # logger.debug(f"Price for {pair_altname} from WebSocket: {price_ws}")
                    return float(price_ws)

            # Ha nincs WS adat, vagy a pár nincs feliratkoztatva, REST API hívás
            # A KrakenAPIClient.get_current_price metódusa már kezeli a REST hívást, ha kell
            price_rest = self.api.get_current_price(pair_altname)
            if price_rest is not None and price_rest > 0:
                # logger.debug(f"Price for {pair_altname} from REST API: {price_rest}")
                return float(price_rest)

            # logger.warning(f"Could not retrieve valid price for {pair_altname} from any source.")
            return None

        except Exception as e:
            logger.error(f"❌ Price fetch error for {pair_altname}: {e}", exc_info=True)
            return None

    def _get_real_current_price(self, pair: str, entry_price: float = None) -> float:
        """Get real current price from API"""
        try:
            # A 'pair' itt feltételezhetően altname formátumú (pl. XBTUSD)
            current_price = self.get_current_price_for_pair(pair)
            if current_price and current_price > 0:
                return current_price
            # Fallback only if API fails
            logger.warning(f"API price fetch failed for {pair} or returned invalid price. Using fallback.")
            return entry_price if entry_price else 1000.0 # Fallback to entry_price or a default
        except Exception as e:
            logger.error(f"Real price fetch failed for {pair}: {e}", exc_info=True)
            return entry_price if entry_price else 1000.0 # Fallback to entry_price or a default


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

        pair_display_name = pair_to_chart_altname # Alapértelmezés, ha nincs a listában
        if self.pair_list and self.pair_list.currentItem():
            pair_display_name = self.pair_list.currentItem().text() # Pl. "XBT/USD"


        current_time_str = pd.Timestamp.now().strftime('%H:%M:%S')
        websocket_active_for_pair = False
        if hasattr(self.api, 'ws_client') and self.api.ws_client and \
           hasattr(self.api.ws_client, 'is_pair_subscribed') and self.api.ws_client.is_pair_subscribed(pair_to_chart_altname):
            websocket_active_for_pair = True

        # Státusz címkék frissítése
        live_trading_panel_active = False
        if self.live_panel and hasattr(self.live_panel, 'is_trading_active'):
            live_trading_panel_active = self.live_panel.is_trading_active()

        label_style_live_ws = "font-size: 11px; font-weight: bold; color: #00FF00;" # Zöld
        label_style_ws_ready = "font-size: 11px; font-weight: bold; color: #00AAFF;" # Kék
        label_style_live_rest = "font-size: 11px; font-weight: bold; color: #27AE60;" # Sötétebb zöld
        label_style_offline_rest = "font-size: 11px; font-weight: bold; color: #FFAA00;" # Narancs
        label_style_no_data = "font-size: 11px; font-weight: bold; color: #FF4444;" # Piros

        current_indicator_text = f"🎯 {pair_display_name} - {current_time_str}"
        current_live_status_text = "N/A"
        current_live_status_style = label_style_no_data


        if websocket_active_for_pair:
            if live_trading_panel_active: # Live trading és WS is aktív
                current_live_status_text = "🟢 LIVE+WS"
                current_live_status_style = label_style_live_ws
            else: # Csak WS aktív, live trading nem
                current_live_status_text = "📡 WS-READY"
                current_live_status_style = label_style_ws_ready
        else: # Nincs aktív WS kapcsolat erre a párra
            if live_trading_panel_active: # Live trading aktív, de WS nem (REST)
                current_live_status_text = "🟠 LIVE (REST)"
                current_live_status_style = label_style_live_rest
            else: # Sem live trading, sem WS (REST alapon offline)
                current_live_status_text = "⚪ OFFLINE (REST)"
                current_live_status_style = label_style_offline_rest


        if self.indicator_label:
            self.indicator_label.setText(current_indicator_text)
        if self.live_status_label:
            self.live_status_label.setText(current_live_status_text)
            self.live_status_label.setStyleSheet(current_live_status_style)

        # Chart címének beállítása még az API hívás előtt
        data_source_for_title = "WS" if websocket_active_for_pair else "REST"
        if self.chart_widget:
            self.chart_widget.setTitle(f"{pair_display_name} - 1 Min Chart ({data_source_for_title}) - Loading...", color='gray')


        try:
            # logger.debug(f"[UI_CHART] Requesting OHLC for altname: {pair_to_chart_altname}")
            # Az API hívás limit paraméterrel, hogy ne kérjünk le túl sok adatot a chartra
            raw_ohlc_data = self.api.get_ohlc(pair_to_chart_altname, interval=1, limit=120) # Utolsó 120 gyertya (2 óra)

            if not raw_ohlc_data or not isinstance(raw_ohlc_data, dict):
                logger.warning(f"[UI_CHART] No OHLC data received for {pair_to_chart_altname}")
                self._show_chart_error(f"No data for {pair_display_name}")
                return

            # A Kraken API a pair altname-jét adja vissza kulcsként, vagy egy 'last' kulcsot is tartalmazhat
            pair_data_key = pair_to_chart_altname
            if pair_to_chart_altname not in raw_ohlc_data and len(raw_ohlc_data) == 1: # Ha csak egy kulcs van, az lesz az
                 pair_data_key = next(iter(raw_ohlc_data))
            elif pair_to_chart_altname not in raw_ohlc_data and 'last' in raw_ohlc_data : # Néha 'last' a kulcs
                # Ebben az esetben a raw_ohlc_data[pair_data_key] maga a lista
                 pass # A pair_data_key már jó, ha az altname nincs benne
            elif pair_to_chart_altname not in raw_ohlc_data:
                logger.warning(f"[UI_CHART] OHLC data received, but key '{pair_to_chart_altname}' not found. Keys: {list(raw_ohlc_data.keys())}")
                self._show_chart_error(f"Data key error for {pair_display_name}")
                return


            ohlc_values = raw_ohlc_data.get(pair_data_key)

            if not ohlc_values or not isinstance(ohlc_values, list) or len(ohlc_values) == 0:
                logger.warning(f"[UI_CHART] Empty or invalid OHLC values for {pair_display_name} (key: {pair_data_key})")
                self._show_chart_error(f"Empty data for {pair_display_name}")
                return

            # logger.debug(f"[UI_CHART] Processing {len(ohlc_values)} candles for {pair_display_name} (API key: {pair_data_key})")

            try:
                df = pd.DataFrame(ohlc_values, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
                df = df.astype({"time": int, "open": float, "high": float, "low": float, "close": float})
                df['time_dt'] = pd.to_datetime(df['time'], unit='s') # Külön oszlop a datetime objektumoknak
                # df = df.set_index('time_dt') # Indexelés datetime objektummal
                df_display = df # A limit miatt már a megfelelő mennyiségű adatunk van
            except Exception as e_df:
                logger.error(f"[UI_CHART] DataFrame processing failed for {pair_display_name}: {e_df}", exc_info=True)
                self._show_chart_error(f"Data processing error for {pair_display_name}")
                return

            if self.chart_widget:
                self.chart_widget.clear()
                self.chart_widget.setTitle(f"{pair_display_name} - 1 Min Chart ({data_source_for_title}) - {current_time_str}", color='white')

                time_values_ts = df_display['time'].values # Unix timestamp floatként a plotoláshoz
                close_array = df_display['close'].values

                self.chart_widget.plot(time_values_ts, close_array, pen=pg.mkPen(color='#00AAFF', width=2), name='Price')

                # Indikátorok hozzáadása (SMA, Bollinger)
                if len(df_display) >= 20:
                    try:
                        sma_20 = df_display['close'].rolling(20, min_periods=1).mean()
                        self.chart_widget.plot(time_values_ts, sma_20.values, pen=pg.mkPen(color='#FFDB58', width=1, style=Qt.DotLine), name='SMA20') # Sárga, pontozott

                        bb_std = df_display['close'].rolling(20, min_periods=1).std()
                        bb_upper = sma_20 + (bb_std * 2)
                        bb_lower = sma_20 - (bb_std * 2)

                        # Kitöltés a Bollinger szalagok között
                        fill_upper_item = pg.PlotDataItem(time_values_ts, bb_upper.values, pen=pg.mkPen(color=(255,107,53,80))) # Javítva: fill_upper -> fill_upper_item
                        fill_lower_item = pg.PlotDataItem(time_values_ts, bb_lower.values, pen=pg.mkPen(color=(255,107,53,80))) # Javítva: fill_lower -> fill_lower_item
                        fill = pg.FillBetweenItem(fill_lower_item, fill_upper_item, brush=pg.mkBrush(255,107,53,30)) # Javítva: Fill μεταξύ -> FillBetweenItem
                        self.chart_widget.addItem(fill)

                        self.chart_widget.plot(time_values_ts, bb_upper.values, pen=pg.mkPen(color=(255,107,53,150), style=Qt.DashLine), name='BB Upper')
                        self.chart_widget.plot(time_values_ts, bb_lower.values, pen=pg.mkPen(color=(255,107,53,150), style=Qt.DashLine), name='BB Lower')
                    except Exception as e_ta:
                        logger.warning(f"[UI_CHART] Technical indicators calculation failed for {pair_display_name}: {e_ta}")

                self._format_chart_time_axis(df_display, time_values_ts) # df_display és time_values_ts átadása
                self._add_position_markers_to_chart(pair_to_chart_altname, time_values_ts) # Időértékek átadása a markerekhez
                self._update_price_display_on_dashboard(pair_display_name, df_display, websocket_active_for_pair)

            # logger.debug(f"[UI_CHART] Chart updated successfully for {pair_display_name}")

        except Exception as e:
            logger.error(f"[UI_CHART] Chart update failed for {pair_display_name}: {e}", exc_info=True)
            self._show_chart_error(f"Chart error for {pair_display_name}: {str(e)[:50]}")


    def _show_chart_error(self, error_message: str):
        """Hibaüzenet megjelenítése a diagramon."""
        if self.chart_widget:
            self.chart_widget.clear()
            self.chart_widget.setTitle(error_message, color='red', size='10pt')
            # Esetleg egy szöveges elem hozzáadása a chart közepére
            text_item = pg.TextItem(error_message, color=(200, 0, 0), anchor=(0.5, 0.5))
            # Hogy középre kerüljön, ismernünk kell a chart határait, vagy egy viewboxot használunk
            # Egyszerűbb megoldás a setTitle.

    def _format_chart_time_axis(self, df_with_datetime: pd.DataFrame, time_values_timestamps: list):
        """Időtengely formázása a diagramon, hogy olvashatóbb legyen."""
        try:
            # MODIFIED LINE:
            if self.chart_widget and not df_with_datetime.empty and len(time_values_timestamps) > 0:
                axis = self.chart_widget.getAxis('bottom')

                # Csak akkor formázzuk, ha van elég adat a tengelyhez
                if len(df_with_datetime) < 2 or len(time_values_timestamps) < 2:
                    axis.setTicks(None) # Alapértelmezett tickek
                    return

                # Tickek kiválasztása (pl. minden 15. vagy 30. gyertya)
                num_candles = len(df_with_datetime)
                tick_spacing = 1  # Alapértelmezett, ha kevés a gyertya
                if num_candles > 100: tick_spacing = 30
                elif num_candles > 50: tick_spacing = 15
                elif num_candles > 20: tick_spacing = 5

                tick_indices = list(range(0, num_candles, tick_spacing))
                if num_candles -1 not in tick_indices and num_candles > 1 : # Utolsó tick hozzáadása
                    tick_indices.append(num_candles-1)

                tick_strings = [df_with_datetime['time_dt'].iloc[idx].strftime('%H:%M') for idx in tick_indices if idx < num_candles]
                tick_values_ts = [time_values_timestamps[idx] for idx in tick_indices if idx < len(time_values_timestamps)]

                if tick_values_ts and tick_strings:
                    ticks = [list(zip(tick_values_ts, tick_strings))]
                    axis.setTicks(ticks)
                else: # Ha valamiért üres lenne a tick lista
                    axis.setTicks(None)


        except Exception as e:
            logger.warning(f"Time axis formatting failed: {e}", exc_info=True)
            if hasattr(self, 'chart_widget') and self.chart_widget:
                 self.chart_widget.getAxis('bottom').setTicks(None) # Hiba esetén alapértelmezett


    def _add_position_markers_to_chart(self, pair_altname: str, time_values_ts: list):
        """Pozíció nyitási és zárási pontok, valamint SL/TP szintek hozzáadása a diagramhoz."""
        try:
            if not self.chart_widget: return

            # Aktuális nyitott pozíció
            current_pos_data = self.position.get_position(pair_altname)
            if current_pos_data and current_pos_data.get('entry_price'):
                entry_price = current_pos_data['entry_price']
                side = current_pos_data.get('side', 'buy')
                entry_time_unix = current_pos_data.get('entry_time_unix') # Ha tároljuk a nyitás idejét

                marker_color = QColor(0, 255, 0, 180) if side == 'buy' else QColor(255, 0, 0, 180) # Zöld/Piros
                sl = current_pos_data.get('stop_loss')
                tp = current_pos_data.get('take_profit')

                # Entry line
                entry_line = pg.InfiniteLine(pos=entry_price, angle=0,
                                             pen=pg.mkPen(marker_color, width=2, style=Qt.SolidLine),
                                             label=f'Entry: {entry_price:.4f}',
                                             labelOpts={'color': marker_color, 'movable': True, 'position': 0.95})
                self.chart_widget.addItem(entry_line)

                # Entry marker (nyíl) - ha van nyitási idő
                if entry_time_unix and time_values_ts and len(time_values_ts) > 0 and entry_time_unix >= time_values_ts[0] and entry_time_unix <= time_values_ts[-1]: # Ellenőrzés, hogy az idő a charton van-e
                    arrow_symbol = 's' # Négyzet, lehetne 't1' (felfele háromszög) vagy 't' (lefele)
                    arrow_brush = pg.mkBrush(marker_color)
                    entry_scatter = pg.ScatterPlotItem(x=[entry_time_unix], y=[entry_price],
                                                       symbol=arrow_symbol, size=12, pen=pg.mkPen(None), brush=arrow_brush)
                    self.chart_widget.addItem(entry_scatter)


                if sl:
                    sl_line = pg.InfiniteLine(pos=sl, angle=0,
                                              pen=pg.mkPen(QColor(255,165,0,180), width=1, style=Qt.DashLine), # Narancs
                                              label=f'SL: {sl:.4f}',
                                              labelOpts={'color': QColor(255,165,0), 'movable': True, 'position': 0.9})
                    self.chart_widget.addItem(sl_line)
                if tp:
                    tp_line = pg.InfiniteLine(pos=tp, angle=0,
                                              pen=pg.mkPen(QColor(60,179,113,180), width=1, style=Qt.DashLine), # Tengerzöld
                                              label=f'TP: {tp:.4f}',
                                              labelOpts={'color': QColor(60,179,113), 'movable': True, 'position': 0.85})
                    self.chart_widget.addItem(tp_line)

            # TODO: Korábbi zárt pozíciók markerjeinek hozzáadása (ha a trade_logger tárolja az időpontokat)

        except Exception as e:
            logger.warning(f"Position markers drawing failed for {pair_altname}: {e}", exc_info=True)

    def _update_price_display_on_dashboard(self, pair_display_name: str, df_display: pd.DataFrame, websocket_active_for_pair: bool):
        """Árfolyamadatok frissítése a dashboardon."""
        try:
            if not df_display.empty and self.dashboard:
                pair_altname_for_price = self.current_chart_pair # Ez már altname

                current_price_ws = None
                if websocket_active_for_pair:
                    current_price_ws = self.get_current_price_for_pair(pair_altname_for_price)

                current_price_chart = df_display['close'].iloc[-1]
                display_price = current_price_ws if current_price_ws is not None and current_price_ws > 0 else current_price_chart

                self.dashboard.update_field("Current Price", f"${display_price:.6f}") # Pontosabb kijelzés
                data_source_text = "WebSocket Live" if websocket_active_for_pair and current_price_ws is not None else "REST API (Chart)"
                self.dashboard.update_field("Price Data Source", data_source_text)
                # A "Selected Pair" már az on_pair_changed-ben frissül
        except Exception as e:
            logger.warning(f"Dashboard price display update failed for {pair_display_name}: {e}", exc_info=True)


    def enhanced_update_cycle(self): # Korábban main_update_cycle
        """Fő frissítési ciklus a GUI elemekhez és adatokhoz."""
        self.update_counter += 1
        # logger.debug(f"Enhanced update cycle: {self.update_counter}")
        try:
            self.update_chart()
            self.refresh_balance()
            self.refresh_open_trades()

            if self.update_counter % 2 == 0: # Minden második ciklusban
                self.update_ai_performance_display()
                if self.stats_panel and hasattr(self.stats_panel, 'update_stats'):
                    self.stats_panel.update_stats()

            if self.update_counter % 3 == 0: # Minden harmadik ciklusban
                self.refresh_stats() # Fő statisztikák frissítése
                if self.position_list_widget and hasattr(self.position_list_widget, 'update_history'):
                    self.position_list_widget.update_history() # Pozíció előzmények frissítése

            # LiveTradingThread paramétereinek frissítése a GUI-n, ha a szál fut
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


    def get_all_coin_data(self) -> list: # Visszatérési típus megadása
        """Coin adatok lekérdezése a scoring panel számára, volumen alapján szűrve."""
        try:
            if not (hasattr(self, 'api') and self.api and hasattr(self.api, 'get_market_data')):
                logger.warning("API client or get_market_data method not available. Using fallback coin data.")
                return self._get_fallback_coin_data()

            logger.info("📊 Getting real coin data with market data for scoring panel...")
            market_data_list = self.api.get_market_data() # Ez a metódus adja vissza a szűrt és pontozott listát

            if not market_data_list:
                logger.warning("⚠️ No market data received from API. Using fallback coin data.")
                return self._get_fallback_coin_data()

            # Az API válasz már tartalmazhatja a szükséges mezőket (pl. 'score', 'volume_usd')
            # Csak átalakítjuk a scoring panel számára elfogadható formátumra, ha szükséges.
            ui_coin_data = []
            for coin_info in market_data_list[:30]: # Max 30 elem a scoring panelnek
                # Ellenőrizzük a szükséges mezőket
                if not all(k in coin_info for k in ['symbol', 'price', 'volume_usd']):
                    logger.warning(f"Skipping coin due to missing data: {coin_info.get('symbol', 'UNKNOWN')}")
                    continue

                coin_entry = {
                    'symbol': coin_info.get('symbol'),
                    'pair': coin_info.get('symbol'), # A scoring panel ezt várhatja
                    'close': coin_info.get('price'),
                    'volume_usd': coin_info.get('volume_usd'),
                    'score_from_scanner': coin_info.get('score'), # Ha az API adja
                    # Dummy értékek a többihez, ha az API nem szolgáltatja
                    'rsi_15m': coin_info.get('rsi_15m', random.uniform(30, 70)),
                    'correl_btc': coin_info.get('correl_btc', random.uniform(0.5, 0.95)),
                    # ... egyéb mezők, amiket a scoring panel használ ...
                }
                # Csak a nem None értékű mezőket adjuk hozzá, hogy elkerüljük a hibákat
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


    def _get_fallback_coin_data(self) -> list: # Visszatérési típus
        """Fallback coin adatok szolgáltatása, ha az API nem elérhető."""
        fallback_data = [
            {'symbol': 'XBTUSD', 'pair': 'XBTUSD', 'close': 60000, 'volume_usd': 50000000, 'score_from_scanner': 0.8, 'rsi_15m': 55.0, 'correl_btc': 1.0},
            {'symbol': 'ETHUSD', 'pair': 'ETHUSD', 'close': 3000, 'volume_usd': 25000000, 'score_from_scanner': 0.7, 'rsi_15m': 48.2, 'correl_btc': 0.85},
            {'symbol': 'ADAUSD', 'pair': 'ADAUSD', 'close': 0.45, 'volume_usd': 2500000, 'score_from_scanner': 0.6, 'rsi_15m': 38.5, 'correl_btc': 0.75},
            {'symbol': 'SOLUSD', 'pair': 'SOLUSD', 'close': 150.0, 'volume_usd': 8000000, 'score_from_scanner': 0.75, 'rsi_15m': 65.2, 'correl_btc': 0.80},
        ]
        logger.info(f"🔄 Using UI fallback data for {len(fallback_data)} pairs")
        return fallback_data

    # --- Live Trading Actions ---
    def get_live_trading_settings(self) -> dict: # Visszatérési típus
        """Aktuális live trading beállítások lekérdezése a LiveTradingPanel-ből."""
        if self.live_panel and hasattr(self.live_panel, 'get_current_settings'):
            settings = self.live_panel.get_current_settings()
            if settings and isinstance(settings, dict):
                return settings
            else:
                logger.warning("LiveTradingPanel.get_current_settings did not return a valid dict. Using default.")
        else:
            logger.warning("LiveTradingPanel not available or no get_current_settings method. Using default settings.")

        # Alapértelmezett beállítások, ha a panel nem elérhető vagy hibás
        return {
            'auto_trading_enabled': False, # Fontos, hogy alapból ne legyen engedélyezve
            'position_size_usd': 50.0,
            'max_active_trades': 1,
            'stop_loss_pct': 4.0,
            'take_profit_target_pct': 0.8,
            'min_score_for_auto_trade': 0.7,
            'scanner_min_volume_usd': 500000,
            'scanner_min_score_candidate': 0.5,
            'scanner_pump_threshold_pct': 10.0,
            'manual_position_size': 50.0, # Manuális kereskedéshez
            'manual_sl_pct': 4.0,
            'manual_tp_pct': 0.8,
            'scan_interval_sec': 180, # Scan intervallum
        }

    def initialize_live_trading_system(self):
        """Live trading rendszer kezdeti állapotának beállítása."""
        try:
            logger.info("🚀 Initializing Live Trading System UI state...")
            # Kezdetben a live trading nem aktív
            self.live_trading_active = False
            if self.ai_status_label:
                self.ai_status_label.setText(f"Live Trading: READY ({'Adv.' if self.ai_mode_active else 'Basic'})")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: #00AAFF;") # Kék
            if self.ai_toggle_btn:
                self.ai_toggle_btn.setText("🚀 Start Live Trading")
                self.ai_toggle_btn.setEnabled(True) # Engedélyezzük a gombot

            logger.info("✅ Live Trading System UI initialized, ready to be started.")
        except Exception as e:
            logger.error(f"❌ Live trading system UI initialization failed: {e}", exc_info=True)
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: INIT ERROR")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: red;")

    def toggle_live_trading(self):
        """Live trading indítása/leállítása a UI gombról."""
        try:
            if not self.live_trading_active: # Ha nem fut, indítjuk
                self.start_live_trading_action()
            else: # Ha fut, leállítjuk
                self.stop_live_trading_action()
        except Exception as e:
            logger.error(f"❌ Live trading toggle failed: {e}", exc_info=True)
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: TOGGLE ERROR")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: red;")


    def start_live_trading_action(self): # Korábban start_live_trading
        """Live trading indítása és a kapcsolódó UI elemek frissítése."""
        if self.live_trading_active:
            logger.info("Live trading is already active.")
            return
        try:
            logger.info("🚀 Attempting to start Live Trading System...")
            if not self.live_trading_thread or not self.live_trading_thread.isRunning():
                self.live_trading_thread = LiveTradingThread(self) # Új szál példányosítása
                # Signálok összekötése
                self.live_trading_thread.opportunity_found.connect(self.handle_live_opportunity)
                self.live_trading_thread.trade_executed.connect(self.handle_live_trade)
                self.live_trading_thread.ai_decision.connect(self.handle_ai_decision)
                self.live_trading_thread.error_occurred.connect(self.handle_thread_error)
                self.live_trading_thread.scan_progress.connect(self.handle_scan_progress)
                self.live_trading_thread.pump_detected.connect(self.handle_pump_detection)
            else:
                logger.info("LiveTradingThread already exists and might be running. Will try to use existing.")


            # Aktuális beállítások átadása a szálnak
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

                # Paraméterek kijelzése a GUI-n
                if self.min_volume_display_label: self.min_volume_display_label.setText(f"Min Volume USD: {self.live_trading_thread.min_volume_usd:,.0f}")
                if self.min_score_display_label: self.min_score_display_label.setText(f"Min Score Candidate: {self.live_trading_thread.min_score_threshold:.2f}")
                if self.pump_thresh_display_label: self.pump_thresh_display_label.setText(f"Pump Threshold %: {self.live_trading_thread.pump_threshold*100:.1f}")


            self.live_trading_thread.start_trading() # A QThread start() metódusát hívja, ami a run()-t indítja
            self.live_trading_active = True

            # UI frissítése
            if self.ai_toggle_btn: self.ai_toggle_btn.setText("🛑 Stop Live Trading")
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: ACTIVE")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: #27AE60;") # Zöld
            if self.live_panel and hasattr(self.live_panel, 'set_trading_active_status'): # Ha a panelnek van ilyen metódusa
                self.live_panel.set_trading_active_status(True)

            logger.info("✅ Live Trading System started with current settings.")
            self.ai_decision_log_append("📈 Live Trading System Started.")

        except Exception as e:
            logger.error(f"❌ Failed to start Live Trading System: {e}", exc_info=True)
            self.live_trading_active = False # Hiba esetén visszaállítjuk
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: START ERROR")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: red;")
            self.ai_decision_log_append(f"❌ Live Trading Start Error: {str(e)[:50]}")


    def stop_live_trading_action(self): # Korábban stop_live_trading
        """Live trading leállítása és a kapcsolódó UI elemek frissítése."""
        if not self.live_trading_active:
            logger.info("Live trading is not active.")
            return
        try:
            logger.info("🔴 Attempting to stop Live Trading System...")
            if self.live_trading_thread and self.live_trading_thread.isRunning():
                self.live_trading_thread.stop_trading()
                # Nem várunk itt a szál leállására, a closeEvent-ben fogunk
            self.live_trading_active = False

            # UI frissítése
            if self.ai_toggle_btn: self.ai_toggle_btn.setText("🚀 Start Live Trading")
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: STOPPED")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: #E74C3C;") # Pirosas
            if self.scan_progress_label: self.scan_progress_label.setText("Scan Progress: Idle")
            if self.live_panel and hasattr(self.live_panel, 'set_trading_active_status'):
                self.live_panel.set_trading_active_status(False)

            logger.info("✅ Live Trading System stopped.")
            self.ai_decision_log_append("📉 Live Trading System Stopped.")

        except Exception as e:
            logger.error(f"❌ Failed to stop Live Trading System: {e}", exc_info=True)
            # Nem állítjuk vissza a live_trading_active-ot, mert a szándék a leállítás volt
            if self.ai_status_label:
                self.ai_status_label.setText("Live Trading: STOP ERROR")
                self.ai_status_label.setStyleSheet("font-weight: bold; color: red;")
            self.ai_decision_log_append(f"❌ Live Trading Stop Error: {str(e)[:50]}")

    def emergency_stop_all(self):
        """Vészleállítás: minden aktivitás azonnali beszüntetése."""
        try:
            logger.critical("🚨 EMERGENCY STOP ACTIVATED!")
            if self.live_trading_active:
                self.stop_live_trading_action() # Először a normál leállítási folyamat

            # Minden nyitott pozíció azonnali zárása piaci áron (ha a PositionManager támogatja)
            if self.position and hasattr(self.position, 'close_all_positions_market'):
                closed_info = self.position.close_all_positions_market()
                logger.info(f"🚨 Emergency close all positions: {closed_info}")
                self.ai_decision_log_append(f"🚨 EMERGENCY CLOSED POSITIONS: {len(closed_info)} closed.")
            elif self.position and hasattr(self.position, 'get_all_positions'): # Manuális zárás, ha nincs market close
                all_pos = self.position.get_all_positions()
                for pair_alt in list(all_pos.keys()):
                    price = self._get_real_current_price(pair_alt) # Valós ár használata
                    if price:
                        self.position.close_position(pair_alt, price, reason="EMERGENCY_STOP")
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
                self.ai_toggle_btn.setEnabled(False) # Tiltsuk le a start gombot vészleállítás után

            logger.info("🚨 Emergency stop procedure completed.")
            self.ai_decision_log_append("🚨 EMERGENCY STOP ACTIVATED - ALL SYSTEMS HALTED.")

        except Exception as e:
            logger.error(f"❌ Emergency stop procedure failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ EMERGENCY STOP FAILED: {str(e)[:50]}")


    def update_trading_settings(self, settings_dict: dict):
        """Kereskedési beállítások frissítése a LiveTradingPanel-ből érkező jelzés alapján."""
        try:
            logger.info(f"⚙️ MainWindow received settings update from UI: {settings_dict}")

            # Beállítások alkalmazása a LiveTradingThread-re, ha fut
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

            # UI elemek frissítése az új beállításokkal (ha vannak ilyen dedikált kijelzők)
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
                    self.ai_decision_log_append("🔄 Force Rescan Requested.")
                else:
                    # Ha nincs dedikált force_scan_now, akkor csak egy progress üzenetet küldünk
                    if hasattr(self.live_trading_thread, 'scan_progress') and hasattr(self.live_trading_thread.scan_progress, 'emit'):
                        self.live_trading_thread.scan_progress.emit("Force scan requested (manual trigger).")
                    logger.warning("🔄 (Note: LiveTradingThread does not have explicit force_scan_now method. Emitting progress.)")
                    self.ai_decision_log_append("🔄 Force Rescan (No dedicated method in thread).")
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


    # --- Signal Handlers from LiveTradingThread ---
    def handle_live_opportunity(self, opportunity_data: dict):
        """A LiveTradingThread által talált kereskedési lehetőség kezelése."""
        try:
            pair = opportunity_data.get('pair', 'Unknown')
            score = opportunity_data.get('score', 0.0)
            reason = opportunity_data.get('reason', 'N/A')

            logger.info(f"🚀 Live Opportunity Received by MainWindow: {pair} (Score: {score:.3f}) Reason: {reason}")

            if self.ai_decision_display: # Középső panel kijelzője
                self.ai_decision_display.setText(f"🎯 Candidate: {pair} Score: {score:.2f}")

            # Frissítjük a legjobb jelöltek listáját a GUI-n
            if self.live_trading_thread and hasattr(self.live_trading_thread, 'final_candidates') and self.live_trading_thread.final_candidates:
                opp_text = f"🏆 TOP SCAN CANDIDATES - {time.strftime('%H:%M:%S')} 🏆\n"
                opp_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                for i, cand in enumerate(self.live_trading_thread.final_candidates[:3]): # Max 3 jelölt kijelzése
                    pump_ind = "🔥" if cand.get('pump_detected') else "📊"
                    opp_text += (f"{i+1}. {pump_ind} {cand.get('altname','N/A')}: Score {cand.get('final_score',0):.3f}, "
                                 f"Vol ${cand.get('volume_usd',0):,.0f}\n")
                if self.ai_opportunities_display: self.ai_opportunities_display.setPlainText(opp_text)
            elif self.ai_opportunities_display:
                self.ai_opportunities_display.setPlainText("Waiting for next scan results from LiveTradingThread...")

            self.ai_decision_log_append(f"🎯 Scan Found: {pair} Score: {score:.3f} ({reason[:30]})")

            # Döntés a kereskedés végrehajtásáról (ha az auto_trading engedélyezve van)
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

            if len(self.position.get_all_positions()) >= max_active_trades:
                logger.warning(f"⚠️ Max active positions ({max_active_trades}) reached. Cannot execute new trade for {pair_altname}.")
                self.ai_decision_log_append(f"⚠️ Max Pos Skip: {pair_altname}")
                return

            # Oldal meghatározása (buy/sell)
            signal_value = trade_setup_data.get('signal', 'buy') # Alapértelmezetten 'buy'
            side = str(signal_value).lower()
            if side not in ['buy', 'sell']:
                logger.error(f"❌ Invalid trade side '{side}' for {pair_altname}. Skipping.")
                self.ai_decision_log_append(f"❌ Invalid Side: {pair_altname} ({side})")
                return

            # Ár, volumen, SL, TP meghatározása a trade_setup_data-ból
            # Ezeket a LiveTradingThread-nek kellene szolgáltatnia
            entry_price = trade_setup_data.get('entry_price')
            volume = trade_setup_data.get('volume')
            stop_loss = trade_setup_data.get('stop_loss')
            take_profit = trade_setup_data.get('take_profit')
            position_size_usd = trade_setup_data.get('position_size_usd', current_settings.get('position_size_usd', 50.0)) # Pozíció méret USD-ben

            if entry_price is None or entry_price <= 0:
                logger.error(f"❌ Invalid or missing entry price for {pair_altname}. Cannot execute.")
                self.ai_decision_log_append(f"❌ Exec Fail: No Price {pair_altname}")
                return

            if volume is None or volume <= 0: # Ha nincs volumen, számoljuk a position_size_usd alapján
                if position_size_usd > 0 and entry_price > 0:
                    volume = position_size_usd / entry_price
                else:
                    logger.error(f"❌ Invalid or missing volume/position_size_usd for {pair_altname}. Cannot execute.")
                    self.ai_decision_log_append(f"❌ Exec Fail: No Volume {pair_altname}")
                    return

            # SL/TP számítása, ha nincsenek megadva (pl. százalékos alapon)
            if stop_loss is None:
                sl_pct = current_settings.get('stop_loss_pct', 4.0) / 100.0
                stop_loss = entry_price * (1 - sl_pct) if side == 'buy' else entry_price * (1 + sl_pct)
            if take_profit is None:
                tp_pct = current_settings.get('take_profit_target_pct', 0.8) / 100.0
                take_profit = entry_price * (1 + tp_pct) if side == 'buy' else entry_price * (1 - tp_pct)


            success = self.position.open_position(
                pair=pair_altname,
                side=side,
                entry_price=entry_price,
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_time_unix=time.time() # Pozíció nyitási idejének hozzáadása
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
            action = trade_data.get('action', 'N/A') # Pl. OPEN, CLOSE, UPDATE_SLTP
            status = trade_data.get('status', 'updated') # Pl. executed, failed, pending
            pnl = trade_data.get('pnl', 0.0)
            self.last_trade_pnl = pnl # Utolsó P&L eltárolása

            if status == 'closed': # Ha egy pozíció lezárult
                if pnl > 0:
                    self.ai_performance['successful_trades'] += 1
                self.ai_performance['ai_profit'] += pnl
                # A total_decisions már a nyitáskor növekszik

            self.update_ai_performance_display()
            self.refresh_open_trades()
            self.refresh_balance()

            log_entry = f"TRADE EVENT: {pair} - {action} - Status: {status}, P&L: ${pnl:.2f}"
            self.ai_decision_log_append(log_entry)
            if self.ai_decision_label: # Bal oldali panel
                self.ai_decision_label.setText(f"Last Trade: {pair} {action} ${pnl:.2f}")


        except Exception as e:
            logger.error(f"❌ Live trade handling in MainWindow failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ Trade Handle Error: {str(e)[:30]}")

    def handle_ai_decision(self, decision_data: dict):
        """A LiveTradingThread által küldött általános AI döntési esemény kezelése."""
        try:
            self.ai_decisions.append(decision_data)
            if len(self.ai_decisions) > 100: # Max 100 döntés tárolása
                self.ai_decisions = self.ai_decisions[-100:]

            decision_text = decision_data.get('decision', 'No decision text')
            pair = decision_data.get('pair', 'N/A')
            score = decision_data.get('score', 0.0)

            if self.ai_decision_label: # Bal oldali panel
                self.ai_decision_label.setText(f"Last Event: {decision_text[:35]}") # Rövidített szöveg

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
        if self.ai_decision_display: # Középső panel
            self.ai_decision_display.setText(f"❌ Scanner Error: {error_message[:100]}")

        self.ai_decision_log_append(f"THREAD ERROR: {error_message}")


    def handle_scan_progress(self, progress_message: str):
        """A LiveTradingThread által küldött szkennelési folyamat üzenetének kezelése."""
        if self.scan_progress_label: # Bal oldali panel
            self.scan_progress_label.setText(f"Scan: {progress_message[:40]}") # Rövidített üzenet
        # Ha a szkennelés befejeződött, vagy fontosabb üzenet van, azt a fő decision display-re is kiírhatjuk
        if "✅ Scan completed" in progress_message or "⚠️ Scan took" in progress_message or "Forcing" in progress_message:
             if self.ai_decision_display: self.ai_decision_display.setText(progress_message) # Középső panel
        # Nem logoljuk minden progress üzenetet az ai_decision_log-ba, csak a fontosabbakat


    def handle_pump_detection(self, pump_data: dict):
        """A LiveTradingThread által észlelt pump esemény kezelése."""
        try:
            pair = pump_data.get('pair', 'Unknown')
            pump_pct = pump_data.get('pump_pct', 0.0)
            volume_usd = pump_data.get('volume_usd', 0)
            logger.info(f"🔥 PUMP DETECTED (Signal to MainWindow): {pair} {pump_pct:.2f}% Volume: ${volume_usd:,.0f}")

            if self.ai_decision_display: # Középső panel
                self.ai_decision_display.setText(f"🔥 PUMP ALERT: {pair} {pump_pct:.1f}%")
                # Esetleg a háttér villogtatása vagy más vizuális jelzés

            log_entry = f"PUMP ALERT: {pair} {pump_pct:.2f}% (Vol: ${volume_usd:,.0f})"
            self.ai_decision_log_append(log_entry)

        except Exception as e:
            logger.error(f"❌ Pump detection handling in MainWindow failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ Pump Handle Error: {str(e)[:30]}")

    # --- Position Monitoring & Management ---
    def monitor_positions_with_ai(self): # Korábban monitor_positions
        """Nyitott pozíciók figyelése és kezelése (SL/TP, esetleg AI alapú zárás)."""
        try:
            if not (self.position and hasattr(self.position, 'get_all_positions')):
                return

            open_positions = self.position.get_all_positions()
            if not open_positions:
                return

            # logger.debug(f"Monitoring {len(open_positions)} open positions...")
            for pair_altname, position_data in list(open_positions.items()): # list() másolat készítéséhez
                # Valós ár használata a pozíció monitorozásához
                current_price = self._get_real_current_price(pair_altname, position_data.get('entry_price'))
                if not current_price:
                    logger.warning(f"⚠️ Cannot monitor position for {pair_altname}, current price unavailable.")
                    continue

                self.monitor_single_position_standard_sl_tp(pair_altname, position_data, current_price)
                # TODO: Itt lehetne hívni egy fejlettebb AI alapú pozíciókezelő logikát is,
                # pl. self.ai_position_manager.check_position(pair_altname, position_data, current_price)

        except Exception as e:
            logger.error(f"❌ Position monitoring cycle failed: {e}", exc_info=True)

    def monitor_single_position_standard_sl_tp(self, pair_altname: str, position_data: dict, current_price: float):
        """Egyetlen pozíció figyelése standard SL/TP alapján."""
        try:
            entry_price = position_data.get('entry_price')
            stop_loss = position_data.get('stop_loss')
            take_profit = position_data.get('take_profit')
            side = position_data.get('side', 'buy').lower()

            if entry_price is None: return # Nincs mi alapján figyelni

            should_close = False
            close_reason = ""

            if side == 'buy':
                if stop_loss and current_price <= stop_loss:
                    should_close = True; close_reason = "STOP_LOSS_HIT"
                elif take_profit and current_price >= take_profit:
                    should_close = True; close_reason = "TAKE_PROFIT_HIT"
            elif side == 'sell':
                if stop_loss and current_price >= stop_loss: # Fordított logika shortnál
                    should_close = True; close_reason = "STOP_LOSS_HIT"
                elif take_profit and current_price <= take_profit: # Fordított logika shortnál
                    should_close = True; close_reason = "TAKE_PROFIT_HIT"

            if should_close:
                logger.info(f"🛡️ Standard monitor triggered close for {pair_altname}: {close_reason} at price {current_price:.4f}")
                self.close_position_programmatically(pair_altname, current_price, close_reason)

        except Exception as e:
            logger.error(f"❌ Standard SL/TP monitoring failed for {pair_altname}: {e}", exc_info=True)


    def close_position_programmatically(self, pair_altname: str, exit_price: float, reason: str): # Korábban close_position_with_ai
        """Pozíció zárása programatikusan (pl. SL/TP, AI döntés alapján)."""
        try:
            position_data_before_close = self.position.get_position(pair_altname) # Adatok lekérése zárás előtt
            if not position_data_before_close:
                logger.warning(f"⚠️ Attempted to close non-existent or already closed position: {pair_altname}")
                return

            closed_position_info = self.position.close_position(pair_altname, exit_price, reason=reason) # reason átadása

            if closed_position_info:
                # A PositionManager.close_position adja vissza a P&L-t és egyéb adatokat
                pnl = closed_position_info.get('pnl', 0.0)
                self.last_trade_pnl = pnl # Utolsó P&L frissítése
                side = closed_position_info.get('side', position_data_before_close.get('side', 'N/A'))
                entry_price_log = closed_position_info.get('entry_price', position_data_before_close.get('entry_price', 0.0))
                volume_log = closed_position_info.get('volume', position_data_before_close.get('volume', 0.0))


                self.trade_logger.log(
                    pair=pair_altname, side=side, entry_price=entry_price_log,
                    exit_price=exit_price, volume=volume_log, pnl=pnl,
                    reason=reason
                )

                # AI teljesítmény frissítése (a total_decisions már nyitáskor nőtt)
                if closed_position_info.get('status', 'closed') == 'closed': # Csak ha tényleg lezárult
                    if pnl > 0:
                        self.ai_performance['successful_trades'] += 1
                    self.ai_performance['ai_profit'] += pnl
                    # Az accuracy-t az update_ai_performance_display számolja

                if self.ai_decision_display: # Középső panel
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

    # --- Manual Trading Actions ---
    def execute_manual_trade(self):
        """Manuális kereskedés végrehajtása a kiválasztott páron."""
        try:
            if not self.pair_list or not self.pair_list.currentItem():
                logger.warning("⚠️ No pair selected for manual trade.")
                self.ai_decision_log_append("Manual Trade: No pair selected.")
                return

            pair_display_name = self.pair_list.currentItem().text() # Pl. "XBT/USD"
            pair_altname = pair_display_name.replace("/", "")      # Pl. "XBTUSD"
            current_price = self._get_real_current_price(pair_altname) # Valós ár használata

            if not current_price or current_price <= 0:
                logger.error(f"❌ Could not get valid price for manual trade on {pair_altname}.")
                self.ai_decision_log_append(f"Manual Trade: No valid price for {pair_altname}.")
                return

            # Beállítások lekérdezése a manuális kereskedéshez
            trade_settings = self.get_live_trading_settings()
            position_size_usd = trade_settings.get('manual_position_size', 50.0)
            stop_loss_pct = trade_settings.get('manual_sl_pct', 4.0) / 100.0
            take_profit_pct = trade_settings.get('manual_tp_pct', 0.8) / 100.0

            # TODO: Lehetőség a felhasználónak kiválasztani a vételi/eladási oldalt
            side = "buy" # Egyelőre csak vétel

            volume = position_size_usd / current_price
            stop_loss = current_price * (1 - stop_loss_pct) if side == 'buy' else current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct) if side == 'buy' else current_price * (1 - take_profit_pct)

            success = self.position.open_position(
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
                self.ai_performance['total_decisions'] +=1 # A manuális trade is beleszámít
                self.update_ai_performance_display()
            else:
                logger.error(f"❌ Manual trade failed for: {pair_display_name} (PositionManager denied).")
                self.ai_decision_log_append(f"Manual Trade Fail (PM): {pair_display_name}")

        except Exception as e:
            logger.error(f"❌ Manual trade execution failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"Manual Trade Error: {str(e)[:30]}")

    def open_test_position(self): # Ez a metódus már létezik, a tartalma frissül
        """Teszt pozíció nyitása $50 értékben a kiválasztott páron."""
        try:
            if not self.pair_list or not self.pair_list.currentItem():
                logger.warning("⚠️ No pair selected for test position.")
                self.ai_decision_log_append("Test Pos: No pair selected.")
                return

            pair_display_name = self.pair_list.currentItem().text()
            pair_altname = pair_display_name.replace("/", "")
            logger.info(f"🧪 Opening test position for {pair_display_name} (alt: {pair_altname})")

            current_price = self._get_real_current_price(pair_altname) # Valós ár használata
            if not current_price or current_price <= 0:
                logger.error(f"❌ Could not get valid price for test position on {pair_altname}.")
                self.ai_decision_log_append(f"Test Pos: No price for {pair_altname}.")
                return

            position_size_usd = 50.0
            volume = position_size_usd / current_price

            # Százalékos SL/TP a teszt pozícióhoz
            sl_pct = 0.04  # 4%
            tp_pct = 0.008 # 0.8%
            side = "buy" # Teszt pozíció legyen mindig vétel

            stop_loss = current_price * (1 - sl_pct)
            take_profit = current_price * (1 + tp_pct)

            success = self.position.open_position(
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

            position_data = self.position.get_position(pair_altname)
            if not position_data:
                logger.warning(f"⚠️ No open position found for {pair_display_name} (alt: {pair_altname}) to close manually.")
                self.ai_decision_log_append(f"Manual Close: No open pos for {pair_display_name}.")
                return

            current_price = self._get_real_current_price(pair_altname, position_data.get('entry_price')) # Valós ár használata
            if not current_price or current_price <= 0:
                logger.error(f"❌ Could not get valid price for manually closing {pair_altname}.")
                self.ai_decision_log_append(f"Manual Close: No price for {pair_altname}.")
                return

            logger.info(f"🚪 Attempting to manually close position for {pair_display_name} at price {current_price:.4f}")
            # A close_position_programmatically metódust használjuk a záráshoz és logoláshoz
            self.close_position_programmatically(pair_altname, current_price, "MANUAL_CLOSE_SELECTED")

            # A frissítések már a close_position_programmatically-ban megtörténnek
            if self.dashboard: self.dashboard.update_field("Last Action", f"MANUAL CLOSE {pair_display_name}")

        except Exception as e:
            logger.error(f"Manual close selected position failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"Manual Close Error: {str(e)[:30]}")


    def manual_close_all_positions(self): # Ez a metódus felülírja a korábbit, ha volt ilyen nevű
        """Minden nyitott pozíció manuális zárása."""
        try:
            logger.info("🚪 Attempting to manually close ALL open positions...")
            if not (self.position and hasattr(self.position, 'get_all_positions')):
                logger.warning("Position manager not available, cannot close all positions.")
                self.ai_decision_log_append("Close All: No Pos. Manager.")
                return

            open_positions = self.position.get_all_positions()
            if not open_positions:
                logger.info("ℹ️ No open positions to close.")
                self.ai_decision_log_append("Close All: No open positions.")
                return

            closed_count = 0
            total_pnl_from_action = 0.0

            for pair_altname in list(open_positions.keys()): # Másolaton iterálunk
                current_price = self._get_real_current_price(pair_altname, open_positions[pair_altname].get('entry_price')) # Valós ár
                if current_price and current_price > 0:
                    # A close_position_programmatically kezeli a logolást és a teljesítmény frissítését
                    # de a P&L-t itt is összegezzük erre a specifikus "close all" akcióra
                    pos_data_before = self.position.get_position(pair_altname) # P&L számításhoz, ha a close_pos nem adja vissza
                    self.close_position_programmatically(pair_altname, current_price, "MANUAL_CLOSE_ALL")
                    closed_count += 1
                    # A P&L számítása itt bonyolultabb lenne, mert a close_programmatically már frissíti
                    # az ai_performance-t. Itt csak a darabszámot logoljuk.
                else:
                    logger.error(f"❌ Could not get price for {pair_altname} during CLOSE ALL. Skipping.")
                    self.ai_decision_log_append(f"Close All: No price for {pair_altname}.")

            logger.info(f"📴 Manual CLOSE ALL action: Attempted to close {len(open_positions)} positions. Successfully initiated close for: {closed_count}.")
            self.ai_decision_log_append(f"🚪 CLOSE ALL: {closed_count} positions closed.")

            if self.dashboard: self.dashboard.update_field("Last Action", f"CLOSED ALL ({closed_count})")
            # Az update_ai_performance_display és egyéb frissítések a close_position_programmatically-ban történnek.

        except Exception as e:
            logger.error(f"Manual close all positions failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"Close All Error: {str(e)[:30]}")

    # --- AI Performance & Logging ---
    def update_ai_performance_display(self):
        """AI kereskedési teljesítményének frissítése a GUI-n."""
        try:
            if not hasattr(self, 'ai_performance_labels') or not self.ai_performance_labels:
                # logger.debug("ai_performance_labels not initialized, skipping display update.")
                return

            total_decisions = self.ai_performance.get('total_decisions', 0)
            successful_trades = self.ai_performance.get('successful_trades', 0)
            total_pnl = self.ai_performance.get('ai_profit', 0.0)

            accuracy = (successful_trades / total_decisions * 100) if total_decisions > 0 else 0.0
            self.ai_performance['ai_accuracy'] = accuracy # Tároljuk az adatstruktúrában is

            if 'Bot Accuracy' in self.ai_performance_labels:
                self.ai_performance_labels['Bot Accuracy'].setText(f"{accuracy:.1f}%")
            if 'Total Trades' in self.ai_performance_labels:
                self.ai_performance_labels['Total Trades'].setText(str(total_decisions))
            if 'Successful Trades' in self.ai_performance_labels:
                self.ai_performance_labels['Successful Trades'].setText(str(successful_trades))
            if 'Total P&L' in self.ai_performance_labels:
                self.ai_performance_labels['Total P&L'].setText(f"${total_pnl:.2f}")

            # Utolsó trade P&L kijelzése
            if 'Last Trade P&L' in self.ai_performance_labels:
                 self.ai_performance_labels['Last Trade P&L'].setText(f"${getattr(self, 'last_trade_pnl', 0.0):.2f}")


            # Összesített statisztika a középső panelen (ai_stats_label)
            if self.ai_stats_label: # Ez a középső panelen lévő "Trading Stats"
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
            self.limit_ai_log_lines() # Log sorok limitálása

    def limit_ai_log_lines(self, max_lines=100): # Növelt max_lines
        """Az AI döntési napló sorainak limitálása a GUI-n."""
        try:
            if not (hasattr(self, 'ai_decision_log') and self.ai_decision_log): return

            current_text = self.ai_decision_log.toPlainText()
            lines = current_text.split('\n')

            if len(lines) > max_lines:
                limited_text = '\n'.join(lines[-max_lines:])
                self.ai_decision_log.setPlainText(limited_text)
                # Kurzor a végére mozgatása, hogy az új üzenetek látszódjanak
                cursor = self.ai_decision_log.textCursor()
                cursor.movePosition(cursor.End)
                self.ai_decision_log.setTextCursor(cursor)

        except Exception as e:
            logger.error(f"❌ AI Log limit failed: {e}")


    # --- Advanced Trading Cycle (Placeholder) ---
    def run_advanced_trading_cycle(self): # Ez a metódus már létezik, a tartalma frissül
        """Fejlett AI kereskedési ciklus futtatása (ha az AI mód aktív)."""
        try:
            if not self.ai_mode_active:
                # logger.debug("Advanced mode not active, skipping advanced trading cycle.")
                return

            logger.info("🤖 Running advanced AI trading cycle...")
            # Ez a rész most csak egy placeholder.
            # Itt kellene integrálni az AdvancedMarketScanner, MLScoringEngine, stb. hívásait.

            # 1. Piac szkennelése az AdvancedMarketScanner-rel
            opportunities = []
            if hasattr(self, 'ai_scanner') and self.ai_scanner and hasattr(self.ai_scanner, 'scan_top_opportunities'):
                opportunities = self.ai_scanner.scan_top_opportunities(max_pairs=10) # Max 10 pár elemzése
                logger.info(f"Advanced Scan: Found {len(opportunities)} potential opportunities.")
                self.ai_decision_log_append(f"Adv. Scan: {len(opportunities)} opps.")

            # 2. Minden lehetőség elemzése az ML motorral és micro stratégiával
            for opp in opportunities:
                if not isinstance(opp, CoinAnalysis): # Típusellenőrzés
                    logger.warning(f"Skipping invalid opportunity object: {opp}")
                    continue

                opp_pair = getattr(opp, 'pair', 'UnknownPair') # Biztonságosabb hozzáférés
                logger.info(f"Analyzing opportunity: {opp_pair}")

                # ML Jellemzők létrehozása
                ml_features = self._create_ml_features(opp)
                if not ml_features:
                    logger.warning(f"Could not create ML features for {opp_pair}. Skipping.")
                    continue

                # ML Predikció
                prediction = None
                if hasattr(self, 'ml_engine') and self.ml_engine and hasattr(self.ml_engine, 'predict_trade_success'):
                    prediction = self.ml_engine.predict_trade_success(ml_features)
                    prob = getattr(prediction, 'probability', 0.0)
                    conf = getattr(prediction, 'confidence', 0.0)
                    logger.info(f"  ML Prediction for {opp_pair}: Prob={prob:.3f}, Conf={conf:.3f}")
                    self.ai_decision_log_append(f"ML {opp_pair}: P={prob:.2f} C={conf:.2f}")


                # Micro Stratégia elemzés (ha van predikció)
                trade_setup = None
                if prediction and hasattr(self, 'micro_strategy') and self.micro_strategy and \
                   hasattr(self.micro_strategy, 'analyze_micro_opportunity'):
                    trade_setup = self.micro_strategy.analyze_micro_opportunity(opp, prediction)
                    if trade_setup:
                        trade_setup_signal = getattr(trade_setup, 'signal', 'N/A')
                        logger.info(f"  Micro Strategy Setup for {opp_pair}: Signal={trade_setup_signal}")
                        self.ai_decision_log_append(f"Micro {opp_pair}: Signal {trade_setup_signal}")


                # Kereskedés végrehajtása, ha a feltételek teljesülnek
                if trade_setup and isinstance(trade_setup, MicroTradeSetup) and \
                   (getattr(trade_setup, 'signal', 'N/A') == "BUY" or getattr(trade_setup, 'signal', 'N/A') == "SELL"): # Enum helyett string
                    # Kockázatkezelés és pozícióméretezés
                    risk_managed_params = None
                    if hasattr(self, 'risk_manager') and self.risk_manager and hasattr(self.risk_manager, 'calculate_trade_parameters'):
                         account_balance = self.wallet_display.get_usd_balance() if self.wallet_display else 500.0 # Dummy balance
                         if account_balance is None: account_balance = 500.0
                         risk_managed_params = self.risk_manager.calculate_trade_parameters(
                             pair=opp_pair,
                             signal=trade_setup.signal,
                             entry_price=trade_setup.entry_price,
                             stop_loss_price=trade_setup.stop_loss,
                             take_profit_price=trade_setup.take_profit,
                             account_balance=account_balance,
                             ml_confidence=getattr(prediction, 'confidence', 0.5)
                         )

                    if risk_managed_params:
                        logger.info(f"  Risk Managed Params for {opp_pair}: SizeUSD={risk_managed_params.get('position_size_usd',0):.0f}, Vol={risk_managed_params.get('volume',0):.6f}")
                        # Adatok előkészítése az execute_ai_opportunity_from_live_thread számára
                        execution_data = {
                            'pair': opp_pair,
                            'signal': trade_setup.signal.lower(), # Kisbetűs buy/sell
                            'entry_price': risk_managed_params.get('entry_price', trade_setup.entry_price),
                            'volume': risk_managed_params.get('volume'),
                            'stop_loss': risk_managed_params.get('stop_loss_price', trade_setup.stop_loss),
                            'take_profit': risk_managed_params.get('take_profit_price', trade_setup.take_profit),
                            'position_size_usd': risk_managed_params.get('position_size_usd'),
                            'total_score': getattr(prediction, 'probability', 0.0) # Vagy valami kombinált score
                        }
                        self.execute_ai_opportunity_from_live_thread(execution_data)
                    else:
                        logger.warning(f"Could not get risk managed parameters for {opp_pair}. Skipping trade.")
                        self.ai_decision_log_append(f"Risk Skip: {opp_pair}")

            logger.info("🤖 Advanced AI trading cycle finished.")

        except Exception as e:
            logger.error(f"Advanced AI trading cycle failed: {e}", exc_info=True)
            self.ai_decision_log_append(f"❌ Adv. Cycle Error: {str(e)[:30]}")


    def _create_ml_features(self, opportunity_analysis: CoinAnalysis) -> MLFeatures or None: # Visszatérési típus
        """MLFeatures objektum létrehozása a CoinAnalysis objektumból."""
        try:
            if not isinstance(opportunity_analysis, CoinAnalysis):
                logger.warning(f"Invalid opportunity_analysis object type for ML feature creation: {type(opportunity_analysis)}")
                return None

            # Hozzáférés az opportunity_analysis attribútumaihoz getattr-ral, alapértelmezett értékekkel
            # Fontos, hogy az MLFeatures osztály konstruktora milyen argumentumokat vár!
            # Az MLFeatures definíciójától függ, hogy mely mezőket kell itt átadni.
            # Tegyük fel, hogy az MLFeatures várja ezeket a mezőket:
            features = MLFeatures(
                rsi_3m=getattr(opportunity_analysis, 'rsi_3m', 50.0),
                rsi_15m=getattr(opportunity_analysis, 'rsi_15m', 50.0),
                macd_3m_signal_diff=getattr(opportunity_analysis, 'macd_3m_signal_diff', 0.0), # Példa, ha ilyen van
                volume_24h_usd=getattr(opportunity_analysis, 'volume_24h_usd', 0.0),
                price_change_pct_1h=getattr(opportunity_analysis, 'price_change_pct_1h', 0.0),
                # ... és így tovább, az MLFeatures osztály definíciója alapján ...
                # Hozzáadok néhány általánosabb mezőt, amik hasznosak lehetnek
                current_price=getattr(opportunity_analysis, 'price', 0.0),
                volatility_15m=getattr(opportunity_analysis, 'atr_15m_pct', 0.01), # ATR százalékosan
                market_trend_btc=getattr(opportunity_analysis, 'btc_trend_短期', 'NEUTRAL'), # Rövid távú BTC trend
            )
            return features
        except Exception as e:
            pair_name_log = getattr(opportunity_analysis, 'pair', 'UnknownPair')
            logger.error(f"ML features creation failed for {pair_name_log}: {e}", exc_info=True)
            return None

    # --- Application Closing ---
    def closeEvent(self, event):
        """Alkalmazás bezárásakor lefutó tisztítási folyamatok."""
        try:
            logger.info("🔄 Shutting down AI trading bot...")

            # Live trading szál leállítása, ha fut
            if self.live_trading_active and self.live_trading_thread and self.live_trading_thread.isRunning():
                logger.info("Stopping LiveTradingThread...")
                self.live_trading_thread.stop_trading()
                if not self.live_trading_thread.wait(5000): # Vár 5 másodpercet a szálra
                    logger.warning("⚠️ LiveTradingThread did not stop gracefully, attempting to terminate.")
                    self.live_trading_thread.terminate() # Végső esetben terminate
                    self.live_trading_thread.wait() # Vár a terminate után is
                else:
                    logger.info("✅ LiveTradingThread stopped.")
            elif self.live_trading_thread and not self.live_trading_thread.isRunning():
                 logger.info("LiveTradingThread was already stopped or not started.")


            # API kliens cleanup
            if hasattr(self.api, 'cleanup') and callable(self.api.cleanup):
                logger.info("Cleaning up API client...")
                self.api.cleanup()
                logger.info("✅ API client cleaned up.")

            # Timerek leállítása
            timers_to_stop = ['timer', 'position_timer'] # A scan_timer már nem itt van
            for timer_name in timers_to_stop:
                if hasattr(self, timer_name):
                    timer_instance = getattr(self, timer_name)
                    if timer_instance and timer_instance.isActive():
                        timer_instance.stop()
                        logger.info(f"Timer '{timer_name}' stopped.")

            # AI teljesítmény mentése (opcionális)
            if hasattr(self, 'ai_performance'):
                try:
                    import json
                    os.makedirs('logs', exist_ok=True) # logs mappa létrehozása, ha nincs
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
            event.accept() # Hiba esetén is záródjon be


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Stílus beállítása (opcionális, de szebbé teheti)
    # app.setStyle("Fusion")
    # Sötét téma beállítása
    # dark_palette = QPalette()
    # dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    # ... (többi szín beállítása)
    # app.setPalette(dark_palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
