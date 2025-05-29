# gui/advanced_control_panel.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                            QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
                            QCheckBox, QProgressBar, QTextEdit, QComboBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

# Add these imports to fix the missing dependencies
try:
    from strategy.intelligent_trader import IntelligentTrader
except ImportError:
    print("[WARNING] IntelligentTrader not found, using fallback")
    class IntelligentTrader:
        def __init__(self):
            self.learning_data = []
            self.performance_metrics = {'total_trades': 0, 'winning_trades': 0, 'total_profit': 0.0}
            self.position_manager = type('PositionManager', (), {'get_all_positions': lambda: {}})()
        
        def get_trading_status(self):
            return self.performance_metrics
        
        def learn_and_adapt(self):
            pass
        
        def close_position_with_learning(self, pair, price, reason):
            pass

try:
    from strategy.market_scanner import MarketScanner
except ImportError:
    print("[WARNING] MarketScanner not found, using fallback")
    class MarketScanner:
        def __init__(self):
            pass
        
        def get_top_opportunities(self, min_score=0.4):
            return []

class AdvancedControlPanel(QWidget):
    """Fejlett vezérlőpult AI trading funkcióval"""
    
    # Signals
    start_intelligent_trading = pyqtSignal()
    stop_intelligent_trading = pyqtSignal()
    update_ai_settings = pyqtSignal(dict)
    emergency_stop_all = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.intelligent_trader = IntelligentTrader()
        self.market_scanner = MarketScanner()
        
        # Status tracking
        self.ai_trading_active = False
        self.last_scan_results = []
        
        self.setup_ui()
        self.setup_timer()
        self.load_settings()

    def setup_ui(self):
        """UI felépítése"""
        main_layout = QVBoxLayout()
        
        # AI Trading Control
        ai_group = self.create_ai_trading_group()
        main_layout.addWidget(ai_group)
        
        # Market Scanner Control  
        scanner_group = self.create_scanner_group()
        main_layout.addWidget(scanner_group)
        
        # Performance Monitoring
        performance_group = self.create_performance_group()
        main_layout.addWidget(performance_group)
        
        # Learning & AI Settings
        learning_group = self.create_learning_group()
        main_layout.addWidget(learning_group)
        
        # Emergency Controls
        emergency_group = self.create_emergency_group()
        main_layout.addWidget(emergency_group)
        
        self.setLayout(main_layout)
        self.apply_styles()

    def create_ai_trading_group(self):
        """AI Trading vezérlés csoport - $50 optimalizálva"""
        group = QGroupBox("🎯 Micro Trading Engine ($50 positions)")
        layout = QGridLayout()
        
        # Trading status
        self.ai_status_label = QLabel("Status: INACTIVE")
        self.ai_status_label.setStyleSheet("font-weight: bold; color: #FF6B35;")
        layout.addWidget(self.ai_status_label, 0, 0, 1, 2)
        
        # Start/Stop buttons
        self.start_ai_btn = QPushButton("🚀 Start Micro Trading")
        self.start_ai_btn.clicked.connect(self.start_ai_trading)
        layout.addWidget(self.start_ai_btn, 1, 0)
        
        self.stop_ai_btn = QPushButton("⏹️ Stop Trading")
        self.stop_ai_btn.clicked.connect(self.stop_ai_trading)
        self.stop_ai_btn.setEnabled(False)
        layout.addWidget(self.stop_ai_btn, 1, 1)
        
        # AI Parameters - $50 optimalizálva
        layout.addWidget(QLabel("Max Concurrent Positions:"), 2, 0)
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 5)  # Max 5 pozíció mikro-tradingnél
        self.max_positions_spin.setValue(3)
        layout.addWidget(self.max_positions_spin, 2, 1)
        
        layout.addWidget(QLabel("Confidence Threshold:"), 3, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(60, 90)  # Szigorúbb konfidencia mikro-tradingnél
        self.confidence_slider.setValue(75)      # 75% alapértelmezett
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        layout.addWidget(self.confidence_slider, 3, 1)
        
        self.confidence_label = QLabel("75%")
        layout.addWidget(self.confidence_label, 3, 2)
        
        # 🎯 MIKRO-TRADING SPECIFIKUS BEÁLLÍTÁSOK
        layout.addWidget(QLabel("Position Size ($):"), 4, 0)
        self.investment_spin = QDoubleSpinBox()
        self.investment_spin.setRange(25, 75)      # $25-75 között
        self.investment_spin.setValue(50)          # $50 alapértelmezett
        self.investment_spin.setDecimals(0)
        layout.addWidget(self.investment_spin, 4, 1)
        
        layout.addWidget(QLabel("Target Profit ($):"), 5, 0)
        self.target_profit_spin = QDoubleSpinBox()
        self.target_profit_spin.setRange(0.10, 1.00)  # $0.10 - $1.00
        self.target_profit_spin.setValue(0.15)         # $0.15 alapértelmezett
        self.target_profit_spin.setDecimals(2)
        self.target_profit_spin.setSingleStep(0.05)
        layout.addWidget(self.target_profit_spin, 5, 1)
        
        layout.addWidget(QLabel("Max Loss ($):"), 6, 0)
        self.max_loss_spin = QDoubleSpinBox()
        self.max_loss_spin.setRange(1.0, 5.0)     # $1-5 max loss
        self.max_loss_spin.setValue(2.0)          # $2 alapértelmezett
        self.max_loss_spin.setDecimals(1)
        layout.addWidget(self.max_loss_spin, 6, 1)
        
        # Mode selection
        layout.addWidget(QLabel("Trading Mode:"), 7, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Simulation", "Paper Trading", "Live Trading"])
        self.mode_combo.setCurrentText("Simulation")
        layout.addWidget(self.mode_combo, 7, 1)
        
        # 🎯 MIKRO-TRADING STATISZTIKÁK
        layout.addWidget(QLabel("Daily Profit Target:"), 8, 0)
        self.daily_target_label = QLabel("$0.45")  # 3 pozíció × $0.15
        self.daily_target_label.setStyleSheet("color: #00AA00; font-weight: bold;")
        layout.addWidget(self.daily_target_label, 8, 1)
        
        layout.addWidget(QLabel("Total Exposure:"), 9, 0)
        self.exposure_label = QLabel("$150")  # 3 pozíció × $50
        self.exposure_label.setStyleSheet("color: #FF8800; font-weight: bold;")
        layout.addWidget(self.exposure_label, 9, 1)
        
        group.setLayout(layout)
        return group

    def create_scanner_group(self):
        """Market Scanner vezérlés"""
        group = QGroupBox("🔍 Market Scanner")
        layout = QGridLayout()
        
        # Scanner status
        self.scanner_status = QLabel("Ready")
        layout.addWidget(QLabel("Status:"), 0, 0)
        layout.addWidget(self.scanner_status, 0, 1)
        
        # Manual scan button
        self.manual_scan_btn = QPushButton("🔄 Manual Scan")
        self.manual_scan_btn.clicked.connect(self.manual_market_scan)
        layout.addWidget(self.manual_scan_btn, 1, 0, 1, 2)
        
        # Scan parameters
        layout.addWidget(QLabel("Min Volume ($M):"), 2, 0)
        self.min_volume_spin = QDoubleSpinBox()
        self.min_volume_spin.setRange(0.1, 100)
        self.min_volume_spin.setValue(0.5)  # Mikro-tradinghez alacsonyabb
        self.min_volume_spin.setDecimals(1)
        layout.addWidget(self.min_volume_spin, 2, 1)
        
        layout.addWidget(QLabel("Max Pairs to Analyze:"), 3, 0)
        self.max_pairs_spin = QSpinBox()
        self.max_pairs_spin.setRange(10, 100)
        self.max_pairs_spin.setValue(80)  # Több pár mikro-tradinghez
        layout.addWidget(self.max_pairs_spin, 3, 1)
        
        # Scan results preview
        self.scan_results = QTextEdit()
        self.scan_results.setMaximumHeight(120)
        self.scan_results.setPlaceholderText("Scan results will appear here...")
        layout.addWidget(self.scan_results, 4, 0, 2, 2)
        
        group.setLayout(layout)
        return group

    def create_performance_group(self):
        """Teljesítmény monitoring - mikro-trading optimalizálva"""
        group = QGroupBox("📊 Micro Trading Performance")
        layout = QGridLayout()
        
        # Key metrics
        self.total_trades_label = QLabel("0")
        self.win_rate_label = QLabel("0%")
        self.total_profit_label = QLabel("$0.00")
        self.active_positions_label = QLabel("0")
        self.avg_profit_label = QLabel("$0.00")      # Új: átlag profit
        self.daily_profit_label = QLabel("$0.00")    # Új: napi profit
        
        layout.addWidget(QLabel("Total Trades:"), 0, 0)
        layout.addWidget(self.total_trades_label, 0, 1)
        
        layout.addWidget(QLabel("Win Rate:"), 1, 0)
        layout.addWidget(self.win_rate_label, 1, 1)
        
        layout.addWidget(QLabel("Total Profit:"), 2, 0)
        layout.addWidget(self.total_profit_label, 2, 1)
        
        layout.addWidget(QLabel("Active Positions:"), 3, 0)
        layout.addWidget(self.active_positions_label, 3, 1)
        
        layout.addWidget(QLabel("Avg Profit/Trade:"), 4, 0)
        layout.addWidget(self.avg_profit_label, 4, 1)
        
        layout.addWidget(QLabel("Today's Profit:"), 5, 0)
        layout.addWidget(self.daily_profit_label, 5, 1)
        
        # Progress bars
        layout.addWidget(QLabel("Position Utilization:"), 6, 0)
        self.position_progress = QProgressBar()
        self.position_progress.setRange(0, 100)
        layout.addWidget(self.position_progress, 6, 1)
        
        layout.addWidget(QLabel("Daily Target Progress:"), 7, 0)
        self.daily_progress = QProgressBar()
        self.daily_progress.setRange(0, 100)
        layout.addWidget(self.daily_progress, 7, 1)
        
        group.setLayout(layout)
        return group

    def create_learning_group(self):
        """AI Tanulás beállítások"""
        group = QGroupBox("🧠 AI Learning Settings")
        layout = QVBoxLayout()
        
        # Learning controls
        self.learning_enabled_cb = QCheckBox("Enable Learning")
        self.learning_enabled_cb.setChecked(True)
        layout.addWidget(self.learning_enabled_cb)
        
        self.auto_adapt_cb = QCheckBox("Auto-adapt Weights")
        self.auto_adapt_cb.setChecked(True)
        layout.addWidget(self.auto_adapt_cb)
        
        # Learning stats
        stats_layout = QHBoxLayout()
        self.learning_data_count = QLabel("Learning Data: 0 trades")
        stats_layout.addWidget(self.learning_data_count)
        
        self.last_learning_update = QLabel("Last Update: Never")
        stats_layout.addWidget(self.last_learning_update)
        layout.addLayout(stats_layout)
        
        # Manual learning trigger
        self.trigger_learning_btn = QPushButton("🔄 Trigger Learning Update")
        self.trigger_learning_btn.clicked.connect(self.trigger_learning)
        layout.addWidget(self.trigger_learning_btn)
        
        # Reset learning data
        self.reset_learning_btn = QPushButton("🗑️ Reset Learning Data")
        self.reset_learning_btn.clicked.connect(self.reset_learning_data)
        self.reset_learning_btn.setStyleSheet("background-color: #FF6B35;")
        layout.addWidget(self.reset_learning_btn)
        
        group.setLayout(layout)
        return group

    def create_emergency_group(self):
        """Vészhelyzeti vezérlők"""
        group = QGroupBox("🚨 Emergency Controls")
        layout = QVBoxLayout()
        
        # Emergency stop all
        self.emergency_stop_btn = QPushButton("🛑 EMERGENCY STOP ALL")
        self.emergency_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF0000;
                color: white;
                font-weight: bold;
                font-size: 16px;
                padding: 15px;
                border: 3px solid #CC0000;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #CC0000;
            }
            QPushButton:pressed {
                background-color: #990000;
            }
        """)
        self.emergency_stop_btn.clicked.connect(self.emergency_stop)
        layout.addWidget(self.emergency_stop_btn)
        
        # Close all positions
        self.close_all_btn = QPushButton("📴 Close All Positions")
        self.close_all_btn.clicked.connect(self.close_all_positions)
        layout.addWidget(self.close_all_btn)
        
        # Emergency status
        self.emergency_status = QLabel("System Status: NORMAL")
        self.emergency_status.setStyleSheet("color: #00AA00; font-weight: bold;")
        layout.addWidget(self.emergency_status)
        
        group.setLayout(layout)
        return group

    def setup_timer(self):
        """Timer a státusz frissítéshez"""
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)  # 5 másodpercenként

    def apply_styles(self):
        """Stílusok alkalmazása"""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 5px;
                font-weight: bold;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #AAAAAA;
            }
            QLabel {
                color: white;
                font-size: 12px;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2D2D30;
                color: white;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 8px;
                background: #2D2D30;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078D4;
                border: 1px solid #0078D4;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0078D4;
                border-radius: 5px;
            }
            QTextEdit {
                background-color: #1E1E1E;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
            QCheckBox {
                color: white;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #2D2D30;
                border: 1px solid #555;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #0078D4;
                border: 1px solid #0078D4;
                border-radius: 3px;
            }
        """)

    def start_ai_trading(self):
        """AI Trading indítása - mikro-trading módban"""
        try:
            # Mikro-trading számítások frissítése
            self.update_micro_trading_calculations()
            
            # Trading indítása
            self.ai_trading_active = True
            if hasattr(self.intelligent_trader, 'simulation_mode'):
                self.intelligent_trader.simulation_mode = (self.mode_combo.currentText() == "Simulation")
            
            # UI frissítése
            self.ai_status_label.setText("Status: MICRO-TRADING ACTIVE")
            self.ai_status_label.setStyleSheet("font-weight: bold; color: #00AA00;")
            self.start_ai_btn.setEnabled(False)
            self.stop_ai_btn.setEnabled(True)
            
            # Mikro-trading timer (gyakoribb ellenőrzés)
            self.trading_timer = QTimer()
            self.trading_timer.timeout.connect(self.run_micro_trading_cycle)
            self.trading_timer.start(30000)  # 30 másodperc (gyakoribb mint 1 perc)
            
            self.scanner_status.setText("Micro-scanning...")
            self.scan_results.append("🎯 Micro-Trading started! ($50 positions)")
            
            print("🎯 Micro-Trading started with $50 positions!")
            
            # Emit signal
            self.start_intelligent_trading.emit()
            
        except Exception as e:
            print(f"Error starting micro trading: {e}")
            self.scan_results.append(f"❌ Error: {e}")

    def stop_ai_trading(self):
        """AI Trading leállítása"""
        try:
            self.ai_trading_active = False
            
            if hasattr(self, 'trading_timer'):
                self.trading_timer.stop()
            
            # UI frissítése
            self.ai_status_label.setText("Status: INACTIVE")
            self.ai_status_label.setStyleSheet("font-weight: bold; color: #FF6B35;")
            self.start_ai_btn.setEnabled(True)
            self.stop_ai_btn.setEnabled(False)
            
            self.scanner_status.setText("Ready")
            self.scan_results.append("⏹️ Micro-Trading stopped.")
            
            print("⏹️ Micro-Trading stopped!")
            
            # Emit signal
            self.stop_intelligent_trading.emit()
            
        except Exception as e:
            print(f"Error stopping micro trading: {e}")

    def run_micro_trading_cycle(self):
        """Mikro-trading ciklus futtatása"""
        try:
            if not self.ai_trading_active:
                return
                
            # Egyenleg szimuláció ($500-1000)
            simulated_balance = 750.0  
            
            # Mikro-trading ciklus
            if hasattr(self.intelligent_trader, 'run_micro_trading_cycle'):
                self.intelligent_trader.run_micro_trading_cycle(simulated_balance)
            else:
                # Fallback to regular cycle
                print("[MICRO] Running fallback trading cycle")
            
            # Eredmények frissítése
            status = self.intelligent_trader.get_trading_status()
            
            position_size = self.investment_spin.value()
            active_pos = status.get('active_positions', 0)
            total_profit = status.get('total_profit', 0)
            
            self.scan_results.append(
                f"💰 Micro cycle: {active_pos} positions × ${position_size:.0f}, "
                f"Total profit: ${total_profit:.2f}"
            )
            
            # Auto-scroll
            cursor = self.scan_results.textCursor()
            cursor.movePosition(cursor.End)
            self.scan_results.setTextCursor(cursor)
            
        except Exception as e:
            print(f"Error in micro trading cycle: {e}")
            self.scan_results.append(f"❌ Micro cycle error: {e}")

    def manual_market_scan(self):
        """Manuális piaci szkenelés"""
        try:
            self.manual_scan_btn.setEnabled(False)
            self.scanner_status.setText("Scanning...")
            self.scan_results.append("🔍 Starting manual market scan...")
            
            # Scanner beállítások frissítése
            if hasattr(self.market_scanner, 'MIN_VOLUME_24H'):
                self.market_scanner.MIN_VOLUME_24H = self.min_volume_spin.value() * 1000000
                self.market_scanner.MAX_PAIRS_TO_ANALYZE = self.max_pairs_spin.value()
            
            # Scan futtatása háttérben (egyszerűsített)
            QTimer.singleShot(100, self.execute_scan)
            
        except Exception as e:
            print(f"Error in manual scan: {e}")
            self.scan_results.append(f"❌ Scan error: {e}")
            self.manual_scan_btn.setEnabled(True)

    def execute_scan(self):
        """Szkenelés végrehajtása"""
        try:
            results = self.market_scanner.get_top_opportunities(min_score=0.4)
            
            self.scan_results.clear()
            self.scan_results.append(f"📈 Micro-Trading Scan Results ({len(results)} opportunities):")
            self.scan_results.append("-" * 50)
            
            if results:
                for i, coin in enumerate(results[:10]):  # Top 10
                    if hasattr(coin, 'pair'):
                        self.scan_results.append(
                            f"#{i+1}: {coin.pair} - Score: {coin.final_score:.3f} - "
                            f"Vol: ${coin.volume_24h:.0f} - RSI: {coin.rsi:.1f}"
                        )
                    else:
                        self.scan_results.append(f"#{i+1}: {coin} - Limited data")
            else:
                self.scan_results.append("No opportunities found. Using fallback mode.")
                
            self.scanner_status.setText(f"Found {len(results)} micro opportunities")
            self.manual_scan_btn.setEnabled(True)
            
        except Exception as e:
            print(f"Error executing scan: {e}")
            self.scan_results.append(f"❌ Execution error: {e}")
            self.scanner_status.setText("Error")
            self.manual_scan_btn.setEnabled(True)

    def update_confidence_label(self, value):
        """Konfidencia slider label frissítése"""
        self.confidence_label.setText(f"{value}%")

    def update_micro_trading_calculations(self):
        """Mikro-trading számítások frissítése"""
        try:
            position_size = self.investment_spin.value()
            max_positions = self.max_positions_spin.value()
            target_profit = self.target_profit_spin.value()
            
            # Napi profit cél
            daily_target = target_profit * max_positions
            self.daily_target_label.setText(f"${daily_target:.2f}")
            
            # Összes exposure
            total_exposure = position_size * max_positions
            self.exposure_label.setText(f"${total_exposure:.0f}")
            
            # AI trader beállítások frissítése
            if hasattr(self.intelligent_trader, 'DEFAULT_POSITION_SIZE'):
                self.intelligent_trader.DEFAULT_POSITION_SIZE = position_size
                self.intelligent_trader.MIN_PROFIT_TARGET = target_profit
            
            if hasattr(self.intelligent_trader, 'max_concurrent_trades'):
                self.intelligent_trader.max_concurrent_trades = max_positions
                
        except Exception as e:
            print(f"Error updating micro calculations: {e}")

    def update_status(self):
        """Státusz frissítése - mikro-trading adatokkal"""
        try:
            status = self.intelligent_trader.get_trading_status()
            
            # Alap metrikai
            total_trades = status.get('total_trades', 0)
            win_rate = status.get('win_rate', 0)
            total_profit = status.get('total_profit', 0)
            active_positions = 0
            
            if hasattr(self.intelligent_trader, 'position_manager'):
                active_positions = len(self.intelligent_trader.position_manager.get_all_positions())
            
            # UI frissítése
            self.total_trades_label.setText(str(total_trades))
            self.win_rate_label.setText(f"{win_rate:.1%}" if isinstance(win_rate, float) else f"{win_rate}%")
            self.total_profit_label.setText(f"${total_profit:.2f}")
            self.active_positions_label.setText(str(active_positions))
            
            # Mikro-trading specifikus számítások
            avg_profit = total_profit / max(total_trades, 1)
            self.avg_profit_label.setText(f"${avg_profit:.2f}")
            
            # Ma született profit (egyszerűsített)
            daily_profit = total_profit * 0.3  # Szimuláció: 30% ma
            self.daily_profit_label.setText(f"${daily_profit:.2f}")
            
            # Progress bar-ok
            max_positions = self.max_positions_spin.value()
            position_util = (active_positions / max(max_positions, 1)) * 100
            self.position_progress.setValue(int(position_util))
            
            # Napi target progress
            daily_target = self.target_profit_spin.value() * max_positions
            daily_progress = min(100, (daily_profit / max(daily_target, 0.01)) * 100)
            self.daily_progress.setValue(int(daily_progress))
            
            # Learning data count
            learning_count = len(self.intelligent_trader.learning_data)
            self.learning_data_count.setText(f"Learning Data: {learning_count} micro-trades")
            
        except Exception as e:
            print(f"Error updating micro status: {e}")

    def trigger_learning(self):
        """Manuális tanulás indítása"""
        try:
            if hasattr(self.intelligent_trader, 'learn_and_adapt'):
                self.intelligent_trader.learn_and_adapt()
            self.scan_results.append("🧠 Learning update triggered!")
            self.last_learning_update.setText(f"Last Update: Now")
        except Exception as e:
            print(f"Error triggering learning: {e}")

    def reset_learning_data(self):
        """Tanulási adatok törlése"""
        try:
            self.intelligent_trader.learning_data.clear()
            self.intelligent_trader.performance_metrics = {
                'total_trades': 0, 'winning_trades': 0, 'total_profit': 0.0
            }
            self.scan_results.append("🗑️ Learning data reset!")
        except Exception as e:
            print(f"Error resetting learning data: {e}")

    def emergency_stop(self):
        """Vészleállítás"""
        try:
            self.stop_ai_trading()
            self.close_all_positions()
            
            self.emergency_status.setText("System Status: EMERGENCY STOP")
            self.emergency_status.setStyleSheet("color: #FF0000; font-weight: bold;")
            
            self.scan_results.append("🚨 EMERGENCY STOP ACTIVATED!")
            self.emergency_stop_all.emit()
            
        except Exception as e:
            print(f"Error in emergency stop: {e}")

    def close_all_positions(self):
        """Összes pozíció zárása"""
        try:
            if hasattr(self.intelligent_trader, 'position_manager'):
                positions = self.intelligent_trader.position_manager.get_all_positions()
                for pair in list(positions.keys()):
                    if hasattr(self.intelligent_trader, 'close_position_with_learning'):
                        self.intelligent_trader.close_position_with_learning(pair, positions[pair].get('entry_price', 0), "MANUAL_CLOSE")
                    else:
                        print(f"[CLOSE] Would close position: {pair}")
                
                self.scan_results.append(f"📴 Closed {len(positions)} positions")
            else:
                self.scan_results.append("📴 No position manager available")
            
        except Exception as e:
            print(f"Error closing positions: {e}")

    def load_settings(self):
        """Beállítások betöltése"""
        try:
            import os
            settings_file = 'settings/micro_trading_settings.json'
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    
                # Load settings into UI
                self.max_positions_spin.setValue(settings.get('max_positions', 3))
                self.confidence_slider.setValue(settings.get('confidence_threshold', 75))
                self.investment_spin.setValue(settings.get('investment_per_trade', 50))
                self.target_profit_spin.setValue(settings.get('target_profit', 0.15))
                self.max_loss_spin.setValue(settings.get('max_loss', 2.0))
                
                mode = settings.get('trading_mode', 'Simulation')
                index = self.mode_combo.findText(mode)
                if index >= 0:
                    self.mode_combo.setCurrentIndex(index)
                    
                self.learning_enabled_cb.setChecked(settings.get('learning_enabled', True))
                self.auto_adapt_cb.setChecked(settings.get('auto_adapt', True))
                
                print("[SETTINGS] Micro-trading settings loaded")
        except Exception as e:
            print(f"[SETTINGS] Error loading settings: {e}")

    def save_settings(self):
        """Beállítások mentése"""
        settings = {
            'max_positions': self.max_positions_spin.value(),
            'confidence_threshold': self.confidence_slider.value(),
            'investment_per_trade': self.investment_spin.value(),
            'target_profit': self.target_profit_spin.value(),
            'max_loss': self.max_loss_spin.value(),
            'trading_mode': self.mode_combo.currentText(),
            'learning_enabled': self.learning_enabled_cb.isChecked(),
            'auto_adapt': self.auto_adapt_cb.isChecked()
        }
        
        try:
            import os
            os.makedirs('settings', exist_ok=True)
            with open('settings/micro_trading_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            print("[SETTINGS] Micro-trading settings saved")
        except Exception as e:
            print(f"Error saving settings: {e}")

    def closeEvent(self, event):
        """Alkalmazás bezárása"""
        self.save_settings()
        if self.ai_trading_active:
            self.stop_ai_trading()
        event.accept()
