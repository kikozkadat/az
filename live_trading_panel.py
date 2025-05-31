# gui/live_trading_panel.py - Továbbfejlesztett verzió

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, 
                            QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
                            QProgressBar, QTextEdit, QComboBox, QGridLayout)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont
import time
import random
import json
from datetime import datetime

class LiveTradingWorker(QThread):
    """Background worker for live trading operations"""
    opportunity_found = pyqtSignal(dict)
    trade_executed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, settings, api_client=None):
        super().__init__()
        self.settings = settings
        self.api_client = api_client
        self.running = False
        self.scan_interval = 30  # seconds
        
    def start_trading(self):
        self.running = True
        self.start()
        
    def stop_trading(self):
        self.running = False
        self.quit()
        self.wait(3000)  # 3 second timeout
        
    def run(self):
        while self.running:
            try:
                if self.settings.get('auto_trading', False):
                    self.scan_for_opportunities()
                self.msleep(self.scan_interval * 1000)
            except Exception as e:
                self.error_occurred.emit(str(e))
                
    def scan_for_opportunities(self):
        """Scan for trading opportunities"""
        try:
            # Mock opportunity generation
            if random.random() < 0.1:  # 10% chance
                opportunity = {
                    'pair': random.choice(['ADAUSD', 'SOLUSD', 'DOTUSD', 'LINKUSD']),
                    'score': random.uniform(0.6, 0.9),
                    'confidence': random.uniform(0.65, 0.85),
                    'volume_usd': random.uniform(1000000, 5000000),
                    'reason': 'Volume spike + Bollinger breakout'
                }
                self.opportunity_found.emit(opportunity)
                
        except Exception as e:
            self.error_occurred.emit(f"Scan error: {e}")

class LiveTradingPanel(QWidget):
    """Enhanced Live Trading Panel for $50 micro-trading"""
    
    # Signals
    start_live_trading = pyqtSignal()
    stop_live_trading = pyqtSignal()
    emergency_stop = pyqtSignal()
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, api_client=None, parent_window=None):
        super().__init__()
        
        # Store references
        self.api_client = api_client
        self.parent_window = parent_window
        
        # Trading state
        self.is_trading = False
        self.worker = None
        
        # Session tracking
        self.session_start_time = None
        self.session_trades = 0
        self.session_profit = 0.0
        self.opportunities = []
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        self.setup_update_timer()
        
        print("✅ LiveTradingPanel initialized successfully")

    def setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Trading controls
        controls = self.create_trading_controls()
        main_layout.addWidget(controls)
        
        # Settings
        settings = self.create_settings_panel()
        main_layout.addWidget(settings)
        
        # Opportunities
        opportunities = self.create_opportunities_panel()
        main_layout.addWidget(opportunities)
        
        # Statistics
        stats = self.create_statistics_panel()
        main_layout.addWidget(stats)
        
        self.setLayout(main_layout)

    def create_header(self):
        """Create header section"""
        group = QGroupBox("🚀 Live Micro-Trading ($50 Positions)")
        layout = QVBoxLayout()
        
        # Status
        self.status_label = QLabel("Status: OFFLINE")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF4444;")
        layout.addWidget(self.status_label)
        
        # Session info
        self.session_label = QLabel("Session: Not started")
        self.session_label.setStyleSheet("font-size: 11px; color: #CCCCCC;")
        layout.addWidget(self.session_label)
        
        group.setLayout(layout)
        return group

    def create_trading_controls(self):
        """Create trading control buttons"""
        group = QGroupBox("Trading Controls")
        layout = QHBoxLayout()
        
        # Start button
        self.start_btn = QPushButton("🟢 Start Trading")
        self.start_btn.clicked.connect(self.start_trading)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2ECC71;
            }
        """)
        layout.addWidget(self.start_btn)
        
        # Stop button
        self.stop_btn = QPushButton("🔴 Stop Trading")
        self.stop_btn.clicked.connect(self.stop_trading)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
        """)
        layout.addWidget(self.stop_btn)
        
        # Emergency stop
        self.emergency_btn = QPushButton("🚨 EMERGENCY")
        self.emergency_btn.clicked.connect(self.emergency_stop_trading)
        self.emergency_btn.setStyleSheet("""
            QPushButton {
                background-color: #8E44AD;
                color: white;
                font-weight: bold;
                padding: 10px 15px;
                border-radius: 5px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #9B59B6;
            }
        """)
        layout.addWidget(self.emergency_btn)
        
        group.setLayout(layout)
        return group

    def create_settings_panel(self):
        """Create settings panel"""
        group = QGroupBox("Micro-Trading Settings")
        layout = QGridLayout()
        
        # Auto trading
        self.auto_trading_cb = QCheckBox("Auto Trading")
        self.auto_trading_cb.setChecked(True)
        layout.addWidget(self.auto_trading_cb, 0, 0, 1, 2)
        
        # Position size
        layout.addWidget(QLabel("Position Size ($):"), 1, 0)
        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(25.0, 75.0)
        self.position_size_spin.setValue(50.0)
        self.position_size_spin.setDecimals(0)
        layout.addWidget(self.position_size_spin, 1, 1)
        
        # Max positions
        layout.addWidget(QLabel("Max Positions:"), 2, 0)
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 3)
        self.max_positions_spin.setValue(1)
        layout.addWidget(self.max_positions_spin, 2, 1)
        
        # Confidence threshold
        layout.addWidget(QLabel("Min Confidence:"), 3, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.5, 0.9)
        self.confidence_spin.setValue(0.7)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setSingleStep(0.05)
        layout.addWidget(self.confidence_spin, 3, 1)
        
        # Stop loss percentage
        layout.addWidget(QLabel("Stop Loss (%):"), 4, 0)
        self.stop_loss_spin = QDoubleSpinBox()
        self.stop_loss_spin.setRange(2.0, 8.0)
        self.stop_loss_spin.setValue(4.0)
        self.stop_loss_spin.setDecimals(1)
        layout.addWidget(self.stop_loss_spin, 4, 1)
        
        # Apply button
        apply_btn = QPushButton("Apply Settings")
        apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(apply_btn, 5, 0, 1, 2)
        
        group.setLayout(layout)
        return group

    def create_opportunities_panel(self):
        """Create opportunities display panel"""
        group = QGroupBox("Market Opportunities")
        layout = QVBoxLayout()
        
        # Opportunities list
        self.opportunities_text = QTextEdit()
        self.opportunities_text.setMaximumHeight(150)
        self.opportunities_text.setReadOnly(True)
        self.opportunities_text.setPlaceholderText("Market opportunities will appear here...")
        self.opportunities_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #00FF00;
                border: 1px solid #555;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.opportunities_text)
        
        # Scan button
        scan_btn = QPushButton("🔍 Manual Scan")
        scan_btn.clicked.connect(self.manual_scan)
        layout.addWidget(scan_btn)
        
        group.setLayout(layout)
        return group

    def create_statistics_panel(self):
        """Create statistics panel"""
        group = QGroupBox("Session Statistics")
        layout = QGridLayout()
        
        # Session stats
        layout.addWidget(QLabel("Session Trades:"), 0, 0)
        self.trades_label = QLabel("0")
        self.trades_label.setStyleSheet("font-weight: bold; color: #00AAFF;")
        layout.addWidget(self.trades_label, 0, 1)
        
        layout.addWidget(QLabel("Session P&L:"), 1, 0)
        self.profit_label = QLabel("$0.00")
        self.profit_label.setStyleSheet("font-weight: bold; color: #00FF00;")
        layout.addWidget(self.profit_label, 1, 1)
        
        layout.addWidget(QLabel("Session Duration:"), 2, 0)
        self.duration_label = QLabel("00:00:00")
        self.duration_label.setStyleSheet("font-weight: bold; color: #FFAA00;")
        layout.addWidget(self.duration_label, 2, 1)
        
        layout.addWidget(QLabel("Success Rate:"), 3, 0)
        self.success_label = QLabel("0%")
        self.success_label.setStyleSheet("font-weight: bold; color: #FF00AA;")
        layout.addWidget(self.success_label, 3, 1)
        
        # Progress bar for daily target
        layout.addWidget(QLabel("Daily Target:"), 4, 0)
        self.daily_progress = QProgressBar()
        self.daily_progress.setRange(0, 100)
        self.daily_progress.setValue(0)
        layout.addWidget(self.daily_progress, 4, 1)
        
        group.setLayout(layout)
        return group

    def setup_connections(self):
        """Setup signal connections"""
        # Connect spinbox changes to settings update
        self.position_size_spin.valueChanged.connect(self.on_settings_changed)
        self.max_positions_spin.valueChanged.connect(self.on_settings_changed)
        self.confidence_spin.valueChanged.connect(self.on_settings_changed)
        self.stop_loss_spin.valueChanged.connect(self.on_settings_changed)
        self.auto_trading_cb.toggled.connect(self.on_settings_changed)

    def setup_update_timer(self):
        """Setup update timer"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # Update every second

    def start_trading(self):
        """Start live trading"""
        try:
            if self.is_trading:
                return
                
            print("🚀 Starting live trading...")
            
            # Create and start worker
            settings = self.get_current_settings()
            self.worker = LiveTradingWorker(settings, self.api_client)
            self.worker.opportunity_found.connect(self.on_opportunity_found)
            self.worker.trade_executed.connect(self.on_trade_executed)
            self.worker.error_occurred.connect(self.on_error)
            
            self.worker.start_trading()
            
            # Update state
            self.is_trading = True
            self.session_start_time = time.time()
            self.session_trades = 0
            self.session_profit = 0.0
            
            # Update UI
            self.status_label.setText("Status: 🟢 LIVE TRADING")
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00FF00;")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            # Emit signal
            self.start_live_trading.emit()
            
            print("✅ Live trading started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start trading: {e}")
            self.on_error(str(e))

    def stop_trading(self):
        """Stop live trading"""
        try:
            if not self.is_trading:
                return
                
            print("🔴 Stopping live trading...")
            
            # Stop worker
            if self.worker:
                self.worker.stop_trading()
                self.worker = None
                
            # Update state
            self.is_trading = False
            
            # Update UI
            self.status_label.setText("Status: 🔴 STOPPED")
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF4444;")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            # Emit signal
            self.stop_live_trading.emit()
            
            print("✅ Live trading stopped")
            
        except Exception as e:
            print(f"❌ Failed to stop trading: {e}")

    def emergency_stop_trading(self):
        """Emergency stop"""
        try:
            print("🚨 EMERGENCY STOP ACTIVATED!")
            
            self.stop_trading()
            
            # Update UI for emergency state
            self.status_label.setText("Status: 🚨 EMERGENCY STOP")
            self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FF0000;")
            
            # Emit emergency signal
            self.emergency_stop.emit()
            
        except Exception as e:
            print(f"❌ Emergency stop failed: {e}")

    def apply_settings(self):
        """Apply current settings"""
        settings = self.get_current_settings()
        self.settings_changed.emit(settings)
        print(f"⚙️ Settings applied: {settings}")

    def on_settings_changed(self):
        """Handle settings change"""
        if self.is_trading:
            self.apply_settings()

    def manual_scan(self):
        """Perform manual market scan"""
        try:
            self.opportunities_text.append("🔍 Manual scan initiated...")
            
            # Simulate scan results
            QTimer.singleShot(1000, self.simulate_scan_results)
            
        except Exception as e:
            self.opportunities_text.append(f"❌ Scan error: {e}")

    def simulate_scan_results(self):
        """Simulate scan results"""
        pairs = ['ADAUSD', 'SOLUSD', 'DOTUSD', 'LINKUSD', 'UNIUSD']
        
        self.opportunities_text.append("📊 Scan results:")
        
        for pair in pairs:
            score = random.uniform(0.3, 0.9)
            confidence = random.uniform(0.6, 0.85)
            volume = random.uniform(500000, 5000000)
            
            if score > 0.6:
                color = "🟢" if score > 0.8 else "🟡"
                self.opportunities_text.append(
                    f"  {color} {pair}: Score {score:.2f}, Conf {confidence:.2f}, Vol ${volume:,.0f}"
                )

    def on_opportunity_found(self, opportunity):
        """Handle opportunity found"""
        try:
            pair = opportunity['pair']
            score = opportunity['score']
            confidence = opportunity['confidence']
            
            self.opportunities_text.append(
                f"🎯 OPPORTUNITY: {pair} (Score: {score:.2f}, Conf: {confidence:.2f})"
            )
            
            # Auto-scroll to bottom
            scrollbar = self.opportunities_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        except Exception as e:
            print(f"Error handling opportunity: {e}")

    def on_trade_executed(self, trade_data):
        """Handle trade execution"""
        try:
            self.session_trades += 1
            self.session_profit += trade_data.get('pnl', 0)
            
            pair = trade_data.get('pair', 'UNKNOWN')
            action = trade_data.get('action', 'TRADE')
            pnl = trade_data.get('pnl', 0)
            
            # Log to opportunities panel
            pnl_color = "🟢" if pnl >= 0 else "🔴"
            self.opportunities_text.append(
                f"{pnl_color} TRADE: {pair} {action} - P&L: ${pnl:.2f}"
            )
            
            # Auto-scroll
            scrollbar = self.opportunities_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        except Exception as e:
            print(f"Error handling trade execution: {e}")

    def on_error(self, error_msg):
        """Handle errors"""
        try:
            self.opportunities_text.append(f"❌ ERROR: {error_msg}")
            print(f"LiveTradingPanel error: {error_msg}")
            
            # Auto-scroll
            scrollbar = self.opportunities_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        except Exception as e:
            print(f"Error handling error: {e}")

    def update_display(self):
        """Update display elements"""
        try:
            # Update session info
            if self.session_start_time:
                duration = time.time() - self.session_start_time
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                
                self.session_label.setText(f"Session: {hours:02d}:{minutes:02d}:{seconds:02d}")
                self.duration_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Update statistics
            self.trades_label.setText(str(self.session_trades))
            self.profit_label.setText(f"${self.session_profit:.2f}")
            
            # Update profit label color
            if self.session_profit > 0:
                self.profit_label.setStyleSheet("font-weight: bold; color: #00FF00;")
            elif self.session_profit < 0:
                self.profit_label.setStyleSheet("font-weight: bold; color: #FF0000;")
            else:
                self.profit_label.setStyleSheet("font-weight: bold; color: #FFAA00;")
            
            # Success rate
            if self.session_trades > 0:
                profitable_trades = 1 if self.session_profit > 0 else 0  # Simplified
                success_rate = (profitable_trades / self.session_trades) * 100
                self.success_label.setText(f"{success_rate:.0f}%")
            
            # Daily target progress (assuming $0.20 daily target)
            daily_target = 0.20
            if daily_target > 0:
                progress = min(100, (self.session_profit / daily_target) * 100)
                self.daily_progress.setValue(max(0, int(progress)))
                
        except Exception as e:
            print(f"Display update error: {e}")

    def update_opportunities(self, opportunities):
        """Update opportunities display (external call)"""
        try:
            if not opportunities:
                return
                
            self.opportunities_text.append("📊 New opportunities detected:")
            
            for opp in opportunities[:5]:  # Show top 5
                pair = opp.get('pair', 'UNKNOWN')
                score = opp.get('score', 0)
                confidence = opp.get('confidence', 0)
                reason = opp.get('reason', 'Unknown')
                
                color = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🔴"
                
                self.opportunities_text.append(
                    f"  {color} {pair}: {score:.2f} ({confidence:.1%}) - {reason}"
                )
            
            # Auto-scroll
            scrollbar = self.opportunities_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        except Exception as e:
            print(f"Update opportunities error: {e}")

    def update_trade_stats(self, trade_data):
        """Update trade statistics (external call)"""
        try:
            pair = trade_data.get('pair', 'UNKNOWN')
            action = trade_data.get('action', 'UNKNOWN')
            pnl = trade_data.get('pnl', 0)
            
            if action in ['open', 'close']:
                self.session_trades += 1
                if action == 'close':
                    self.session_profit += pnl
                    
        except Exception as e:
            print(f"Trade stats update error: {e}")

    def get_current_settings(self):
        """Get current settings dictionary"""
        try:
            return {
                'auto_trading': self.auto_trading_cb.isChecked(),
                'position_size': self.position_size_spin.value(),
                'max_positions': self.max_positions_spin.value(),
                'confidence_threshold': self.confidence_spin.value(),
                'stop_loss_pct': self.stop_loss_spin.value(),
                'trading_mode': 'live',
                'scan_interval': 30,
                'risk_per_trade': 4.0,
                'take_profit_target': 0.3
            }
        except Exception as e:
            print(f"Get settings error: {e}")
            return {
                'auto_trading': False,
                'position_size': 50.0,
                'max_positions': 1,
                'confidence_threshold': 0.7,
                'stop_loss_pct': 4.0
            }

    def get_session_stats(self):
        """Get session statistics"""
        try:
            duration_minutes = 0
            if self.session_start_time:
                duration_minutes = (time.time() - self.session_start_time) / 60
                
            return {
                'session_trades': self.session_trades,
                'session_profit': self.session_profit,
                'session_duration_minutes': duration_minutes,
                'is_trading': self.is_trading,
                'success_rate': (1 if self.session_profit > 0 else 0) if self.session_trades > 0 else 0
            }
        except Exception as e:
            print(f"Session stats error: {e}")
            return {
                'session_trades': 0,
                'session_profit': 0.0,
                'session_duration_minutes': 0,
                'is_trading': False,
                'success_rate': 0
            }

    def is_trading_active(self):
        """Check if trading is currently active"""
        return self.is_trading

    def closeEvent(self, event):
        """Handle widget close event"""
        try:
            if self.worker:
                self.worker.stop_trading()
            event.accept()
        except Exception as e:
            print(f"Close event error: {e}")
            event.accept()


# Standalone testing
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Test the panel
    panel = LiveTradingPanel()
    panel.show()
    
    # Test with mock opportunities
    test_opportunities = [
        {'pair': 'ADAUSD', 'score': 0.85, 'confidence': 0.75, 'reason': 'Volume spike'},
        {'pair': 'SOLUSD', 'score': 0.72, 'confidence': 0.68, 'reason': 'Bollinger breakout'},
        {'pair': 'DOTUSD', 'score': 0.91, 'confidence': 0.82, 'reason': 'Perfect storm'}
    ]
    
    # Simulate opportunity update after 3 seconds
    QTimer.singleShot(3000, lambda: panel.update_opportunities(test_opportunities))
    
    sys.exit(app.exec_())
