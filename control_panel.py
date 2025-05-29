from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                            QSpinBox, QHBoxLayout, QCheckBox, QDoubleSpinBox, QGroupBox)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont

class ControlPanel(QWidget):
    # Signals for communication with main window
    toggle_trading = pyqtSignal(bool)
    set_max_positions = pyqtSignal(int)
    force_refresh = pyqtSignal()
    close_all_positions = pyqtSignal()
    update_sl_tp_settings = pyqtSignal(float, float)  # stop_loss%, take_profit%
    emergency_stop = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.trading_enabled = False
        self.setup_ui()
        self.apply_styles()

    def setup_ui(self):
        """Initialize the control panel UI"""
        layout = QVBoxLayout()
        
        # Trading control group
        trading_group = self.create_trading_controls()
        layout.addWidget(trading_group)
        
        # Position management group
        position_group = self.create_position_controls()
        layout.addWidget(position_group)
        
        # Risk management group
        risk_group = self.create_risk_controls()
        layout.addWidget(risk_group)
        
        # Emergency controls
        emergency_group = self.create_emergency_controls()
        layout.addWidget(emergency_group)
        
        self.setLayout(layout)

    def create_trading_controls(self):
        """Create trading enable/disable controls"""
        group = QGroupBox("Kereskedési Vezérlés")
        layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Kereskedés: KIKAPCSOLVA")
        self.status_label.setStyleSheet("font-weight: bold; color: #FF4444;")
        layout.addWidget(self.status_label)
        
        # Toggle button
        self.toggle_btn = QPushButton("Kereskedés BE/KI")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self._toggle_trading)
        layout.addWidget(self.toggle_btn)
        
        # Auto trading checkbox
        self.auto_trading_cb = QCheckBox("Automatikus kereskedés")
        self.auto_trading_cb.setChecked(False)
        layout.addWidget(self.auto_trading_cb)
        
        group.setLayout(layout)
        return group

    def create_position_controls(self):
        """Create position management controls"""
        group = QGroupBox("Pozíció Kezelés")
        layout = QVBoxLayout()
        
        # Max positions
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Max pozíciók:"))
        self.max_pos_spin = QSpinBox()
        self.max_pos_spin.setMinimum(1)
        self.max_pos_spin.setMaximum(20)
        self.max_pos_spin.setValue(3)
        self.max_pos_spin.valueChanged.connect(lambda val: self.set_max_positions.emit(val))
        pos_layout.addWidget(self.max_pos_spin)
        layout.addLayout(pos_layout)
        
        # Refresh button
        self.refresh_btn = QPushButton("Piacok frissítése")
        self.refresh_btn.clicked.connect(self.force_refresh.emit)
        layout.addWidget(self.refresh_btn)
        
        # Close all button
        self.close_all_btn = QPushButton("Összes pozíció zárása")
        self.close_all_btn.clicked.connect(self.close_all_positions.emit)
        self.close_all_btn.setStyleSheet("background-color: #FF6B35; color: white;")
        layout.addWidget(self.close_all_btn)
        
        group.setLayout(layout)
        return group

    def create_risk_controls(self):
        """Create risk management controls"""
        group = QGroupBox("Kockázatkezelés")
        layout = QVBoxLayout()
        
        # Stop Loss setting
        sl_layout = QHBoxLayout()
        sl_layout.addWidget(QLabel("Stop Loss (%):"))
        self.sl_spin = QDoubleSpinBox()
        self.sl_spin.setMinimum(0.1)
        self.sl_spin.setMaximum(10.0)
        self.sl_spin.setValue(1.5)
        self.sl_spin.setDecimals(1)
        self.sl_spin.setSingleStep(0.1)
        sl_layout.addWidget(self.sl_spin)
        layout.addLayout(sl_layout)
        
        # Take Profit setting
        tp_layout = QHBoxLayout()
        tp_layout.addWidget(QLabel("Take Profit (%):"))
        self.tp_spin = QDoubleSpinBox()
        self.tp_spin.setMinimum(0.1)
        self.tp_spin.setMaximum(20.0)
        self.tp_spin.setValue(2.5)
        self.tp_spin.setDecimals(1)
        self.tp_spin.setSingleStep(0.1)
        tp_layout.addWidget(self.tp_spin)
        layout.addLayout(tp_layout)
        
        # Apply risk settings button
        apply_risk_btn = QPushButton("Alkalmaz")
        apply_risk_btn.clicked.connect(self._update_risk_settings)
        layout.addWidget(apply_risk_btn)
        
        # Risk per trade
        risk_layout = QHBoxLayout()
        risk_layout.addWidget(QLabel("Kockázat/trade (%):"))
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setMinimum(0.1)
        self.risk_spin.setMaximum(5.0)
        self.risk_spin.setValue(1.0)
        self.risk_spin.setDecimals(1)
        risk_layout.addWidget(self.risk_spin)
        layout.addLayout(risk_layout)
        
        group.setLayout(layout)
        return group

    def create_emergency_controls(self):
        """Create emergency stop controls"""
        group = QGroupBox("Vészhelyzeti Vezérlés")
        layout = QVBoxLayout()
        
        # Emergency stop button
        self.emergency_btn = QPushButton("VÉSZLEÁLLÍTÁS")
        self.emergency_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF0000;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border: 2px solid #CC0000;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
            QPushButton:checked {
                background-color: #00AA00;
            }
            QLabel {
                color: white;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2D2D30;
                color: white;
                border: 1px solid #555;
                padding: 2px;
            }
            QCheckBox {
                color: white;
            }
        """)

    def _toggle_trading(self):
        """Handle trading toggle"""
        self.trading_enabled = not self.trading_enabled
        
        if self.trading_enabled:
            self.status_label.setText("Kereskedés: BEKAPCSOLVA")
            self.status_label.setStyleSheet("font-weight: bold; color: #00AA00;")
            self.toggle_btn.setText("Kereskedés KIKAPCSOLÁSA")
        else:
            self.status_label.setText("Kereskedés: KIKAPCSOLVA")
            self.status_label.setStyleSheet("font-weight: bold; color: #FF4444;")
            self.toggle_btn.setText("Kereskedés BEKAPCSOLÁSA")
            
        self.toggle_trading.emit(self.trading_enabled)

    def _update_risk_settings(self):
        """Update risk management settings"""
        sl_percent = self.sl_spin.value()
        tp_percent = self.tp_spin.value()
        self.update_sl_tp_settings.emit(sl_percent, tp_percent)
        print(f"[CONTROL] Risk settings updated: SL={sl_percent}%, TP={tp_percent}%")

    def _emergency_stop(self):
        """Handle emergency stop"""
        self.trading_enabled = False
        self.toggle_btn.setChecked(False)
        self.status_label.setText("Kereskedés: VÉSZLEÁLLÍTÁS")
        self.status_label.setStyleSheet("font-weight: bold; color: #FF0000;")
        self.emergency_status.setText("Rendszer: VÉSZLEÁLLÍTÁS AKTÍV")
        self.emergency_status.setStyleSheet("color: #FF0000; font-weight: bold;")
        
        # Disable all trading controls
        self.toggle_btn.setEnabled(False)
        self.auto_trading_cb.setEnabled(False)
        
        self.emergency_stop.emit()
        print("[EMERGENCY] Emergency stop activated!")

    def reset_emergency(self):
        """Reset emergency stop state"""
        self.emergency_status.setText("Rendszer: Normál")
        self.emergency_status.setStyleSheet("color: #00AA00; font-weight: bold;")
        self.toggle_btn.setEnabled(True)
        self.auto_trading_cb.setEnabled(True)
        self.status_label.setText("Kereskedés: KIKAPCSOLVA")
        self.status_label.setStyleSheet("font-weight: bold; color: #FF4444;")
        print("[CONTROL] Emergency state reset")

    def get_settings(self):
        """Get current control panel settings"""
        return {
            "trading_enabled": self.trading_enabled,
            "auto_trading": self.auto_trading_cb.isChecked(),
            "max_positions": self.max_pos_spin.value(),
            "stop_loss_percent": self.sl_spin.value(),
            "take_profit_percent": self.tp_spin.value(),
            "risk_per_trade": self.risk_spin.value()
        }

    def set_position_count(self, current_positions, max_positions):
        """Update position count display"""
        self.max_pos_spin.setValue(max_positions)
        # You could add a label to show current vs max positions

    def update_trading_status(self, is_trading, reason=""):
        """Update trading status from external source"""
        if not is_trading and self.trading_enabled:
            self.trading_enabled = False
            self.toggle_btn.setChecked(False)
            self.status_label.setText(f"Kereskedés: LEÁLLÍTVA - {reason}")
            self.status_label.setStyleSheet("font-weight: bold; color: #FF8800;")
                background-color: #CC0000;
            }
            QPushButton:pressed {
                background-color: #990000;
            }
        """)
        self.emergency_btn.clicked.connect(self._emergency_stop)
        layout.addWidget(self.emergency_btn)
        
        # Emergency status
        self.emergency_status = QLabel("Rendszer: Normál")
        self.emergency_status.setStyleSheet("color: #00AA00; font-weight: bold;")
        layout.addWidget(self.emergency_status)
        
        group.setLayout(layout)
        return group

    def apply_styles(self):
        """Apply consistent styling"""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
