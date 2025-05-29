from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QFormLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt

class SettingsPanel(QGroupBox):
    def __init__(self):
        super().__init__("Beállítások")
        self.setStyleSheet("QGroupBox { font-weight: bold; font-size: 15px; color: white; }")
        self.setMaximumWidth(250)

        layout = QVBoxLayout()
        form = QFormLayout()

        self.usd_input = QLineEdit("30")  # új mező USD-ben
        self.sl_input = QLineEdit("1.5")
        self.tp_input = QLineEdit("2.5")

        form.addRow(QLabel("USD érték (vétel):"), self.usd_input)
        form.addRow(QLabel("Stop Loss (%):"), self.sl_input)
        form.addRow(QLabel("Take Profit (%):"), self.tp_input)

        layout.addLayout(form)

        self.save_button = QPushButton("Mentés")
        layout.addWidget(self.save_button)
        self.setLayout(layout)

    def get_settings(self):
        return {
            "usd_value": float(self.usd_input.text()),
            "stoploss": float(self.sl_input.text()),
            "takeprofit": float(self.tp_input.text())
        }

