from PyQt5.QtWidgets import QVBoxLayout, QLabel, QGroupBox, QGridLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class DashboardPanel(QGroupBox):
    def __init__(self):
        super().__init__("Jelentés és pozíció")
        self.setStyleSheet("QGroupBox { font-weight: bold; font-size: 15px; color: white; }")
        layout = QGridLayout()

        font = QFont()
        font.setPointSize(10)

        self.labels = {}
        fields = [
            "Számla egyenleg", "Aktuális pozíció", "P/L", "P/L (összes)",
            "StopLoss", "TakeProfit", "Jel", "RSI", "EMA gyors", "EMA lassú",
            "Trade-ek száma", "Win %", "Átlag PnL"
        ]
        for i, field in enumerate(fields):
            key = QLabel(field + ":")
            key.setFont(font)
            key.setStyleSheet("color: #CCCCCC")
            value = QLabel("-")
            value.setFont(font)
            value.setStyleSheet("color: #00FFAA")
            layout.addWidget(key, i, 0)
            layout.addWidget(value, i, 1)
            self.labels[field] = value

        self.setLayout(layout)

    def update_field(self, field, value):
        if field in self.labels:
            self.labels[field].setText(str(value))

