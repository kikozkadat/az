from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
import os
import csv
from datetime import datetime

class StatsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.labels = {
            "total": QLabel("Összes trade: 0"),
            "wins": QLabel("Win %: 0%"),
            "avg": QLabel("Átlag PnL: 0"),
            "max_win": QLabel("Max profit: 0"),
            "max_loss": QLabel("Max veszteség: 0"),
            "today": QLabel("Mai PnL: 0")
        }

        for lbl in self.labels.values():
            self.layout.addWidget(lbl)

        self.update_stats()

    def update_stats(self):
        path = "logs/trade_log.csv"
        if not os.path.exists(path):
            return

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return

        pnls = [float(r["pnl"]) for r in rows]
        wins = [p for p in pnls if p > 0]

        today = datetime.now().strftime("%Y-%m-%d")
        today_pnls = [
            float(r["pnl"]) for r in rows
            if r["timestamp"].startswith(today)
        ]

        self.labels["total"].setText(f"Összes trade: {len(rows)}")
        self.labels["wins"].setText(f"Win %: {len(wins) / len(rows) * 100:.1f}%")
        self.labels["avg"].setText(f"Átlag PnL: {sum(pnls) / len(pnls):.2f}")
        self.labels["max_win"].setText(f"Max profit: {max(pnls):.2f}")
        self.labels["max_loss"].setText(f"Max veszteség: {min(pnls):.2f}")
        self.labels["today"].setText(f"Mai PnL: {sum(today_pnls):.2f}")

