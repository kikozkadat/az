# utils/history_analyzer.py

import csv
import os

class HistoryAnalyzer:
    def __init__(self, log_path="logs/trade_log.csv"):
        self.log_path = log_path

    def analyze_pair(self, pair, limit=1000):
        """
        Visszaadja a megadott párhoz:
        - win_rate: nyereséges tradek aránya
        - avg_pnl: átlagos profit/loss
        """
        if not os.path.exists(self.log_path):
            return {"win_rate": 0.0, "avg_pnl": 0.0}

        with open(self.log_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r["pair"] == pair]

        if not rows:
            return {"win_rate": 0.0, "avg_pnl": 0.0}

        rows = rows[-limit:]
        pnls = [float(r["pnl"]) for r in rows]
        wins = [p for p in pnls if p > 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0

        return {
            "win_rate": round(win_rate, 4),
            "avg_pnl": round(avg_pnl, 4)
        }

