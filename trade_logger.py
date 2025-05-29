import os
import csv
import time

class TradeLogger:
    def __init__(self, log_path="logs/trade_log.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.headers = ["timestamp", "pair", "side", "entry_price", "exit_price", "volume", "pnl", "reason", "fee"]

        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, pair, side, entry_price, exit_price, volume, pnl, reason, fee="N/A"):
        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                pair,
                side,
                round(entry_price, 6),
                round(exit_price, 6) if exit_price else "N/A",
                round(volume, 6),
                round(pnl, 4),
                reason,
                fee
            ])

