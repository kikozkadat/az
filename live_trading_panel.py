from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QMainWindow, QApplication
from PyQt5.QtCore import QThread, pyqtSignal
import time
import random
import sys

# Dummy scorer and API client for simulation
class DummyAPIClient:
    def get_market_data(self):
        # Random market data simulation
        coins = [
            {"symbol": "BTCUSD", "score": random.uniform(0, 1)},
            {"symbol": "ETHUSD", "score": random.uniform(0, 1)},
            {"symbol": "XRPUSD", "score": random.uniform(0, 1)}
        ]
        return coins

# Background thread that fetches market data
class TradingWorker(QThread):
    update_signal = pyqtSignal(list)

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self.running = True

    def run(self):
        while self.running:
            try:
                data = self.api_client.get_market_data()
                self.update_signal.emit(data)
            except Exception as e:
                print(f"[Worker Error]: {e}")
            time.sleep(5)  # update every 5 seconds

    def stop(self):
        self.running = False

# GUI panel displaying live market scores
class LiveTradingPanel(QWidget):
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Live Trading Panel")

        self.layout = QVBoxLayout()
        self.label = QLabel("Live Coin Scores")
        self.list_widget = QListWidget()

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.list_widget)
        self.setLayout(self.layout)

        # Setup background worker
        self.api_client = api_client
        self.worker = TradingWorker(self.api_client)
        self.worker.update_signal.connect(self.update_list)
        self.worker.start()

        self._trading_active = False  # New internal state

    def update_list(self, coin_data):
        self.list_widget.clear()
        for coin in coin_data:
            text = f"{coin['symbol']}: score={coin['score']:.3f}"
            self.list_widget.addItem(text)

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()

    def is_trading_active(self):
        return self._trading_active

    def start_live_trading(self):
        self._trading_active = True

    def stop_live_trading(self):
        self._trading_active = False

# Main window that includes the LiveTradingPanel
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot Dashboard")
        api_client = DummyAPIClient()
        self.panel = LiveTradingPanel(api_client)
        self.setCentralWidget(self.panel)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

