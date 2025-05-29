# gui/pair_list_widget.py

from PyQt5.QtWidgets import QListWidget

class PairListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(100)
        self.setStyleSheet("background-color: #1c1c1c; color: white; font-weight: bold;")
        self.add_default_pairs()

    def add_default_pairs(self):
        pairs = ["XBT/USD", "ETH/USD", "ADA/USD", "XRP/USD", "SOL/USD"]
        self.addItems(pairs)

    def get_selected_pair(self):
        item = self.currentItem()
        return item.text() if item else "XBT/USD"

