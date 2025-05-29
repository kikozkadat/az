# gui/profit_chart.py – Candlestick chart, Bollinger, EMA, SL/TP, aktuális ár

import sys
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import mplfinance as mpf

class ProfitChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(Figure())
        self.layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)
        self.data = None
        self.sl = None
        self.tp = None
        self.price = None
        self.bbands = None
        self.ema = None
        self.setMinimumHeight(380)

    def update_chart(self, ohlc_df, sl=None, tp=None, price=None, bbands=None, ema=None):
        self.ax.clear()
        self.data = ohlc_df
        self.sl = sl
        self.tp = tp
        self.price = price
        self.bbands = bbands
        self.ema = ema
        # Candlestick chart with mplfinance
        addplots = []
        if self.bbands is not None:
            addplots += [mpf.make_addplot(self.bbands['upper'], color='orange'),
                         mpf.make_addplot(self.bbands['lower'], color='orange')]
        if self.ema is not None:
            addplots.append(mpf.make_addplot(self.ema, color='cyan'))
        mpf.plot(self.data, type='candle', ax=self.ax, addplot=addplots, volume=True, style='charles', warn_too_much_data=10000)
        # SL/TP/Price lines
        if self.sl is not None:
            self.ax.axhline(self.sl, color='red', linestyle='--', lw=1, label='Stop Loss')
        if self.tp is not None:
            self.ax.axhline(self.tp, color='lime', linestyle='--', lw=1, label='Take Profit')
        if self.price is not None:
            self.ax.axhline(self.price, color='white', linestyle=':', lw=1, label='Current Price')
        self.ax.legend()
        self.canvas.draw()

