# core/trailing_stop.py

class TrailingStopManager:
    def __init__(self, target_profit=0.005):
        self.target_profit = target_profit
        self.pair = None
        self.position_manager = None
        self.trader = None
        self.active = False
        self.entry_price = 0.0
        self.max_price = 0.0
        self.current_stop_price = 0.0

    def link(self, pair, position_manager, trader):
        self.pair = pair
        self.position_manager = position_manager
        self.trader = trader
        pos = self.position_manager.get_position(pair)
        if not pos:
            return

        self.entry_price = pos["entry_price"]
        self.max_price = self.entry_price
        self.active = True

    def update(self, current_price):
        if not self.active or self.pair is None:
            return

        pos = self.position_manager.get_position(self.pair)
        if not pos:
            self.active = False
            return

        pnl = current_price - self.entry_price if pos["side"] == "buy" else self.entry_price - current_price

        if current_price > self.max_price:
            self.max_price = current_price

        trailing_trigger_price = self.entry_price * (1 + self.target_profit) if pos["side"] == "buy" else self.entry_price * (1 - self.target_profit)

        if self.max_price >= trailing_trigger_price:
            if pos["side"] == "buy":
                self.current_stop_price = self.max_price * (1 - self.target_profit)
            else:
                self.current_stop_price = self.max_price * (1 + self.target_profit)

        if pos["side"] == "buy" and current_price < self.current_stop_price:
            self.trader.close_market_position(self.pair, "buy", pos["volume"])
            self.position_manager.close_position(self.pair)
            self.active = False
            print("Trailing stop executed -> BUY pozíció lezárva")

        elif pos["side"] == "sell" and current_price > self.current_stop_price:
            self.trader.close_market_position(self.pair, "sell", pos["volume"])
            self.position_manager.close_position(self.pair)
            self.active = False
            print("Trailing stop executed -> SELL pozíció lezárva")

        print(f"[TS DEBUG] Ár: {current_price:.4f}, PnL: {pnl:.4f}, Max: {self.max_price:.4f}, Aktív: {self.active}")

