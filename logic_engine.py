class LogicEngine:
    def __init__(self):
        self.history = {}

    def evaluate(self, df):
        if df is None or len(df) < 30:
            return "HOLD"

        for col in ['rsi', 'ema_fast', 'ema_slow']:
            if col not in df.columns:
                return "HOLD"

        last = df.iloc[-1]
        rsi = last['rsi']
        ema_fast = last['ema_fast']
        ema_slow = last['ema_slow']

        if rsi < 30 and ema_fast > ema_slow:
            return "BUY"
        elif rsi > 70 and ema_fast < ema_slow:
            return "SELL"
        else:
            return "HOLD"

    def update_history(self, pair, signal):
        if pair not in self.history:
            self.history[pair] = []
        self.history[pair].append(signal)
        if len(self.history[pair]) > 1000:
            self.history[pair] = self.history[pair][-1000:]

