from strategy.indicator_engine import IndicatorEngine

class StrategyLogic:
    def __init__(self):
        self.indicators = IndicatorEngine()

    def generate_signal(self, df):
        if df is None or len(df) < 30:
            return "NO_DATA"

        rsi = self.indicators.compute_rsi(df)
        stoch_rsi = self.indicators.compute_stoch_rsi(df)
        ema_fast, ema_slow = self.indicators.compute_ema_crossover(df)
        upper_band, lower_band = self.indicators.compute_bollinger(df)

        last_close = df['close'].iloc[-1]
        last_rsi = rsi.iloc[-1]
        last_stoch = stoch_rsi.iloc[-1]
        last_ema_fast = ema_fast.iloc[-1]
        last_ema_slow = ema_slow.iloc[-1]

        buy = (
            last_rsi < 30 and
            last_stoch < 0.2 and
            last_ema_fast > last_ema_slow and
            last_close < lower_band.iloc[-1]
        )

        sell = (
            last_rsi > 70 and
            last_stoch > 0.8 and
            last_ema_fast < last_ema_slow and
            last_close > upper_band.iloc[-1]
        )

        if buy:
            return "BUY"
        elif sell:
            return "SELL"
        else:
            return "HOLD"

