class PairManager:
    def __init__(self, pairs=None):
        if pairs is None:
            pairs = ["BTCUSD", "ETHUSD"]
        self.pairs = pairs
        self.data_cache = {pair: None for pair in pairs}
        self.last_signals = {pair: "" for pair in pairs}

    def update_pair_data(self, pair, ohlc_data):
        self.data_cache[pair] = ohlc_data

    def get_cached_data(self, pair):
        return self.data_cache.get(pair)

    def set_signal(self, pair, signal):
        self.last_signals[pair] = signal

    def get_signal(self, pair):
        return self.last_signals.get(pair)

