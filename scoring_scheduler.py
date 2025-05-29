from strategy.scorer import CoinScorer
import time

class ScoringScheduler:
    def __init__(self):
        self.scorer = CoinScorer()

    def fetch_and_score(self, coin_list):
        """
        Végigmegy a kapott coin_list-en (list of dict), pontozza őket,
        és visszaadja a score-olt listát.
        """
        scored_coins = []
        for coin in coin_list:
            score = self.scorer.score_coin(coin)
            scored_coin = coin.copy()
            scored_coin['score'] = score
            scored_coins.append(scored_coin)
        # Legjobb score szerinti rendezés
        scored_coins.sort(key=lambda c: c['score'], reverse=True)
        return scored_coins

# Példa használat
if __name__ == "__main__":
    # Fake adatokkal
    coins = [
        {'symbol': 'BTCUSD', 'volume_15m_avg': 10000, 'volume_last': 11000, 'rsi_3m': 48, 'rsi_15m': 50, 'close': 70000, 'boll_3m_upper': 70100, 'boll_3m_lower': 69500},
        {'symbol': 'XRPUSD', 'volume_15m_avg': 800, 'volume_last': 2400, 'rsi_3m': 55, 'rsi_15m': 45, 'close': 0.55, 'boll_3m_upper': 0.56, 'boll_3m_lower': 0.52}
    ]
    scheduler = ScoringScheduler()
    results = scheduler.fetch_and_score(coins)
    for c in results:
        print(c['symbol'], c['score'])

