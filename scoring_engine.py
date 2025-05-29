# Ezt a fájlt scoring_engine.py néven kell elmenteni a projekt gyökerébe vagy a megfelelő modulba.
# Ez a pontozási logika kiegészíti az AI döntéshozatalt.

from indicators import calculate_bollinger_bands, calculate_atr, calculate_rsi
from utils import calculate_correlation, estimate_profitability

MIN_USD_VOLUME = 5000
MIN_ATR_RATIO = 1.05
BOLLINGER_THRESHOLD = 0.98


def filter_and_score_assets(market_data, btc_data, eth_data):
    filtered_assets = []

    for coin in market_data:
        data = market_data[coin]

        if data['usd_volume'] < MIN_USD_VOLUME:
            continue

        if data['atr'] < data['atr_mean'] * MIN_ATR_RATIO:
            continue

        price = data['price']
        upper_band = data['bollinger_upper']
        if price < upper_band * BOLLINGER_THRESHOLD:
            continue

        btc_trigger = btc_data['breakout_prob'] > 0.7 and data['correlation_btc'] > 0.85
        eth_trigger = eth_data['breakout_prob'] > 0.7 and data['correlation_eth'] > 0.85

        if not (btc_trigger or eth_trigger):
            continue

        profit_estimate = estimate_profitability(price, data['spread'], data['fee'], target_profit=0.2)
        if not profit_estimate['feasible']:
            continue

        filtered_assets.append({
            'symbol': coin,
            'score': profit_estimate['score'],
            'trigger': 'BTC' if btc_trigger else 'ETH',
            'expected_profit': profit_estimate['expected']
        })

    filtered_assets.sort(key=lambda x: x['score'], reverse=True)
    return filtered_assets

