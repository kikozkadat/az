import time
from data.pair_filter import get_usd_pairs
from core.scoring_scheduler import ScoringScheduler
from core.trade_manager import TradeManager
from core.position_manager import PositionManager
from core.sl_tp_helper import calculate_sl_tp
from data.kraken_api_client import KrakenAPIClient
from utils.trade_logger import TradeLogger
from config import TRADE_ENABLED, PER_TRADE_USD, MAX_PARALLEL_POSITIONS

scheduler = ScoringScheduler()
trader = TradeManager()
positions = PositionManager()
kraken = KrakenAPIClient()
logger = TradeLogger()

CHECK_INTERVAL = 300  # 5 perc
FULL_SCAN_INTERVAL = 3

cycle = 0

while True:
    if not TRADE_ENABLED:
        print("[INFO] Kereskedés inaktív.")
        time.sleep(CHECK_INTERVAL)
        continue

    cycle += 1
    pairs = get_usd_pairs()

    if cycle % FULL_SCAN_INTERVAL != 0:
        pairs = pairs[:50]

    print(f"[INFO] Értékelés {len(pairs)} párra...")
    scored = scheduler.fetch_and_score(pairs)

    # --- SL/TP pozíciófigyelés ---
    pos = positions.active_position
    if pos:
        ticker = kraken.get_ticker(pos['pair'])
        current_price = float(ticker[pos['pair']]['c'][0])
        direction = pos['side']

        if direction == "BUY":
            if current_price <= pos['sl']:
                print(f"[ZÁRÁS - STOPLOSS] {pos['pair']} @ {current_price}")
                trader.close_market_position(pos['pair'], direction, pos['volume'])
                positions.close_position(current_price)
                pnl = (current_price - pos['entry_price']) * pos['volume']
                logger.log(pos['pair'], direction, pos['entry_price'], current_price, pos['volume'], pnl, "SL")
                time.sleep(CHECK_INTERVAL)
                continue
            elif current_price >= pos['tp']:
                print(f"[ZÁRÁS - TAKEPROFIT] {pos['pair']} @ {current_price}")
                trader.close_market_position(pos['pair'], direction, pos['volume'])
                positions.close_position(current_price)
                pnl = (current_price - pos['entry_price']) * pos['volume']
                logger.log(pos['pair'], direction, pos['entry_price'], current_price, pos['volume'], pnl, "TP")
                time.sleep(CHECK_INTERVAL)
                continue

    open_positions = 1 if positions.active_position else 0
    if open_positions >= MAX_PARALLEL_POSITIONS:
        print("[INFO] Max pozíció nyitva. Várakozás...")
        time.sleep(CHECK_INTERVAL)
        continue

    for pair, score in scored:
        if score >= 3:
            ticker = kraken.get_ticker(pair)
            price = float(ticker[pair]['c'][0])
            volume = round(PER_TRADE_USD / price, 6)
            sl, tp = calculate_sl_tp(price, direction="BUY")

            print(f"[TRADE] Nyitás: {pair} ár={price}, SL={sl}, TP={tp}")
            trader.place_order(pair, side="BUY", volume=volume)
            positions.open_position(pair, "BUY", price, volume, sl, tp)
            break

    time.sleep(CHECK_INTERVAL)

