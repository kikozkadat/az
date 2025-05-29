# trade_executor.py – végrehajtási logika (vétel)

from kraken_data import place_market_order
from wallet_tracker import get_balance

FEE_RATE = 0.0026  # Kraken taker fee pl. 0.26%


def execute_trade(symbol, amount_usd):
    balance = get_balance("USD")

    if balance < amount_usd:
        print(f"Nincs elég egyenleg. Egyenleg: {balance:.2f} USD")
        return False

    # Lekérdezzük az aktuális piaci árat (egyszerűsítve)
    price_data = place_market_order(symbol, amount_usd, simulate=True)  # simulate: True = csak próbálja

    if not price_data:
        print("Sikertelen árlekérdezés vagy megbízás.")
        return False

    fill_price = price_data["avg_price"]
    amount_coin = amount_usd / fill_price
    fee = amount_usd * FEE_RATE
    estimated_net = amount_usd - fee

    print(f"Vétel: {symbol}, Ár: {fill_price:.4f}, Mennyiség: {amount_coin:.6f}, Költség: {fee:.4f}, Tiszta: {estimated_net:.4f}")

    # Végleges megbízás leadása
    confirmed = place_market_order(symbol, amount_usd, simulate=False)
    return confirmed is not None

