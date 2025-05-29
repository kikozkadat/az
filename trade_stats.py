import pandas as pd
import os

LOG_FILE = "logs/trade_log.csv"

def summarize_trades():
    if not os.path.exists(LOG_FILE):
        print("Nincs trade_log.csv fájl.")
        return

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print("Nincs adat a logban.")
        return

    total = len(df)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    zero = df[df['pnl'] == 0]

    print("--- Kereskedési statisztika ---")
    print(f"Összes ügylet: {total}")
    print(f"Nyereséges: {len(wins)}")
    print(f"Veszteséges: {len(losses)}")
    print(f"Nulla eredményű: {len(zero)}")
    print(f"Találati arány: {len(wins) / total * 100:.2f}%")
    print(f"Átlag profit: {df['pnl'].mean():.2f} USD")
    print(f"Max nyereség: {df['pnl'].max():.2f} USD")
    print(f"Max veszteség: {df['pnl'].min():.2f} USD")

if __name__ == "__main__":
    summarize_trades()

