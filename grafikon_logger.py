import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE = "logs/trade_log.csv"

def plot_total_pnl():
    if not os.path.exists(LOG_FILE):
        print("Nincs trade_log.csv fájl.")
        return

    df = pd.read_csv(LOG_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['cumulative'] = df['pnl'].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['cumulative'], label="Összesített PnL", color='green')
    plt.title("Profit alakulása")
    plt.xlabel("Idő")
    plt.ylabel("USD")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_total_pnl()

