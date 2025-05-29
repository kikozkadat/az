from ai.ollama_engine import OllamaEngine

# Teszt adatok: dummy kereskedési log
example_trades = [
    {"pair": "BTCUSD", "action": "BUY", "price": 29500, "rsi": 27.5, "ema_fast": 29480, "ema_slow": 29300},
    {"pair": "BTCUSD", "action": "SELL", "price": 29900, "rsi": 71.2, "ema_fast": 29890, "ema_slow": 29700},
    {"pair": "ETHUSD", "action": "BUY", "price": 1840, "rsi": 30.1, "ema_fast": 1835, "ema_slow": 1820},
    {"pair": "ETHUSD", "action": "SELL", "price": 1895, "rsi": 76.4, "ema_fast": 1890, "ema_slow": 1870},
]

engine = OllamaEngine("mistral")
recommendation = engine.analyze_logs(example_trades)

print("AI javaslat:")
print(recommendation)

