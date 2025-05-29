import requests

def get_usd_pairs():
    url = "https://api.kraken.com/0/public/AssetPairs"
    response = requests.get(url)
    data = response.json()

    pairs = []
    for symbol, details in data.get("result", {}).items():
        if details.get("quote") == "ZUSD":
            pairs.append(symbol)

    return sorted(pairs)

if __name__ == "__main__":
    for p in get_usd_pairs():
        print(p)

