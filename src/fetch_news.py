import requests
import pandas as pd
import os

def fetch_nifty50_news(api_key, save_path="data/nifty_news.csv"):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "NSEI",
        "apikey": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    articles = data.get("feed", [])
    if not articles:
        print("❌ No articles returned.")
        return

    df = pd.DataFrame(articles)
    df["Date"] = pd.to_datetime(df["time_published"].str[:10])
    df["Sentiment"] = df["overall_sentiment_score"]

    df = df[["Date", "title", "summary", "Sentiment"]]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ News data saved to {save_path}")

if __name__ == "__main__":
    api_key = "YOUR_API_KEY"  # Replace with your Alpha Vantage API key
    fetch_nifty50_news(api_key)