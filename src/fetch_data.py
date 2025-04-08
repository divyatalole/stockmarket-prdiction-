import yfinance as yf
import pandas as pd
import os

def fetch_nifty50_yf(save_path="data/nifty_prices.csv"):
    symbol = "^NSEI"  # NIFTY 50 index
    df = yf.download(symbol, start="2015-01-01", progress=False)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    df.reset_index(inplace=True)
    df.rename(columns={"Date": "Date"}, inplace=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ NIFTY 50 data saved to {save_path}")

if __name__ == "__main__":
    fetch_nifty50_yf()
