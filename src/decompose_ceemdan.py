import pandas as pd
import numpy as np
from PyEMD import CEEMDAN
import os

def load_stock_data(path="data/nifty_prices.csv", downsample_factor=5):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    
    df = pd.read_csv(path, skiprows=[1, 2])  # Skip metadata rows
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Close']]
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Downsample and reset index
    df = df.iloc[::downsample_factor].reset_index(drop=True)
    
    # Save matching rows for later alignment
    df.to_csv("data/ceemdan_aligned_prices.csv", index=False)
    
    return df

def apply_ceemdan(series, max_imfs=8):
    ceemdan = CEEMDAN()
    ceemdan.trials = 100
    ceemdan.noise_width = 0.05
    
    imfs = ceemdan.ceemdan(series.values)
    
    if max_imfs:
        imfs = imfs[:max_imfs]
    
    return imfs.T  # Transpose: (timesteps, num_imfs)

def save_imfs(imfs, save_path="data/imfs.csv"):
    df = pd.DataFrame(imfs, columns=[f"IMF_{i+1}" for i in range(imfs.shape[1])])
    df.to_csv(save_path, index=False)
    print(f"✅ Saved {imfs.shape[1]} IMFs to {save_path}")

if __name__ == "__main__":
    df = load_stock_data()
    close_prices = df['Close']
    imfs = apply_ceemdan(close_prices)
    save_imfs(imfs)
