import pandas as pd
import os

def merge_all_sources(
    price_path="data/ceemdan_aligned_prices.csv",  # Use aligned prices
    imf_path="data/imfs.csv",
    sentiment_path="data/sentiment_scores.csv",
    save_path="data/final_dataset.csv"
):
    # Load and prepare stock prices
    stock_df = pd.read_csv(price_path)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    stock_df = stock_df[['Date', 'Close']].dropna().reset_index(drop=True)

    # Load IMFs
    imfs_df = pd.read_csv(imf_path)
    # Ensure we have same number of rows
    imfs_df = imfs_df.iloc[:len(stock_df)]
    merged_df = pd.concat([stock_df, imfs_df], axis=1)

    # Load sentiment and merge
    sentiment_df = pd.read_csv(sentiment_path)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    merged_df = pd.merge(merged_df, sentiment_df, on='Date', how='left')

    # Fill missing sentiment values (if any)
    merged_df['Sentiment'] = merged_df['Sentiment'].ffill().bfill()

    # Save the final dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged_df.to_csv(save_path, index=False)
    print(f"✅ Final merged dataset saved to {save_path}")
    print(f"📐 Shape: {merged_df.shape}")
    print(merged_df.head())

if __name__ == "__main__":
    merge_all_sources()
