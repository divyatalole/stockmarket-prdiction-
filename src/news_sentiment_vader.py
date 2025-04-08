import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

def apply_vader_sentiment(input_path="data/nifty_scraped_news.csv", output_path="data/sentiment_scores.csv"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found!")

    df = pd.read_csv(input_path)
    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    for _, row in df.iterrows():
        text = f"{row['Title']} {row['Summary']}"
        score = analyzer.polarity_scores(text)['compound']
        sentiments.append(score)

    df['Sentiment'] = sentiments

    # Group sentiment per day (mean)
    daily_sentiment = df.groupby("Date")["Sentiment"].mean().reset_index()
    daily_sentiment.to_csv(output_path, index=False)

    print(f"✅ Sentiment scores saved to {output_path}")
    print(daily_sentiment.tail())

if __name__ == "__main__":
    apply_vader_sentiment()
