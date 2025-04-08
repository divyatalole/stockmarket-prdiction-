import feedparser
import pandas as pd
from datetime import datetime, timedelta
import os
from urllib.parse import quote

def scrape_google_news(query="NIFTY 50", days_back=30, save_path="data/nifty_scraped_news.csv"):
    all_articles = []

    for i in range(days_back):
        date = (datetime.today() - timedelta(days=i)).strftime('%Y-%m-%d')
        print(f"🔍 Fetching for {date}...")

        # Google News RSS for date-specific search
        url = f"https://news.google.com/rss/search?q={quote(query)}+when:{i}d&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)

        for entry in feed.entries:
            all_articles.append({
                "Date": date,
                "Title": entry.title,
                "Summary": entry.get("summary", "")
            })

    # Save results
    df = pd.DataFrame(all_articles)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n✅ Scraped {len(df)} articles → {save_path}")

if __name__ == "__main__":
    scrape_google_news(query="NIFTY 50", days_back=30)
