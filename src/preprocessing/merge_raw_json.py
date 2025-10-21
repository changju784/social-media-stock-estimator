'''
Merge all crawled raw JSON files into a single CSV file.
'''
import os, json
import pandas as pd

RAW_PATH = "data/raw"
OUT_PATH = "data/interim/merged_reddit.csv"

def merge_raw_files():
    records = []
    for file in os.listdir(RAW_PATH):
        if not file.endswith(".json"):
            continue
        # infer ticker from filename (e.g. investing_amazon.json, wallstreetbets_apple.json)
        fname = file.lower()
        ticker_map = {
            "amazon": "AMZN",
            "apple": "AAPL",
            "google": "GOOG",
            "meta": "META",
            "netflix": "NFLX",
        }
        ticker = None
        for key, val in ticker_map.items():
            if key in fname:
                ticker = val
                break
        with open(os.path.join(RAW_PATH, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for post in data:
                records.append({
                    "id": post.get("id"),
                    "title": post.get("title"),
                    "selftext": post.get("selftext", ""),
                    "score": post.get("score", 0),
                    "created_utc": post.get("created_utc"),
                    "subreddit": post.get("subreddit"),
                    "ticker": ticker,
                })
    df = pd.DataFrame(records).drop_duplicates(subset=["id"])
    df.to_csv(OUT_PATH, index=False)
    print(f"[merge_raw_json] merged {len(df)} posts → {OUT_PATH}")

if __name__ == "__main__":
    merge_raw_files()
