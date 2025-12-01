"""
Merge all crawled raw JSON files from:
    data/raw/<company>/<subreddit>.json
into a single unified CSV.
"""

import os
import json
import pandas as pd

RAW_DIR = "data/raw"
OUT_PATH = "data/interim/merged_reddit.csv"


def merge_raw_files():
    records = []

    # Each folder under data/raw corresponds to a company name
    for company in os.listdir(RAW_DIR):
        company_dir = os.path.join(RAW_DIR, company)
        if not os.path.isdir(company_dir):
            continue

        ticker = company.upper()  # e.g., "amazon" -> "AMAZON" (you can switch to real tickers later)

        for file in os.listdir(company_dir):
            if not file.endswith(".json"):
                continue

            filepath = os.path.join(company_dir, file)
            subreddit = file.replace(".json", "")

            with open(filepath, "r", encoding="utf-8") as f:
                posts = json.load(f)

            for post in posts:
                records.append({
                    "id": post.get("id"),
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", ""),
                    "comments": " ".join(post.get("comments", [])),
                    "score": post.get("score", 0),
                    "subreddit": subreddit,
                    "company_raw": company,
                    "ticker": ticker,
                    "created_utc": post.get("created_utc"),
                })

            print(f"Loaded {len(posts)} posts from {filepath}")

    df = pd.DataFrame(records)

    # remove duplicates by post id
    df = df.drop_duplicates(subset=["id"], keep="first")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n[merge_raw_json] merged {len(df)} total posts → {OUT_PATH}")


if __name__ == "__main__":
    merge_raw_files()
