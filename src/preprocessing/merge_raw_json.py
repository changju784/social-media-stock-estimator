"""
Merge v2 raw Reddit dataset from:
    data/raw/v2/<company>/<subreddit>.json
into a single CSV.

Outputs:
    data/interim/merged_reddit.csv
"""

import os
import json
import pandas as pd
from src.config import RAW_DIR, INTERIM_DIR

OUT_PATH = f"{INTERIM_DIR}/merged_reddit.csv"


def merge_raw_files():
    records = []

    for company in os.listdir(RAW_DIR):
        company_path = os.path.join(RAW_DIR, company)
        if not os.path.isdir(company_path):
            continue

        ticker = company.upper()   # placeholder until you add a ticker mapping

        for file in os.listdir(company_path):
            if not file.endswith(".json"):
                continue

            subreddit = file.replace(".json", "")
            filepath = os.path.join(company_path, file)

            with open(filepath, "r", encoding="utf-8") as f:
                posts = json.load(f)

            for post in posts:
                records.append({
                    "id": post.get("id"),
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", ""),
                    "comments": " ".join(post.get("comments", [])),
                    "score": post.get("score", 0),
                    "created_utc": post.get("created_utc"),
                    "subreddit": subreddit,
                    "company": company,
                    "ticker": ticker,
                })

            print(f"Loaded {len(posts)} posts from {filepath}")

    df = pd.DataFrame(records).drop_duplicates(subset=["id"])
    os.makedirs(INTERIM_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[merge_raw_json] merged {len(df)} posts → {OUT_PATH}")


if __name__ == "__main__":
    merge_raw_files()
