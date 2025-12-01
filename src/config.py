# src/config.py

import os

RAW_DIR = "data/raw/v2"
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"

# Automatically detect all company folders under RAW_DIR
COMPANIES = [
    c for c in os.listdir(RAW_DIR)
    if os.path.isdir(os.path.join(RAW_DIR, c))
]

# Optional: add your subreddit list here
SUBREDDITS = ["stocks", "StockMarket", "investing", "wallstreetbets"]
