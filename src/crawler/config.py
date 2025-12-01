# config.py

# ------------------------------------------
# Reddit Sources
# ------------------------------------------
SUBREDDITS = [
    "stocks",
    "wallstreetbets",
    "investing",
    "StockMarket"
]

# ------------------------------------------
# Target Companies (20 tickers)
# ------------------------------------------
COMPANIES = [
    "meta", "apple", "amazon", "google", "netflix",
    "tesla", "nvidia", "microsoft", "amd", "intel",
    "jpmorgan", "bankofamerica", "walmart", "cocacola", "mcdonalds",
    "exxon", "chevron", "boeing", "ford", "nike"
]

# ------------------------------------------
# Data Save Location
# ------------------------------------------
SAVE_DIR = "data/raw"

# ------------------------------------------
# Crawler Settings (shared by crawler)
# ------------------------------------------
POST_LIMIT = 120            # Posts per (subreddit, company)
COMMENT_LIMIT = 20          # Comments per post
TIME_FILTER = "year"        # Time window for Reddit search
SLEEP_SEC = 0.4             # Throttle to avoid rate-limit
