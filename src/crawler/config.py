SUBREDDITS = ["stocks", "wallstreetbets", "investing"]
COMPANIES = ["meta", "apple", "amazon", "google", "netflix"]

AFTER_DAYS = 90         # crawl posts from past 3 months
POST_LIMIT = 100         # number of posts per (subreddit, company)
COMMENT_LIMIT = 20       # number of comments per post
SLEEP_SEC = 1            # throttle to avoid rate-limit
SAVE_DIR = "data/raw/"
