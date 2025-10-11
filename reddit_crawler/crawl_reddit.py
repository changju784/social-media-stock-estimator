import os, json, time
from dotenv import load_dotenv
import praw
from config import SUBREDDITS, COMPANIES, SAVE_DIR

load_dotenv() 

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def crawl_praw():
    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        for comp in COMPANIES:
            print(f"\n🔍 Crawling {sub} for {comp}")
            posts = []
            # Fetch top 100 posts from last month mentioning the company
            for post in subreddit.search(comp, sort="top", time_filter="month", limit=100):
                post.comments.replace_more(limit=0)
                comments = [c.body for c in post.comments[:20] if hasattr(c, "body")]
                item = {
                    "id": post.id,
                    "title": post.title,
                    "selftext": post.selftext,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": post.created_utc,
                    "url": post.url,
                    "permalink": post.permalink,
                    "company": comp,
                    "comments": comments
                }
                posts.append(item)
                time.sleep(0.3)
            out = f"{SAVE_DIR}/{sub}_{comp}.json"
            os.makedirs(SAVE_DIR, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved {len(posts)} posts to {out}")

if __name__ == "__main__":
    crawl_praw()
