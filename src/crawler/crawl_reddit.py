import os
import json
import time
import traceback
from dotenv import load_dotenv
import praw

from config import SUBREDDITS, COMPANIES, SAVE_DIR

load_dotenv()

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# Configurations for scalable crawling
POST_LIMIT = 120            # number of posts per (subreddit, company)
COMMENT_LIMIT = 20          # number of comments per post
SLEEP_SEC = 0.4             # throttle to avoid rate limits
TIME_FILTER = "year"        # expand historical window (month → year)


def crawl_praw():
    print("🚀 Starting Reddit crawler...\n")

    for sub in SUBREDDITS:
        subreddit = reddit.subreddit(sub)

        for comp in COMPANIES:
            print(f"\n📌 Crawling r/{sub} for company: {comp}")

            posts = []
            count = 0

            try:
                # Search Reddit posts
                for post in subreddit.search(
                    query=comp,
                    sort="top",
                    time_filter=TIME_FILTER,
                    limit=POST_LIMIT,
                ):
                    count += 1
                    print(f"[{count}/{POST_LIMIT}] Post ID: {post.id}")

                    # Fetch comments
                    try:
                        post.comments.replace_more(limit=0)
                        comments = [
                            c.body
                            for c in post.comments[:COMMENT_LIMIT]
                            if hasattr(c, "body")
                        ]
                    except Exception as e:
                        print(f"❗ Failed to fetch comments on {post.id}: {e}")
                        comments = []

                    # Store post data
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
                        "subreddit": sub,
                        "comments": comments,
                    }

                    posts.append(item)
                    time.sleep(SLEEP_SEC)

            except Exception as e:
                print(f"❗ ERROR while crawling {comp} in {sub}:")
                traceback.print_exc()

            # Save results only if posts exist
            save_company_dir = os.path.join(SAVE_DIR, comp)
            os.makedirs(save_company_dir, exist_ok=True)

            out_path = os.path.join(save_company_dir, f"{sub}.json")

            if len(posts) == 0:
                print(f"⚠️ No posts found for {comp} in r/{sub}. Skipping save.")
                continue

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=False, indent=2)

            print(f"✅ Saved {len(posts)} posts → {out_path}")

    print("\n🎉 Finished crawling all subreddits & companies!")


if __name__ == "__main__":
    crawl_praw()
