import pandas as pd, json, os
from collections import Counter

DATA_PATH = "data/interim/labeled_texts.csv"
LOG_PATH = "data/logs/stats.json"

def log_stats():
    df = pd.read_csv(DATA_PATH)
    stats = {
        "total_posts": len(df),
        "label_distribution": df["label"].value_counts().to_dict(),
        "avg_length": df["clean_text"].str.split().apply(len).mean(),
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[stats_logger] logged summary → {LOG_PATH}")

if __name__ == "__main__":
    log_stats()
