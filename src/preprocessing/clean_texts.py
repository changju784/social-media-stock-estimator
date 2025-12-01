"""
Clean and normalize title + selftext + comments.
"""
import pandas as pd
import re

IN_PATH = "data/interim/merged_reddit.csv"
OUT_PATH = "data/interim/cleaned_texts.csv"


def clean(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def clean_dataset():
    df = pd.read_csv(IN_PATH)

    df["raw_text"] = (
        df["title"].fillna("") + " "
        + df["selftext"].fillna("") + " "
        + df["comments"].fillna("")
    )

    df["clean_text"] = df["raw_text"].apply(clean)
    df = df[df["clean_text"].str.strip() != ""]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"[clean_texts] cleaned {len(df)} → {OUT_PATH}")


if __name__ == "__main__":
    clean_dataset()
