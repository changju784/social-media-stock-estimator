'''
Normalize text (strip URLs, emojis/symbols, lowercasing) and concatenate title + selftext.
'''
import pandas as pd, re

IN_PATH = "data/interim/merged_reddit.csv"
OUT_PATH = "data/interim/cleaned_texts.csv"

def clean_text(text):
    text = re.sub(r"http\S+", "", text)            # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)     # remove symbols
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def clean_dataset():
    df = pd.read_csv(IN_PATH)
    df["clean_text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]
    df.to_csv(OUT_PATH, index=False)
    print(f"[clean_texts] cleaned {len(df)} posts → {OUT_PATH}")

if __name__ == "__main__":
    clean_dataset()
