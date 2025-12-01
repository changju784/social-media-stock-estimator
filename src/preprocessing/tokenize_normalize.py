"""
Tokenize with NLTK and remove stopwords.
"""
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

IN_PATH = "data/interim/cleaned_texts.csv"
OUT_PATH = "data/interim/tokenized_texts.csv"

def tokenize(t):
    sw = set(stopwords.words("english"))
    tokens = word_tokenize(t)
    return [x for x in tokens if x not in sw]


def process():
    df = pd.read_csv(IN_PATH)
    df["tokens"] = df["clean_text"].apply(tokenize)
    df.to_csv(OUT_PATH, index=False)
    print(f"[tokenize_normalize] {len(df)} posts → {OUT_PATH}")


if __name__ == "__main__":
    process()
