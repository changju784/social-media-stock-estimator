'''
Tokenize and remove stopwords from cleaned text.
'''
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

IN_PATH = "data/interim/cleaned_texts.csv"
OUT_PATH = "data/interim/tokenized_texts.csv"

def tokenize_text(text):
    tokens = word_tokenize(text)
    stop = set(stopwords.words('english'))
    return [w for w in tokens if w not in stop]

def process():
    df = pd.read_csv(IN_PATH)
    df["tokens"] = df["clean_text"].apply(tokenize_text)
    df.to_csv(OUT_PATH, index=False)
    print(f"[tokenize_normalize] tokenized {len(df)} posts → {OUT_PATH}")

if __name__ == "__main__":
    process()
