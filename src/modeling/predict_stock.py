import os
import numpy as np
import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ---------------------------------------------------------
# Universal paths
# ---------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# ---------------------------------------------------------
# Helper functions (identical to training)
# ---------------------------------------------------------
def get_finbert_embeddings(texts, model, tokenizer, max_len=128):
    embs = []
    for t in texts:
        toks = tokenizer(t, truncation=True, padding="max_length",
                         max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            out = model(**toks)
        embs.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.array(embs)

def encode_metadata(df):
    meta = pd.DataFrame(index=df.index)
    meta["score_log1p"] = np.log1p(df["score"].clip(lower=0))
    sub = pd.get_dummies(df["subreddit"], prefix="sub")
    tic = pd.get_dummies(df["ticker"], prefix="tick")
    dt = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    dow = dt.dt.dayofweek.fillna(0).astype(int)
    meta["day_sin"] = np.sin(2 * np.pi * dow / 7)
    meta["day_cos"] = np.cos(2 * np.pi * dow / 7)
    return np.hstack([meta.values, sub.values, tic.values])

# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict_from_posts(posts_df):
    """
    posts_df must contain:
    ['clean_text','neg_prob','neu_prob','pos_prob','sentiment_score',
     'score','subreddit','ticker','created_utc']
    """
    pca = joblib.load(os.path.join(MODEL_DIR, "pca_finbert_128.pkl"))
    pipe = joblib.load(os.path.join(MODEL_DIR, "ridge_pipeline.pkl"))

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = AutoModel.from_pretrained("ProsusAI/finbert")

    # FinBERT → PCA
    X_text = get_finbert_embeddings(posts_df["clean_text"].tolist(), finbert, tokenizer)
    X_text_pca = pca.transform(X_text)

    # sentiment + metadata
    X_sent = posts_df[["neg_prob", "neu_prob", "pos_prob", "sentiment_score"]].values
    X_meta = encode_metadata(posts_df)

    X = np.hstack([X_text_pca, X_sent, X_meta]).astype(np.float32)

    # mean aggregation across posts for stability
    X_agg = X.mean(axis=0, keepdims=True)
    pred = pipe.predict(X_agg)[0]
    return float(pred)

# ---------------------------------------------------------
# Example CLI usage
# ---------------------------------------------------------
if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "interim", "labeled_texts_with_prices.csv"))

    # simulate “recent posts for AAPL”
    aapl_df = df[df["ticker"] == "AAPL"].tail(10)
    prediction = predict_from_posts(aapl_df)
    print(f"Predicted 7-day price change for AAPL: {prediction:+.3f}%")
