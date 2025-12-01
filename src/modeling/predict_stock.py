import os
import numpy as np
import pandas as pd
import torch
import joblib
from transformers import AutoTokenizer, AutoModel

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "elasticnet_pca_pipeline.pkl")


# --------------------------
# FinBERT
# --------------------------
def get_finbert_embeddings(texts, model, tokenizer, max_len=128):
    embs = []
    for t in texts:
        tok = tokenizer(t, padding="max_length", truncation=True,
                        max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            out = model(**tok)
        embs.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.array(embs)


# --------------------------
# Feature Engineering (same as training)
# --------------------------
def prepare_features(df, embeddings):
    df = df.copy()
    df["text_length"] = df["clean_text"].str.len()

    df["sentiment_strength"] = df[["neg_prob", "pos_prob"]].max(axis=1)
    df["sentiment_conf"] = 1 - df["neu_prob"]
    df["bull_bear_ratio"] = df["pos_prob"] / (df["neg_prob"] + 1e-6)

    df["sentiment_x_price"] = df["sentiment_score"] * df["price_post"]
    df["pos_x_price"] = df["pos_prob"] * df["price_post"]
    df["neg_x_price"] = df["neg_prob"] * df["price_post"]

    base = df[[
        "neg_prob", "neu_prob", "pos_prob",
        "sentiment_score", "text_length",
        "sentiment_strength", "sentiment_conf", "bull_bear_ratio",
        "sentiment_x_price", "pos_x_price", "neg_x_price",
        "price_post", "price_7d", "price_diff"
    ]].copy()

    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )

    return pd.concat([base, emb_df], axis=1)


# --------------------------
# Predict function
# --------------------------
def predict_from_posts(posts_df):
    pipeline = joblib.load(MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = AutoModel.from_pretrained("ProsusAI/finbert")

    embeddings = get_finbert_embeddings(posts_df["clean_text"].tolist(), finbert, tokenizer)
    X = prepare_features(posts_df, embeddings)

    # Aggregate all posts for ticker → mean features
    X_agg = X.mean(axis=0, keepdims=True)

    pred = pipeline.predict(X_agg)[0]
    return float(pred)


if __name__ == "__main__":
    print("Run this via another script (predict_from_posts)")
