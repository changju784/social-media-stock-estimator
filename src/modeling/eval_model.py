import os
import numpy as np
import pandas as pd
import joblib
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModel

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "reddit_test.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "elasticnet_pca_pipeline.pkl")


# -------------------------------
# FinBERT embedding
# -------------------------------
def get_finbert_embeddings(texts, model, tokenizer, max_len=128):
    embs = []
    for t in tqdm(texts, desc="Encoding FinBERT"):
        tok = tokenizer(t, padding="max_length", truncation=True,
                        max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            out = model(**tok)
        embs.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.array(embs)


# -------------------------------
# Feature Engineering (same as training)
# -------------------------------
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


# -------------------------------
# Evaluation
# -------------------------------
def evaluate():
    print("🔍 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["clean_text", "price_pct_change"])

    y_true = df["price_pct_change"].clip(-0.10, 0.10).values

    print("🔍 Loading model...")
    pipeline = joblib.load(MODEL_PATH)

    print("🔍 Loading FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = AutoModel.from_pretrained("ProsusAI/finbert")

    print("🔍 Encoding text...")
    embeddings = get_finbert_embeddings(df["clean_text"].tolist(), finbert, tokenizer)

    print("🔍 Building features...")
    X = prepare_features(df, embeddings)

    print("🔍 Predicting...")
    y_pred = pipeline.predict(X)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    direction = (np.sign(y_true) == np.sign(y_pred)).mean()

    print("\n📈 Evaluation Results")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Corr: {corr:.4f}")
    print(f"Directional Accuracy: {direction:.2%}")

    # Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.grid(True)
    plt.show()

    # Residual Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    evaluate()
