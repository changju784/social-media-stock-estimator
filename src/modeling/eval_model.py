import os
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Universal paths
# ---------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(ROOT_DIR, "data", "interim", "labeled_texts_with_prices.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# ---------------------------------------------------------
# Helper functions (match train_model.py)
# ---------------------------------------------------------
def get_finbert_embeddings(texts, model, tokenizer, max_len=128):
    embs = []
    for t in tqdm(texts, desc="Encoding texts"):
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
# Evaluation
# ---------------------------------------------------------
def evaluate_model():
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["clean_text", "price_pct_change"]).copy()
    df["price_pct_change"] = df["price_pct_change"].clip(lower=-0.10, upper=0.10)
    y_true = df["price_pct_change"].values

    # 2. Load models
    pca = joblib.load(os.path.join(MODEL_DIR, "pca_finbert_128.pkl"))
    pipe = joblib.load(os.path.join(MODEL_DIR, "ridge_pipeline.pkl"))

    # 3. Build feature set
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = AutoModel.from_pretrained("ProsusAI/finbert")
    X_text = get_finbert_embeddings(df["clean_text"].tolist(), finbert, tokenizer)
    X_text_pca = pca.transform(X_text)
    X_sent = df[["neg_prob", "neu_prob", "pos_prob", "sentiment_score"]].values
    X_meta = encode_metadata(df)
    X = np.hstack([X_text_pca, X_sent, X_meta]).astype(np.float32)

    # 4. Predict
    y_pred = pipe.predict(X)

    # 5. Metrics
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"MAE: {mae:.4f} | R²: {r2:.3f} | Corr: {corr:.3f}")

    # 6. Plot
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual % Change")
    plt.ylabel("Predicted % Change")
    plt.title("Predicted vs Actual Stock Price Change (7d)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_model()
