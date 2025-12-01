import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import joblib

# ---------- paths ----------
ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(ROOT_DIR, "data", "interim", "labeled_texts_with_prices.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models"); os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- helpers ----------
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["clean_text", "price_pct_change"]).copy()
    # clip extreme outcomes to stabilize training
    df["price_pct_change"] = df["price_pct_change"].clip(lower=-0.10, upper=0.10)
    return df

def get_finbert_embeddings(texts, model, tokenizer, max_len=128):
    embs = []
    for t in tqdm(texts, desc="Encoding texts"):
        tok = tokenizer(t, truncation=True, padding="max_length",
                        max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            out = model(**tok)
        embs.append(out.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.array(embs)

def encode_metadata(df):
    meta = pd.DataFrame(index=df.index)
    meta["score_log1p"] = np.log1p(df["score"].clip(lower=0))

    # subreddit / ticker one-hots (lean)
    sub = pd.get_dummies(df["subreddit"], prefix="sub")
    tic = pd.get_dummies(df["ticker"],   prefix="tick")

    # simple cyclic time
    dt = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    dow = dt.dt.dayofweek.fillna(0).astype(int)
    meta["day_sin"] = np.sin(2*np.pi*dow/7)
    meta["day_cos"] = np.cos(2*np.pi*dow/7)

    return np.hstack([meta.values, sub.values, tic.values])

def train():
    # ----- data -----
    df = load_dataset()
    print(f"Loaded {len(df)} samples")

    # FinBERT
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert   = AutoModel.from_pretrained("ProsusAI/finbert")

    X_text = get_finbert_embeddings(df["clean_text"].tolist(), finbert, tokenizer)   # (N, 768)
    X_sent = df[["neg_prob", "neu_prob", "pos_prob", "sentiment_score"]].values      # (N, 4)
    
    # --- Encode metadata and save category columns for inference ---
    X_meta = encode_metadata(df)

    # Save column order to keep metadata encoding consistent at inference
    subreddit_cols = pd.get_dummies(df["subreddit"], prefix="sub").columns.tolist()
    ticker_cols    = pd.get_dummies(df["ticker"], prefix="tick").columns.tolist()

    meta_info = {
        "subreddit_cols": subreddit_cols,
        "ticker_cols": ticker_cols,
    }
    joblib.dump(meta_info, os.path.join(MODEL_DIR, "meta_columns.pkl"))                                                     # (~N, ~10-12)

    # ----- PCA on embeddings to shrink capacity -----
    pca = PCA(n_components=128, random_state=42)
    X_text_pca = pca.fit_transform(X_text)  # (N, 128)

    # concat final features
    X = np.hstack([X_text_pca, X_sent, X_meta]).astype(np.float32)
    y = df["price_pct_change"].values.astype(np.float32)

    # ----- pipeline: scale + ridge -----
    ridge = RidgeCV(alphas=np.logspace(-4, 3, 20), cv=5)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("reg",    ridge),
    ])

    # CV score (MAE negative in sklearn → invert sign)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_cv = -cross_val_score(pipe, X, y, scoring="neg_mean_absolute_error", cv=kf).mean()
    print(f"CV MAE: {mae_cv:.4f}")

    # holdout for quick sanity
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    y_pr = pipe.predict(X_te)
    print(f"Holdout  MAE: {mean_absolute_error(y_te, y_pr):.4f} | R²: {r2_score(y_te, y_pr):.3f}")
    print(f"Chosen alpha: {pipe.named_steps['reg'].alpha_:.6f}")

    # save artifacts
    joblib.dump(pca,  os.path.join(MODEL_DIR, "pca_finbert_128.pkl"))
    joblib.dump(pipe, os.path.join(MODEL_DIR, "ridge_pipeline.pkl"))
    print("✅ Saved PCA + Ridge pipeline to models/")

if __name__ == "__main__":
    train()
