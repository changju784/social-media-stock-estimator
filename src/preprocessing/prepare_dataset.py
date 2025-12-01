import os
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score

from transformers import AutoTokenizer, AutoModel


# ---------- Paths ----------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PROC_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(PROC_DIR, "reddit_train.csv")
VAL_PATH   = os.path.join(PROC_DIR, "reddit_val.csv")

torch.set_grad_enabled(False)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------
def load_train_val():
    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)

    # Clip price targets
    train["price_pct_change"] = train["price_pct_change"].clip(-0.10, 0.10)
    val["price_pct_change"]   = val["price_pct_change"].clip(-0.10, 0.10)

    return train, val



# ---------------------------------------------------------
# FinBERT Embeddings
# ---------------------------------------------------------
def load_finbert():
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def embed_texts(texts, tokenizer, model):
    embeddings = []
    for text in tqdm(texts, desc="Encoding FinBERT"):
        tokens = tokenizer(text, padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
        outputs = model(**tokens)
        cls = outputs.last_hidden_state[:, 0, :].detach().numpy().squeeze()
        embeddings.append(cls)
    return np.array(embeddings)



# ---------------------------------------------------------
# Prepare Features (ONLY use the columns that exist)
# ---------------------------------------------------------
def prepare_features(df, embeddings):
    df["text_length"] = df["clean_text"].str.len()

    # Only columns that exist
    feature_df = pd.DataFrame({
        "neg_prob": df["neg_prob"],
        "neu_prob": df["neu_prob"],
        "pos_prob": df["pos_prob"],
        "sentiment_score": df["sentiment_score"],
        "text_length": df["text_length"],
        "price_post": df["price_post"],
        "price_7d": df["price_7d"],
        "price_diff": df["price_diff"],
    })

    emb_df = pd.DataFrame(
        embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )

    return pd.concat([feature_df, emb_df], axis=1)



# ---------------------------------------------------------
# Build Model Pipeline (RidgeCV)
# ---------------------------------------------------------
def build_pipeline(n_emb_dim):

    num_features = [
        "neg_prob", "neu_prob", "pos_prob",
        "sentiment_score", "text_length",
        "price_post", "price_7d", "price_diff",
    ]

    emb_features = [f"emb_{i}" for i in range(n_emb_dim)]
    numeric_cols = num_features + emb_features

    categorical_cols = ["ticker"]   # ticker is the ONLY categorical field you have

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    model = RidgeCV(alphas=np.logspace(-3, 3, 20))

    return Pipeline([
        ("preprocessor", preprocessor),
        ("ridge", model)
    ])



# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def train_model():
    train, val = load_train_val()
    tokenizer, model = load_finbert()

    print("🔹 Encoding training texts")
    train_emb = embed_texts(train["clean_text"].tolist(), tokenizer, model)

    print("🔹 Encoding validation texts")
    val_emb = embed_texts(val["clean_text"].tolist(), tokenizer, model)

    X_train = prepare_features(train, train_emb)
    X_val   = prepare_features(val, val_emb)

    y_train = train["price_pct_change"]
    y_val   = val["price_pct_change"]

    full_train = pd.concat([train[["ticker"]], X_train], axis=1)
    full_val   = pd.concat([val[["ticker"]], X_val], axis=1)

    pipeline = build_pipeline(train_emb.shape[1])

    print("🔹 Training RidgeCV...")
    pipeline.fit(full_train, y_train)

    preds = pipeline.predict(full_val)

    mae = mean_absolute_error(y_val, preds)
    r2  = r2_score(y_val, preds)
    corr = np.corrcoef(y_val, preds)[0, 1]

    print("\n📈 Validation Results (RidgeCV)")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"Corr: {corr:.4f}")

    joblib.dump(pipeline, os.path.join(MODEL_DIR, "ridge_pipeline.pkl"))
    print("\n✅ Saved RidgeCV model")


if __name__ == "__main__":
    train_model()
