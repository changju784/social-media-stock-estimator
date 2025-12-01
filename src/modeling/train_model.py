import os
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.decomposition import PCA
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

    # Clip for stability
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
# Feature Engineering (ONLY using existing columns)
# ---------------------------------------------------------
def prepare_features(df, embeddings):
    df["text_length"] = df["clean_text"].str.len()

    # ------- Sentiment engineering -------
    df["sentiment_strength"] = df[["neg_prob", "pos_prob"]].max(axis=1)
    df["sentiment_conf"] = 1 - df["neu_prob"]
    df["bull_bear_ratio"] = df["pos_prob"] / (df["neg_prob"] + 1e-6)

    # ------- Interaction features -------
    df["sentiment_x_price"] = df["sentiment_score"] * df["price_post"]
    df["pos_x_price"] = df["pos_prob"] * df["price_post"]
    df["neg_x_price"] = df["neg_prob"] * df["price_post"]

    # ------- Base numeric features -------
    feature_df = df[[
        "neg_prob", "neu_prob", "pos_prob",
        "sentiment_score", "text_length",
        "sentiment_strength", "sentiment_conf", "bull_bear_ratio",
        "sentiment_x_price", "pos_x_price", "neg_x_price",
        "price_post", "price_7d", "price_diff"
    ]].copy()

    # ------- Embeddings -------
    emb_df = pd.DataFrame(
        embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )

    return pd.concat([feature_df, emb_df], axis=1)



# ---------------------------------------------------------
# Build Model Pipeline (PCA + ElasticNet)
# ---------------------------------------------------------
def build_pipeline(n_emb_dim):

    base_features = [
        "neg_prob", "neu_prob", "pos_prob",
        "sentiment_score", "text_length",
        "sentiment_strength", "sentiment_conf", "bull_bear_ratio",
        "sentiment_x_price", "pos_x_price", "neg_x_price",
        "price_post", "price_7d", "price_diff",
    ]

    emb_features = [f"emb_{i}" for i in range(n_emb_dim)]

    numeric_cols = base_features + emb_features
    categorical_cols = ["ticker"]

    # PCA reduces dimensionality of embeddings
    pca = PCA(n_components=128, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-3, 3, 20),
        max_iter=5000,
        cv=5,
        n_jobs=-1
    )

    return Pipeline([
        ("pre", preprocessor),
        ("pca", pca),
        ("model", model),
    ])



# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def train_model():
    train, val = load_train_val()
    tokenizer, model = load_finbert()

    print("🔹 Encoding training texts")
    train_emb = embed_texts(train["clean_text"].tolist(), tokenizer, model)

    print("🔹 Encoding val texts")
    val_emb = embed_texts(val["clean_text"].tolist(), tokenizer, model)

    print("🔹 Building features")
    X_train = prepare_features(train, train_emb)
    X_val   = prepare_features(val, val_emb)

    y_train = train["price_pct_change"]
    y_val   = val["price_pct_change"]

    full_train = pd.concat([train[["ticker"]], X_train], axis=1)
    full_val   = pd.concat([val[["ticker"]], X_val], axis=1)

    print("🔹 Building PCA + ElasticNet pipeline")
    pipeline = build_pipeline(train_emb.shape[1])

    print("🔹 Training model...")
    pipeline.fit(full_train, y_train)

    preds = pipeline.predict(full_val)

    mae = mean_absolute_error(y_val, preds)
    r2  = r2_score(y_val, preds)
    corr = np.corrcoef(y_val, preds)[0, 1]

    print("\n📈 Validation Results (ElasticNet + PCA)")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    print(f"Corr: {corr:.4f}")

    joblib.dump(pipeline, os.path.join(MODEL_DIR, "elasticnet_pca_pipeline.pkl"))
    print("\n✅ Saved ElasticNet PCA pipeline")


if __name__ == "__main__":
    train_model()
