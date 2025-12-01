import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import torch.nn.functional as F

MODEL = "ProsusAI/finbert"
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label_map = {"negative": -1, "neutral": 0, "positive": 1}

def pseudo_label_finbert(in_path="data/interim/cleaned_texts.csv",
                         out_path="data/interim/labeled_texts.csv",
                         batch_size=32, 
                         max_length=256):
    print("[finbert_labeler] Loading FinBERT...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL)
    mdl.eval()

    df = pd.read_csv(in_path)
    texts = df["clean_text"].astype(str).tolist()

    labels, negs, neus, poss = [], [], [], []

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting sentiment"):
        batch = texts[i:i + batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**enc).logits
            probs = F.softmax(logits, dim=-1)
            for p in probs:
                negs.append(float(p[0]))
                neus.append(float(p[1]))
                poss.append(float(p[2]))
            preds = probs.argmax(dim=-1).tolist()
            labels.extend([label_map[id2label[j]] for j in preds])

    df["neg_prob"] = negs
    df["neu_prob"] = neus
    df["pos_prob"] = poss
    df["label"] = labels
    df["sentiment_score"] = df["pos_prob"] - df["neg_prob"]

    df.to_csv(out_path, index=False)
    print(f"[finbert_labeler] Saved pseudo-labeled data with probabilities → {out_path}")

if __name__ == "__main__":
    pseudo_label_finbert()
