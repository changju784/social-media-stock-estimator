import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH = "data/interim/labeled_texts.csv"
OUT_DIR = "data/processed"

def prepare():
    df = pd.read_csv(IN_PATH)

    # Keep ticker in final dataset
    cols = ["clean_text", "ticker", "neg_prob", "neu_prob", "pos_prob",
            "label", "sentiment_score"]
    df = df[cols]

    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

    os.makedirs(OUT_DIR, exist_ok=True)
    train.to_csv(f"{OUT_DIR}/reddit_train.csv", index=False)
    val.to_csv(f"{OUT_DIR}/reddit_val.csv", index=False)
    test.to_csv(f"{OUT_DIR}/reddit_test.csv", index=False)

    print(f"[prepare_dataset] train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    prepare()
