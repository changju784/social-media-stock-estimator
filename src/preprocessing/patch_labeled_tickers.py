import pandas as pd

INPUT = "data/interim/labeled_texts.csv"
OUTPUT = "data/interim/labeled_texts_fixed.csv"

# Map folder-style names → real Yahoo Finance tickers
TICKER_MAP = {
    "amazon": "AMZN",
    "apple": "AAPL",
    "google": "GOOGL",
    "meta": "META",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "amd": "AMD",
    "netflix": "NFLX",
    "walmart": "WMT",
    "intel": "INTC",
    "jpmorgan": "JPM",
    "ford": "F",
    "boeing": "BA",
    "nike": "NKE",
    "chevron": "CVX",
    "exxon": "XOM",
    "mcdonalds": "MCD"
}

def main():
    print("[PATCH] Loading labeled sentiment file...")
    df = pd.read_csv(INPUT)

    if "company" not in df.columns:
        raise ValueError(
            "ERROR: labeled_texts.csv does not contain a 'company' column. "
            "Ensure merge_raw_json.py included it."
        )

    print("[PATCH] Fixing tickers...")
    df["ticker"] = df["company"].str.lower().map(TICKER_MAP)

    # Check for any missing mappings
    missing = df[df["ticker"].isna()]
    if len(missing) > 0:
        print("[WARN] Missing ticker mapping for companies:")
        print(missing["company"].unique())

    df.to_csv(OUTPUT, index=False)
    print(f"[PATCH] Saved → {OUTPUT}")

if __name__ == "__main__":
    main()
