"""
Augment Reddit sentiment dataset with historical price movement using yfinance.

This script:
- Reads a CSV with columns including 'ticker' and 'created_utc'
- Downloads historical stock data for each ticker in one batch
- For each Reddit post:
    price_post       = nearest previous trading close before post_date
    price_7d         = nearest next trading close after post_date + 7 days
    price_diff       = price_7d - price_post
    price_pct_change = (price_7d - price_post) / price_post

Output:
    A CSV with added price target columns.

Usage:
    python src/preprocessing/augment_with_yfinance.py \
        --input data/interim/labeled_texts.csv \
        --output data/interim/labeled_texts_with_prices.csv
"""

import argparse
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# ======================================================================
# Helper Functions
# ======================================================================

def unix_to_date(unix_ts):
    """Convert UNIX timestamp (seconds) → datetime.date."""
    try:
        ts = float(unix_ts)
        return datetime.utcfromtimestamp(ts).date()
    except:
        return None


def nearest_previous(prices_df, target_date):
    """Return nearest trading day <= target_date."""
    all_dates = prices_df.index.date
    valid = [d for d in all_dates if d <= target_date]
    return max(valid) if valid else None


def nearest_next(prices_df, target_date):
    """Return nearest trading day >= target_date."""
    all_dates = prices_df.index.date
    valid = [d for d in all_dates if d >= target_date]
    return min(valid) if valid else None


# ======================================================================
# Core Augmentation Logic
# ======================================================================

def augment_with_prices(df):
    df = df.copy()

    # Convert post timestamps → date
    df["post_date"] = df["created_utc"].apply(unix_to_date)

    # Prepare new columns
    new_cols = [
        "price_post", "price_post_date",
        "price_7d", "price_7d_date",
        "price_diff", "price_pct_change"
    ]
    for col in new_cols:
        df[col] = None

    tickers = sorted(df["ticker"].dropna().unique().tolist())
    print(f"[INFO] Found {len(tickers)} unique tickers: {tickers}")

    for ticker in tickers:
        print(f"\nFetching price history for {ticker}...")

        idxs = df.index[df["ticker"] == ticker].tolist()
        post_dates = [
            df.at[i, "post_date"]
            for i in idxs if df.at[i, "post_date"] is not None
        ]

        if not post_dates:
            print(f"[WARN] No valid dates for ticker {ticker}")
            continue

        min_d = min(post_dates) - timedelta(days=5)
        max_d = max(post_dates) + timedelta(days=10)

        # Batch download for efficiency
        prices = yf.download(
            ticker,
            start=min_d.isoformat(),
            end=max_d.isoformat(),
            auto_adjust=True,
            progress=False
        )

        if prices.empty:
            print(f"[WARN] No price data found for {ticker}. Skipping.")
            continue

        # Process each Reddit post for this ticker
        for i in idxs:
            post_date = df.at[i, "post_date"]
            if post_date is None:
                continue

            target_7 = post_date + timedelta(days=7)

            prev_date = nearest_previous(prices, post_date)
            next_date = nearest_next(prices, target_7)

            if prev_date:
                df.at[i, "price_post"] = float(prices.loc[str(prev_date)]["Close"])
                df.at[i, "price_post_date"] = prev_date.isoformat()

            if next_date:
                df.at[i, "price_7d"] = float(prices.loc[str(next_date)]["Close"])
                df.at[i, "price_7d_date"] = next_date.isoformat()

            # Compute price movement
            p0 = df.at[i, "price_post"]
            p1 = df.at[i, "price_7d"]

            if p0 is not None and p1 is not None:
                df.at[i, "price_diff"] = p1 - p0
                df.at[i, "price_pct_change"] = (p1 - p0) / p0 if p0 != 0 else None

    return df


# ======================================================================
# Script Entry Point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Augment Reddit dataset with stock price data")
    parser.add_argument("--input", required=True, help="input CSV (labeled_texts.csv)")
    parser.add_argument("--output", required=True, help="output CSV path")
    args = parser.parse_args()

    print(f"[INFO] Loading data from {args.input}")
    df = pd.read_csv(args.input)

    augmented = augment_with_prices(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    augmented.to_csv(args.output, index=False)

    print(f"\n[DONE] Saved augmented dataset → {args.output}")


if __name__ == "__main__":
    main()
