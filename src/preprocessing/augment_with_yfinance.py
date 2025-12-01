"""
Augment labeled Reddit dataset with stock price movements using yfinance.

Assumes input CSV contains:
    - 'ticker'         (uppercase company code)
    - 'created_utc'    (UNIX timestamp, seconds)
    - 'sentiment_score'
    - 'label'
    - 'clean_text'

Produces:
    price_post
    price_post_date
    price_7d
    price_7d_date
    price_diff
    price_pct_change
"""

import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os


# ------------------------------
# Helpers
# ------------------------------

def unix_to_date(ts):
    """Convert UNIX timestamp to datetime.date."""
    try:
        return datetime.utcfromtimestamp(float(ts)).date()
    except:
        return None


def nearest_prev(df, target_date):
    """Nearest previous trading date."""
    dates = df.index.date
    valid = [d for d in dates if d <= target_date]
    return max(valid) if valid else None


def nearest_next(df, target_date):
    """Nearest next trading date."""
    dates = df.index.date
    valid = [d for d in dates if d >= target_date]
    return min(valid) if valid else None


# ------------------------------
# Core augmentation function
# ------------------------------

def augment(df):
    df = df.copy()

    # Convert timestamps
    df["post_date"] = df["created_utc"].apply(unix_to_date)

    # Prepare empty columns
    for col in ["price_post", "price_7d", "price_post_date",
                "price_7d_date", "price_diff", "price_pct_change"]:
        df[col] = None

    tickers = sorted(df["ticker"].dropna().unique().tolist())
    print(f"[INFO] Found {len(tickers)} tickers to fetch.\n{tickers}")

    for ticker in tickers:
        print(f"\nFetching price data for {ticker}...")

        # All posts for this ticker
        idx = df.index[df["ticker"] == ticker].tolist()
        post_dates = [df.at[i, "post_date"] for i in idx if df.at[i, "post_date"]]

        if not post_dates:
            continue

        min_date = min(post_dates) - timedelta(days=5)
        max_date = max(post_dates) + timedelta(days=10)

        price_df = yf.download(
            ticker,
            start=min_date.isoformat(),
            end=max_date.isoformat(),
            auto_adjust=True,
            progress=False
        )

        if price_df.empty:
            print(f"[WARN] No price data for {ticker}.")
            continue

        for i in idx:
            post_date = df.at[i, "post_date"]
            if post_date is None:
                continue

            target_7d = post_date + timedelta(days=7)

            prev = nearest_prev(price_df, post_date)
            nxt = nearest_next(price_df, target_7d)

            if prev:
                df.at[i, "price_post"] = float(price_df.loc[str(prev)]["Close"])
                df.at[i, "price_post_date"] = prev.isoformat()

            if nxt:
                df.at[i, "price_7d"] = float(price_df.loc[str(nxt)]["Close"])
                df.at[i, "price_7d_date"] = nxt.isoformat()

            if df.at[i, "price_post"] and df.at[i, "price_7d"]:
                p0 = df.at[i, "price_post"]
                p1 = df.at[i, "price_7d"]
                df.at[i, "price_diff"] = p1 - p0
                df.at[i, "price_pct_change"] = (p1 - p0) / p0

    return df


# ------------------------------
# Entry script
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Augment dataset with yfinance prices")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"[INFO] Loaded {len(df)} rows.")

    augmented = augment(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    augmented.to_csv(args.output, index=False)

    print(f"[DONE] Saved augmented dataset → {args.output}")


if __name__ == "__main__":
    main()
