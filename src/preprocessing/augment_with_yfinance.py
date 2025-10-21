"""Augment processed reddit CSVs with historical stock prices using yfinance.

This script:
- Reads a CSV with columns including 'id', 'title', 'selftext', 'created_utc'
- Extracts a single ticker per post using $TICKER pattern or heuristic (uppercase 1-5 letter tokens)
- Uses yfinance to download historical daily OHLC data around the post date and 7 days after
- Picks the nearest available close price for the post date and for post_date + 7 days
- Computes absolute and percent price changes and writes an augmented CSV with new columns

Usage:
    python src/preprocessing/augment_with_yfinance.py \
        --input data/processed/reddit_train.csv \
        --output data/processed/reddit_train_with_prices.csv \
        --date-col created_utc

Notes / assumptions:
- `created_utc` is a Unix timestamp (seconds) as present in the processed CSVs.
- If multiple tickers are found, the script picks the first one. You can modify extraction logic as needed.
- For market-closed dates (weekends/holidays), the script selects the nearest previous available trading day for the post date price,
  and the nearest next available trading day for the 7-day price (so we measure forward movement after the post).
- Network access required to fetch data via yfinance. yfinance caches are not used here.
"""

import argparse
import re
from datetime import datetime, timedelta
import time
import json
from glob import glob
import pandas as pd

try:
    import yfinance as yf
except Exception:
    raise SystemExit("yfinance is required. Please install with `pip install yfinance` or update requirements.txt and install.")

TICKER_REGEX = re.compile(r"\$([A-Za-z]{1,5})(?:\b|$)")
# Fallback heuristic: standalone uppercase words 1-5 letters (may produce false positives)
FALLBACK_REGEX = re.compile(r"\b([A-Z]{1,5})\b")


def build_ticker_whitelist(raw_dir='data/raw'):
    """Scan JSON files and filenames in raw_dir to build a candidate ticker whitelist.
    Returns a set of uppercase tokens that look like tickers.
    """
    tickers = set()
    # include filenames like stocks_aapl.json -> AAPL
    for path in glob(f"{raw_dir}/*.json"):
        name = path.split('/')[-1]
        # pick tokens after underscore or before .json
        parts = re.split(r'[_\.-]', name)
        for p in parts:
            if 1 <= len(p) <= 5 and p.isalpha():
                tickers.add(p.upper())
        # also try to parse file content cheaply for $TICKER mentions
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            for m in TICKER_REGEX.findall(text):
                tickers.add(m.upper())
            for m in re.findall(r"\(([A-Za-z]{1,5})\)", text):
                tickers.add(m.upper())
        except Exception:
            continue
    return tickers


def extract_ticker(text):
    """Extract a ticker from text using $TICKER first, then fallback heuristic.
    Returns uppercased ticker symbol or None.
    """
    if not isinstance(text, str):
        return None
    # 1) $TICKER pattern
    m = TICKER_REGEX.search(text)
    if m:
        return m.group(1).upper()
    # 2) common form like (AAPL) or $AAPL in parentheses or plain uppercase tickers
    # look for tokens like (AAPL)
    m2 = re.search(r"\(([A-Z]{1,5})\)", text)
    if m2:
        return m2.group(1).upper()
    # 3) fallback: return first uppercase token of length 1-5 that is not a common word
    for tok in FALLBACK_REGEX.findall(text):
        # filter out common English words that happen to be uppercase (I,AM,...). Keep simple blacklist
        if tok in {"I", "A", "AM", "THE", "IN", "ON", "TO", "FOR", "AND", "BUT"}:
            continue
        return tok.upper()
    return None


def unix_to_date_str(unix_ts):
    """Convert unix timestamp (seconds) to date string YYYY-MM-DD"""
    try:
        ts = float(unix_ts)
    except Exception:
        return None
    return datetime.utcfromtimestamp(ts).date().isoformat()


def nearest_previous_trading_date(prices_df, target_date):
    """Given a dataframe indexed by date (datetime.date or Timestamp), return the nearest date <= target_date present in index."""
    # ensure index is datetime.date
    idx_dates = pd.to_datetime(prices_df.index).date
    # filter
    le = [d for d in idx_dates if d <= target_date]
    if not le:
        return None
    return max(le)


def nearest_next_trading_date(prices_df, target_date):
    """Return nearest date >= target_date present in index."""
    idx_dates = pd.to_datetime(prices_df.index).date
    ge = [d for d in idx_dates if d >= target_date]
    if not ge:
        return None
    return min(ge)


def fetch_prices_for_ticker(ticker, start_date, end_date):
    """Download daily historical data for ticker between start_date and end_date (inclusive).
    start_date and end_date are date strings YYYY-MM-DD.
    Returns a DataFrame or None on error.
    """
    try:
        # yfinance expects YYYY-MM-DD strings
        # set auto_adjust=True to match expected behaviour and avoid future warnings
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False, auto_adjust=True)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"yfinance download error for {ticker}: {e}")
        return None


def augment_dataframe(df, date_col='created_utc', text_cols=('title', 'selftext')):
    """Augment a pandas DataFrame with ticker and price columns.
    Returns augmented DataFrame.
    """
    # Prepare columns
    df = df.copy()
    df['post_date'] = df[date_col].apply(unix_to_date_str)

    # If the dataframe already has a `ticker` column with expected values, prefer it.
    expected = {'AMZN', 'AAPL', 'GOOG', 'META', 'NFLX'}
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].apply(lambda x: x if str(x).upper() in expected else pd.NA)
        missing_mask = df['ticker'].isna()
    else:
        missing_mask = pd.Series([True] * len(df), index=df.index)

    # For rows still missing a ticker, fall back to whitelist/extraction
    if missing_mask.any():
        whitelist = build_ticker_whitelist()
        print(f"Ticker whitelist contains {len(whitelist)} candidates (sample: {list(whitelist)[:10]})")

        def find_in_row(row):
            for col in text_cols:
                txt = row.get(col, '') or ''
                t = extract_ticker(txt)
                if t and t in whitelist and t in expected:
                    return t
            # try fallback: search for any whitelist token present in title/selftext
            combined = ' '.join([str(row.get(c, '') or '') for c in text_cols])
            for token in whitelist:
                if token in expected and re.search(rf"\b{re.escape(token)}\b", combined):
                    return token
            return None

        df.loc[missing_mask, 'ticker'] = df.loc[missing_mask].apply(find_in_row, axis=1)

    # columns to add
    df['price_post'] = pd.NA
    df['price_post_date'] = pd.NA
    df['price_7d'] = pd.NA
    df['price_7d_date'] = pd.NA
    df['price_diff'] = pd.NA
    df['price_pct_change'] = pd.NA

    # Process unique tickers to batch yfinance calls where possible
    tickers = df['ticker'].dropna().unique().tolist()
    print(f"Found {len(tickers)} unique tickers to fetch (sample: {tickers[:10]})")

    for ticker in tickers:
        subset_idx = df.index[df['ticker'] == ticker].tolist()
        # determine min start and max end to download
        # gather dates
        post_dates = [datetime.fromisoformat(d).date() for d in df.loc[subset_idx, 'post_date'] if d]
        if not post_dates:
            continue
        min_date = min(post_dates)
        max_date = max(post_dates)
        # we need data from min_date - 5 days (in case nearest previous) up to max_date + 10 days
        start = (min_date - timedelta(days=5)).isoformat()
        end = (max_date + timedelta(days=10)).isoformat()
        prices = fetch_prices_for_ticker(ticker, start, end)
        if prices is None:
            print(f"No price data for ticker {ticker}")
            continue

        # For each post for this ticker, pick nearest previous for post_date and nearest next for post_date+7
        for idx in subset_idx:
            post_date_str = df.at[idx, 'post_date']
            if not post_date_str:
                continue
            post_date = datetime.fromisoformat(post_date_str).date()
            target_forward = post_date + timedelta(days=7)

            prev_date = nearest_previous_trading_date(prices, post_date)
            next_date = nearest_next_trading_date(prices, target_forward)

            if prev_date:
                prev_close = prices.loc[str(prev_date)]['Close']
                # prices.loc[...] may return a single-element Series; use .iloc[0] to avoid future deprecation
                df.at[idx, 'price_post'] = float(prev_close.iloc[0]) if hasattr(prev_close, 'iloc') else float(prev_close)
                df.at[idx, 'price_post_date'] = prev_date.isoformat()
            if next_date:
                next_close = prices.loc[str(next_date)]['Close']
                df.at[idx, 'price_7d'] = float(next_close.iloc[0]) if hasattr(next_close, 'iloc') else float(next_close)
                df.at[idx, 'price_7d_date'] = next_date.isoformat()

            # compute diffs if both available
            try:
                if pd.notna(df.at[idx, 'price_post']) and pd.notna(df.at[idx, 'price_7d']):
                    p0 = float(df.at[idx, 'price_post'])
                    p1 = float(df.at[idx, 'price_7d'])
                    df.at[idx, 'price_diff'] = p1 - p0
                    df.at[idx, 'price_pct_change'] = (p1 - p0) / p0 if p0 != 0 else pd.NA
            except Exception as e:
                print(f"Error computing diff for index {idx}, ticker {ticker}: {e}")

            # be polite to yfinance
            time.sleep(0.05)

    return df


def main():
    parser = argparse.ArgumentParser(description='Augment reddit CSV with yfinance prices')
    parser.add_argument('--input', required=True, help='input CSV path')
    parser.add_argument('--output', required=True, help='output CSV path')
    parser.add_argument('--date-col', default='created_utc', help='column name for post unix timestamp (seconds)')
    parser.add_argument('--text-cols', default='title,selftext', help='comma separated text columns to search for ticker')
    parser.add_argument('--full-output', action='store_true', help='when set, include all original input columns in the output (in addition to price columns)')
    args = parser.parse_args()

    text_cols = [c.strip() for c in args.text_cols.split(',') if c.strip()]

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    aug = augment_dataframe(df, date_col=args.date_col, text_cols=text_cols)

    # Price columns to ensure exist
    price_cols = ['price_post_date', 'price_post', 'price_7d_date', 'price_7d',
                  'price_diff', 'price_pct_change']
    for c in price_cols:
        if c not in aug.columns:
            aug[c] = pd.NA

    if args.full_output:
        # include all original input columns followed by the price columns
        input_cols = list(df.columns)
        # avoid duplicate 'ticker' or price cols if present in input_cols
        # ensure price_cols present at the end
        out_cols = input_cols + [c for c in price_cols if c not in input_cols]
    else:
        out_cols = ['title', 'label', 'sentiment_score', 'ticker'] + price_cols

    aug[out_cols].to_csv(args.output, index=False)
    print(f"Wrote augmented data to {args.output} with columns: {out_cols}")


if __name__ == '__main__':
    main()
