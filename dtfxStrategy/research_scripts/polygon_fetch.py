import os
import sys
import time
import json
import numpy as np
import pandas as pd
import requests

API_KEY = os.environ["POLYGON_API_KEY"]  # set this before running
CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"

SYMBOL_MAP = {"BTC-USDT": "X:BTCUSD", "ETH-USDT": "X:ETHUSD", "SOL-USDT": "X:SOLUSD", "XAUT-USDT": "X:XAUTUSD"}


def fetch_with_retry(url, params=None, retries=6, backoff=15):
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = backoff * (attempt + 1)
            print(f"    429 rate-limited, waiting {wait}s (attempt {attempt+1}/{retries})", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def fetch_polygon_year(ticker, year=2025):
    symbol = SYMBOL_MAP[ticker]
    cache_path = f"{CACHE_DIR}/polygon_{ticker.replace('-','_')}_{year}.csv"

    all_rows = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{year}-01-01/{year}-12-31"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}

    page = 0
    while url:
        page += 1
        d = fetch_with_retry(url, params=params if page == 1 else {"apiKey": API_KEY})
        results = d.get("results", [])
        all_rows.extend(results)
        print(f"  {ticker} page {page}: +{len(results)} bars (total {len(all_rows)})", flush=True)
        url = d.get("next_url")
        params = None
        if url:
            time.sleep(13)  # stay under the free-tier ~5 calls/minute limit

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "vol"})
    df = df[["date", "open", "high", "low", "close", "vol"]].drop_duplicates(subset="date").sort_values("date")
    df.to_csv(cache_path, index=False)
    print(f"Saved {ticker}: {len(df)} bars, {df['date'].min()} to {df['date'].max()} -> {cache_path}", flush=True)
    return df


if __name__ == "__main__":
    tickers_to_fetch = sys.argv[1:] if len(sys.argv) > 1 else list(SYMBOL_MAP.keys())
    for t in tickers_to_fetch:
        print(f"=== Fetching {t} for 2025 ===", flush=True)
        fetch_polygon_year(t)
