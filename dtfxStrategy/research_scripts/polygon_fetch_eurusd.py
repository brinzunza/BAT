import os
import time
import pandas as pd
import requests

API_KEY = os.environ["POLYGON_API_KEY"]  # set this before running
CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"
SYMBOL = "C:EURUSD"


def fetch_with_retry(url, params=None, retries=8, backoff=15):
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


def fetch_polygon_range(start, end, label):
    cache_path = f"{CACHE_DIR}/polygon_EURUSD_{label}.csv"
    all_rows = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{SYMBOL}/range/1/minute/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": API_KEY}
    page = 0
    while url:
        page += 1
        d = fetch_with_retry(url, params=params if page == 1 else {"apiKey": API_KEY})
        results = d.get("results", [])
        all_rows.extend(results)
        print(f"  EURUSD page {page}: +{len(results)} bars (total {len(all_rows)})", flush=True)
        url = d.get("next_url")
        params = None
        if url:
            time.sleep(13)
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "vol"})
    df = df[["date", "open", "high", "low", "close", "vol"]].drop_duplicates(subset="date").sort_values("date")
    df.to_csv(cache_path, index=False)
    print(f"Saved EURUSD {label}: {len(df)} bars, {df['date'].min()} to {df['date'].max()} -> {cache_path}", flush=True)


if __name__ == "__main__":
    fetch_polygon_range("2024-01-01", "2024-12-31", "2024")
    fetch_polygon_range("2025-01-01", "2025-12-31", "2025")
    fetch_polygon_range("2026-01-01", "2026-08-18", "2026")
