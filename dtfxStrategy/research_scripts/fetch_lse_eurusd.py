"""Fetch EUR/USD 1-minute history from the London Strategic Edge vault API.

This is the original data vendor used throughout this project's full-history backtests
(see ../CHECKPOINT_NOTES.md). The raw file is intentionally NOT committed to git (see
.gitignore) -- it lives in ../data/, a normal on-disk project folder instead of an OS
temp directory, specifically so it survives temp-directory cleanup the way a prior copy
did not.

Usage:
    LSE_API_KEY=lse_live_... python3 fetch_lse_eurusd.py

Resumable: if the output file already has partial progress, picks up where it left off.
"""
import os
import sys
import time
import requests
import pandas as pd

API_KEY = os.environ.get("LSE_API_KEY")
if not API_KEY:
    sys.exit("Set LSE_API_KEY in the environment before running this script.")

BASE_URL = "https://api.londonstrategicedge.com/vault"
SYMBOL = "EUR/USD"
TIMEFRAME = "1m"
START = "2009-09-25T00:00:00"  # matches this project's established data window
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lse_eurusd_1min_full.csv")
PAGE_LIMIT = 5000
SLEEP_BETWEEN_REQUESTS = 0.65  # stays under the 100 calls/minute plan limit

headers = {"x-api-key": API_KEY}

resume_from = START
if os.path.exists(OUT_PATH):
    existing = pd.read_csv(OUT_PATH, usecols=["date"], parse_dates=["date"])
    if len(existing) > 0:
        resume_from = (existing["date"].max() + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
        print(f"Resuming from existing partial file: {len(existing):,} rows, last={existing['date'].max()}", flush=True)

cursor = resume_from
total_written = 0
batch_num = 0
write_header = not os.path.exists(OUT_PATH)

while True:
    params = {"symbol": SYMBOL, "timeframe": TIMEFRAME, "start": cursor, "order": "asc", "limit": PAGE_LIMIT}
    try:
        r = requests.get(f"{BASE_URL}/candles", params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"request failed at cursor={cursor}: HTTP {r.status_code} {r.text[:200]}, retrying in 5s", flush=True)
            time.sleep(5)
            continue
        rows = r.json()
    except Exception as e:
        print(f"request failed at cursor={cursor}: {e}, retrying in 5s", flush=True)
        time.sleep(5)
        continue

    if not rows:
        print(f"no more candles returned at cursor={cursor}, stopping (reached present)", flush=True)
        break

    df = pd.DataFrame(rows).rename(columns={"ts": "date"})[["date", "open", "high", "low", "close"]]
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(OUT_PATH, mode="a", header=write_header, index=False)
    write_header = False
    total_written += len(df)
    batch_num += 1

    last_time = df["date"].iloc[-1]
    cursor = (last_time + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")

    if batch_num % 20 == 0:
        print(f"batch {batch_num}: {total_written:,} rows written so far, up to {last_time}", flush=True)

    if len(rows) < PAGE_LIMIT:
        print(f"final partial page ({len(rows)} rows) at {last_time}, reached present -- stopping", flush=True)
        break

    time.sleep(SLEEP_BETWEEN_REQUESTS)

print(f"\nDone. Total rows written this run: {total_written:,}", flush=True)
final = pd.read_csv(OUT_PATH)
print(f"Final file: {len(final):,} total rows, {OUT_PATH}", flush=True)
