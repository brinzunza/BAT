"""Updates the 4 tradeable pairs' 1-minute LSE vault files to the present, resuming from
each file's own existing max timestamp (same pattern as fetch_lse_eurusd.py / vault API
--x-api-key auth). EUR/USD, GBP/USD, EUR/GBP use the "_full" (2009-09-25-on) files;
USD/JPY uses the "_2018on" file, matching what's already on disk for each.

Usage:
    LSE_API_KEY=lse_live_... python3 fetch_lse_update_tradeable.py
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
PAGE_LIMIT = 5000
SLEEP_BETWEEN_REQUESTS = 0.65
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

JOBS = [
    ("EUR/USD", "2009-09-25T00:00:00", os.path.join(DATA_DIR, "lse_eurusd_1min_full.csv")),
    ("GBP/USD", "2009-09-25T00:00:00", os.path.join(DATA_DIR, "lse_gbpusd_1min_full.csv")),
    ("EUR/GBP", "2009-09-25T00:00:00", os.path.join(DATA_DIR, "lse_eurgbp_1min_full.csv")),
    ("USD/JPY", "2018-01-01T00:00:00", os.path.join(DATA_DIR, "lse_usdjpy_1min_2018on.csv")),
]

headers = {"x-api-key": API_KEY}

for symbol, default_start, out_path in JOBS:
    resume_from = default_start
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, usecols=["date"], parse_dates=["date"])
        if len(existing) > 0:
            resume_from = (existing["date"].max() + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
            print(f"[{symbol}] resuming from {resume_from} (existing: {len(existing):,} rows, "
                  f"last={existing['date'].max()})", flush=True)

    cursor = resume_from
    total_written = 0
    batch_num = 0
    write_header = not os.path.exists(out_path)

    while True:
        params = {"symbol": symbol, "timeframe": "1m", "start": cursor, "order": "asc", "limit": PAGE_LIMIT}
        try:
            r = requests.get(f"{BASE_URL}/candles", params=params, headers=headers, timeout=30)
            if r.status_code != 200:
                print(f"[{symbol}] HTTP {r.status_code} {r.text[:200]}, retry in 5s", flush=True)
                time.sleep(5)
                continue
            rows = r.json()
        except Exception as e:
            print(f"[{symbol}] error {e}, retry in 5s", flush=True)
            time.sleep(5)
            continue

        if not rows:
            print(f"[{symbol}] no more candles at {cursor}, done", flush=True)
            break

        df = pd.DataFrame(rows).rename(columns={"ts": "date"})[["date", "open", "high", "low", "close"]]
        df["date"] = pd.to_datetime(df["date"])
        df.to_csv(out_path, mode="a", header=write_header, index=False)
        write_header = False
        total_written += len(df)
        batch_num += 1
        last_time = df["date"].iloc[-1]
        cursor = (last_time + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")

        if batch_num % 40 == 0:
            print(f"[{symbol}] batch {batch_num}: {total_written:,} rows, up to {last_time}", flush=True)

        if len(rows) < PAGE_LIMIT:
            print(f"[{symbol}] final partial page ({len(rows)}) at {last_time}, reached present", flush=True)
            break
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"[{symbol}] DONE: {total_written:,} rows written this run -> {out_path}\n", flush=True)

print("ALL 4 TRADEABLE PAIRS UPDATED", flush=True)
