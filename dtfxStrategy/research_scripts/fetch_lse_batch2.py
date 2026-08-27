"""Batch 2: full-history gold (settles the earlier 'no edge on gold' question against the
newer rule), plus SPY, AAPL, and ES.F futures as requested -- genuinely different asset
classes/market structures (regular-hours equities, an equity-index future) that the COMB
search never touched.
"""
import os
import time
import requests
import pandas as pd

API_KEY = os.environ.get("LSE_API_KEY")
BASE_URL = "https://api.londonstrategicedge.com/vault"
PAGE_LIMIT = 5000
SLEEP_BETWEEN_REQUESTS = 0.5
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

JOBS = [
    ("xauusd_full", "XAU/USD", None, "2006-01-01T00:00:00"),
    ("spy", "SPY", "etf", "2003-09-10T00:00:00"),
    ("aapl", "AAPL", "stocks", "2003-09-10T00:00:00"),
    ("esf", "ES.F", "futures", "2016-05-29T00:00:00"),
]

headers = {"x-api-key": API_KEY}

for out_name, symbol, dataset, start in JOBS:
    out_path = os.path.join(DATA_DIR, f"lse_{out_name}_1min.csv")
    resume_from = start
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, usecols=["date"], parse_dates=["date"])
        if len(existing) > 0:
            resume_from = (existing["date"].max() + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
            print(f"[{symbol}] resuming from {resume_from}", flush=True)

    cursor = resume_from
    total_written = 0
    batch_num = 0
    write_header = not os.path.exists(out_path)

    while True:
        params = {"symbol": symbol, "timeframe": "1m", "start": cursor, "order": "asc", "limit": PAGE_LIMIT}
        if dataset:
            params["dataset"] = dataset
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
        df["date"] = pd.to_datetime(df["date"], format="mixed")
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

print("BATCH 2 COMPLETE", flush=True)
