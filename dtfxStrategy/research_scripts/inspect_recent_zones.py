"""Inspect recent detected zones, active resting levels, and price ranges across all 4 tickers.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

ROOT = '/Users/brunoinzunza/Documents/github/PRODLIVETRADING'
sys.path.insert(0, ROOT)
import config
from broker.oanda_client import OANDAClient
from strategy.pivots import find_pivots_strict
from strategy.engine import run_structure_fsm

client = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG, environment=config.OANDA_ENVIRONMENT)
symbols = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]

now_utc = datetime.now(timezone.utc)
since_24h = now_utc - timedelta(hours=24)

print("=" * 110)
print(f"  ZONE & PRICE ACTION INSPECTION (LAST 24 HOURS: {since_24h.strftime('%Y-%m-%d %H:%M UTC')} to Present)")
print("=" * 110)

for sym in symbols:
    df = client.fetch_warmup_bars(sym, count=4500)
    df_24h = df[df.index >= since_24h]
    
    low_24h = df_24h["low"].min() if len(df_24h) > 0 else 0
    high_24h = df_24h["high"].max() if len(df_24h) > 0 else 0
    curr_price = df["close"].iloc[-1]
    
    zigzag = find_pivots_strict(df, min_retracement=3)
    _, zones = run_structure_fsm(zigzag)

    print(f"\n🔹 {sym:7s} | Current Price: {curr_price:.5f} | 24h Range: [{low_24h:.5f} - {high_24h:.5f}] ({len(df_24h)} bars)")
    print(f"   • Total Active / Detected Zones in Warmup: {len(zones)}")
    
    since_naive = (now_utc - timedelta(hours=48)).replace(tzinfo=None)
    recent_zones = []
    for z in zones:
        t_z = pd.Timestamp(z["created_at"])
        if t_z.tzinfo is not None:
            t_z = t_z.tz_localize(None)
        if t_z >= since_naive:
            recent_zones.append(z)
    print(f"   • Zones Created in Last 48 Hours: {len(recent_zones)}")
    for z in recent_zones[-5:]:
        created_pdt = (pd.Timestamp(z["created_at"]) - pd.Timedelta(hours=7)).strftime('%b %d %I:%M %p')
        print(f"     [{z['side'].upper():4s}] Created: {created_pdt} PDT | Top: {z['top']:.5f} | Mid: {z['mid']:.5f} | Bottom: {z['bottom']:.5f}")

print("\n" + "=" * 110 + "\n")
