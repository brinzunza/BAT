"""Inspect SymbolEngine active zones after seed_history on live OANDA warmup bars.
"""
import os
import sys
import pandas as pd
from datetime import datetime, timezone, timedelta

ROOT = '/Users/brunoinzunza/Documents/github/PRODLIVETRADING'
sys.path.insert(0, ROOT)
import config
from broker.oanda_client import OANDAClient
from strategy.engine import SymbolEngine

client = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG, environment=config.OANDA_ENVIRONMENT)
symbols = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]

print("=" * 115)
print("  LIVE BOT WARMUP & PAST VALID ZONE RECONSTRUCTION AUDIT")
print("=" * 115)

for sym in symbols:
    engine = SymbolEngine(sym, warmup_bars=config.WARMUP_BARS)
    df = client.fetch_warmup_bars(sym, count=config.WARMUP_BARS)
    engine.seed_history(df)
    
    curr_price = df["close"].iloc[-1]
    active_zones = list(engine.active_zones.values())
    
    print(f"\n🔹 {sym:7s} | Current Price: {curr_price:.5f} | Warmup Bars Seeded: {len(df)} (~{len(df)/60:.1f} hours of history)")
    print(f"   • Active / Valid Uninvalidated Zones in Memory: {len(active_zones)}")
    
    if active_zones:
        for z in active_zones[-5:]:
            t_created = pd.Timestamp(z["created_at"])
            t_last = pd.Timestamp(df.index[-1]).tz_localize(None) if pd.Timestamp(df.index[-1]).tzinfo is not None else pd.Timestamp(df.index[-1])
            t_c_naive = t_created.tz_localize(None) if t_created.tzinfo is not None else t_created
            pdt_time = (t_c_naive - pd.Timedelta(hours=7)).strftime('%b %d %I:%M %p')
            age_h = (t_last - t_c_naive).total_seconds() / 3600.0
            print(f"     [{z['side'].upper():4s}] Created: {pdt_time} PDT (Age: {age_h:.1f}h) | Mid: {z['mid']:.5f} | Top: {z['top']:.5f} | Bot: {z['bottom']:.5f} | Touches: {z['touch_count']}/4")
    else:
        print("     (No uninvalidated zones currently active)")

print("\n" + "=" * 115 + "\n")
