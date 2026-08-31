"""Fetch the absolute latest live market candles from OANDA and display all currently active valid zones across all 4 tickers with pips distance, stops, targets, and age.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

ROOT = '/Users/brunoinzunza/Documents/github/PRODLIVETRADING'
sys.path.insert(0, ROOT)
import config
from broker.oanda_client import OANDAClient
from strategy.engine import SymbolEngine, N_MULTIPLIER

client = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG, environment=config.OANDA_ENVIRONMENT)
symbols = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]

now_utc = datetime.now(timezone.utc)
now_pdt = datetime.now()

print("=" * 135)
print(f"  LIVE ACTIVE VALID STRUCTURAL ZONES AUDIT ({now_pdt.strftime('%b %d, %Y %I:%M:%S %p %Z')} / {now_utc.strftime('%H:%M:%S UTC')})")
print("=" * 135)

all_active_rows = []

for sym in symbols:
    engine = SymbolEngine(sym, warmup_bars=config.WARMUP_BARS)
    df = client.fetch_warmup_bars(sym, count=config.WARMUP_BARS)
    engine.seed_history(df)
    
    curr_price = df["close"].iloc[-1]
    active_zones = list(engine.active_zones.values())
    pip_mult = 100.0 if "JPY" in sym else 10000.0

    print(f"\n📊 {sym:7s} | Live Price: {curr_price:.5f} | Total Active Zones: {len(active_zones)} | 75h History Seeded")
    print("-" * 135)

    if not active_zones:
        print("   (No uninvalidated zones currently active)")
        continue

    rows = []
    for z in active_zones:
        t_created = pd.Timestamp(z["created_at"])
        t_last = pd.Timestamp(df.index[-1]).tz_localize(None) if pd.Timestamp(df.index[-1]).tzinfo is not None else pd.Timestamp(df.index[-1])
        t_c_naive = t_created.tz_localize(None) if t_created.tzinfo is not None else t_created
        pdt_time = (t_c_naive - pd.Timedelta(hours=7)).strftime('%b %d %I:%M %p')
        age_h = (t_last - t_c_naive).total_seconds() / 3600.0
        
        is_buy = z["side"] == "buy"
        mid = z["mid"]
        top = z["top"]
        bottom = z["bottom"]
        width_pips = (top - bottom) * pip_mult
        
        risk_amt = N_MULTIPLIER * (top - bottom) / 2.0
        stop_p = mid - risk_amt if is_buy else mid + risk_amt
        target_p = mid + risk_amt if is_buy else mid - risk_amt
        
        dist_pips = (mid - curr_price) * pip_mult if is_buy else (curr_price - mid) * pip_mult
        # If is_buy and price is above mid, distance to entry is negative or positive:
        dist_to_entry_pips = abs(curr_price - mid) * pip_mult

        touches_used = z["touch_count"]
        rem_touches = 4 - touches_used

        rows.append({
            "Side": z["side"].upper(),
            "Midpoint (Entry)": f"{mid:.5f}",
            "Stop Price": f"{stop_p:.5f}",
            "Target Price": f"{target_p:.5f}",
            "Width (Pips)": f"{width_pips:.1f}p",
            "Distance": f"{dist_to_entry_pips:.1f} pips",
            "Created At (PDT)": pdt_time,
            "Age": f"{age_h:.1f}h",
            "Touches": f"{touches_used}/4 (Rem: {rem_touches})",
            "Status": "ARMED / WATCHING" if age_h >= 3.33 else f"FILTERED ({3.33-age_h:.1f}h left)"
        })

    df_rows = pd.DataFrame(rows)
    print(df_rows.to_string(index=False))

print("\n" + "=" * 135 + "\n")
