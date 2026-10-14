"""Inspect exact trade frequency and active trading hours per symbol in UTC, EST, and PDT.
"""
import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_periodicity_profitability_analysis import generate_all_trades

trades = generate_all_trades()
trades = trades[trades["exit_time"] >= pd.Timestamp("2005-09-25")].sort_values("exit_time").reset_index(drop=True)
trades["hour_utc"] = trades["entry_time"].dt.hour
trades["hour_est"] = (trades["hour_utc"] - 4) % 24  # EDT (UTC-4)
trades["hour_pdt"] = (trades["hour_utc"] - 7) % 24  # PDT (UTC-7)

print("=" * 90)
print("  HOURLY TRADE DISTRIBUTION & ACTIVE WINDOWS PER SYMBOL")
print("=" * 90)

for sym in ["EURUSD", "EURGBP", "GBPUSD", "USDJPY"]:
    s_df = trades[trades["symbol"] == sym]
    n_tot = len(s_df)
    hours_utc = s_df["hour_utc"].value_counts().sort_index()
    active_utc_range = f"{hours_utc.index.min():02d}:00 - {hours_utc.index.max():02d}:00 UTC"
    
    # Peak hours
    top3_utc = hours_utc.nlargest(3).index.tolist()
    top3_str_utc = ", ".join([f"{h:02d}:00 UTC" for h in top3_utc])
    top3_str_pdt = ", ".join([f"{(h-7)%24:02d}:00 PDT" for h in top3_utc])
    top3_str_est = ", ".join([f"{(h-4)%24:02d}:00 EDT" for h in top3_utc])

    print(f"\n🔹 {sym} (Total Trades: {n_tot:,}):")
    print(f"   • Active Window (UTC):      {active_utc_range}")
    print(f"   • Active Window (US West / PDT): {(hours_utc.index.min()-7)%24:02d}:00 - {(hours_utc.index.max()-7)%24:02d}:00 PDT")
    print(f"   • Active Window (US East / EDT): {(hours_utc.index.min()-4)%24:02d}:00 - {(hours_utc.index.max()-4)%24:02d}:00 EDT")
    print(f"   • Peak Entry Hours:         {top3_str_pdt} ({top3_str_est})")

print("\n" + "=" * 90)
