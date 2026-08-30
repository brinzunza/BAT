"""Calculate exact daily trade frequency, averages, and distributions across the portfolio and per symbol.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_periodicity_profitability_analysis import generate_all_trades

trades = generate_all_trades()
trades = trades[trades["exit_time"] >= pd.Timestamp("2005-09-25")].sort_values("entry_time").reset_index(drop=True)
trades["date"] = trades["entry_time"].dt.date
trades["year"] = trades["entry_time"].dt.year

# Total days in sample
min_date = trades["date"].min()
max_date = trades["date"].max()
all_calendar_days = (max_date - min_date).days + 1
# FX trading days (approx 260 days per year or count distinct weekdays)
date_range = pd.date_range(min_date, max_date)
trading_days_approx = (date_range.dayofweek < 5).sum() # Mon-Fri count

# Actual active days where trades occurred
daily_counts = trades.groupby("date").size()
active_trading_days = len(daily_counts)

print("=" * 95)
print("  PORTFOLIO DAILY TRADE FREQUENCY AUDIT (2005 - 2026)")
print("=" * 95)
print(f"  • Total Portfolio Trades:         {len(trades):,} trades")
print(f"  • Total Sample Span:              {min_date} to {max_date} ({all_calendar_days:,} calendar days)")
print(f"  • Estimated FX Trading Days:      {trading_days_approx:,} days (approx 252-260/yr)")
print(f"  • Active Trading Days (with >=1 trade): {active_trading_days:,} days ({active_trading_days / trading_days_approx * 100:.1f}% of all trading days)")
print("-" * 95)

avg_per_active_day = len(trades) / active_trading_days
avg_per_trading_day = len(trades) / trading_days_approx
avg_per_calendar_day = len(trades) / all_calendar_days

print(f"  📊 AVERAGE TRADES PER DAY:")
print(f"     • Per Active Trading Day:      {avg_per_active_day:.2f} trades / day (Median: {daily_counts.median():.0f} trades)")
print(f"     • Per Standard FX Trading Day: {avg_per_trading_day:.2f} trades / day")
print(f"     • Per Calendar Day (incl wknds):{avg_per_calendar_day:.2f} trades / day")
print(f"     • Max Trades on Single Day:    {daily_counts.max()} trades (Occurred during high volatility)")
print("-" * 95)

# Per-symbol daily averages
print("  🔹 DAILY TRADES BY SYMBOL:")
for sym in ["EURUSD", "EURGBP", "GBPUSD", "USDJPY"]:
    s_df = trades[trades["symbol"] == sym]
    s_daily = s_df.groupby("date").size()
    s_tot = len(s_df)
    s_avg_active = s_tot / len(s_daily)
    s_avg_all_days = s_tot / trading_days_approx
    print(f"     • {sym:7s}: {s_tot:6,d} trades | Avg when active: {s_avg_active:.2f}/day | Avg overall: {s_avg_all_days:.2f}/day")

print("-" * 95)
# Distribution of daily trade count
dist = daily_counts.value_counts().sort_index()
print("  📈 DAILY TRADE COUNT FREQUENCY DISTRIBUTION:")
bins = [1, 2, 3, 4, 5, 6, 8, 10, 15, 999]
bin_labels = ["1 trade", "2 trades", "3 trades", "4 trades", "5 trades", "6-7 trades", "8-9 trades", "10-14 trades", "15+ trades"]
daily_binned = pd.cut(daily_counts, bins=[0, 1, 2, 3, 4, 5, 7, 9, 14, 999], labels=bin_labels).value_counts().sort_index()

for lbl, count in daily_binned.items():
    pct = count / active_trading_days * 100
    print(f"     • {lbl:15s}: {count:5,d} days ({pct:5.1f}% of active days)")

print("=" * 95 + "\n")
