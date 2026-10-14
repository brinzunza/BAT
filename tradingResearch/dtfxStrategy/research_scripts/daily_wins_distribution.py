"""Calculate daily win counts, distributions, and dollar expectations.
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

min_date = trades["date"].min()
max_date = trades["date"].max()
all_weekdays = pd.date_range(min_date, max_date, freq="B").date

daily = trades.groupby("date").agg(
    total_trades=("pnl_r", "count"),
    wins=("is_win", lambda x: (x == 1).sum()),
    losses=("is_win", lambda x: (x == 0).sum()),
    net_r=("pnl_r", "sum")
).reindex(all_weekdays, fill_value=0)

total_days = len(daily)
active_days = (daily["total_trades"] > 0).sum()

avg_wins_active = daily[daily["total_trades"] > 0]["wins"].mean()
avg_losses_active = daily[daily["total_trades"] > 0]["losses"].mean()
avg_net_r_active = daily[daily["total_trades"] > 0]["net_r"].mean()

avg_wins_all = daily["wins"].mean()
avg_losses_all = daily["losses"].mean()
avg_net_r_all = daily["net_r"].mean()

print("=" * 90)
print("  DAILY WINS & EXPECTED DAILY EARNINGS (2005 - 2026)")
print("=" * 90)
print(f"  • Average Wins per Active Day:    {avg_wins_active:.2f} Wins / day (Losses: {avg_losses_active:.2f})")
print(f"  • Average Net Edge per Active Day:{avg_net_r_active:+.2f} R / day (+{avg_net_r_active*5.0:.1f}%)")
print(f"  • Average Wins across All Mon-Fri:{avg_wins_all:.2f} Wins / day (Losses: {avg_losses_all:.2f})")
print(f"  • Average Net Edge All Mon-Fri:   {avg_net_r_all:+.2f} R / day (+{avg_net_r_all*5.0:.1f}%)")
print("-" * 90)

win_dist = daily["wins"].value_counts().sort_index()
print("  📈 DAILY WIN COUNT FREQUENCY (HOW MANY WINS HAPPEN IN A DAY):")
for w_cnt, days in win_dist.items():
    if w_cnt <= 8:
        print(f"     • {w_cnt} Wins in a day: {days:4d} days ({days/total_days*100:5.1f}% of all days)")
print("=" * 90 + "\n")
