"""Detailed Past 60 Days Breakdown Table for DTFX SOMEGA Multi-Touch Strategy.

Outputs a clean day-by-day table of the most recent 60 calendar days with:
- Date
- Trades
- Wins
- Losses
- Win Rate %
- Net Return (R)
- Profit Factor
- Avg Trade Length (minutes)
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

# Compute trade duration in minutes
trades["duration_minutes"] = (trades["exit_time"] - trades["entry_time"]).dt.total_seconds() / 60.0

# Get the most recent 60 calendar days of data
max_date = trades["exit_time"].max()
cutoff_date = max_date - pd.Timedelta(days=60)

sub = trades[trades["exit_time"] >= cutoff_date].copy()
sub["date"] = sub["exit_time"].dt.date

# Group by day
daily = sub.groupby("date").agg(
    trades=("pnl_r", "count"),
    wins=("is_win", lambda x: (x == 1).sum()),
    losses=("is_win", lambda x: (x == 0).sum()),
    net_r=("pnl_r", "sum"),
    avg_duration_min=("duration_minutes", "mean")
).reset_index()

daily["win_rate"] = daily["wins"] / daily["trades"] * 100
daily["profit_factor"] = daily.apply(
    lambda r: r["wins"] / r["losses"] if r["losses"] > 0 else (float("inf") if r["wins"] > 0 else 1.0),
    axis=1
)

# Format daily table
daily_display = daily.copy()
daily_display.columns = ["Date", "Trades", "Wins", "Losses", "Net Return (R)", "Avg Trade Length (min)", "Win Rate %", "Profit Factor"]
daily_display = daily_display[["Date", "Trades", "Wins", "Losses", "Win Rate %", "Net Return (R)", "Profit Factor", "Avg Trade Length (min)"]]

daily_display["Win Rate %"] = daily_display["Win Rate %"].map(lambda x: f"{x:.1f}%")
daily_display["Net Return (R)"] = daily_display["Net Return (R)"].map(lambda x: f"{x:+.1f} R")
daily_display["Profit Factor"] = daily_display["Profit Factor"].map(lambda x: "inf" if x == float("inf") else f"{x:.2f}")
daily_display["Avg Trade Length (min)"] = daily_display["Avg Trade Length (min)"].map(lambda x: f"{x:.0f}m ({x/60:.1f}h)")

print("\n" + "=" * 90)
print(f"  PAST 60 DAYS SCORECARD ({cutoff_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})")
print("=" * 90)
print(daily_display.to_string(index=False))

# Total 60-day aggregate
tot_trades = len(sub)
tot_wins = len(sub[sub["is_win"] == 1])
tot_losses = len(sub[sub["is_win"] == 0])
tot_wr = tot_wins / tot_trades * 100 if tot_trades else 0
tot_pf = tot_wins / tot_losses if tot_losses else float("inf")
tot_net_r = sub["pnl_r"].sum()
tot_avg_dur = sub["duration_minutes"].mean()

print("\n" + "=" * 90)
print("  60-DAY AGGREGATE SUMMARY")
print("=" * 90)
print(f"  • Date Range:             {cutoff_date.strftime('%b %d, %Y')} -> {max_date.strftime('%b %d, %Y')}")
print(f"  • Total Trades:           {tot_trades:,}")
print(f"  • Wins / Losses:          {tot_wins} Wins / {tot_losses} Losses")
print(f"  • Win Rate:               {tot_wr:.1f}%")
print(f"  • Profit Factor:          {tot_pf:.2f}")
print(f"  • Net Return:             {tot_net_r:+.1f} R (+{tot_net_r * 5.0:.1f}% Account Equity @ 5% Risk)")
print(f"  • Average Trade Length:   {tot_avg_dur:.0f} minutes ({tot_avg_dur/60:.1f} hours)")
print(f"  • Active Trading Days:    {len(daily)} days")
print(f"  • Profitable Days %:      {len(daily[daily['net_r'] > 0]) / len(daily) * 100:.1f}% ({len(daily[daily['net_r'] > 0])} Green / {len(daily[daily['net_r'] < 0])} Red)")
print("=" * 90)
