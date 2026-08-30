"""Calculate exact weekly trade statistics, win counts, and weekly dollar profit simulation on a $2.0M account.
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

# Group trades by ISO calendar week
trades["year_week"] = trades["entry_time"].dt.to_period("W")

weekly = trades.groupby("year_week").agg(
    total_trades=("pnl_r", "count"),
    wins=("is_win", lambda x: (x == 1).sum()),
    losses=("is_win", lambda x: (x == 0).sum()),
    net_r=("pnl_r", "sum")
).reset_index()

total_weeks = len(weekly)
avg_trades_per_week = weekly["total_trades"].mean()
median_trades_per_week = weekly["total_trades"].median()
avg_wins_per_week = weekly["wins"].mean()
avg_losses_per_week = weekly["losses"].mean()
avg_net_r_per_week = weekly["net_r"].mean()
median_net_r_per_week = weekly["net_r"].median()
profitable_weeks_pct = (weekly["net_r"] > 0).mean() * 100

START_AUM = 2000000.0
RISK_PCT = 0.05
DOLLAR_RISK = START_AUM * RISK_PCT # $100,000

avg_gross_win_usd = avg_wins_per_week * DOLLAR_RISK
avg_gross_loss_usd = avg_losses_per_week * DOLLAR_RISK
avg_net_profit_linear = avg_net_r_per_week * DOLLAR_RISK

# Compounded weekly return simulation per week
def calc_compounded_week(df_sub):
    bal = START_AUM
    for pnl in df_sub["pnl_r"]:
        bal += bal * RISK_PCT * pnl
    return bal - START_AUM

compounded_profits = trades.groupby("year_week").apply(calc_compounded_week)
avg_net_profit_compounded = compounded_profits.mean()
median_net_profit_compounded = compounded_profits.median()

print("=" * 100)
print("  WEEKLY PORTFOLIO PERFORMANCE & $2,000,000 ACCOUNT PROFIT SIMULATION")
print("=" * 100)
print(f"  • Total Historical Sample:        1,091 Weeks (22 Years: 2005 - 2026)")
print(f"  • Average Trades per Week:        {avg_trades_per_week:.1f} trades / week (Median: {median_trades_per_week:.0f})")
print(f"  • Average Wins per Week:          {avg_wins_per_week:.1f} WINS / week ({avg_wins_per_week/avg_trades_per_week*100:.1f}% WR)")
print(f"  • Average Losses per Week:        {avg_losses_per_week:.1f} LOSSES / week")
print(f"  • Average Net Edge per Week:      {avg_net_r_per_week:+.2f} R / week (Median: {median_net_r_per_week:+.1f} R)")
print(f"  • Profitable Weeks Win Rate:      {profitable_weeks_pct:.1f}% of all weeks are profitable")
print("-" * 100)
print(f"  💰 ON A $2,000,000 STARTING ACCOUNT AT 5.0% RISK ($100,000 / TRADE):")
print(f"     • Gross Winning Payout:        +${avg_gross_win_usd:,.2f} / week ({avg_wins_per_week:.1f} wins x $100k)")
print(f"     • Gross Loss Deductions:       -${avg_gross_loss_usd:,.2f} / week ({avg_losses_per_week:.1f} losses x $100k)")
print(f"     ------------------------------------------------------------------")
print(f"     • AVERAGE NET PROFIT / WEEK:   +${avg_net_profit_linear:,.2f} / week (+{avg_net_r_per_week*5.0:.1f}% / week)")
print(f"     • COMPOUNDED NET PROFIT / WEEK:+${avg_net_profit_compounded:,.2f} / week (Median: +${median_net_profit_compounded:,.2f})")
print(f"     • BEST SINGLE WEEK:            +${weekly['net_r'].max() * DOLLAR_RISK:,.2f} (+{weekly['net_r'].max():.1f} R)")
print("=" * 100 + "\n")
