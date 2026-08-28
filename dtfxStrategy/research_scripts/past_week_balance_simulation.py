"""Simulate trade-by-trade compounding balance progression over the past week.
Tracks Starting Balance, Dollar Risk, Dollar Profit, Ending Balance After Trade.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from past_week_detailed_trades import generate_full_trade_records

df = generate_full_trade_records()
max_date = df["entry_time_utc"].max()
start_date = max_date - pd.Timedelta(days=8)

recent_df = df[df["entry_time_utc"] >= start_date].copy().sort_values("entry_time_utc").reset_index(drop=True)

START_BALANCE = 100.00
RISK_PCT = 0.05

rows = []
curr_bal = START_BALANCE

for idx, r in recent_df.iterrows():
    risk_dollars = curr_bal * RISK_PCT
    is_win = r["is_win"] == 1
    pnl_r = 1.0 if is_win else -1.0
    pnl_dollars = risk_dollars if is_win else -risk_dollars
    bal_after = curr_bal + pnl_dollars

    rows.append({
        "Trade #": idx + 1,
        "Date": r["entry_time_pdt"].strftime("%b %d"),
        "Entry (PDT)": r["entry_time_pdt"].strftime("%I:%M %p"),
        "Exit (PDT)": r["exit_time_pdt"].strftime("%I:%M %p"),
        "Duration": r["duration"],
        "Ticker": r["symbol"],
        "Side": r["side"],
        "Touch": f"#{r['touch_count']}",
        "Bal Before": f"${curr_bal:.2f}",
        "Risk ($)": f"${risk_dollars:.2f}",
        "Outcome": "WIN" if is_win else "LOSS",
        "Profit / Loss": f"{pnl_dollars:+.2f} ({pnl_r:+.0f}R)",
        "Bal After": f"${bal_after:.2f}",
        "Growth %": f"{((bal_after - START_BALANCE) / START_BALANCE) * 100:+.1f}%"
    })
    curr_bal = bal_after

df_res = pd.DataFrame(rows)

print("=" * 135)
print(f"  PAST WEEK COMPLIANT BALANCE SIMULATION (STARTING CAPITAL: ${START_BALANCE:.2f} @ {RISK_PCT*100:.1f}% RISK)")
print("=" * 135)
print(df_res.to_string(index=False))
print("=" * 135)
print(f"  • Starting Balance:    ${START_BALANCE:.2f}")
print(f"  • Final Total Balance: ${curr_bal:.2f} ({((curr_bal - START_BALANCE) / START_BALANCE) * 100:+.2f}% Net Gain)")
print(f"  • Total Net R:         {recent_df['pnl_r'].sum():+.1f} R")
print(f"  • Win Rate:            {(recent_df['is_win'] == 1).mean() * 100:.1f}% ({(recent_df['is_win'] == 1).sum()}W / {(recent_df['is_win'] == 0).sum()}L)")
print("=" * 135 + "\n")
