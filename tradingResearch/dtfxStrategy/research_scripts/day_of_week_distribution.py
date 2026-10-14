"""Calculate Day of Week distribution (Trades, Win Rate %, PF, Net Return) in Local Time (PDT) and UTC.
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
trades["entry_pdt"] = trades["entry_time"] - pd.Timedelta(hours=7)

trades["dow_pdt"] = trades["entry_pdt"].dt.day_name()
trades["dow_pdt_num"] = trades["entry_pdt"].dt.dayofweek # 0=Mon, 6=Sun

trades["dow_utc"] = trades["entry_time"].dt.day_name()
trades["dow_utc_num"] = trades["entry_time"].dt.dayofweek

days_order = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

def analyze_dow(df, col_name, label="PDT"):
    rows = []
    tot_trades = len(df)
    for d in days_order:
        sub = df[df[col_name] == d]
        n = len(sub)
        if n == 0:
            continue
        w = (sub["is_win"] == 1).sum()
        l = (sub["is_win"] == 0).sum()
        wr = w / n * 100.0
        pf = w / l if l > 0 else float("inf")
        tot_r = sub["pnl_r"].sum()
        avg_r = sub["pnl_r"].mean()
        
        # Max loss streak on this day
        streaks = []
        curr = 0
        for win in sub["is_win"]:
            if win == 0:
                curr += 1
            else:
                if curr > 0:
                    streaks.append(curr)
                    curr = 0
        if curr > 0:
            streaks.append(curr)
        max_ls = max(streaks) if streaks else 0

        rows.append({
            "Day of Week": d,
            "Trades": f"{n:,}",
            "% of Total": f"{n / tot_trades * 100:.1f}%",
            "Wins": f"{w:,}",
            "Losses": f"{l:,}",
            "Win Rate %": f"{wr:.1f}%",
            "Profit Factor": f"{pf:.2f}",
            "Avg R / Trade": f"{avg_r:+.3f} R",
            "Total Net R": f"{tot_r:+.1f} R",
            "Max Loss Streak": f"{max_ls} trades"
        })
    return pd.DataFrame(rows)

df_pdt = analyze_dow(trades, "dow_pdt", "Local Time (PDT)")
df_utc = analyze_dow(trades, "dow_utc", "UTC")

print("=" * 115)
print("  1. DAY OF THE WEEK DISTRIBUTION (LOCAL US PACIFIC TIME / PDT)")
print("=" * 115)
print(df_pdt.to_string(index=False))

print("\n" + "=" * 115)
print("  2. DAY OF THE WEEK DISTRIBUTION (UTC)")
print("=" * 115)
print(df_utc.to_string(index=False))

# Breakdown by Symbol across days of week (PDT)
print("\n" + "=" * 115)
print("  3. TRADES PER DAY OF WEEK BY CURRENCY PAIR (LOCAL PDT)")
print("=" * 115)
pivot_trades = trades.pivot_table(index="dow_pdt", columns="symbol", values="pnl_r", aggfunc="count", fill_value=0)
pivot_trades = pivot_trades.reindex(days_order)
pivot_trades["Total"] = pivot_trades.sum(axis=1)
print(pivot_trades.to_string())

print("\n" + "=" * 115)
print("  4. WIN RATE % PER DAY OF WEEK BY CURRENCY PAIR (LOCAL PDT)")
print("=" * 115)
pivot_wr = trades.pivot_table(index="dow_pdt", columns="symbol", values="is_win", aggfunc=lambda x: f"{x.mean()*100:.1f}%")
pivot_wr = pivot_wr.reindex(days_order)
print(pivot_wr.to_string())
print("=" * 115 + "\n")
