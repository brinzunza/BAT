"""Calculate the exact historical frequency of 2+ consecutive losses at:
1. Trade level
2. Day level
3. Week level
4. Month level
across the 22-year dataset (2005 - 2026).
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

# 1. TRADE LEVEL STREAKS
is_loss_trade = (trades["is_win"] == 0).astype(int).values

def get_streak_counts(binary_series):
    streaks = []
    curr = 0
    for val in binary_series:
        if val == 1:
            curr += 1
        else:
            if curr > 0:
                streaks.append(curr)
                curr = 0
    if curr > 0:
        streaks.append(curr)
    return streaks

trade_streaks = get_streak_counts(is_loss_trade)
total_loss_streaks = len(trade_streaks)
s_series = pd.Series(trade_streaks)

# 2. DAY LEVEL STREAKS
trades["date"] = trades["entry_time"].dt.date
daily_pnl = trades.groupby("date")["pnl_r"].sum()
# Filter for days where trades occurred
daily_loss = (daily_pnl < 0).astype(int).values
day_streaks = pd.Series(get_streak_counts(daily_loss))

# 3. WEEK LEVEL STREAKS
trades["week"] = trades["entry_time"].dt.to_period("W")
weekly_pnl = trades.groupby("week")["pnl_r"].sum()
weekly_loss = (weekly_pnl < 0).astype(int).values
week_streaks = pd.Series(get_streak_counts(weekly_loss))

# 4. MONTH LEVEL STREAKS
trades["month"] = trades["entry_time"].dt.to_period("M")
monthly_pnl = trades.groupby("month")["pnl_r"].sum()
monthly_loss = (monthly_pnl < 0).astype(int).values
month_streaks = pd.Series(get_streak_counts(monthly_loss))

years_span = (trades["entry_time"].max() - trades["entry_time"].min()).days / 365.25
total_weeks = len(weekly_pnl)
total_months = len(monthly_pnl)
total_days = len(daily_pnl)
total_trades = len(trades)

print("=" * 110)
print(f"  CONSECUTIVE LOSSES AUDIT ACROSS 22 YEARS (2005 - 2026: {years_span:.1f} YEARS)")
print("=" * 110)

# Helper function to summarize
def summarize_level(streaks_series, total_units, unit_name, years):
    print(f"\n🔹 {unit_name.upper()} LEVEL CONSECUTIVE LOSSES (Total {total_units:,} {unit_name}s in {years:.1f} yrs):")
    print("-" * 110)
    for k in [2, 3, 4, 5, 6, 7]:
        ge_k = (streaks_series >= k).sum()
        if ge_k > 0:
            freq_per_yr = ge_k / years
            interval_str = f"Every {1/freq_per_yr:.1f} yrs" if freq_per_yr < 1.0 else f"{freq_per_yr:.1f} times / yr"
            if "Trade" in unit_name:
                interval_str = f"{freq_per_yr/52:.1f} times / wk (~{freq_per_yr/12:.1f}/mo)"
            elif "Day" in unit_name:
                interval_str = f"{freq_per_yr/12:.1f} times / mo (~{freq_per_yr:.1f}/yr)"
            print(f"   • >= {k} Consecutive Losses: {ge_k:5,d} occurrences | Frequency: {interval_str}")
        else:
            print(f"   • >= {k} Consecutive Losses:     0 occurrences (NEVER HAPPENED in 22 years)")

summarize_level(s_series, total_trades, "Trade", years_span)
summarize_level(day_streaks, total_days, "Active Day", years_span)
summarize_level(week_streaks, total_weeks, "Week", years_span)
summarize_level(month_streaks, total_months, "Month", years_span)

print("\n" + "=" * 110)
print("  DETAILED BREAKDOWN MATRIX:")
print("=" * 110)

matrix_data = []

# Trade level exactly
for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    cnt_eq = (s_series == k).sum()
    cnt_ge = (s_series >= k).sum()
    pct = cnt_ge / len(s_series) * 100
    matrix_data.append({
        "Level": "Trade",
        "Streak Length": f"{k} in a row",
        "Exact Occurrences": f"{cnt_eq:,}",
        "At Least (>= k) Occurrences": f"{cnt_ge:,}",
        "% of Loss Sequences": f"{pct:.1f}%",
        "How Often": f"{cnt_ge/years_span/12:.1f} times/mo" if cnt_ge/years_span >= 12 else (f"{cnt_ge/years_span:.1f} times/yr" if cnt_ge/years_span >= 1.0 else f"Once every {years_span/cnt_ge:.1f} yrs")
    })

# Day level
for k in [1, 2, 3, 4, 5]:
    cnt_eq = (day_streaks == k).sum()
    cnt_ge = (day_streaks >= k).sum()
    pct = cnt_ge / len(day_streaks) * 100 if len(day_streaks) > 0 else 0
    how_often = f"{cnt_ge/years_span/12:.1f} times/mo" if cnt_ge/years_span >= 12 else (f"{cnt_ge/years_span:.1f} times/yr" if cnt_ge/years_span >= 1.0 else f"Once every {years_span/cnt_ge:.1f} yrs") if cnt_ge > 0 else "Never"
    matrix_data.append({
        "Level": "Day",
        "Streak Length": f"{k} in a row",
        "Exact Occurrences": f"{cnt_eq:,}",
        "At Least (>= k) Occurrences": f"{cnt_ge:,}",
        "% of Loss Sequences": f"{pct:.1f}%",
        "How Often": how_often
    })

# Week level
for k in [1, 2, 3, 4]:
    cnt_eq = (week_streaks == k).sum()
    cnt_ge = (week_streaks >= k).sum()
    pct = cnt_ge / len(week_streaks) * 100 if len(week_streaks) > 0 else 0
    how_often = f"{cnt_ge/years_span:.1f} times/yr" if cnt_ge/years_span >= 1.0 else (f"Once every {years_span/cnt_ge:.1f} yrs" if cnt_ge > 0 else "Never")
    matrix_data.append({
        "Level": "Week",
        "Streak Length": f"{k} in a row",
        "Exact Occurrences": f"{cnt_eq:,}",
        "At Least (>= k) Occurrences": f"{cnt_ge:,}",
        "% of Loss Sequences": f"{pct:.1f}%",
        "How Often": how_often
    })

# Month level
for k in [1, 2, 3]:
    cnt_eq = (month_streaks == k).sum()
    cnt_ge = (month_streaks >= k).sum()
    pct = cnt_ge / len(month_streaks) * 100 if len(month_streaks) > 0 else 0
    how_often = f"Once every {years_span/cnt_ge:.1f} yrs" if cnt_ge > 0 else "Never"
    matrix_data.append({
        "Level": "Month",
        "Streak Length": f"{k} in a row",
        "Exact Occurrences": f"{cnt_eq:,}",
        "At Least (>= k) Occurrences": f"{cnt_ge:,}",
        "% of Loss Sequences": f"{pct:.1f}%",
        "How Often": how_often
    })

print(pd.DataFrame(matrix_data).to_string(index=False))
print("=" * 110 + "\n")
