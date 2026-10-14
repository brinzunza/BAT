"""Empirical Research on Maximum Zone Age / Expiry Filter (2005-2026).

Analyzes:
1. Zone Age Cohort Analysis (Win Rate % and PF bucketed by actual elapsed hours from zone creation to first touch).
2. Parameter Sweep of Maximum Zone Age Filter (from 6 hours to 720 hours vs Unlimited).
3. Impact on Total Trades, Win Rate %, Profit Factor, Annual Return, Daily Compounding Sharpe, and Max Drawdown.
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_periodicity_profitability_analysis import (
    load_instrument_universal, _prep_zones_universal,
    INSTRUMENTS, BASE_PARAMS, SESSION_HOUR_FILTER,
    MAX_BARS_TO_RESOLVE, MAX_TOUCHES
)

@njit
def sim_multitouch_with_max_age_numba(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                                      lows, highs, n_multiplier, min_fill_bars, max_fill_bars,
                                      max_bars, max_touches=4):
    n = len(zone_top)
    n_times = len(lows)

    touch_entry_idx = np.full((n, max_touches), -1, dtype=np.int64)
    touch_exit_idx = np.full((n, max_touches), -1, dtype=np.int64)
    touch_status = np.zeros((n, max_touches), dtype=np.int8)

    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        zone_width = top - bottom
        risk = n_multiplier * zone_width / 2.0
        invalidation = bottom if is_buy else top
        stop_level = mid - risk if is_buy else mid + risk
        target_level = mid + risk if is_buy else mid - risk

        start = zone_start_idx[i]
        if start >= n_times:
            continue

        earliest_valid_entry = start + min_fill_bars
        latest_valid_entry = start + max_fill_bars if max_fill_bars > 0 else n_times

        if earliest_valid_entry >= n_times:
            continue

        # Find first valid midpoint touch
        current_bar = start
        first_touch_bar = -1
        while current_bar < n_times:
            if max_fill_bars > 0 and current_bar > latest_valid_entry:
                break
            if is_buy and lows[current_bar] <= invalidation:
                break
            elif (not is_buy) and highs[current_bar] >= invalidation:
                break

            mid_hit = (lows[current_bar] <= mid) if is_buy else (highs[current_bar] >= mid)
            if mid_hit:
                if current_bar >= earliest_valid_entry:
                    first_touch_bar = current_bar
                    break
                else:
                    # Early touch invalidates zone
                    break
            current_bar += 1

        if first_touch_bar == -1:
            continue

        sim_bar = first_touch_bar
        for t in range(max_touches):
            touch_entry_idx[i, t] = sim_bar
            after_start = sim_bar + 1
            after_end = min(after_start + max_bars, n_times)

            sl_bar, tp_bar = -1, -1
            for j in range(after_start, after_end):
                if is_buy:
                    if sl_bar == -1 and lows[j] <= stop_level:
                        sl_bar = j
                    if tp_bar == -1 and highs[j] >= target_level:
                        tp_bar = j
                else:
                    if sl_bar == -1 and highs[j] >= stop_level:
                        sl_bar = j
                    if tp_bar == -1 and lows[j] <= target_level:
                        tp_bar = j
                if sl_bar != -1 and tp_bar != -1:
                    break

            if sl_bar == -1 and tp_bar == -1:
                break

            if sl_bar != -1 and (tp_bar == -1 or sl_bar <= tp_bar):
                touch_status[i, t] = 1  # stopped out
                touch_exit_idx[i, t] = sl_bar
                break
            else:
                touch_status[i, t] = 2  # target hit
                touch_exit_idx[i, t] = tp_bar

            # Search for re-entry on remaining touches
            re_search_start = tp_bar + 1
            re_search_end = min(re_search_start + max_bars, n_times)
            next_entry_bar = -1
            search_bar = re_search_start
            while search_bar < re_search_end:
                if is_buy and lows[search_bar] <= invalidation:
                    break
                elif (not is_buy) and highs[search_bar] >= invalidation:
                    break
                re_mid_hit = (lows[search_bar] <= mid) if is_buy else (highs[search_bar] >= mid)
                if re_mid_hit:
                    next_entry_bar = search_bar
                    break
                search_bar += 1

            if next_entry_bar == -1:
                break
            sim_bar = next_entry_bar

    return touch_entry_idx, touch_exit_idx, touch_status

def run_simulation(max_age_hours=0):
    min_retracement = int(BASE_PARAMS["min_retracement"])
    n_multiplier = float(BASE_PARAMS["n_multiplier"])
    min_risk_pct = float(BASE_PARAMS["min_risk_pct"])
    min_fill_bars = int(round(BASE_PARAMS["min_fill_age_hours"] * 60))
    max_fill_bars = int(round(max_age_hours * 60)) if max_age_hours > 0 else 0

    all_trades = []
    for symbol in INSTRUMENTS.keys():
        times, lows, highs, df = load_instrument_universal(symbol)
        packed = _prep_zones_universal(symbol, min_retracement)
        zone_top = packed["top"]
        zone_bottom = packed["bottom"]
        zone_mid = packed["mid"]
        zone_is_buy = packed["is_buy"]
        zone_start_idx = packed["start_idx"]

        zone_widths = zone_top - zone_bottom
        risks = n_multiplier * zone_widths / 2.0
        risk_pcts = risks / zone_mid
        valid_mask = risk_pcts >= min_risk_pct

        z_top = np.ascontiguousarray(zone_top[valid_mask], dtype=np.float64)
        z_bottom = np.ascontiguousarray(zone_bottom[valid_mask], dtype=np.float64)
        z_mid = np.ascontiguousarray(zone_mid[valid_mask], dtype=np.float64)
        z_buy = np.ascontiguousarray(zone_is_buy[valid_mask], dtype=np.bool_)
        z_start = np.ascontiguousarray(zone_start_idx[valid_mask], dtype=np.int64)

        e_idx, x_idx, status = sim_multitouch_with_max_age_numba(
            z_top, z_bottom, z_mid, z_buy, z_start,
            lows, highs, n_multiplier, min_fill_bars, max_fill_bars,
            MAX_BARS_TO_RESOLVE, max_touches=MAX_TOUCHES
        )

        somega_hours = SESSION_HOUR_FILTER.get(symbol)

        for i in range(len(z_top)):
            for t in range(MAX_TOUCHES):
                if e_idx[i, t] != -1 and status[i, t] in (1, 2):
                    e_time = pd.Timestamp(times[e_idx[i, t]])
                    x_time = pd.Timestamp(times[x_idx[i, t]])
                    z_created = pd.Timestamp(times[z_start[i]])
                    if somega_hours is None or e_time.hour in somega_hours:
                        stat = status[i, t]
                        age_h = (e_time - z_created).total_seconds() / 3600.0
                        all_trades.append({
                            "symbol": symbol,
                            "entry_time": e_time,
                            "exit_time": x_time,
                            "created_time": z_created,
                            "zone_age_hours": age_h,
                            "touch_count": t + 1,
                            "is_win": 1 if stat == 2 else 0,
                            "pnl_r": 1.0 if stat == 2 else -1.0,
                        })

    df = pd.DataFrame(all_trades)
    if len(df) > 0:
        df = df[df["exit_time"] >= pd.Timestamp("2005-09-25")].sort_values("exit_time").reset_index(drop=True)
    return df

print("Pre-loading data and baseline run (Unlimited Max Age)...")
base_df = run_simulation(max_age_hours=0)
base_df["date"] = base_df["exit_time"].dt.date
years_in_sample = (base_df["exit_time"].max() - base_df["exit_time"].min()).total_seconds() / (365.25 * 86400)

# 1. ZONE AGE COHORT ANALYSIS
bins = [3.33, 6, 12, 24, 48, 72, 120, 240, 720, 999999]
labels = [
    "3.3h to 6h",
    "6h to 12h",
    "12h to 24h (1d)",
    "24h to 48h (2d)",
    "48h to 72h (3d)",
    "72h to 120h (5d)",
    "120h to 240h (10d)",
    "240h to 720h (1mo)",
    "> 720h (>1 month)"
]

base_df["age_cohort"] = pd.cut(base_df["zone_age_hours"], bins=bins, labels=labels, right=False)

cohort_rows = []
for c in labels:
    sub = base_df[base_df["age_cohort"] == c]
    n = len(sub)
    if n == 0:
        continue
    w = (sub["is_win"] == 1).sum()
    l = (sub["is_win"] == 0).sum()
    wr = w / n * 100.0
    pf = w / l if l > 0 else float("inf")
    tot_r = sub["pnl_r"].sum()
    avg_r = sub["pnl_r"].mean()
    cohort_rows.append({
        "Zone Age Cohort": c,
        "Trades": f"{n:,}",
        "% of Total": f"{n / len(base_df) * 100:.1f}%",
        "Win Rate %": f"{wr:.1f}%",
        "Profit Factor": f"{pf:.2f}",
        "Avg R / Trade": f"{avg_r:+.3f} R",
        "Total Return": f"{tot_r:+.1f} R"
    })

df_cohort = pd.DataFrame(cohort_rows)
print("\n" + "=" * 115)
print("  1. ZONE AGE COHORT ANALYSIS (HOW WIN RATE & PF DECAY WITH ZONE AGE)")
print("=" * 115)
print(df_cohort.to_string(index=False))


# 2. PARAMETER SWEEP OF MAXIMUM ZONE AGE FILTER
sweep_hours = [12, 24, 36, 48, 72, 96, 120, 168, 240, 336, 504, 720, 0]
sweep_rows = []

for max_h in sweep_hours:
    label = f"{max_h} Hours ({max_h/24:.1f}d)" if max_h > 0 else "Unlimited (No Max Age)"
    sim_df = run_simulation(max_age_hours=max_h)
    n = len(sim_df)
    if n == 0:
        continue
    sim_df["date"] = sim_df["exit_time"].dt.date
    w = (sim_df["is_win"] == 1).sum()
    l = (sim_df["is_win"] == 0).sum()
    wr = w / n * 100.0
    pf = w / l if l > 0 else float("inf")
    tot_r = sim_df["pnl_r"].sum()
    ann_r = tot_r / years_in_sample

    daily = sim_df.groupby("date")["pnl_r"].sum() * 0.01
    std_d = daily.std()
    sharpe = (daily.mean() / std_d * np.sqrt(252)) if std_d > 0 else 0.0

    eq = [100.0]
    for r in sim_df["pnl_r"]:
        eq.append(eq[-1] * (1.0 + r * 0.01))
    eq_arr = np.array(eq)
    pk = np.maximum.accumulate(eq_arr)
    max_dd = ((pk - eq_arr) / pk * 100.0).max()

    sweep_rows.append({
        "Max Zone Age": label,
        "Total Trades": f"{n:,}",
        "Win Rate %": f"{wr:.1f}%",
        "Profit Factor": f"{pf:.2f}",
        "Total Net R": f"{tot_r:+.1f} R",
        "Annual Return (R)": f"{ann_r:+.1f} R",
        "Daily Sharpe": f"{sharpe:.2f}",
        "Max DD (1%)": f"{max_dd:.1f}%"
    })

df_sweep = pd.DataFrame(sweep_rows)
print("\n" + "=" * 115)
print("  2. PARAMETER SWEEP: MAXIMUM ZONE AGE EXPIRY FILTER COMPARISON")
print("=" * 115)
print(df_sweep.to_string(index=False))
print("=" * 115 + "\n")
