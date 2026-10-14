"""Comprehensive Buy vs Sell Directional Performance & Asymmetry Audit (2005-2026).

Analyzes:
1. Global Buy vs Sell aggregate performance metrics (Trades, Win Rate %, PF, Net R, Sharpe, Max DD)
2. Currency Pair Standalone Buy vs Sell breakdown (EURUSD, GBPUSD, EURGBP, USDJPY)
3. Trading Session Buy vs Sell breakdown (Asia, London, London/NY Overlap, SOMEGA, Late Off-Hours)
4. Multi-Touch Buy vs Sell breakdown (Touch 1, Touch 2, Touch 3, Touch 4)
5. Macro Decade / Year-by-Year Buy vs Sell consistency
6. Longest one-sided loss streaks (Buy-only streaks vs Sell-only streaks)
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_periodicity_profitability_analysis import (
    load_instrument_universal, _prep_zones_universal,
    sim_multitouch_4_numba, INSTRUMENTS, BASE_PARAMS,
    SESSION_HOUR_FILTER, MAX_BARS_TO_RESOLVE, MAX_TOUCHES
)

def generate_directional_trades():
    min_retracement = int(BASE_PARAMS["min_retracement"])
    n_multiplier = float(BASE_PARAMS["n_multiplier"])
    min_risk_pct = float(BASE_PARAMS["min_risk_pct"])
    min_fill_bars = int(round(BASE_PARAMS["min_fill_age_hours"] * 60))

    all_trades = []
    for symbol in INSTRUMENTS.keys():
        print(f"Loading {symbol}...")
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

        e_idx, x_idx, status = sim_multitouch_4_numba(
            z_top, z_bottom, z_mid, z_buy, z_start,
            lows, highs, n_multiplier, min_fill_bars, MAX_BARS_TO_RESOLVE, max_touches=MAX_TOUCHES
        )

        somega_hours = SESSION_HOUR_FILTER.get(symbol)

        for i in range(len(z_top)):
            for t in range(MAX_TOUCHES):
                if e_idx[i, t] != -1 and status[i, t] in (1, 2):
                    e_time = pd.Timestamp(times[e_idx[i, t]])
                    x_time = pd.Timestamp(times[x_idx[i, t]])
                    if somega_hours is None or e_time.hour in somega_hours:
                        stat = status[i, t]
                        all_trades.append({
                            "symbol": symbol,
                            "is_buy": bool(z_buy[i]),
                            "entry_time": e_time,
                            "exit_time": x_time,
                            "touch_count": t + 1,
                            "is_win": 1 if stat == 2 else 0,
                            "pnl_r": 1.0 if stat == 2 else -1.0,
                        })

    df = pd.DataFrame(all_trades).sort_values("exit_time").reset_index(drop=True)
    return df

trades = generate_directional_trades()
trades = trades[trades["exit_time"] >= pd.Timestamp("2005-09-25")].sort_values("exit_time").reset_index(drop=True)
trades["year"] = trades["exit_time"].dt.year
trades["date"] = trades["exit_time"].dt.date
trades["hour"] = trades["entry_time"].dt.hour

years_in_sample = (trades["exit_time"].max() - trades["exit_time"].min()).total_seconds() / (365.25 * 86400)

buys = trades[trades["is_buy"] == True].copy().reset_index(drop=True)
sells = trades[trades["is_buy"] == False].copy().reset_index(drop=True)

def compute_metrics(df, label="All"):
    n = len(df)
    if n == 0:
        return {}
    wins = (df["is_win"] == 1).sum()
    losses = (df["is_win"] == 0).sum()
    wr = wins / n * 100.0
    pf = wins / losses if losses > 0 else float("inf")
    tot_r = df["pnl_r"].sum()
    avg_r = df["pnl_r"].mean()
    ann_r = tot_r / years_in_sample

    # Daily compounding Sharpe & Max DD at 1% risk
    daily = df.groupby("date")["pnl_r"].sum() * 0.01
    std_d = daily.std()
    sharpe = (daily.mean() / std_d * np.sqrt(252)) if std_d > 0 else 0.0

    eq = [100.0]
    for r in df["pnl_r"]:
        eq.append(eq[-1] * (1.0 + r * 0.01))
    eq_arr = np.array(eq)
    pk = np.maximum.accumulate(eq_arr)
    max_dd = ((pk - eq_arr) / pk * 100.0).max()

    # Max loss streak
    streaks = []
    curr = 0
    for w in df["is_win"]:
        if w == 0:
            curr += 1
        else:
            if curr > 0:
                streaks.append(curr)
                curr = 0
    if curr > 0:
        streaks.append(curr)
    max_loss_streak = max(streaks) if streaks else 0
    avg_loss_streak = np.mean(streaks) if streaks else 0.0

    return {
        "Direction": label,
        "Trades": f"{n:,}",
        "Wins": f"{wins:,}",
        "Losses": f"{losses:,}",
        "Win Rate %": f"{wr:.1f}%",
        "Profit Factor": f"{pf:.2f}",
        "Avg R / Trade": f"{avg_r:+.3f} R",
        "Total Return": f"{tot_r:+.1f} R",
        "Annual Return (R)": f"{ann_r:+.1f} R",
        "Daily Sharpe (1%)": f"{sharpe:.2f}",
        "Max DD (1%)": f"{max_dd:.1f}%",
        "Max Loss Streak": f"{max_loss_streak} trades",
        "Avg Loss Streak": f"{avg_loss_streak:.2f} trades"
    }

# 1. Overall Buy vs Sell
row_all = compute_metrics(trades, "Combined (Buys + Sells)")
row_buy = compute_metrics(buys, "BUY Trades (Longs)")
row_sell = compute_metrics(sells, "SELL Trades (Shorts)")

df_global = pd.DataFrame([row_all, row_buy, row_sell])

print("\n" + "=" * 115)
print("  1. GLOBAL BUY VS SELL PERFORMANCE SCORECARD (2005 - 2026)")
print("=" * 115)
print(df_global.to_string(index=False))


# 2. Currency Pair Standalone Breakdown
pair_rows = []
for sym in ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]:
    sym_df = trades[trades["symbol"] == sym]
    sym_buy = sym_df[sym_df["is_buy"] == True]
    sym_sell = sym_df[sym_df["is_buy"] == False]
    
    r_b = compute_metrics(sym_buy, f"{sym} - BUY")
    r_s = compute_metrics(sym_sell, f"{sym} - SELL")
    pair_rows.extend([r_b, r_s])

df_pair = pd.DataFrame(pair_rows)
print("\n" + "=" * 115)
print("  2. CURRENCY PAIR STANDALONE BUY VS SELL BREAKDOWN")
print("=" * 115)
cols_to_show = ["Direction", "Trades", "Win Rate %", "Profit Factor", "Avg R / Trade", "Total Return", "Max Loss Streak", "Max DD (1%)"]
print(df_pair[cols_to_show].to_string(index=False))


# 3. Session Regime Breakdown
def get_session(hour):
    if 0 <= hour < 7:
        return "Asia (00-07 UTC)"
    elif 7 <= hour < 12:
        return "London Open (07-12 UTC)"
    elif 12 <= hour < 16:
        return "London/NY Overlap (12-16 UTC)"
    elif 16 <= hour < 21:
        return "NY Afternoon / SOMEGA (16-21 UTC)"
    else:
        return "Late Off-Hours (21-24 UTC)"

trades["session"] = trades["hour"].apply(get_session)
session_rows = []
for sess in ["Asia (00-07 UTC)", "London Open (07-12 UTC)", "London/NY Overlap (12-16 UTC)", "NY Afternoon / SOMEGA (16-21 UTC)", "Late Off-Hours (21-24 UTC)"]:
    s_df = trades[trades["session"] == sess]
    s_b = s_df[s_df["is_buy"] == True]
    s_s = s_df[s_df["is_buy"] == False]
    session_rows.append(compute_metrics(s_b, f"{sess} [BUY]"))
    session_rows.append(compute_metrics(s_s, f"{sess} [SELL]"))

df_sess = pd.DataFrame(session_rows)
print("\n" + "=" * 115)
print("  3. TRADING SESSION BUY VS SELL BREAKDOWN")
print("=" * 115)
print(df_sess[cols_to_show].to_string(index=False))


# 4. Multi-Touch Buy vs Sell Breakdown
touch_rows = []
for touch in range(1, 5):
    t_df = trades[trades["touch_count"] == touch]
    t_b = t_df[t_df["is_buy"] == True]
    t_s = t_df[t_df["is_buy"] == False]
    touch_rows.append(compute_metrics(t_b, f"Touch #{touch} [BUY]"))
    touch_rows.append(compute_metrics(t_s, f"Touch #{touch} [SELL]"))

df_touch = pd.DataFrame(touch_rows)
print("\n" + "=" * 115)
print("  4. MULTI-TOUCH ZONE (1 to 4) BUY VS SELL BREAKDOWN")
print("=" * 115)
print(df_touch[cols_to_show].to_string(index=False))


# 5. Longest Consecutive One-Sided Loss Streaks in the Same Direction
buy_losses = [w == 0 for w in buys["is_win"]]
sell_losses = [w == 0 for w in sells["is_win"]]

def calc_streaks(loss_bool_list):
    res = []
    curr = 0
    for l in loss_bool_list:
        if l:
            curr += 1
        else:
            if curr > 0:
                res.append(curr)
                curr = 0
    if curr > 0:
        res.append(curr)
    return res

b_streaks = calc_streaks(buy_losses)
s_streaks = calc_streaks(sell_losses)

print("\n" + "=" * 115)
print("  5. ISOLATED SUB-ACCOUNT DIRECTIONAL LOSS STREAKS (CONSECUTIVE BUY-ONLY vs SELL-ONLY LOSSES)")
print("=" * 115)
print(f"  • Long Account (Buy Trades Only):   Max Loss Streak = {max(b_streaks)} consecutive losses (Avg: {np.mean(b_streaks):.2f})")
print(f"  • Short Account (Sell Trades Only):  Max Loss Streak = {max(s_streaks)} consecutive losses (Avg: {np.mean(s_streaks):.2f})")
print("=" * 115 + "\n")
