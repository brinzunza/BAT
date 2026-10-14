"""Extract and display detailed trade-by-trade breakdown over the past week of historical data.
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

def generate_full_trade_records():
    min_retracement = int(BASE_PARAMS["min_retracement"])
    n_multiplier = float(BASE_PARAMS["n_multiplier"])
    min_risk_pct = float(BASE_PARAMS["min_risk_pct"])
    min_fill_bars = int(round(BASE_PARAMS["min_fill_age_hours"] * 60))

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
                        dur_mins = int(round((x_time - e_time).total_seconds() / 60.0))
                        dur_h = dur_mins // 60
                        dur_m = dur_mins % 60
                        all_trades.append({
                            "symbol": symbol,
                            "side": "BUY" if z_buy[i] else "SELL",
                            "entry_time_utc": e_time,
                            "exit_time_utc": x_time,
                            "entry_time_pdt": e_time - pd.Timedelta(hours=7),
                            "exit_time_pdt": x_time - pd.Timedelta(hours=7),
                            "duration": f"{dur_h}h {dur_m:02d}m",
                            "duration_mins": dur_mins,
                            "touch_count": t + 1,
                            "outcome": "WIN (+1.0R)" if stat == 2 else "LOSS (-1.0R)",
                            "is_win": 1 if stat == 2 else 0,
                            "pnl_r": 1.0 if stat == 2 else -1.0,
                            "entry_price": float(z_mid[i]),
                        })

    df = pd.DataFrame(all_trades).sort_values("entry_time_utc").reset_index(drop=True)
    return df

df = generate_full_trade_records()
max_date = df["entry_time_utc"].max()
start_date = max_date - pd.Timedelta(days=8)

recent_df = df[df["entry_time_utc"] >= start_date].copy().reset_index(drop=True)
recent_df["day_pdt"] = recent_df["entry_time_pdt"].dt.strftime("%Y-%m-%d (%a)")

print("=" * 125)
print(f"  PAST WEEK DETAILED TRADE-BY-TRADE LOG ({start_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')})")
print("=" * 125)

total_w = (recent_df["is_win"] == 1).sum()
total_l = (recent_df["is_win"] == 0).sum()
total_n = len(recent_df)
wr = total_w / total_n * 100.0 if total_n > 0 else 0
pf = total_w / total_l if total_l > 0 else float("inf")
tot_r = recent_df["pnl_r"].sum()

print(f"  • Total Trades:     {total_n}")
print(f"  • Wins / Losses:    {total_w} Wins / {total_l} Losses")
print(f"  • Win Rate %:       {wr:.1f}%")
print(f"  • Profit Factor:    {pf:.2f}")
print(f"  • Net Return (R):   {tot_r:+.1f} R")
print(f"  • Avg Trade Length: {recent_df['duration_mins'].mean() / 60.0:.1f} hours ({recent_df['duration_mins'].median():.0f} mins median)")
print("=" * 125)

# Group by day and print
cum_r = 0.0
for day, grp in recent_df.groupby("day_pdt", sort=False):
    day_w = (grp["is_win"] == 1).sum()
    day_l = (grp["is_win"] == 0).sum()
    day_r = grp["pnl_r"].sum()
    cum_r += day_r
    print(f"\n📅 {day.upper()} — Trades: {len(grp)} | {day_w}W / {day_l}L | Daily: {day_r:+.1f} R | Running Total: {cum_r:+.1f} R")
    print("-" * 125)
    
    display_rows = []
    for idx, r in grp.iterrows():
        display_rows.append({
            "Pair": r["symbol"],
            "Side": r["side"],
            "Touch": f"#{r['touch_count']}",
            "Entry Time (PDT)": r["entry_time_pdt"].strftime("%I:%M %p"),
            "Exit Time (PDT)": r["exit_time_pdt"].strftime("%I:%M %p"),
            "Entry Time (UTC)": r["entry_time_utc"].strftime("%H:%M UTC"),
            "Exit Time (UTC)": r["exit_time_utc"].strftime("%H:%M UTC"),
            "Duration": r["duration"],
            "Entry Price": f"{r['entry_price']:.5f}",
            "Outcome": r["outcome"],
            "P&L": f"{r['pnl_r']:+.1f} R"
        })
    print(pd.DataFrame(display_rows).to_string(index=False))

print("\n" + "=" * 125 + "\n")
