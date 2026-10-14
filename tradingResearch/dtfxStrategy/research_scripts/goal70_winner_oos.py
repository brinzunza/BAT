"""OOS validation of the winning combined portfolio: EURGBP+GBPUSD+USDJPY (own SOMEGA
session windows) + EURUSD (hours 19-22, correlation-driven quality window) + XAUUSD
(hours 12-17 London/NY overlap, stop ceiling <=10 gold-pips/$1.00). All 1:1 R:R."""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv", "XAUUSD": "lse_xauusd_full_1min.csv"})
import prod_filter_grid_search as pg

for inst in ["EURUSD", "GBPUSD", "EURGBP", "USDJPY", "XAUUSD"]:
    rs.load_instrument(inst)
    rs._prep_zones(inst, 2)


def build_with_stoppips(inst, spread_pips, pip):
    z = rs._prep_zones(inst, 2)
    times, lows, highs, df = rs.load_instrument(inst)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
    risk_pct = (2.0 * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
    ok = risk_pct >= pg.MIN_RISK_PCT
    top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
    is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
    touch_num, entry_idx, is_win, entry_mid, stop_risk = pg._simulate_generic(
        top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
        int(3.366682 * 3600 * 1e9), 2.0, 2.0, True, pg.PROBE_DEPTH, 1.0, pg.MAX_ZONE_TOUCHES, pg.MAX_WINDOW_BARS)
    spread_price = spread_pips * pip
    r_gross = np.where(is_win, 1.0, -1.0)
    cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
    net_r = r_gross - cost_in_r
    return pd.DataFrame({"entry_time": pd.to_datetime(times[entry_idx]), "net_r": net_r,
                          "is_win": is_win, "stop_pips": stop_risk / pip})


def session_mask(inst_df, inst):
    hrs = {"EURGBP": set(range(18, 24)), "GBPUSD": {20, 21, 22}, "USDJPY": {19, 20, 21, 22}}[inst]
    return inst_df[inst_df.entry_time.dt.hour.isin(hrs)]


core_parts = [session_mask(build_with_stoppips(i, 1.6, 1e-4), i) for i in ["EURGBP", "GBPUSD"]]
core_parts.append(session_mask(build_with_stoppips("USDJPY", 1.6, 1e-2), "USDJPY"))
eurusd = build_with_stoppips("EURUSD", 1.6, 1e-4)
eurusd = eurusd[eurusd.entry_time.dt.hour.isin([19, 20, 21, 22])]
gold = build_with_stoppips("XAUUSD", 3.5, 0.1)
gold = gold[gold.entry_time.dt.hour.isin(range(12, 18))]
gold = gold[gold.stop_pips <= 10]

full = pd.concat(core_parts + [eurusd, gold], ignore_index=True).sort_values("entry_time").reset_index(drop=True)
years = (full.entry_time.max() - full.entry_time.min()).days / 365.25
w = full.loc[full.net_r > 0, "net_r"].sum(); l = -full.loc[full.net_r <= 0, "net_r"].sum()
print(f"FULL HISTORY: {len(full):,} trades ({len(full)/years:,.0f}/yr) WR={100*full.is_win.mean():.2f}% PF={w/l:.3f}")

split = pd.Timestamp("2020-01-01")
for label, sub in [("TRAIN 2009-2020", full[full.entry_time < split]), ("TEST 2020-2026 (OOS)", full[full.entry_time >= split])]:
    yrs = (sub.entry_time.max() - sub.entry_time.min()).days / 365.25
    w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
    print(f"  {label:22s} trades/yr={len(sub)/yrs:6,.0f}  WR={100*sub.is_win.mean():.2f}%  PF={(w/l if l>0 else float('inf')):.3f}")

full.to_csv(os.path.join(DATA_DIR, "goal70_winner_trades.csv"), index=False)
print(f"\nSaved: {os.path.join(DATA_DIR, 'goal70_winner_trades.csv')}")
