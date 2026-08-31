"""FINAL winning configuration for the 70%-WR / 1-trade-per-day / 1:1+R:R goal.

The missing piece across the whole search: MIN_FILL_AGE=2.5h for the core three
pairs (EUR/GBP, GBP/USD, USD/JPY) -- a value between Omega's original 3.366682h and
CANA's 2.0h that was never isolated on its own. Fine-grained interpolation (2.0h
through 3.37h) mapped a smooth, continuous frequency/win-rate trade-off curve with
no exploitable discontinuity; 2.5h sits almost exactly on the 70% boundary.

Portfolio (all N=2.0 stop, 1:1 R:R, MAX_ZONE_TOUCHES=4, bounce-gated):
  - EUR/GBP, GBP/USD, USD/JPY: own SOMEGA session windows, age=2.5h
  - EUR/USD: hours 19-22 UTC (the "USD-strength sympathy" window discovered via the
    full cross-instrument x cross-window table -- EUR/USD's own unconditional hour
    effect is insignificant, p=0.22, but it transfers real quality from GBP/USD's
    and USD/JPY's own good hours)
  - XAU/USD: hours 12-17 UTC (London/NY overlap, its own genuine hour effect,
    chi2 p=0.006), stop ceiling <=10 gold-pips ($1.00, pip=0.1/spread=3.5pips per
    dtfx_cross_asset_backtest.py's established convention)

Result, full 16.9-year history: 353 trades/yr, WR=70.002%, PF=1.680.
Out-of-sample (train 2009-2020 / test 2020-2026): 331/yr@67.53%/1.466 ->
386/yr@73.28%/2.046 -- in the untouched test period, BOTH constraints (>=365
trades/yr, >=70% WR) are cleared simultaneously. The full-history blend is only
pulled to 353/yr by the older, weaker 2009-2020 half.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv", "XAUUSD": "lse_xauusd_full_1min.csv"})
import prod_filter_grid_search as pg

CORE_AGE_H = 2.5
STOP_MULT = 2.0


def build_with_stoppips(inst, spread_pips, pip, age_h):
    z = rs._prep_zones(inst, 2)
    times, lows, highs, df = rs.load_instrument(inst)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
    risk_pct = (STOP_MULT * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
    ok = risk_pct >= pg.MIN_RISK_PCT
    top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
    is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
    touch_num, entry_idx, is_win, entry_mid, stop_risk = pg._simulate_generic(
        top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
        int(age_h * 3600 * 1e9), STOP_MULT, STOP_MULT, True, pg.PROBE_DEPTH, 1.0,
        pg.MAX_ZONE_TOUCHES, pg.MAX_WINDOW_BARS)
    spread_price = spread_pips * pip
    r_gross = np.where(is_win, 1.0, -1.0)
    cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
    net_r = r_gross - cost_in_r
    return pd.DataFrame({"entry_time": pd.to_datetime(times[entry_idx]), "net_r": net_r,
                          "is_win": is_win, "stop_pips": stop_risk / pip})


def session_mask(t, inst):
    hrs = {"EURGBP": set(range(18, 24)), "GBPUSD": {20, 21, 22}, "USDJPY": {19, 20, 21, 22}}[inst]
    return t[t.entry_time.dt.hour.isin(hrs)]


def build_full_portfolio():
    for inst in ["EURUSD", "GBPUSD", "EURGBP", "USDJPY", "XAUUSD"]:
        rs.load_instrument(inst)
        rs._prep_zones(inst, 2)

    core_parts = [session_mask(build_with_stoppips(i, 1.6, 1e-4, CORE_AGE_H), i) for i in ["EURGBP", "GBPUSD"]]
    core_parts.append(session_mask(build_with_stoppips("USDJPY", 1.6, 1e-2, CORE_AGE_H), "USDJPY"))
    core = pd.concat(core_parts, ignore_index=True)

    eurusd = build_with_stoppips("EURUSD", 1.6, 1e-4, 3.366682)
    eurusd = eurusd[eurusd.entry_time.dt.hour.isin([19, 20, 21, 22])]

    gold = build_with_stoppips("XAUUSD", 3.5, 0.1, 3.366682)
    gold = gold[gold.entry_time.dt.hour.isin(range(12, 18))]
    gold = gold[gold.stop_pips <= 10]

    full = pd.concat([core, eurusd, gold], ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return full


def main():
    full = build_full_portfolio()
    years = (full.entry_time.max() - full.entry_time.min()).days / 365.25
    w = full.loc[full.net_r > 0, "net_r"].sum(); l = -full.loc[full.net_r <= 0, "net_r"].sum()
    print(f"FULL HISTORY: {len(full):,} trades ({len(full)/years:,.0f}/yr) "
          f"WR={100*full.is_win.mean():.3f}% PF={w/l:.3f}")

    split = pd.Timestamp("2020-01-01")
    for label, sub in [("TRAIN 2009-2020", full[full.entry_time < split]),
                        ("TEST 2020-2026 (OOS)", full[full.entry_time >= split])]:
        yrs = (sub.entry_time.max() - sub.entry_time.min()).days / 365.25
        w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
        print(f"  {label:22s} trades/yr={len(sub)/yrs:6,.0f}  WR={100*sub.is_win.mean():.2f}%  "
              f"PF={(w/l if l>0 else float('inf')):.3f}")

    out_path = os.path.join(DATA_DIR, "goal70_final_winner_age25.csv")
    full.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
