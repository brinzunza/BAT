"""Maximize total portfolio geometric-growth rate g = (1/years) * sum(log(1+0.05*net_r))
by hour-of-entry, per instrument, INSTEAD of the earlier win-rate/chi-square-driven
session filter. g is additive across trades, so the log-growth-optimal inclusion rule
is simple and rigorous: include every (instrument, hour) bucket whose own summed
log-growth contribution is positive; drop only buckets that net-destroy growth. This
is a fundamentally different objective from "maximize win rate" -- it explicitly wants
MORE volume at any hour with positive (even thin) edge, not just the best hours.

Base pool: age=2.0h (CANA's shorter gate, for more raw signal), STOP_MULT=2.0 (1:1 R:R
preserved), max_touches=8, require_bounce=True (bounce-gated multi-touch retained).
"""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv", "XAUUSD": "lse_xauusd_full_1min.csv"})
import prod_filter_grid_search as pg

STOP_MULT = 2.0
AGE_H = 2.0
MAX_TOUCHES = 8
RISK_FRAC = 0.05
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2, "XAUUSD": 0.1}
SPREAD_PIPS = {"EURUSD": 1.6, "GBPUSD": 1.6, "EURGBP": 1.6, "USDJPY": 1.6, "XAUUSD": 3.5}


def build_all(inst):
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
        int(AGE_H * 3600 * 1e9), STOP_MULT, STOP_MULT, True, pg.PROBE_DEPTH, 1.0,
        MAX_TOUCHES, pg.MAX_WINDOW_BARS)
    pip = PIP[inst]
    spread_price = SPREAD_PIPS[inst] * pip
    r_gross = np.where(is_win, 1.0, -1.0)
    cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
    net_r = r_gross - cost_in_r
    return pd.DataFrame({"entry_time": pd.to_datetime(times[entry_idx]), "net_r": net_r,
                          "is_win": is_win, "stop_pips": stop_risk / pip, "instrument": inst})


def main():
    instruments = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY", "XAUUSD"]
    for inst in instruments:
        rs.load_instrument(inst)
        rs._prep_zones(inst, 2)

    pools = {}
    for inst in instruments:
        t = build_all(inst)
        t["glog"] = np.log1p(RISK_FRAC * t.net_r)
        t["hour"] = t.entry_time.dt.hour
        pools[inst] = t
        years = (t.entry_time.max()-t.entry_time.min()).days/365.25
        print(f"{inst}: {len(t):,} raw trades ({len(t)/years:,.0f}/yr) before any hour filter, "
              f"WR={100*t.is_win.mean():.1f}% avg_r={t.net_r.mean():.4f}")

    print("\nPer-instrument, per-hour g-contribution (sum of log1p(0.05*net_r)) -- "
          "positive = keep, negative = drop:")
    optimal_hours = {}
    for inst, t in pools.items():
        byh = t.groupby("hour").agg(n=("net_r", "size"), wr=("is_win", "mean"), avg_r=("net_r", "mean"), gsum=("glog", "sum"))
        byh["wr"] = 100*byh["wr"]
        keep = byh[byh.gsum > 0].index.tolist()
        optimal_hours[inst] = keep
        print(f"\n{inst} (keep hours: {sorted(keep)}):")
        print(byh.round(4).to_string())

    # Build optimal portfolio
    frames = []
    for inst, t in pools.items():
        keep_hours = optimal_hours[inst]
        sub = t[t.hour.isin(keep_hours)]
        frames.append(sub)
    full = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)

    years = (full.entry_time.max()-full.entry_time.min()).days/365.25
    tpy = len(full)/years
    wr = 100*full.is_win.mean()
    w = full.loc[full.net_r>0,"net_r"].sum(); l = -full.loc[full.net_r<=0,"net_r"].sum()
    pf = w/l if l>0 else float("inf")
    g = tpy * full.glog.mean()
    y1m = np.log(1_000_000/200)/g
    y10m = np.log(10_000_000/200)/g
    print(f"\n=== OPTIMAL-HOUR PORTFOLIO (age=2.0h, touches<=8) ===")
    print(f"n={len(full):,} ({tpy:.0f}/yr) WR={wr:.2f}% PF={pf:.3f} avg_r={full.net_r.mean():.4f} "
          f"g={g:.3f} yrs->1M={y1m:.2f} yrs->10M={y10m:.2f}")
    print(f"(compare: SOMEGA g=5.301 yrs->1M=1.61 | CANA g=8.304 yrs->1M=1.03 | goal70-orig g=4.424 yrs->1M=1.93)")

    full.to_csv(os.path.join(DATA_DIR, "goal2_hour_opt_trades.csv"), index=False)
    return full, optimal_hours


if __name__ == "__main__":
    main()
