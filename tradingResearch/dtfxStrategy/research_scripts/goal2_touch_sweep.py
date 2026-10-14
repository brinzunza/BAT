"""Sweep MAX_ZONE_TOUCHES on the goal70_final_winner portfolio to see if letting
already-good zones re-arm more times (still bounce-gated, still retiring on any loss)
adds trades without hurting quality. Later touches showed HIGHER win rate in CANA's
by-touch breakdown (touch1 58.8% -> touch3 68.75%), so this is the first, lowest-risk
lever to test before anything structural.

Screening metric: g = trades_per_year * E[log(1 + 0.05*net_r)], the annualized
continuously-compounded growth rate at 5% fixed-fractional risk -- this is what
actually determines time-to-$1M/$10M under compounding, not WR/PF alone.
SOMEGA g=5.301 (yrs->1M=1.61), CANA g=8.304 (yrs->1M=1.03), goal70 baseline g=4.424 (yrs->1M=1.93).
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

CORE_AGE_H = 2.5
STOP_MULT = 2.0
RISK_FRAC = 0.05


def build_with_stoppips(inst, spread_pips, pip, age_h, max_touches):
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
        max_touches, pg.MAX_WINDOW_BARS)
    spread_price = spread_pips * pip
    r_gross = np.where(is_win, 1.0, -1.0)
    cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
    net_r = r_gross - cost_in_r
    return pd.DataFrame({"entry_time": pd.to_datetime(times[entry_idx]), "net_r": net_r,
                          "is_win": is_win, "touch_num": touch_num, "stop_pips": stop_risk / pip})


def session_mask(t, inst):
    hrs = {"EURGBP": set(range(18, 24)), "GBPUSD": {20, 21, 22}, "USDJPY": {19, 20, 21, 22}}[inst]
    return t[t.entry_time.dt.hour.isin(hrs)]


def build_full_portfolio(max_touches):
    core_parts = [session_mask(build_with_stoppips(i, 1.6, 1e-4, CORE_AGE_H, max_touches), i) for i in ["EURGBP", "GBPUSD"]]
    core_parts.append(session_mask(build_with_stoppips("USDJPY", 1.6, 1e-2, CORE_AGE_H, max_touches), "USDJPY"))
    core = pd.concat(core_parts, ignore_index=True)

    eurusd = build_with_stoppips("EURUSD", 1.6, 1e-4, 3.366682, max_touches)
    eurusd = eurusd[eurusd.entry_time.dt.hour.isin([19, 20, 21, 22])]

    gold = build_with_stoppips("XAUUSD", 3.5, 0.1, 3.366682, max_touches)
    gold = gold[gold.entry_time.dt.hour.isin(range(12, 18))]
    gold = gold[gold.stop_pips <= 10]

    full = pd.concat([core, eurusd, gold], ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return full


def score(full, label):
    years = (full.entry_time.max() - full.entry_time.min()).days / 365.25
    w = full.loc[full.net_r > 0, "net_r"].sum(); l = -full.loc[full.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    wr = 100 * full.is_win.mean()
    tpy = len(full) / years
    g_log = np.log1p(RISK_FRAC * full.net_r.values).mean()
    g = tpy * g_log
    y1m = np.log(1_000_000/200) / g
    y10m = np.log(10_000_000/200) / g
    print(f"{label:14s} n={len(full):6,} ({tpy:6.0f}/yr)  WR={wr:5.2f}%  PF={pf:.3f}  avg_r={full.net_r.mean():.4f}  "
          f"g={g:6.3f}  yrs->1M={y1m:.2f}  yrs->10M={y10m:.2f}")
    return dict(max_touches=label, trades=len(full), tpy=tpy, wr=wr, pf=pf, avg_r=full.net_r.mean(), g=g, y1m=y1m, y10m=y10m)


def main():
    for inst in ["EURUSD", "GBPUSD", "EURGBP", "USDJPY", "XAUUSD"]:
        rs.load_instrument(inst)
        rs._prep_zones(inst, 2)

    rows = []
    for mt in [4, 6, 8, 12, 20, 40]:
        full = build_full_portfolio(mt)
        rows.append(score(full, f"max_touches={mt}"))

    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "goal2_touch_sweep.csv"), index=False)


if __name__ == "__main__":
    main()
