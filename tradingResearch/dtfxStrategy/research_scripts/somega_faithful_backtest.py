"""Faithful SOMEGA backtest using the exact same engine/cost model as
goal70_final_winner.py and cana_faithful_backtest.py, for direct comparison.

SOMEGA = live Omega zone/stop/target logic (N_MULTIPLIER=3.343496, MIN_FILL_AGE=3.366682h,
MIN_RISK_PCT=0.000170, symmetric 1:1 R:R) + per-instrument session-hour filter
(EURGBP/GBPUSD/USDJPY filtered, EURUSD unfiltered) + unconditional re-arm on win up to
MAX_ZONE_TOUCHES=4 (require_bounce=False) -- this is prod_filter_grid_search.py's own
documented "PROD's current unconditional re-arm behavior", i.e. exactly what
PRODLIVETRADING/strategy/engine.py does (mark_position_closed: re-arm on target hit,
retire on any loss, no bounce/probe condition).
"""
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
import prod_filter_grid_search as pg

INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
N_MULTIPLIER = 3.343496
MIN_FILL_AGE_H = 3.366682


def main():
    frames = []
    for inst in INSTRUMENTS:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        t = pg.build_trades(inst, N_MULTIPLIER, N_MULTIPLIER, MIN_FILL_AGE_H,
                             require_bounce=False, session_filter=True)
        years = (t["entry_time"].max() - t["entry_time"].min()).days / 365.25
        wins = t.loc[t.net_r > 0, "net_r"].sum()
        losses = -t.loc[t.net_r <= 0, "net_r"].sum()
        pf = wins / losses if losses > 0 else float("inf")
        print(f"  {inst}: {len(t):,} trades ({len(t)/years:,.0f}/yr), "
              f"WR={100*t['is_win'].mean():.2f}%, PF={pf:.3f}")
        frames.append(t)

    combined = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    years = (combined["entry_time"].max() - combined["entry_time"].min()).days / 365.25
    wins = combined.loc[combined.net_r > 0, "net_r"].sum()
    losses = -combined.loc[combined.net_r <= 0, "net_r"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    print(f"\nCOMBINED (full history): {len(combined):,} trades ({len(combined)/years:,.0f}/yr), "
          f"WR={100*combined['is_win'].mean():.2f}%, PF={pf:.3f}, avg_r={combined['net_r'].mean():.4f}")

    split = pd.Timestamp("2020-01-01")
    for label, sub in [("TRAIN 2009-2020", combined[combined.entry_time < split]),
                        ("TEST 2020-2026 (OOS)", combined[combined.entry_time >= split])]:
        yrs = (sub.entry_time.max() - sub.entry_time.min()).days / 365.25
        w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
        print(f"  {label:22s} trades/yr={len(sub)/yrs:6,.0f}  WR={100*sub.is_win.mean():.2f}%  "
              f"PF={(w/l if l>0 else float('inf')):.3f}")

    out_path = os.path.join(DATA_DIR, "somega_faithful_trades.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return combined


if __name__ == "__main__":
    main()
