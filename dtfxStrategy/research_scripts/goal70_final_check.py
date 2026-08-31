"""OOS check on the two Pareto-frontier endpoints found: high-WR/low-freq vs
frequency-clearing/moderate-WR, both at 1:1 R:R."""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
import prod_filter_grid_search as pg
import goal70_1to1_search as g70

for inst in ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]:
    rs.load_instrument(inst)
    rs._prep_zones(inst, 2)


def build(instset, stop_mult, age, session, max_stop):
    pools = []
    for inst in instset:
        old = pg.INSTRUMENTS
        pg.INSTRUMENTS = [inst]
        pool = g70.build_pool_with_stop_pips(2, stop_mult, age, True, session)
        pg.INSTRUMENTS = old
        if len(pool):
            pools.append(pool)
    full = pd.concat(pools, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    if max_stop is not None:
        full = full[full.stop_pips <= max_stop]
    return full


CANDIDATES = [
    ("E: high-WR (EURGBP+GBPUSD+USDJPY, session, <=7pips)", ["EURGBP","GBPUSD","USDJPY"], 2.0, 3.366682, True, 7),
    ("F: frequency-clearing (EURGBP+GBPUSD+USDJPY, 24h, no ceiling)", ["EURGBP","GBPUSD","USDJPY"], 2.5, 3.366682, False, None),
]

for name, instset, stop_mult, age, session, max_stop in CANDIDATES:
    print("=" * 90); print(name); print("=" * 90)
    full = build(instset, stop_mult, age, session, max_stop)
    years = (full.entry_time.max()-full.entry_time.min()).days/365.25
    w = full.loc[full.net_r>0,"net_r"].sum(); l = -full.loc[full.net_r<=0,"net_r"].sum()
    print(f"Full: {len(full):,} trades ({len(full)/years:,.0f}/yr) WR={100*full.is_win.mean():.2f}% PF={w/l:.3f}")
    split = pd.Timestamp("2020-01-01")
    for label, sub in [("TRAIN 2009-2020", full[full.entry_time<split]), ("TEST 2020-2026", full[full.entry_time>=split])]:
        yrs = (sub.entry_time.max()-sub.entry_time.min()).days/365.25
        w = sub.loc[sub.net_r>0,"net_r"].sum(); l = -sub.loc[sub.net_r<=0,"net_r"].sum()
        print(f"  {label:18s} trades/yr={len(sub)/yrs:6,.0f} WR={100*sub.is_win.mean():.2f}% PF={(w/l if l>0 else float('inf')):.3f}")
    print()
