"""Refine the 70%-WR search: (1) a MAX stop-pip ceiling instead of a floor, (2)
restricting to the historically strongest instruments (EUR/GBP, GBP/USD) instead of
blending in EUR/USD and USD/JPY, both at 1:1 R:R, session-filtered, bounce-gated."""
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

ALL_INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
STRONG_INSTRUMENTS = ["EURGBP", "GBPUSD"]

for inst in ALL_INSTRUMENTS:
    rs.load_instrument(inst)
    rs._prep_zones(inst, 2)

configs = [
    (2.0, 3.366682, True, True),
    (2.5, 3.366682, True, True),
    (2.0, 2.0, True, True),
    (2.5, 2.0, True, True),
    (3.343496, 3.366682, True, True),
]

for instset_name, instset in [("ALL 4", ALL_INSTRUMENTS), ("EUR/GBP + GBP/USD only", STRONG_INSTRUMENTS)]:
    print("=" * 90)
    print(instset_name)
    print("=" * 90)
    for stop_mult, age, bounce, session in configs:
        pools = []
        for inst in instset:
            old = pg.INSTRUMENTS
            pg.INSTRUMENTS = [inst]
            pool = g70.build_pool_with_stop_pips(2, stop_mult, age, bounce, session)
            pg.INSTRUMENTS = old
            if len(pool):
                pools.append(pool)
        if not pools:
            continue
        full = pd.concat(pools, ignore_index=True)
        years = (full.entry_time.max() - full.entry_time.min()).days / 365.25
        for max_stop in [None, 15, 10, 7, 5]:
            sub = full if max_stop is None else full[full.stop_pips <= max_stop]
            if len(sub) < 30:
                continue
            w = sub.loc[sub.net_r>0,"net_r"].sum(); l = -sub.loc[sub.net_r<=0,"net_r"].sum()
            pf = w/l if l>0 else float("inf")
            tag = "no ceiling" if max_stop is None else f"<= {max_stop}pips"
            print(f"  stop={stop_mult:.2f} age={age:.2f} {tag:12s}: trades={len(sub):6,d} "
                  f"({len(sub)/years:6,.0f}/yr)  WR={100*sub.is_win.mean():.2f}%  PF={pf:.3f}")
        print()
