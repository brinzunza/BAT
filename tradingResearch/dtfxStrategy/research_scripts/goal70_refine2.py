"""Push for >=365 trades/yr while keeping WR as high as possible: EUR/GBP+GBP/USD
without the session filter (24h), plus a 3-instrument blend (add USDJPY) with stop
ceilings, searching for a combination that clears BOTH constraints simultaneously."""
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


def run(instset_name, instset, stop_mult, age, session, ceilings):
    pools = []
    for inst in instset:
        old = pg.INSTRUMENTS
        pg.INSTRUMENTS = [inst]
        pool = g70.build_pool_with_stop_pips(2, stop_mult, age, True, session)
        pg.INSTRUMENTS = old
        if len(pool):
            pools.append(pool)
    if not pools:
        print(f"  {instset_name}: no trades")
        return
    full = pd.concat(pools, ignore_index=True)
    years = (full.entry_time.max() - full.entry_time.min()).days / 365.25
    for max_stop in ceilings:
        sub = full if max_stop is None else full[full.stop_pips <= max_stop]
        if len(sub) < 30:
            continue
        w = sub.loc[sub.net_r>0,"net_r"].sum(); l = -sub.loc[sub.net_r<=0,"net_r"].sum()
        pf = w/l if l>0 else float("inf")
        tag = "no ceiling" if max_stop is None else f"<= {max_stop}pips"
        flag = " <-- clears 365/yr" if len(sub)/years >= 365 else ""
        print(f"  {instset_name:28s} stop={stop_mult:.2f} age={age:.2f} session={session} {tag:12s}: "
              f"trades={len(sub):6,d} ({len(sub)/years:6,.0f}/yr)  WR={100*sub.is_win.mean():.2f}%  PF={pf:.3f}{flag}")


print("=== EUR/GBP + GBP/USD, 24h (session=False) ===")
for stop_mult, age in [(2.0, 3.366682), (2.5, 3.366682), (2.0, 2.0)]:
    run("EURGBP+GBPUSD 24h", ["EURGBP", "GBPUSD"], stop_mult, age, False, [None, 15, 10, 7, 5])

print("\n=== EUR/GBP + GBP/USD + USDJPY, session=True ===")
for stop_mult, age in [(2.0, 3.366682), (2.5, 3.366682)]:
    run("EURGBP+GBPUSD+USDJPY session", ["EURGBP", "GBPUSD", "USDJPY"], stop_mult, age, True, [None, 15, 10, 7])

print("\n=== EUR/GBP + GBP/USD + USDJPY, 24h ===")
for stop_mult, age in [(2.0, 3.366682), (2.5, 3.366682)]:
    run("EURGBP+GBPUSD+USDJPY 24h", ["EURGBP", "GBPUSD", "USDJPY"], stop_mult, age, False, [None, 15, 10, 7])
