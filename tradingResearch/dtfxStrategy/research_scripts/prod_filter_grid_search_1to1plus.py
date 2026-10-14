"""Same faithful multi-touch grid search as prod_filter_grid_search.py, constrained to
R:R >= 1:1 (target distance >= stop distance) -- i.e. excludes the narrow-target trick,
testing only symmetric-or-better reward configs across stop size / age / bounce / session."""
import os, sys, itertools
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
import prod_filter_grid_search as pg

for inst in pg.INSTRUMENTS:
    rs.load_instrument(inst)
    rs._prep_zones(inst, pg.MIN_RETRACEMENT)

grid = list(itertools.product(
    [2.0, 3.343496],           # stop_mult
    [1.0, 1.5, 2.0],           # rr >= 1.0 (constraint)
    [2.0, 3.366682],           # age
    [False, True],              # require_bounce
    [False, True],              # session filter
))
print(f"Running {len(grid)} configs x {len(pg.INSTRUMENTS)} instruments (R:R >= 1:1 only)...\n", flush=True)
rows = []
for stop_mult, rr, age, bounce, session in grid:
    target_mult = stop_mult * rr
    frames = [pg.build_trades(inst, stop_mult, target_mult, age, bounce, session) for inst in pg.INSTRUMENTS]
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) < 30:
        continue
    years = (combined.entry_time.max() - combined.entry_time.min()).days / 365.25
    s = pg.score(combined, years)
    if s is None:
        continue
    s.update(stop_mult=stop_mult, rr=rr, age_h=age, bounce=bounce, session=session)
    rows.append(s)

res = pd.DataFrame(rows)
cols = ["stop_mult", "rr", "age_h", "bounce", "session", "trades", "trades_yr", "win_rate", "pf", "avg_r"]
res = res[cols].sort_values("pf", ascending=False)
out_path = os.path.join(DATA_DIR, "prod_filter_grid_search_1to1plus.csv")
res.to_csv(out_path, index=False)
print("Top 20 by PF:")
print(res.head(20).to_string(index=False))
print("\nTop 10 by trades_yr (among PF > 1.15):")
print(res[res.pf > 1.15].sort_values("trades_yr", ascending=False).head(10).to_string(index=False))
print(f"\nSaved: {out_path}")
