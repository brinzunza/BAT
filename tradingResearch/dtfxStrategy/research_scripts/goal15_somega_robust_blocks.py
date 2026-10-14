"""Re-verifies SOMEGA Robust's full-history ("deploy-today") session-block selection
against the fixed entry_time -- the one row left unverified after
goal14_rederive_five_fixed_entrytime.py, since no standalone script previously
existed for this specific derivation (goal8_unified_holdout.py only computes the
TRUE-holdout, first-half-selected version for the ranking table; this mirrors
goal4_no_gold.py's build_dtfx_robust_no_gold(), same standard >=3-of-4 vote over all
4 quarters, but on SOMEGA's own base params instead of the DTFX-skeleton).
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal2_hour_opt as h
import goal2_robust_hours as rh
import prod_filter_grid_search as pg
import goal5_rr_sweep as g5

TRADEABLE = g5.TRADEABLE
N_MULTIPLIER = 3.343496
MIN_FILL_AGE_H = 3.366682

BLOCK_LABELS = {0: "00-05", 1: "06-11", 2: "12-17", 3: "18-23"}


def build_somega_robust_full_history():
    for inst in TRADEABLE:
        h.rs.load_instrument(inst)
        h.rs._prep_zones(inst, 2)

    pools = {}
    tmin = tmax = None
    for inst in TRADEABLE:
        t = pg.build_trades(inst, N_MULTIPLIER, N_MULTIPLIER, MIN_FILL_AGE_H,
                             require_bounce=False, session_filter=False)
        t["glog"] = np.log1p(h.RISK_FRAC * t.net_r)
        t["hour"] = t.entry_time.dt.hour // 6
        pools[inst] = t
        tmin = t.entry_time.min() if tmin is None else min(tmin, t.entry_time.min())
        tmax = t.entry_time.max() if tmax is None else max(tmax, t.entry_time.max())

    quarters4 = rh.make_quarters(tmin, tmax, k=4)
    final_blocks, _ = rh.select_hours(pools, quarters4)
    for inst in TRADEABLE:
        print(f"  {inst}: blocks kept: {[BLOCK_LABELS[b] for b in sorted(final_blocks[inst])]}")

    frames = [t[t.hour.isin(final_blocks[inst])] for inst, t in pools.items()]
    full = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    r = g5.score_row(full, "SOMEGA Robust (full-history deploy, post entry_time-fix)")
    g5.print_row(r)

    full.to_csv(os.path.join(DATA_DIR, "goal15_somega_robust_fullhistory_trades.csv"), index=False)
    return full, final_blocks


if __name__ == "__main__":
    build_somega_robust_full_history()
