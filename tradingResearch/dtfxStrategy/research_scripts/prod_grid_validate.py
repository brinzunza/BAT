"""OOS + Monte Carlo validation for the top candidates from prod_filter_grid_search.py."""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
import prod_filter_grid_search as pg
from goal_70pct_time_to_target_mc import simulate, N_PATHS, BLOCK_LEN, START_CAPITAL, CAP, \
    TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED
from goal_70pct_monte_carlo import block_bootstrap_paths, summarize, RISK_LEVELS

for inst in pg.INSTRUMENTS:
    rs.load_instrument(inst)
    rs._prep_zones(inst, pg.MIN_RETRACEMENT)

CANDIDATES = [
    ("A: session-filtered, bounce-gated (1179/yr)",  3.343496, 0.35, 2.0, True,  True),
    ("B: 24h, no bounce, high-freq (4405/yr)",        3.343496, 0.35, 2.0, False, False),
]

for name, stop_mult, rr, age, bounce, session in CANDIDATES:
    print("=" * 90)
    print(name)
    print("=" * 90)
    target_mult = stop_mult * rr
    frames = [pg.build_trades(inst, stop_mult, target_mult, age, bounce, session) for inst in pg.INSTRUMENTS]
    combined = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    pool = np.ascontiguousarray(combined["net_r"].values, dtype=np.float64)
    years = (combined.entry_time.max() - combined.entry_time.min()).days / 365.25
    freq = len(pool) / years

    w = combined.loc[combined.net_r>0,"net_r"].sum(); l = -combined.loc[combined.net_r<=0,"net_r"].sum()
    print(f"Full history: {len(pool):,} trades ({freq:,.0f}/yr), WR={100*(pool>0).mean():.2f}%, PF={w/l:.3f}")

    split = pd.Timestamp("2020-01-01")
    train = combined[combined.entry_time < split]
    test = combined[combined.entry_time >= split]
    for label, sub in [("TRAIN 2009-2020", train), ("TEST 2020-2026 (OOS)", test)]:
        yrs = (sub.entry_time.max()-sub.entry_time.min()).days/365.25
        w = sub.loc[sub.net_r>0,"net_r"].sum(); l = -sub.loc[sub.net_r<=0,"net_r"].sum()
        print(f"  {label:22s} trades/yr={len(sub)/yrs:6,.0f}  WR={100*sub.is_win.mean():.2f}%  PF={w/l:.3f}")

    print("\nTime to $1M/$10M: 25,000 paths, $200 start, 5% risk, $2M cap+vault")
    reach1, reach2, ruined = simulate(pool, len(pool), BLOCK_LEN, N_PATHS, START_CAPITAL, CAP,
                                       TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED)
    for tname, reach, target in [("$1M", reach1, TARGET1), ("$10M", reach2, TARGET2)]:
        hit = reach[reach > 0]
        months = hit / freq * 12.0
        n_ruined = int(((reach == -1) & ruined).sum())
        print(f"  {tname}: reached={100*len(hit)/N_PATHS:.1f}%  ruined={100*n_ruined/N_PATHS:.2f}%  "
              f"median={np.median(months) if len(hit) else float('nan'):.1f}mo")

    print("\n3yr drawdown probability, UNCAPPED, block-bootstrap:")
    rng = np.random.default_rng(20260830)
    horizon = int(freq * 3)
    for risk in RISK_LEVELS:
        paths_r = block_bootstrap_paths(pool, horizon, N_PATHS, BLOCK_LEN, rng)
        s = summarize(paths_r, risk)
        print(f"  risk={risk*100:4.1f}%  medianDD={s['median_maxdd']:.1f}%  "
              f"P(DD>50%)={s['prob_dd_gt_50']:.1f}%  P(DD>80%)={s['prob_dd_gt_80']:.1f}%")
    print()
