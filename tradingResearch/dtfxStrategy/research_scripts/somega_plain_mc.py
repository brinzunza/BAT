"""SOMEGA (plain, symmetric 1:1 R:R) run through the exact same Monte Carlo
methodology as CANA and the narrow-target candidate, for a fair 3-way comparison."""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
import goal_70pct_winrate_search as g
from goal_70pct_time_to_target_mc import simulate, N_PATHS, BLOCK_LEN, START_CAPITAL, CAP, \
    TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED
from goal_70pct_monte_carlo import block_bootstrap_paths, summarize, RISK_LEVELS

for inst in g.INSTRUMENTS:
    rs.load_instrument(inst)
    rs._prep_zones(inst, g.MIN_RETRACEMENT)

all_t = [g.build_trades_asym(inst, 3.343496, 3.343496, g.MIN_FILL_AGE_H, True) for inst in g.INSTRUMENTS]
combined = pd.concat(all_t, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
pool = np.ascontiguousarray(combined["net_r"].values, dtype=np.float64)
years = (combined.entry_time.max() - combined.entry_time.min()).days / 365.25
SOMEGA_TRADES_PER_YEAR = len(pool) / years

print(f"SOMEGA pool: {len(pool):,} trades ({SOMEGA_TRADES_PER_YEAR:,.0f}/yr), "
      f"WR={100*(pool>0).mean():.2f}%, avg_r={pool.mean():.4f}\n")

print("--- Time to $1M/$10M: 25,000 paths, $200 start, 5% risk, $2M cap+vault ---")
reach1, reach2, ruined = simulate(pool, len(pool), BLOCK_LEN, N_PATHS, START_CAPITAL, CAP,
                                   TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED)
for name, reach, target in [("$1,000,000", reach1, TARGET1), ("$10,000,000", reach2, TARGET2)]:
    hit = reach[reach > 0]
    n_hit = len(hit)
    n_ruined = int(((reach == -1) & ruined).sum())
    months = hit / SOMEGA_TRADES_PER_YEAR * 12.0
    print(f"  {name}: reached={100*n_hit/N_PATHS:.1f}%  ruined={100*n_ruined/N_PATHS:.2f}%  "
          f"median={np.median(months) if n_hit else float('nan'):.1f}mo")

print("\n--- 3yr drawdown probability, UNCAPPED, block-bootstrap ---")
rng = np.random.default_rng(20260830)
horizon = int(SOMEGA_TRADES_PER_YEAR * 3)
for risk in RISK_LEVELS:
    paths_r = block_bootstrap_paths(pool, horizon, N_PATHS, BLOCK_LEN, rng)
    s = summarize(paths_r, risk)
    print(f"  risk={risk*100:4.1f}%  medianDD={s['median_maxdd']:.1f}%  "
          f"P(DD>50%)={s['prob_dd_gt_50']:.1f}%  P(DD>80%)={s['prob_dd_gt_80']:.1f}%")
