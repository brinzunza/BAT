"""Same 3-year drawdown-probability block-bootstrap Monte Carlo as goal_70pct_monte_carlo.py,
run on the corrected CANA trade pool, for direct comparison."""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from goal_70pct_monte_carlo import block_bootstrap_paths, summarize, BLOCK_LEN, N_PATHS, RISK_LEVELS

CANA_TRADES_PER_YEAR = 2848
HORIZON_3YR = CANA_TRADES_PER_YEAR * 3

pool = pd.read_csv(os.path.join(DATA_DIR, "cana_faithful_trades.csv"))["net_r"].values
pool = np.ascontiguousarray(pool, dtype=np.float64)
rng = np.random.default_rng(20260830)

print(f"CANA pool: {len(pool):,} trades. 3yr horizon = {HORIZON_3YR:,} trades ({N_PATHS:,} paths)\n")
for risk in RISK_LEVELS:
    paths_r = block_bootstrap_paths(pool, HORIZON_3YR, N_PATHS, BLOCK_LEN, rng)
    s = summarize(paths_r, risk)
    print(f"[3yr, risk={risk*100:4.1f}%] median={s['median_multiple']:.2f}x "
          f"(p5={s['p5_multiple']:.2f}x, p95={s['p95_multiple']:.2f}x)  "
          f"medianDD={s['median_maxdd']:.1f}%  P(DD>50%)={s['prob_dd_gt_50']:.1f}%  "
          f"P(DD>80%)={s['prob_dd_gt_80']:.1f}%  P(end<start)={s['prob_below_start']:.1f}%")
