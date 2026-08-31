"""Same 25,000-path, $200-start, $2M-cap-with-vault-sweep, 5%-risk Monte Carlo, run on
the CORRECTED CANA trade pool from cana_faithful_backtest.py (both bugs fixed)."""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from goal_70pct_time_to_target_mc import simulate, N_PATHS, BLOCK_LEN, START_CAPITAL, CAP, \
    TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED

CANA_TRADES_PER_YEAR = 2848.0

pool = pd.read_csv(os.path.join(DATA_DIR, "cana_faithful_trades.csv"))["net_r"].values
pool = np.ascontiguousarray(pool, dtype=np.float64)
print(f"CANA corrected trade pool: {len(pool):,} trades, WR={100*(pool>0).mean():.2f}%, "
      f"avg_r={pool.mean():.4f}, freq={CANA_TRADES_PER_YEAR:.0f}/yr\n")
print(f"Running {N_PATHS:,} paths, risk={RISK_PCT*100:.0f}%/trade, cap=${CAP:,.0f}, "
      f"start=${START_CAPITAL:.0f} ...", flush=True)

reach1, reach2, ruined = simulate(pool, len(pool), BLOCK_LEN, N_PATHS, START_CAPITAL, CAP,
                                   TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED)


def report(name, reach, target):
    hit = reach[reach > 0]
    n_hit = len(hit)
    n_ruined = int(((reach == -1) & ruined).sum())
    n_timeout = N_PATHS - n_hit - n_ruined
    months = hit / CANA_TRADES_PER_YEAR * 12.0
    print(f"\n=== Time to ${target:,.0f} net worth (CANA, corrected) ===")
    print(f"  Reached: {n_hit:,}/{N_PATHS:,} ({100*n_hit/N_PATHS:.1f}%)   "
          f"Ruined: {n_ruined:,} ({100*n_ruined/N_PATHS:.2f}%)   "
          f"Timed out: {n_timeout:,} ({100*n_timeout/N_PATHS:.2f}%)")
    if n_hit == 0:
        return
    print(f"  Quickest: {months.min():7.2f} mo   Average: {months.mean():7.2f} mo   "
          f"Median: {np.median(months):7.2f} mo   Slowest: {months.max():7.2f} mo")


report("$1,000,000", reach1, TARGET1)
report("$10,000,000", reach2, TARGET2)
