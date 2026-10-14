"""Runs the SAME 25,000-path, $200-start, $2M-cap-with-vault-sweep, 5%-risk Monte
Carlo as goal_70pct_time_to_target_mc.py, but on the real CANA trade pool built by
cana_time_to_target_mc.py (actual live parameters, faithful multi-touch simulation) --
direct comparison, same methodology, different (real) input.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from goal_70pct_time_to_target_mc import simulate, N_PATHS, BLOCK_LEN, START_CAPITAL, CAP, \
    TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED

CANA_TRADES_PER_YEAR = 25090.0  # this config's actual observed combined frequency

pool = pd.read_csv(os.path.join(DATA_DIR, "cana_actual_trades.csv"))["net_r"].values
pool = np.ascontiguousarray(pool, dtype=np.float64)
print(f"CANA real trade pool: {len(pool):,} trades, WR={100*(pool>0).mean():.2f}%, avg_r={pool.mean():.4f}\n")
print(f"Running {N_PATHS:,} paths, risk={RISK_PCT*100:.0f}%/trade, cap=${CAP:,.0f}, "
      f"start=${START_CAPITAL:.0f}, freq={CANA_TRADES_PER_YEAR:,.0f}/yr ...", flush=True)

reach1, reach2, ruined = simulate(pool, len(pool), BLOCK_LEN, N_PATHS, START_CAPITAL, CAP,
                                   TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED)


def report(name, reach, target):
    hit = reach[reach > 0]
    n_hit = len(hit)
    n_ruined = int(((reach == -1) & ruined).sum())
    n_timeout = N_PATHS - n_hit - n_ruined
    months = hit / CANA_TRADES_PER_YEAR * 12.0
    print(f"\n=== Time to ${target:,.0f} net worth (CANA, actual parameters) ===")
    print(f"  Reached within {MAX_TRADES:,} trades: {n_hit:,}/{N_PATHS:,} ({100*n_hit/N_PATHS:.1f}%)")
    print(f"  Ruined (net worth < ${RUIN_FLOOR:.0f}) before reaching it: {n_ruined:,} ({100*n_ruined/N_PATHS:.2f}%)")
    print(f"  Neither reached nor ruined within {MAX_TRADES:,}-trade horizon: {n_timeout:,} ({100*n_timeout/N_PATHS:.2f}%)")
    if n_hit == 0:
        return
    print(f"  Quickest: {months.min():7.2f} months   Average: {months.mean():7.2f} months   "
          f"Median: {np.median(months):7.2f} months   Slowest: {months.max():7.2f} months")


report("$1,000,000", reach1, TARGET1)
report("$10,000,000", reach2, TARGET2)

out = pd.DataFrame({"reach_1m_trades": reach1, "reach_10m_trades": reach2, "ruined": ruined})
out.to_csv(os.path.join(DATA_DIR, "cana_time_to_target_mc.csv"), index=False)
print(f"\nSaved: {os.path.join(DATA_DIR, 'cana_time_to_target_mc.csv')}")
