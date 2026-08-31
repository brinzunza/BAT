"""25,000-path Monte Carlo: time to $1M and $10M net worth, starting from $200, with a
HARD $2,000,000 trading-account cap -- growth above the cap is swept to an external,
non-compounding vault (the account resets to exactly $2M and keeps trading/risking at
that base; vault + account = net worth). This is the mechanism CANA.md gestured at
("$2M cap and external vault sweeps") without ever actually simulating it -- here it's
modeled explicitly, trade by trade, per path, not asserted.

Strategy: the validated narrow-target R:R candidate (SOMEGA session filter, stop
unchanged at N=3.343496, target=0.35x stop; see NARROW_TARGET_WINRATE_RESEARCH.md).
Cost model: 1.6-pip OANDA-standard spread, charged ONLY on stop-outs (matches the live
hybrid-stop design). Risk: 5% of the trading-account base per trade (the actual live
RISK_PCT) -- this is an explicit assumption since the request didn't restate it; the
2%/1% comparison from the prior Monte Carlo still applies for how much safer this gets
at lower risk.

Resampling: 20-trade block bootstrap from the actual 2009-2026 historical trade-return
pool (13,221 trades, 83.3% win rate) -- preserves real streakiness. A path that hasn't
reached $10M within 50,000 simulated trades (~64 years at this config's ~782 trades/yr)
is marked "not reached" rather than extrapolated -- no path is silently truncated into
a fake answer.
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
import goal_70pct_winrate_search as g

N_PATHS = 25000
BLOCK_LEN = 20
START_CAPITAL = 200.0
CAP = 2_000_000.0
TARGET1 = 1_000_000.0
TARGET2 = 10_000_000.0
RISK_PCT = 0.05
MAX_TRADES = 50_000
RUIN_FLOOR = 20.0          # net worth below this = treated as effectively ruined
TRADES_PER_YEAR = 782.0    # this config's actual observed frequency (SOMEGA session filter)
SEED = 20260830


@njit(cache=True)
def simulate(pool, n_pool, block_len, n_paths, start_capital, cap, target1, target2,
             risk_pct, max_trades, ruin_floor, seed):
    np.random.seed(seed)
    reach1 = np.full(n_paths, -1, dtype=np.int64)
    reach2 = np.full(n_paths, -1, dtype=np.int64)
    ruined = np.zeros(n_paths, dtype=np.bool_)
    for p in range(n_paths):
        account = start_capital
        vault = 0.0
        t = 0
        done = False
        while t < max_trades and not done:
            start = np.random.randint(0, n_pool - block_len)
            for k in range(block_len):
                r = pool[start + k]
                account = account * (1.0 + risk_pct * r)
                if account > cap:
                    vault += account - cap
                    account = cap
                net_worth = account + vault
                t += 1
                if net_worth < ruin_floor:
                    ruined[p] = True
                    done = True
                    break
                if reach1[p] == -1 and net_worth >= target1:
                    reach1[p] = t
                if net_worth >= target2:
                    reach2[p] = t
                    done = True
                    break
                if t >= max_trades:
                    done = True
                    break
    return reach1, reach2, ruined


def main():
    for inst in g.INSTRUMENTS:
        rs.load_instrument(inst)
        rs._prep_zones(inst, g.MIN_RETRACEMENT)

    stop_mult, target_mult = 3.343496, 3.343496 * 0.35
    all_t = [g.build_trades_asym(inst, stop_mult, target_mult, g.MIN_FILL_AGE_H, True) for inst in g.INSTRUMENTS]
    combined = pd.concat(all_t, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    pool = np.ascontiguousarray(combined["net_r"].values, dtype=np.float64)
    print(f"Trade pool: {len(pool)} trades, win rate={100*(pool>0).mean():.1f}%, "
          f"avg_r={pool.mean():.4f}, assumed frequency={TRADES_PER_YEAR:.0f}/yr\n")
    print(f"Running {N_PATHS:,} block-bootstrap paths, risk={RISK_PCT*100:.0f}%/trade, "
          f"cap=${CAP:,.0f}, start=${START_CAPITAL:.0f} ...", flush=True)

    reach1, reach2, ruined = simulate(pool, len(pool), BLOCK_LEN, N_PATHS, START_CAPITAL,
                                       CAP, TARGET1, TARGET2, RISK_PCT, MAX_TRADES, RUIN_FLOOR, SEED)

    def report(name, reach, target):
        hit = reach[reach > 0]
        n_hit = len(hit)
        n_ruined_before = int(((reach == -1) & ruined).sum())
        n_timeout = N_PATHS - n_hit - n_ruined_before
        months = hit / TRADES_PER_YEAR * 12.0
        print(f"\n=== Time to ${target:,.0f} net worth ===")
        print(f"  Reached within {MAX_TRADES:,} trades: {n_hit:,}/{N_PATHS:,} ({100*n_hit/N_PATHS:.1f}%)")
        print(f"  Ruined (net worth < ${RUIN_FLOOR:.0f}) before reaching it: {n_ruined_before:,} ({100*n_ruined_before/N_PATHS:.2f}%)")
        print(f"  Neither reached nor ruined within {MAX_TRADES:,}-trade horizon: {n_timeout:,} ({100*n_timeout/N_PATHS:.2f}%)")
        if n_hit == 0:
            return
        print(f"  Quickest (min):    {months.min():7.1f} months  ({months.min()/12:.2f} yr)")
        print(f"  Average (mean):    {months.mean():7.1f} months  ({months.mean()/12:.2f} yr)")
        print(f"  Median:            {np.median(months):7.1f} months  ({np.median(months)/12:.2f} yr)")
        print(f"  Slowest (max):     {months.max():7.1f} months  ({months.max()/12:.2f} yr)")
        print(f"  Range (max-min):   {months.max()-months.min():7.1f} months")
        for q in (5, 25, 75, 95):
            print(f"  p{q:<3d}:              {np.percentile(months, q):7.1f} months")

    report("$1,000,000", reach1, TARGET1)
    report("$10,000,000", reach2, TARGET2)

    out = pd.DataFrame({"reach_1m_trades": reach1, "reach_10m_trades": reach2, "ruined": ruined})
    out_path = os.path.join(DATA_DIR, "goal_70pct_time_to_target_mc.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved per-path raw results: {out_path}")


if __name__ == "__main__":
    main()
