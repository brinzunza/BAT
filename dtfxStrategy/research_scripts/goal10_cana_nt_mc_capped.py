"""25,000-run block-bootstrap Monte Carlo for CANA-NarrowTarget (R:R=0.8), with a
REALISTIC $2M account-size cap on position sizing: once equity exceeds $2,000,000,
risk per trade is 5% of $2,000,000 (flat), not 5% of the (much larger, otherwise
absurd) running equity. This is the fix for the "naive uncapped compounding produces
a meaningless 10^96 number" issue flagged throughout this project -- growth becomes
linear, not exponential, past the cap, which is what actually makes "time to $10M" a
meaningful question instead of a rounding error on the way to $1M.

Also tracks max drawdown (peak-to-trough equity %) per simulated path.
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

START = 200.0
RISK_FRAC = 0.05
CAP = 2_000_000.0
T1 = 1_000_000.0
T10 = 10_000_000.0
BLOCK = 20
N_SIMS = 25_000
MAX_TRADES = 60_000
RUIN_FLOOR = 1.0


@njit(cache=True)
def run_mc_capped(net_r, n_sims, block, max_trades, seed):
    np.random.seed(seed)
    n = len(net_r)
    trades_to_1m = np.full(n_sims, -1, dtype=np.int64)
    trades_to_10m = np.full(n_sims, -1, dtype=np.int64)
    max_dd = np.zeros(n_sims, dtype=np.float64)
    ruined = np.zeros(n_sims, dtype=np.bool_)

    for s in range(n_sims):
        eq = START
        peak = START
        dd_worst = 0.0
        trades = 0
        t1 = -1
        t10 = -1
        is_ruined = False
        done = False
        while trades < max_trades and not done:
            start_idx = np.random.randint(0, n)
            for k in range(block):
                idx = (start_idx + k) % n
                r = net_r[idx]
                risk_base = eq if eq < CAP else CAP
                risk = risk_base * RISK_FRAC
                eq = eq + risk * r
                if eq < RUIN_FLOOR:
                    eq = RUIN_FLOOR
                trades += 1
                if eq > peak:
                    peak = eq
                dd = (eq - peak) / peak
                if dd < dd_worst:
                    dd_worst = dd
                if eq <= RUIN_FLOOR:
                    is_ruined = True
                    done = True
                    break
                if t1 == -1 and eq >= T1:
                    t1 = trades
                if t1 != -1 and t10 == -1 and eq >= T10:
                    t10 = trades
                    done = True
                    break
                if trades >= max_trades:
                    done = True
                    break
        trades_to_1m[s] = t1
        trades_to_10m[s] = t10
        max_dd[s] = dd_worst
        ruined[s] = is_ruined

    return trades_to_1m, trades_to_10m, max_dd, ruined


def main():
    t = pd.read_csv(os.path.join(DATA_DIR, "goal9_cana_narrowtarget_full.csv"), parse_dates=["entry_time"])
    t = t.sort_values("entry_time").reset_index(drop=True)
    net_r = t.net_r.values.astype(np.float64)
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    tpy = len(t) / years
    print(f"CANA-NarrowTarget: {len(t):,} trades, {tpy:.1f}/yr, mean net_r={net_r.mean():.4f}")
    print(f"Running {N_SIMS:,} block-bootstrap sims, $2M position-sizing cap, block={BLOCK}...")

    t1_raw, t10_raw, max_dd, ruined = run_mc_capped(net_r, N_SIMS, BLOCK, MAX_TRADES, seed=42)

    t1_hit = t1_raw[t1_raw > 0]
    t10_hit = t10_raw[t10_raw > 0]
    t1_years = t1_hit / tpy
    t10_years = t10_hit / tpy

    print(f"\n=== Time to $1M (n={len(t1_years):,} of {N_SIMS:,} reached it) ===")
    print(f"  Quickest: {t1_years.min():.3f} yrs   Slowest: {t1_years.max():.3f} yrs")
    print(f"  Average:  {t1_years.mean():.3f} yrs   Median: {np.median(t1_years):.3f} yrs")
    print(f"  Range:    {t1_years.max()-t1_years.min():.3f} yrs")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:2d}: {np.percentile(t1_years,p):.3f} yrs")

    print(f"\n=== Time to $10M, WITH $2M position-sizing cap (n={len(t10_years):,} of {N_SIMS:,} reached it) ===")
    if len(t10_years):
        print(f"  Quickest: {t10_years.min():.3f} yrs   Slowest: {t10_years.max():.3f} yrs")
        print(f"  Average:  {t10_years.mean():.3f} yrs   Median: {np.median(t10_years):.3f} yrs")
        print(f"  Range:    {t10_years.max()-t10_years.min():.3f} yrs")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"    P{p:2d}: {np.percentile(t10_years,p):.3f} yrs")
    pct_reached_10m = 100 * len(t10_years) / N_SIMS
    pct_reached_1m = 100 * len(t1_years) / N_SIMS
    print(f"\n  % of sims reaching $1M:  {pct_reached_1m:.2f}%")
    print(f"  % of sims reaching $10M: {pct_reached_10m:.2f}%")
    print(f"  % ruined (equity hit $1 floor): {100*ruined.mean():.3f}%")

    dd_pct = max_dd * 100
    print(f"\n=== Max drawdown (peak-to-trough %, per simulated path) ===")
    print(f"  Best (smallest) DD: {dd_pct.max():.2f}%   Worst DD: {dd_pct.min():.2f}%")
    print(f"  Average DD: {dd_pct.mean():.2f}%   Median DD: {np.median(dd_pct):.2f}%")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p:2d}: {np.percentile(dd_pct,p):.2f}%")

    out = pd.DataFrame({
        "trades_to_1m": t1_raw, "trades_to_10m": t10_raw,
        "years_to_1m": np.where(t1_raw > 0, t1_raw / tpy, np.nan),
        "years_to_10m": np.where(t10_raw > 0, t10_raw / tpy, np.nan),
        "max_drawdown_pct": dd_pct, "ruined": ruined,
    })
    out.to_csv(os.path.join(DATA_DIR, "goal10_cana_nt_mc_25000_results.csv"), index=False)
    print(f"\nSaved: goal10_cana_nt_mc_25000_results.csv ({len(out):,} rows)")


if __name__ == "__main__":
    main()
