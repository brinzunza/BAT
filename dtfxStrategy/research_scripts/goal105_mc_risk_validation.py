"""Follow-up to goal104: the deterministic single-historical-path sweep found a sharp
cliff into near-total ruin at certain risk_pct values (e.g. 10%+ for 5/10-pip zone
heights) sitting right next to risk levels that looked spectacular (5-7%, CAGR ~100%+).
That kind of discontinuity is a warning sign, not good news -- it means the "growth-optimal"
risk found there is fragile to the exact historical ordering of wins and losses, the
classic failure mode of picking a risk fraction by maximizing return on ONE historical
path (full-Kelly sizing off a single sequence). This validates those risk levels properly:
many block-bootstrap-resampled paths (goal79's `run_monte_carlo_two_account`), tracking
ruin probability directly, instead of trusting the one sequence that happened to occur.

Note: `run_monte_carlo_two_account` reads risk_pct from goal79's module-level `RISK_PCT`
global (not a function parameter) -- monkey-patched here per sweep point since Python
resolves module globals at call time, not at import time.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal79_two_account_portfolio as g79
import goal104_optimal_risk_two_account as g104

ZONE_HEIGHTS_TO_VALIDATE = [5, 10]
RISK_GRID_FINE = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
N_SIMS = 1000
MAX_TRADES = 50_000
TARGET = 1_000_000.0
SEED = 42


def main():
    t0 = time.time()
    print("Loading + building zones once...")
    cache = g104.load_and_build_zones()

    rows = []
    for min_pips in ZONE_HEIGHTS_TO_VALIDATE:
        trades = g104.build_candidate_trades(cache, min_pips)
        print(f"\nmin_zone={min_pips}pips: {len(trades):,} candidate trades")

        for risk_pct in RISK_GRID_FINE:
            t1 = time.time()
            g79.RISK_PCT = risk_pct  # module global the MC function actually reads
            days, trades_ct, max_dds, ruin_rate = g79.run_monte_carlo_two_account(
                trades, N_SIMS, TARGET, max_trades=MAX_TRADES, seed=SEED)
            pct_reached = 100.0 * len(days) / N_SIMS
            med_years = float(np.median(days) / 365.25) if len(days) else float("nan")
            med_dd = float(np.median(max_dds))
            p95_dd = float(np.percentile(max_dds, 5))  # worst 5% (most negative)
            rows.append(dict(min_pips=min_pips, risk_pct=risk_pct, pct_reached_1m=pct_reached,
                              median_years_to_1m=med_years, median_max_dd=med_dd,
                              p95_worst_dd=p95_dd, ruin_rate=100.0 * ruin_rate))
            print(f"  risk={risk_pct:>5.0%}  reached $1M={pct_reached:>6.1f}%  "
                  f"median yrs={med_years:>6.2f}  median maxDD={med_dd:>7.2f}%  "
                  f"P95 worst DD={p95_dd:>7.2f}%  ruin={100*ruin_rate:>6.2f}%  ({time.time()-t1:.1f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA_DIR, "goal105_mc_risk_validation.csv"), index=False)
    print(f"\nWrote goal105_mc_risk_validation.csv")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
