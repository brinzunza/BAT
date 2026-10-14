"""Cost-sensitivity stress test: how much worse can execution be before the honestly-
validated edge (10-pip min zone, PRE-2021 data only) disappears? Real spreads vary with
liquidity/session/broker; this checks whether the strategy needs the exact 1.6-pip
assumption to survive, or has real margin against worse execution.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal95_every_swing_zones as g95
import goal99_dtfx_base_backtest as g99
import goal104_optimal_risk_two_account as g104

PRE_END = pd.Timestamp("2021-01-01")
MIN_ZONE_PIPS = 10
SPREAD_SWEEP_PIPS = [1.6, 2.0, 2.5, 3.0, 4.0, 5.0]
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440


def main():
    t0 = time.time()
    print("Loading + building zones once...")
    cache = g104.load_and_build_zones()

    print(f"\nSpread stress test, min_zone={MIN_ZONE_PIPS}pips, PRE-2021 data only:")
    rows = []
    for spread_pips in SPREAD_SWEEP_PIPS:
        pooled_r = []
        for inst, d in cache.items():
            pip = d["pip"]
            spread_price = spread_pips * pip
            min_stop_price = MIN_STOP_PIPS * pip
            min_zone_height = MIN_ZONE_PIPS * pip

            entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
                d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
                d["last_high_of_bar"], d["last_low_of_bar"], spread_price, min_stop_price,
                MAX_WINDOW, False, min_zone_height)

            resolved = outcome > 0
            ei = entry_idx[resolved]
            entry_times = d["times"].values[ei]
            pre_mask = entry_times < np.datetime64(PRE_END)
            pooled_r.append(net_r[resolved][pre_mask])

        r = np.concatenate(pooled_r)
        n = len(r)
        wr = 100.0 * (r > 0).mean()
        gl = -r[r <= 0].sum()
        pf = r[r > 0].sum() / gl if gl > 0 else float("inf")
        avg_r = r.mean()
        rows.append(dict(spread_pips=spread_pips, n=n, wr=wr, pf=pf, avg_r=avg_r))
        print(f"  spread={spread_pips:.1f}pip  n={n:,}  WR={wr:.2f}%  PF={pf:.3f}  avgR={avg_r:+.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "goal109_spread_stress.csv"), index=False)
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
