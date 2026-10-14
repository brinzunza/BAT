"""Answers: 'is max drawdown the common flaw across the fastest-to-$1M rankings?'
Runs the same $2M-cap block-bootstrap Monte Carlo (goal10's methodology) across
every strategy in STRATEGY_RANKINGS.md for a fair, apples-to-apples drawdown
comparison against each one's own speed (median years to $1M)."""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from goal10_cana_nt_mc_capped import run_mc_capped, BLOCK, MAX_TRADES

N_SIMS = 4000  # smaller than goal10's 25,000 -- this is a comparative sweep across 9 strategies

STRATEGIES = [
    ("CANA-NarrowTarget (rr=0.8)", "goal9_cana_narrowtarget_full.csv"),
    ("SOMEGA-NarrowTarget (rr=0.7)", "goal9_somega_narrowtarget_full.csv"),
    ("DTFX-NarrowTarget (rr=0.7)", "goal9_dtfx_narrowtarget_full.csv"),
    ("CANA+ (light filter)", "goal6_cana_base_thresh2_trades.csv"),
    ("DTFX Robust (no gold)", "goal4_dtfx_nogold_full_trades.csv"),
    ("SOMEGA Robust", "goal3_somega_v2_full_trades.csv"),
    ("CANA (faithful)", "cana_faithful_trades.csv"),
    ("SOMEGA (live)", "somega_faithful_trades.csv"),
    ("goal70_final_winner (no gold)", "goal4_goal70_nogold_trades.csv"),
]


def main():
    rows = []
    for i, (label, fname) in enumerate(STRATEGIES):
        t = pd.read_csv(os.path.join(DATA_DIR, fname), parse_dates=["entry_time"]).sort_values("entry_time")
        net_r = t.net_r.values.astype(np.float64)
        years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
        tpy = len(t) / years

        t1_raw, t10_raw, max_dd, ruined = run_mc_capped(net_r, N_SIMS, BLOCK, MAX_TRADES, seed=7000 + i)
        t1_hit = t1_raw[t1_raw > 0]
        years_1m = t1_hit / tpy
        dd_pct = max_dd * 100

        row = dict(
            label=label, tpy=round(tpy), avg_r=round(net_r.mean(), 4),
            wr=round(100 * (net_r > 0).mean(), 2),
            median_yrs_1m=round(np.median(years_1m), 2) if len(years_1m) else None,
            median_dd=round(np.median(dd_pct), 1),
            mean_dd=round(dd_pct.mean(), 1),
            worst_dd=round(dd_pct.min(), 1),
            best_dd=round(dd_pct.max(), 1),
            pct_hit_1m=round(100 * len(years_1m) / N_SIMS, 1),
            pct_ruined=round(100 * ruined.mean(), 2),
        )
        rows.append(row)
        print(f"{label:32s} tpy={tpy:6.0f}  yrs->1M(med)={row['median_yrs_1m']}  "
              f"DD med/mean/worst={row['median_dd']}%/{row['mean_dd']}%/{row['worst_dd']}%  "
              f"ruin={row['pct_ruined']}%")

    df = pd.DataFrame(rows).sort_values("median_yrs_1m")
    df.to_csv(os.path.join(DATA_DIR, "goal11_drawdown_comparison.csv"), index=False)
    print(f"\nSaved: goal11_drawdown_comparison.csv")
    print("\nSorted by speed (fastest first):")
    print(df[["label", "median_yrs_1m", "median_dd", "worst_dd", "wr", "avg_r"]].to_string(index=False))


if __name__ == "__main__":
    main()
