"""Search for a genuinely profitable DTFX zone-fade parameterization under the corrected,
realistic fill model (goal53's own docstring has the full derivation) -- goal53 already
confirmed EVERY existing leaderboard strategy fails this model (best PF=0.378), which
traces to a specific, checkable cause: MIN_RISK_PCT=0.000170 lets in zones whose stop_risk
is only barely larger than a single 1.6-pip spread, so charging that spread against every
trade (now universal, not just stop-outs) can consume most or all of a trade's entire R
unit before any edge gets a chance to show. Raising the minimum-risk floor (filtering to
zones with real room between stop and spread) is the more promising lever than N/RR alone
-- swept jointly here, not assumed.

EUR/GBP only, matching PRODLIVETRADING's current deployment scope.
"""
import os
import sys
import itertools
import time

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal53_realistic_fill_leaderboard_and_search as g53

N_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
RR_GRID = [0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
MIN_RISK_PCT_GRID = [0.000170, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008]
AGE_H = 0.08
TOUCHES = 20
MIN_TRADES_PER_YEAR = 50  # below this, not a usable live strategy regardless of PF


def main():
    combos = list(itertools.product(N_GRID, RR_GRID, MIN_RISK_PCT_GRID))
    print(f"Sweeping {len(combos)} (N, RR, min_risk_pct) combinations on EUR/GBP...")
    results = []
    t0 = time.time()
    for i, (n, rr, mrp) in enumerate(combos):
        r = g53.evaluate(n, rr, AGE_H, TOUCHES, min_risk_pct=mrp)
        if r is None or r["freq_tpy"] < MIN_TRADES_PER_YEAR:
            continue
        r.update(n=n, rr=rr, min_risk_pct=mrp)
        results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(combos)} [{time.time()-t0:.0f}s]")

    df = pd.DataFrame(results).sort_values("pf", ascending=False).reset_index(drop=True)
    out_path = os.path.join(DATA_DIR, "goal54_realistic_fill_search.csv")
    df.to_csv(out_path, index=False)
    print(f"\nTotal runtime: {time.time()-t0:.0f}s. Saved: {out_path}")

    print(f"\nTop 15 by PF (>= {MIN_TRADES_PER_YEAR} trades/yr):")
    cols = ["n", "rr", "min_risk_pct", "n_trades", "wr", "pf", "mean_r", "freq_tpy", "shallow_pct", "years"]
    print(df[cols].head(15).to_string(index=False))

    profitable = df[df.pf > 1.0]
    print(f"\nConfigs with PF > 1.0: {len(profitable)} / {len(df)}")
    if len(profitable):
        print(profitable[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
