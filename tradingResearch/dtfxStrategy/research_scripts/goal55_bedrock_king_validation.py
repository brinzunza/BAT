"""Persists the two pieces of evidence REALISTIC_FILL_RANKINGS.md's "Bedrock-King" section
relies on, both originally run ad hoc: why the extreme-N region (N up to 50) was rejected,
and the 2018 train/holdout check that separated Bedrock-King from that rejected region.
Same corrected fill model as goal53/goal54 (full 1.6-pip fill depth, universal spread cost).
"""
import os
import sys
import itertools

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal51_single_symbol_closest_price_backtest as g51
import goal53_realistic_fill_leaderboard_and_search as g53

INST = "EURGBP"
AGE_H, TOUCHES = 0.08, 20
SPLIT = pd.Timestamp("2018-01-01")


def pf_of(t):
    if len(t) < 20:
        return None, len(t)
    gw, gl = t[t.net_r > 0].net_r.sum(), -t[t.net_r <= 0].net_r.sum()
    return (gw / gl if gl > 0 else float("inf")), len(t)


def extreme_n_sweep():
    """Extends goal54's grid to N up to 50 -- PF keeps climbing but sample size collapses,
    the setup for the train/holdout rejection below."""
    n_grid = [20.0, 25.0, 30.0, 40.0, 50.0]
    rr_grid = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    mrp_grid = [0.020, 0.025, 0.030, 0.040, 0.050]
    rows = []
    for n, rr, mrp in itertools.product(n_grid, rr_grid, mrp_grid):
        r = g53.evaluate(n, rr, AGE_H, TOUCHES, min_risk_pct=mrp)
        if r is None or r["freq_tpy"] < 10:
            continue
        r.update(n=n, rr=rr, min_risk_pct=mrp)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("pf", ascending=False)
    out = os.path.join(DATA_DIR, "goal55_extreme_n_sweep.csv")
    df.to_csv(out, index=False)
    print(f"Extreme-N sweep: {len(df)} valid configs, saved {out}")
    print(df[["n", "rr", "min_risk_pct", "n_trades", "wr", "pf", "freq_tpy"]].head(10).to_string(index=False))
    return df


def train_holdout_check(candidates):
    rows = []
    for n, rr, mrp in candidates:
        parts = [g51.build_touches_with_metadata(INST, n, rr, AGE_H, False, TOUCHES, side,
                                                   min_risk_pct=mrp)
                 for side in (True, False)]
        t = pd.concat(parts, ignore_index=True)
        train, holdout = t[t.entry_time < SPLIT], t[t.entry_time >= SPLIT]
        pf_train, n_train = pf_of(train)
        pf_hold, n_hold = pf_of(holdout)
        rows.append(dict(n=n, rr=rr, min_risk_pct=mrp, n_train=n_train, pf_train=pf_train,
                          n_hold=n_hold, pf_hold=pf_hold,
                          robust_pf=min(pf_train, pf_hold) if pf_train and pf_hold else None))
    df = pd.DataFrame(rows).sort_values("robust_pf", ascending=False, na_position="last")
    out = os.path.join(DATA_DIR, "goal55_train_holdout_validation.csv")
    df.to_csv(out, index=False)
    print(f"\nTrain/holdout validation: saved {out}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    extreme_df = extreme_n_sweep()
    # Rejected extreme-N candidates (in-sample winners from the sweep above) alongside
    # Bedrock-King and its immediate neighborhood, for direct side-by-side comparison.
    candidates = [
        (10.0, 0.15, 0.008), (10.0, 0.50, 0.008), (10.0, 1.50, 0.008),
        (22.0, 0.75, 0.020),  # Bedrock-King
        (15.0, 1.00, 0.013), (25.0, 0.75, 0.020), (25.0, 0.75, 0.023),
        (30.0, 0.50, 0.025), (50.0, 0.30, 0.050), (50.0, 0.50, 0.050),
    ]
    train_holdout_check(candidates)
