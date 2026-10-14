"""/goal (2026-09-04): "find a strategy ... that is net profitable and quickest time to 1
million. follow all the rules and restrictions i have mentioned." goal56 already searched
EUR/GBP exhaustively (1,152 configs, every risk level 2-60%) and found nothing reaches
$1M within the available 16-23 year history -- Bedrock-King itself (the best validated
EUR/GBP candidate) only compounds to ~$207 combined at its best risk. This script checks
whether that's a EUR/GBP-specific ceiling or true of the whole family, by running the
same corrected-fill screen (goal51's model: full 1.6-pip fill depth, universal spread
cost) across the other three symbols PRODLIVETRADING has ever traded. The single-ticker
restriction is about the LIVE deployment's current architecture, not a hard limit on
which instrument to search for research purposes.

Same account rules throughout: two $100 sub-accounts, 50x leverage, min(buy,sell) sizing,
max 1 open position per side (goal30::simulate_two_account, reused unmodified).
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal30_apex_king_deployment as g30
import goal51_single_symbol_closest_price_backtest as g51

AGE_H, TOUCHES = 0.08, 20
N_GRID = [4, 6, 8, 10, 15, 20, 25, 30]
RR_GRID = [0.15, 0.3, 0.5, 0.75, 1.0, 1.5]
MRP_GRID = [0.00017, 0.001, 0.003, 0.005, 0.008, 0.013, 0.020]
RISK_GRID = np.arange(0.02, 0.61, 0.02)


def build(inst, n, rr, mrp):
    parts = [g51.build_touches_with_metadata(inst, n, rr, AGE_H, False, TOUCHES, side,
                                              min_risk_pct=mrp)
             for side in (True, False)]
    return pd.concat(parts, ignore_index=True)


def screen_symbol(inst):
    rows = []
    for n, rr, mrp in itertools.product(N_GRID, RR_GRID, MRP_GRID):
        t = build(inst, n, rr, mrp)
        if len(t) < 100:
            continue
        gw, gl = t[t.net_r > 0].net_r.sum(), -t[t.net_r <= 0].net_r.sum()
        pf = gw / gl if gl > 0 else float("inf")
        if pf <= 1.0 or pf > 3.0:  # >3.0 at this sample scale is almost certainly noise
            continue
        years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
        rows.append(dict(inst=inst, n=n, rr=rr, mrp=mrp, n_trades=len(t), pf=pf,
                          wr=100 * (t.net_r > 0).mean(), freq_tpy=len(t) / years))
    return pd.DataFrame(rows).sort_values("pf", ascending=False) if rows else pd.DataFrame()


def validate_and_speed(inst, n, rr, mrp, split):
    """Train/holdout PF at the given split, plus best-risk extrapolated years-to-$1M."""
    t = build(inst, n, rr, mrp)
    train, holdout = t[t.entry_time < split], t[t.entry_time >= split]

    def pf_of(x):
        if len(x) < 20:
            return None, len(x)
        gw, gl = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return (gw / gl if gl > 0 else float("inf")), len(x)

    pf_tr, n_tr = pf_of(train)
    pf_ho, n_ho = pf_of(holdout)
    robust = pf_tr is not None and pf_ho is not None and pf_tr > 1.0 and pf_ho > 1.0

    years_elapsed = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    best = None
    for rp in RISK_GRID:
        _, eq_buy, eq_sell, dd, _ = g30.simulate_two_account(t, rp)
        cagr = (((eq_buy + eq_sell) / 200.0) ** (1 / years_elapsed)) - 1
        if best is None or cagr > best[0]:
            best = (cagr, rp, eq_buy + eq_sell, dd)
    cagr, best_risk, final_eq, maxdd = best
    years_to_1m = np.log(1_000_000 / 200.0) / np.log(1 + cagr) if cagr > 0 else float("inf")
    return dict(inst=inst, n=n, rr=rr, mrp=mrp, n_trades=len(t), pf_train=pf_tr,
                n_train=n_tr, pf_holdout=pf_ho, n_holdout=n_ho, robust=robust,
                best_risk=best_risk, cagr_pct=100 * cagr, final_equity=final_eq,
                max_dd_pct=maxdd, extrapolated_years_to_1m=years_to_1m)


def main():
    for inst in ("EURUSD", "GBPUSD", "USDJPY"):
        df = screen_symbol(inst)
        out = os.path.join(DATA_DIR, f"goal57_{inst.lower()}_screen.csv")
        df.to_csv(out, index=False)
        print(f"{inst}: {len(df)} profitable configs, saved {out}")
        if len(df):
            print(df.head(8).to_string(index=False))

    # USDJPY only has data from 2018 -- a fixed 2018 split leaves it with no train period,
    # so its own midpoint is used instead of the 2018-01-01 split the other symbols use.
    finalists = [
        ("EURGBP", 22.0, 0.75, 0.020, pd.Timestamp("2018-01-01")),   # Bedrock-King, for comparison
        ("USDJPY", 4.0, 0.75, 0.005, pd.Timestamp("2023-02-10")),     # Yen-Punch
        ("USDJPY", 4.0, 0.15, 0.005, pd.Timestamp("2023-02-10")),     # Yen-Steady
        ("EURUSD", 15.0, 1.00, 0.020, pd.Timestamp("2018-01-01")),    # Dollar-Wide
        ("GBPUSD", 15.0, 1.50, 0.020, pd.Timestamp("2018-01-01")),    # Cable-Wide
    ]
    rows = [validate_and_speed(*f) for f in finalists]
    comp = pd.DataFrame(rows).sort_values("extrapolated_years_to_1m")
    out = os.path.join(DATA_DIR, "goal57_cross_symbol_speed_comparison.csv")
    comp.to_csv(out, index=False)
    print(f"\nFinal cross-symbol comparison, saved {out}:")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
