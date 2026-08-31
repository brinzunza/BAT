"""Searches a genuinely untried axis: fill-age gate (MIN_FILL_AGE), jointly with
(N, R:R), instead of only ever testing the two legacy ages inherited from CANA
(2.0h) and SOMEGA (3.366682h). Every prior sweep in this project (goal10's Ultra
family included) held age_h fixed at one of those two values and only varied
(N, R:R) -- age itself was never actually optimized.

Two re-arm rules are searched independently (bounce-gated touches=8, unconditional
touches=4), each first coarse-sweeping age_h alone (holding N=6.0/R:R=0.4 fixed,
since the Ultra family already showed this point is robust across every prior
age/rearm combination), then refining (N, R:R) locally around whatever age wins,
exactly the same selection discipline as goal10 (first-half g-score selects,
untouched second half scores).
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal10_ultra_family as g10
import goal5_rr_sweep as g5

SPLIT = g10.SPLIT
AGE_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.366682, 4.0, 5.0, 6.0]
N_REFINE = [4.0, 5.0, 6.0, 7.0, 8.0]
RR_REFINE = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]


def g_score_first_half(trades):
    first = trades[trades.entry_time < SPLIT]
    if len(first) < 100:
        return -1e9
    glog = np.log1p(g5.RISK_FRAC * first.net_r)
    years = (first.entry_time.max() - first.entry_time.min()).days / 365.25
    return (len(first) / years) * glog.mean()


def search_one_rearm(require_bounce, touches, label):
    print(f"\n=== {label}: age_h sweep (N=6.0, R:R=0.4 fixed) ===")
    best_age, best_g = None, -1e9
    for age_h in AGE_GRID:
        trades = g10.build(6.0, 0.4, age_h=age_h, require_bounce=require_bounce, touches=touches)
        g = g_score_first_half(trades)
        tag = " <- best so far" if g > best_g else ""
        print(f"  age_h={age_h:8.4f}  first-half g={g:.3f}{tag}")
        if g > best_g:
            best_g, best_age = g, age_h
    print(f"  -> best age_h for {label}: {best_age}")

    print(f"\n=== {label}: refine (N, R:R) at age_h={best_age} ===")
    best_n, best_rr, best_g2 = None, None, -1e9
    for n in N_REFINE:
        for rr in RR_REFINE:
            trades = g10.build(n, rr, age_h=best_age, require_bounce=require_bounce, touches=touches)
            g = g_score_first_half(trades)
            if g > best_g2:
                best_g2, best_n, best_rr = g, n, rr
    print(f"  -> best (N, R:R) at age_h={best_age}: N={best_n}, R:R={best_rr} (g={best_g2:.3f})")

    full = g10.build(best_n, best_rr, age_h=best_age, require_bounce=require_bounce, touches=touches)
    full.to_csv(os.path.join(DATA_DIR, f"goal16_{label}_full.csv"), index=False)
    r_full = g5.score_row(full, f"{label} (full-history, age={best_age}, N={best_n}, R:R={best_rr})")
    g5.print_row(r_full)

    second = full[full.entry_time >= SPLIT].reset_index(drop=True)
    second.to_csv(os.path.join(DATA_DIR, f"goal16_{label}_true_holdout.csv"), index=False)
    r_holdout = g5.score_row(second, f"{label} (TRUE holdout, age={best_age}, N={best_n}, R:R={best_rr})")
    g5.print_row(r_holdout)

    return dict(label=label, age_h=best_age, n=best_n, rr=best_rr, holdout=r_holdout, full=r_full)


def main():
    for inst in g5.TRADEABLE:
        g5.rs.load_instrument(inst)
        g5.rs._prep_zones(inst, g5.pg.MIN_RETRACEMENT)

    results = []
    results.append(search_one_rearm(require_bounce=True, touches=8, label="BounceGated-AgeOpt"))
    results.append(search_one_rearm(require_bounce=False, touches=4, label="Unconditional-AgeOpt"))

    print("\n" + "#" * 90)
    print("  SUMMARY vs current top 2 (Hybrid-Ultra 0.53yr/92.4, DTFX-Ultra 0.61yr/88.1)")
    print("#" * 90)
    for r in results:
        h = r["holdout"]
        print(f"  {r['label']:24s} age={r['age_h']:.4f} N={r['n']} R:R={r['rr']}  "
              f"tpy={h['tpy']:.0f} WR={h['wr']:.2f}% PF={h['pf']:.3f} Sharpe={h['sharpe']:.2f} "
              f"median={h['y1m']:.3f}yr")


if __name__ == "__main__":
    main()
