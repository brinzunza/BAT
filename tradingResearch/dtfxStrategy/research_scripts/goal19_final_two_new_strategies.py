"""Finalizes the 2 new strategies found by the age-optimization search
(goal16->goal17->goal18): every prior sweep in this project held fill-age fixed at
one of two legacy values (2.0h from CANA, 3.366682h from SOMEGA) and only ever
varied (N, R:R). Jointly optimizing fill-age together with (N, R:R) finds a
genuinely better region for both re-arm rules -- a much SHORTER fill-age (0.175-0.4h,
vs 2.0-3.37h used everywhere else) than anything previously tried in this project.

  Speed-King: unconditional re-arm, age_h=0.175 (fine neighborhood peak -- beats the
  coarser grid's 0.20h), N=7.0, R:R=0.20, touches=4.
  Quality-King: bounce-gated re-arm, age_h=0.40, N=7.0, R:R=0.35, touches=8 (this
  point was confirmed stable across two independent grid refinements).

Both validated via the same TRUE holdout discipline as every other strategy in this
project (parameters picked on first-half data only, scored on the untouched second
half) plus a year-by-year breakdown to check for single-year dependency before
being reported as beating the current top 2.
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

CONFIGS = [
    ("Speed-King", dict(n=7.0, rr=0.20, age_h=0.175, require_bounce=False, touches=4)),
    ("Quality-King", dict(n=7.0, rr=0.35, age_h=0.40, require_bounce=True, touches=8)),
]


def year_by_year(full_trades, label):
    t = full_trades.copy()
    t["year"] = t.entry_time.dt.year
    print(f"  {label} year-by-year:")
    losing_years = []
    for yr, sub in t.groupby("year"):
        if len(sub) < 20:
            continue
        wr = 100 * sub.is_win.mean()
        w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        print(f"    {yr}: n={len(sub):6,}  WR={wr:5.2f}%  PF={pf:.3f}")
        if pf < 1.0:
            losing_years.append(yr)
    print(f"    Losing years (PF<1.0): {losing_years if losing_years else 'none'}")
    return losing_years


def main():
    for inst in g5.TRADEABLE:
        g5.rs.load_instrument(inst)
        g5.rs._prep_zones(inst, g5.pg.MIN_RETRACEMENT)

    for label, kw in CONFIGS:
        print(f"\n{'='*90}\n{label}: N={kw['n']}, R:R={kw['rr']}, age_h={kw['age_h']}, "
              f"bounce={kw['require_bounce']}, touches={kw['touches']}\n{'='*90}")

        full = g10.build(kw["n"], kw["rr"], age_h=kw["age_h"], require_bounce=kw["require_bounce"],
                          touches=kw["touches"])
        full.to_csv(os.path.join(DATA_DIR, f"goal19_{label.lower().replace('-','_')}_full.csv"), index=False)
        r_full = g5.score_row(full, f"{label} (full-history)")
        g5.print_row(r_full)

        second = full[full.entry_time >= SPLIT].reset_index(drop=True)
        second.to_csv(os.path.join(DATA_DIR, f"goal19_{label.lower().replace('-','_')}_true_holdout.csv"), index=False)
        r_holdout = g5.score_row(second, f"{label} (TRUE holdout)")
        g5.print_row(r_holdout)

        year_by_year(full, label)

    print("\n" + "#" * 90)
    print("  vs current top 2: Hybrid-Ultra (4,135/yr, 79.38% WR, PF 1.374, Sharpe 6.43, 0.53yr)")
    print("                     DTFX-Ultra   (3,289/yr, 79.86% WR, PF 1.415, Sharpe 6.55, 0.61yr)")
    print("#" * 90)


if __name__ == "__main__":
    main()
