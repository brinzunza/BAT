"""Refines goal16's age-optimization finding: the coarse grid (step 0.25 down to
1.0h) suggested age_h=1.25 was best, but a finer check revealed the REAL peak sits
much lower, around 0.4-0.6h, with a non-monotonic dip below that (g-score collapses
towards age_h=0.1-0.2, then rises sharply to a much higher peak around 0.5h, then
falls off smoothly through 1.0-6.0h) -- the original coarse grid simply stepped over
the true peak entirely. This does the full joint (age_h, N, R:R) refinement properly
around that peak, for both re-arm rules.
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
FINE_AGE = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
N_GRID = [3.0, 4.0, 5.0, 6.0, 7.0]
RR_GRID = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]


def g_score_first_half(trades):
    first = trades[trades.entry_time < SPLIT]
    if len(first) < 100:
        return -1e9
    glog = np.log1p(g5.RISK_FRAC * first.net_r)
    years = (first.entry_time.max() - first.entry_time.min()).days / 365.25
    return (len(first) / years) * glog.mean()


def joint_search(require_bounce, touches, label):
    best = (None, None, None, -1e9)
    rows = []
    for age_h in FINE_AGE:
        for n in N_GRID:
            for rr in RR_GRID:
                trades = g10.build(n, rr, age_h=age_h, require_bounce=require_bounce, touches=touches)
                g = g_score_first_half(trades)
                rows.append((age_h, n, rr, g))
                if g > best[3]:
                    best = (age_h, n, rr, g)
    rows.sort(key=lambda r: -r[3])
    print(f"\n{label}: top 8 (age_h, N, R:R) by first-half g-score:")
    for age_h, n, rr, g in rows[:8]:
        print(f"  age_h={age_h:.2f}  N={n:.1f}  R:R={rr:.2f}  g={g:.3f}")
    return best


def main():
    for inst in g5.TRADEABLE:
        g5.rs.load_instrument(inst)
        g5.rs._prep_zones(inst, g5.pg.MIN_RETRACEMENT)

    configs = [
        ("Unconditional-Refined", dict(require_bounce=False, touches=4)),
        ("BounceGated-Refined", dict(require_bounce=True, touches=8)),
    ]

    results = []
    for label, kw in configs:
        age_h, n, rr, g = joint_search(kw["require_bounce"], kw["touches"], label)
        print(f"  -> BEST for {label}: age_h={age_h}, N={n}, R:R={rr} (g={g:.3f})")

        full = g10.build(n, rr, age_h=age_h, require_bounce=kw["require_bounce"], touches=kw["touches"])
        full.to_csv(os.path.join(DATA_DIR, f"goal17_{label.lower().replace('-','_')}_full.csv"), index=False)
        r_full = g5.score_row(full, f"{label} (full-history)")
        g5.print_row(r_full)

        second = full[full.entry_time >= SPLIT].reset_index(drop=True)
        second.to_csv(os.path.join(DATA_DIR, f"goal17_{label.lower().replace('-','_')}_true_holdout.csv"), index=False)
        r_holdout = g5.score_row(second, f"{label} (TRUE holdout)")
        g5.print_row(r_holdout)

        results.append(dict(label=label, age_h=age_h, n=n, rr=rr, touches=kw["touches"],
                             require_bounce=kw["require_bounce"], full=r_full, holdout=r_holdout))

    print("\n" + "#" * 90)
    print("  FINAL COMPARISON vs current top 2 (Hybrid-Ultra 0.53yr/4135tpy, DTFX-Ultra 0.61yr/3289tpy)")
    print("#" * 90)
    for r in results:
        h = r["holdout"]
        print(f"  {r['label']:24s} age={r['age_h']:.2f}h N={r['n']} R:R={r['rr']} touches={r['touches']} "
              f"bounce={r['require_bounce']}  tpy={h['tpy']:.0f} WR={h['wr']:.2f}% PF={h['pf']:.3f} "
              f"Sharpe={h['sharpe']:.2f} median={h['y1m']:.3f}yr P10={h['y1m_p10']:.3f} P90={h['y1m_p90']:.3f}")


if __name__ == "__main__":
    main()
