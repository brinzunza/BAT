"""goal17's refined search found its best point sitting at the EDGE of the tested
grid (age_h=0.35 was the 2nd-lowest tested, R:R=0.25 was the lowest tested) -- both
still climbing, not yet a genuine interior peak. Extends the grid further toward
shorter age and narrower R:R, tracking profit factor explicitly alongside g-score,
since this project's own established "failure cliff" pattern (win rate keeps
climbing as R:R shrinks, but PF eventually falls under 1.0 and the strategy becomes
a net loser) means the true optimum for a RACE metric (g-score) is not necessarily
where PF is highest -- must find where g-score itself peaks and turns over, and
verify PF stays safely above 1.0 there, not just report whatever wins the grid.

The lower age bound is set at 0.15h, not lower: a preliminary fine sweep (see
goal16_output.log) found a sharp, genuine collapse in edge below ~0.2h (g-score
falls to near-zero at age_h=0.1) -- likely zones filling almost immediately after
formation, before the level has had time to matter -- so the search deliberately
does not chase that collapse zone.
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
AGE_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
N_GRID = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
RR_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def eval_point(n, rr, age_h, require_bounce, touches):
    trades = g10.build(n, rr, age_h=age_h, require_bounce=require_bounce, touches=touches)
    first = trades[trades.entry_time < SPLIT]
    if len(first) < 100:
        return -1e9, None
    glog = np.log1p(g5.RISK_FRAC * first.net_r)
    years = (first.entry_time.max() - first.entry_time.min()).days / 365.25
    g = (len(first) / years) * glog.mean()
    w = first.loc[first.net_r > 0, "net_r"].sum(); l = -first.loc[first.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    return g, pf


def joint_search(require_bounce, touches, label):
    rows = []
    for age_h in AGE_GRID:
        for n in N_GRID:
            for rr in RR_GRID:
                g, pf = eval_point(n, rr, age_h, require_bounce, touches)
                rows.append((age_h, n, rr, g, pf))
    rows.sort(key=lambda r: -r[3])
    print(f"\n{label}: top 12 (age_h, N, R:R) by first-half g-score, with first-half PF shown:")
    for age_h, n, rr, g, pf in rows[:12]:
        edge = " <-- GRID EDGE" if (age_h in (AGE_GRID[0], AGE_GRID[-1]) or n in (N_GRID[0], N_GRID[-1])
                                     or rr in (RR_GRID[0], RR_GRID[-1])) else ""
        pf_s = f"{pf:.3f}" if pf is not None else "n/a"
        print(f"  age_h={age_h:.2f}  N={n:.1f}  R:R={rr:.2f}  g={g:.3f}  PF={pf_s}{edge}")
    return rows[0]


def main():
    for inst in g5.TRADEABLE:
        g5.rs.load_instrument(inst)
        g5.rs._prep_zones(inst, g5.pg.MIN_RETRACEMENT)

    configs = [
        ("Unconditional-Ext", dict(require_bounce=False, touches=4)),
        ("BounceGated-Ext", dict(require_bounce=True, touches=8)),
    ]

    results = []
    for label, kw in configs:
        age_h, n, rr, g, pf = joint_search(kw["require_bounce"], kw["touches"], label)
        print(f"  -> BEST for {label}: age_h={age_h}, N={n}, R:R={rr} (g={g:.3f}, first-half PF={pf:.3f})")

        full = g10.build(n, rr, age_h=age_h, require_bounce=kw["require_bounce"], touches=kw["touches"])
        full.to_csv(os.path.join(DATA_DIR, f"goal18_{label.lower().replace('-','_')}_full.csv"), index=False)
        r_full = g5.score_row(full, f"{label} (full-history)")
        g5.print_row(r_full)

        second = full[full.entry_time >= SPLIT].reset_index(drop=True)
        second.to_csv(os.path.join(DATA_DIR, f"goal18_{label.lower().replace('-','_')}_true_holdout.csv"), index=False)
        r_holdout = g5.score_row(second, f"{label} (TRUE holdout)")
        g5.print_row(r_holdout)

        results.append(dict(label=label, age_h=age_h, n=n, rr=rr, touches=kw["touches"],
                             require_bounce=kw["require_bounce"], full=r_full, holdout=r_holdout))

    print("\n" + "#" * 90)
    print("  FINAL COMPARISON vs current top 2 (Hybrid-Ultra 0.53yr/4135tpy, DTFX-Ultra 0.61yr/3289tpy)")
    print("#" * 90)
    for r in results:
        h = r["holdout"]
        print(f"  {r['label']:20s} age={r['age_h']:.2f}h N={r['n']} R:R={r['rr']} touches={r['touches']} "
              f"bounce={r['require_bounce']}  tpy={h['tpy']:.0f} WR={h['wr']:.2f}% PF={h['pf']:.3f} "
              f"Sharpe={h['sharpe']:.2f} median={h['y1m']:.3f}yr P10={h['y1m_p10']:.3f} P90={h['y1m_p90']:.3f} "
              f"ruin={h['ruin_pct']:.1f}%")


if __name__ == "__main__":
    main()
