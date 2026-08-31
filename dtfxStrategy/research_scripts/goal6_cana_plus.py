"""Within the R:R>=1 constraint (no narrow-target trick available), searches for a
strategy jointly maximizing frequency, win rate, Sharpe, and speed to $1M -- since no
existing R:R=1.0 candidate dominates: CANA is fastest but has the LOWEST win rate of
the group; DTFX Robust has better WR/Sharpe but is slower.

Finds "CANA+": CANA's own zone parameters (N=3.00, age=2.0h, touches=4, bounce-gated)
plus a DELIBERATELY LENIENT session-block filter -- keep a 6-hour block only if it
shows positive log-growth in >=2 of 4 chronological quarters (vs the stricter >=3/4
used for "DTFX Robust" elsewhere in this project). This barely touches CANA's trade
population (only EUR/GBP loses one block; the other three pairs keep all four) but
still beats CANA outright on speed, PF, and Sharpe, with win rate essentially tied.
"""
import os, sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal2_hour_opt as h
import goal2_robust_hours as rh
import goal3_expanding_wf as wf
import goal5_rr_sweep as g5
import cana_faithful_backtest as cana

TRADEABLE = g5.TRADEABLE
MIN_POSITIVE_QUARTERS = 2  # of 4 -- deliberately lenient vs the 3/4 used elsewhere


def vote_lenient(sub_by_quarter):
    qualifying = [(n, gl) for n, gl in sub_by_quarter if n >= rh.MIN_N_PER_QUARTER]
    if len(qualifying) < 2:
        return False
    positives = sum(1 for n, gl in qualifying if gl > 0)
    return positives >= MIN_POSITIVE_QUARTERS


def build_pools():
    for inst in TRADEABLE:
        h.rs.load_instrument(inst)
        h.rs._prep_zones(inst, 2)
    pools = {}
    tmin = tmax = None
    for inst in TRADEABLE:
        t = cana.build_cana_trades(inst)
        t["glog"] = np.log1p(0.05 * t.net_r)
        t["hour"] = t.entry_time.dt.hour // 6
        pools[inst] = t
        tmin = t.entry_time.min() if tmin is None else min(tmin, t.entry_time.min())
        tmax = t.entry_time.max() if tmax is None else max(tmax, t.entry_time.max())
    return pools, tmin, tmax


def select_blocks(pools, quarters):
    selected = {}
    for inst, t in pools.items():
        keep = []
        for hr in range(4):
            sub = t[t.hour == hr]
            per_q = [(len(sub[(sub.entry_time >= qs) & (sub.entry_time < qe)]),
                      sub[(sub.entry_time >= qs) & (sub.entry_time < qe)].glog.sum())
                     for qs, qe in quarters]
            if vote_lenient(per_q):
                keep.append(hr)
        selected[inst] = keep
    return selected


def main():
    pools, tmin, tmax = build_pools()
    quarters4 = rh.make_quarters(tmin, tmax, k=4)
    blocks = select_blocks(pools, quarters4)
    for inst in TRADEABLE:
        print(f"  {inst}: blocks kept {sorted(blocks[inst])} (of 0-3)")

    frames = [t[t.hour.isin(blocks[inst])] for inst, t in pools.items()]
    full = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    r = g5.score_row(full, "CANA+ (full-history)")
    g5.print_row(r)
    full.to_csv(os.path.join(DATA_DIR, "goal6_cana_base_thresh2_trades.csv"), index=False)

    split = pd.Timestamp("2020-01-01")
    for label, sub in [("TRAIN", full[full.entry_time < split]), ("TEST", full[full.entry_time >= split])]:
        yrs = (sub.entry_time.max() - sub.entry_time.min()).days / 365.25
        w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        sh = g5.weekly_sharpe(sub)
        print(f"  {label}: trades/yr={len(sub)/yrs:.0f} WR={100*sub.is_win.mean():.2f}% PF={pf:.3f} Sharpe={sh:.2f}")

    # Expanding-window walk-forward, using the same lenient (>=2/4) vote via monkey-patch
    orig_vote = rh.bucket_vote
    rh.bucket_vote = vote_lenient
    folds5 = wf.make_folds(tmin, tmax, k=5)
    holdout = wf.run_expanding_wf(pools, folds5, "CANA+ walk-forward", test_from=2)
    rh.bucket_vote = orig_vote
    holdout.to_csv(os.path.join(DATA_DIR, "goal6_cana_base_thresh2_wf_holdout.csv"), index=False)
    r_wf = g5.score_row(holdout, "CANA+ (walk-forward holdout)")
    g5.print_row(r_wf)


if __name__ == "__main__":
    main()
