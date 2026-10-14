"""Proper expanding-window walk-forward: split history into 5 chronological folds.
For each test fold k=3..5, select hours using ONLY folds 1..k-1 as the consistency-vote
periods (unanimity when 2 periods, supermajority when 3+), apply that rule to fold k
(never touched during selection), and keep fold k's actual trades. Concatenating folds
3,4,5 gives ~10-13 years of genuinely out-of-sample, expanding-window validated trades
-- a much stronger test than one 50/50 split (more coverage, and later folds' selection
benefits from more prior data, same as a live deployment accumulating history over time).
"""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal2_hour_opt as h
import goal2_robust_hours as rh

RISK_FRAC = h.RISK_FRAC
MIN_N_PER_FOLD = 30


def make_folds(tmin, tmax, k=5):
    edges = pd.date_range(tmin, tmax, periods=k + 1)
    return [(edges[i], edges[i + 1]) for i in range(k)]


def vote_from_periods(per_period, require_unanimous_if_2=True):
    qualifying = [(n, g) for n, g in per_period if n >= MIN_N_PER_FOLD]
    if len(qualifying) < 2:
        return False
    positives = sum(1 for n, g in qualifying if g > 0)
    if len(qualifying) == 2:
        return positives == 2 if require_unanimous_if_2 else positives >= 1
    return positives >= int(np.ceil(0.6667 * len(qualifying)))  # >=2/3 supermajority


def select_hours_from_folds(pools, prior_folds):
    selected = {}
    for inst, t in pools.items():
        keep = []
        for hr in range(24):
            sub = t[t.hour == hr]
            per_period = []
            for fs, fe in prior_folds:
                fsub = sub[(sub.entry_time >= fs) & (sub.entry_time < fe)]
                per_period.append((len(fsub), fsub.glog.sum() if len(fsub) else 0.0))
            if vote_from_periods(per_period):
                keep.append(hr)
        selected[inst] = keep
    return selected


def run_expanding_wf(pools, folds, label, test_from=2):
    """test_from: index of first fold to treat as held-out (0-based); folds before it
    are pure training data, never scored."""
    held_out_frames = []
    print(f"\n=== {label}: {len(folds)} folds, testing from fold {test_from+1} ===")
    for k in range(test_from, len(folds)):
        prior = folds[:k]
        sel = select_hours_from_folds(pools, prior)
        n_hours = {inst: len(hrs) for inst, hrs in sel.items()}
        fs, fe = folds[k]
        frames = []
        for inst, t in pools.items():
            sub = t[(t.entry_time >= fs) & (t.entry_time < fe) & (t.hour.isin(sel[inst]))]
            frames.append(sub)
        fold_trades = pd.concat(frames, ignore_index=True)
        years = (fe - fs).days / 365.25
        tpy = len(fold_trades) / years if years > 0 else 0
        wr = 100 * fold_trades.is_win.mean() if len(fold_trades) else float("nan")
        w = fold_trades.loc[fold_trades.net_r > 0, "net_r"].sum()
        l = -fold_trades.loc[fold_trades.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        print(f"  Fold {k+1} [{fs.date()}..{fe.date()}] hours/inst={n_hours}  "
              f"n={len(fold_trades):5,} ({tpy:6.0f}/yr) WR={wr:5.2f}% PF={pf:.3f}")
        held_out_frames.append(fold_trades)

    wf = pd.concat(held_out_frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    years = (wf.entry_time.max() - wf.entry_time.min()).days / 365.25
    tpy = len(wf) / years
    wr = 100 * wf.is_win.mean()
    w = wf.loc[wf.net_r > 0, "net_r"].sum(); l = -wf.loc[wf.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    g = tpy * wf.glog.mean()
    y1m = np.log(5000) / g if g > 0 else float("inf")
    print(f"  COMBINED expanding-WF holdout ({folds[test_from][0].date()}..{folds[-1][1].date()}): "
          f"n={len(wf):,} ({tpy:.0f}/yr) WR={wr:.2f}% PF={pf:.3f} g={g:.3f} yrs->1M={y1m:.2f}")
    return wf


def main():
    instruments = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY", "XAUUSD"]
    for inst in instruments:
        h.rs.load_instrument(inst)
        h.rs._prep_zones(inst, 2)

    pools = {}
    tmin = tmax = None
    for inst in instruments:
        t = h.build_all(inst)
        t["glog"] = np.log1p(RISK_FRAC * t.net_r)
        t["hour"] = t.entry_time.dt.hour
        pools[inst] = t
        tmin = t.entry_time.min() if tmin is None else min(tmin, t.entry_time.min())
        tmax = t.entry_time.max() if tmax is None else max(tmax, t.entry_time.max())

    folds5 = make_folds(tmin, tmax, k=5)
    print("Folds:", [(f[0].date(), f[1].date()) for f in folds5])
    wf = run_expanding_wf(pools, folds5, "DTFX (5-fold expanding)", test_from=2)
    wf.to_csv(os.path.join(DATA_DIR, "goal3_dtfx_expanding_wf_trades.csv"), index=False)


if __name__ == "__main__":
    main()
