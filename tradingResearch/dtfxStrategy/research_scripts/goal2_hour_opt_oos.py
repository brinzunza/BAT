"""OOS-validate the log-growth-optimal hour selection: pick each instrument's keep-hours
using ONLY the train split (2009-2020), then score strictly on the untouched test split
(2020-2026). If this is real signal (not noise from many small-n hour buckets), the test
period's WR/PF/g should hold up reasonably, not collapse.
"""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal2_hour_opt as h
RISK_FRAC = h.RISK_FRAC
SPLIT = pd.Timestamp("2020-01-01")


def score(full, label):
    if len(full) < 30:
        print(f"{label}: too few trades ({len(full)})")
        return
    years = (full.entry_time.max()-full.entry_time.min()).days/365.25
    tpy = len(full)/years
    wr = 100*full.is_win.mean()
    w = full.loc[full.net_r>0,"net_r"].sum(); l = -full.loc[full.net_r<=0,"net_r"].sum()
    pf = w/l if l>0 else float("inf")
    g = tpy * full.glog.mean()
    y1m = np.log(1_000_000/200)/g if g>0 else float("inf")
    y10m = np.log(10_000_000/200)/g if g>0 else float("inf")
    print(f"{label:28s} n={len(full):6,} ({tpy:6.0f}/yr) WR={wr:5.2f}% PF={pf:.3f} avg_r={full.net_r.mean():.4f} "
          f"g={g:6.3f} yrs->1M={y1m:.2f} yrs->10M={y10m:.2f}")


def main():
    instruments = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY", "XAUUSD"]
    for inst in instruments:
        h.rs.load_instrument(inst)
        h.rs._prep_zones(inst, 2)

    pools = {}
    for inst in instruments:
        t = h.build_all(inst)
        t["glog"] = np.log1p(RISK_FRAC * t.net_r)
        t["hour"] = t.entry_time.dt.hour
        pools[inst] = t

    train_frames, test_frames_full, test_frames_optimal = [], [], []
    for inst, t in pools.items():
        train = t[t.entry_time < SPLIT]
        test = t[t.entry_time >= SPLIT]
        byh = train.groupby("hour")["glog"].sum()
        keep_hours = byh[byh > 0].index.tolist()
        print(f"{inst}: train-selected hours ({len(keep_hours)}/24): {sorted(keep_hours)}")
        train_frames.append(train[train.hour.isin(keep_hours)])
        test_frames_optimal.append(test[test.hour.isin(keep_hours)])
        test_frames_full.append(test)

    train_sel = pd.concat(train_frames, ignore_index=True).sort_values("entry_time")
    test_sel = pd.concat(test_frames_optimal, ignore_index=True).sort_values("entry_time")
    test_all = pd.concat(test_frames_full, ignore_index=True).sort_values("entry_time")

    print()
    score(train_sel, "TRAIN (hours picked here)")
    score(test_sel, "TEST, train-picked hours (OOS)")
    score(test_all, "TEST, ALL hours (no filter, ref)")


if __name__ == "__main__":
    main()
