"""Replace goal2_hour_opt.py's hindsight bucket selection (keep hour if its summed
log-growth over the WHOLE history is positive) with a temporally-consistent, walk-
forward-validated version.

The flaw being fixed: picking 1-of-120 buckets by how each already performed over the
full 2009-2026 sample lets pure noise (a bucket that got lucky once) look like edge.
The fix has two parts:

1. CONSISTENCY REQUIREMENT (for the actual deployed rule): split history into 4
   chronological quarters. A bucket only qualifies if it has positive log-growth AND
   n>=30 trades in a supermajority (>=3 of however many quarters have n>=30) of
   quarters -- i.e. an edge that shows up repeatedly over time, not once in aggregate.
   This uses all available history (as any real deployment decision should), but
   requires the signal to be temporally stable, not just net-positive when summed.

2. WALK-FORWARD HOLDOUT (to prove the METHOD, not just this one bucket list,
   generalizes): the consistency rule itself is re-derived using ONLY the first half
   of history (quarters 1-2, split into two sub-halves for its own consistency check),
   then applied unchanged to the second half (quarters 3-4) -- data the selection
   never touched. Quarters 3-4's performance is the honest, hindsight-free number.
"""
import os, sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal2_hour_opt as h

RISK_FRAC = h.RISK_FRAC
MIN_N_PER_QUARTER = 30


def make_quarters(t_all_min, t_all_max, k=4):
    edges = pd.date_range(t_all_min, t_all_max, periods=k + 1)
    return [(edges[i], edges[i + 1]) for i in range(k)]


def bucket_vote(sub_by_quarter):
    """sub_by_quarter: list of (n, gsum) per quarter. Returns True if positive in a
    supermajority of quarters with enough data, and at least 2 qualifying quarters."""
    qualifying = [(n, gsum) for n, gsum in sub_by_quarter if n >= MIN_N_PER_QUARTER]
    if len(qualifying) < 2:
        return False
    positives = sum(1 for n, gsum in qualifying if gsum > 0)
    return positives >= max(2, int(np.ceil(0.75 * len(qualifying))))


def select_hours(pools, quarters):
    """Consistency-based selection using the given quarters (list of (start,end))."""
    selected = {}
    detail = {}
    for inst, t in pools.items():
        byh_detail = {}
        keep = []
        for hr in range(24):
            sub = t[t.hour == hr]
            per_q = []
            for qs, qe in quarters:
                q = sub[(sub.entry_time >= qs) & (sub.entry_time < qe)]
                per_q.append((len(q), q.glog.sum() if len(q) else 0.0))
            byh_detail[hr] = per_q
            if bucket_vote(per_q):
                keep.append(hr)
        selected[inst] = keep
        detail[inst] = byh_detail
    return selected, detail


def score(sub, label=""):
    if len(sub) < 30:
        return dict(label=label, n=len(sub), tpy=None, wr=None, pf=None, g=None)
    years = (sub.entry_time.max() - sub.entry_time.min()).days / 365.25
    tpy = len(sub) / years
    wr = 100 * sub.is_win.mean()
    w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    g = tpy * sub.glog.mean()
    return dict(label=label, n=len(sub), tpy=tpy, wr=wr, pf=pf, g=g)


def print_score(s):
    if s["tpy"] is None:
        print(f"{s['label']:34s} n={s['n']} (too few trades)")
        return
    y1m = np.log(5000) / s["g"] if s["g"] and s["g"] > 0 else float("inf")
    print(f"{s['label']:34s} n={s['n']:7,} ({s['tpy']:6.0f}/yr) WR={s['wr']:5.2f}% "
          f"PF={s['pf']:.3f} g={s['g']:6.3f} yrs->1M={y1m:.2f}")


def main():
    instruments = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY", "XAUUSD"]
    for inst in instruments:
        h.rs.load_instrument(inst)
        h.rs._prep_zones(inst, 2)

    pools = {}
    tmin, tmax = None, None
    for inst in instruments:
        t = h.build_all(inst)
        t["glog"] = np.log1p(RISK_FRAC * t.net_r)
        t["hour"] = t.entry_time.dt.hour
        pools[inst] = t
        tmin = t.entry_time.min() if tmin is None else min(tmin, t.entry_time.min())
        tmax = t.entry_time.max() if tmax is None else max(tmax, t.entry_time.max())

    # ---------- Part 1: FINAL deployed rule -- consistency across 4 full-history quarters ----------
    quarters4 = make_quarters(tmin, tmax, k=4)
    print("Full-history quarters:", [(q[0].date(), q[1].date()) for q in quarters4])
    final_hours, _ = select_hours(pools, quarters4)
    for inst in instruments:
        print(f"  {inst}: consistency-selected hours ({len(final_hours[inst])}): {sorted(final_hours[inst])}")

    frames = []
    for inst, t in pools.items():
        frames.append(t[t.hour.isin(final_hours[inst])])
    robust_full = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    print()
    print_score(score(robust_full, "ROBUST (consistency, full-history-selected)"))
    split = pd.Timestamp("2020-01-01")
    print_score(score(robust_full[robust_full.entry_time < split], "  -> train 2009-2020 slice"))
    print_score(score(robust_full[robust_full.entry_time >= split], "  -> test 2020-2026 slice"))

    # ---------- Part 2: WALK-FORWARD holdout -- derive the rule on first half only ----------
    half = tmin + (tmax - tmin) / 2
    first_half_quarters = make_quarters(tmin, half, k=4)  # sub-quarters of the FIRST half only
    wf_hours, _ = select_hours(pools, first_half_quarters)
    print()
    print("Walk-forward: rule derived using ONLY", tmin.date(), "to", half.date())
    for inst in instruments:
        print(f"  {inst}: first-half-selected hours ({len(wf_hours[inst])}): {sorted(wf_hours[inst])}")

    frames_wf_first = []
    frames_wf_second = []
    for inst, t in pools.items():
        sel = t[t.hour.isin(wf_hours[inst])]
        frames_wf_first.append(sel[sel.entry_time < half])
        frames_wf_second.append(sel[sel.entry_time >= half])
    wf_first = pd.concat(frames_wf_first, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    wf_second = pd.concat(frames_wf_second, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    print()
    print_score(score(wf_first, f"  first half {tmin.date()}..{half.date()} (in-sample for this rule)"))
    print_score(score(wf_second, f"  second half {half.date()}..{tmax.date()} (TRUE holdout, never seen)"))

    robust_full.to_csv(os.path.join(DATA_DIR, "goal2_robust_candidate_trades.csv"), index=False)
    wf_second.to_csv(os.path.join(DATA_DIR, "goal2_robust_walkforward_holdout_trades.csv"), index=False)
    print(f"\nSaved: goal2_robust_candidate_trades.csv, goal2_robust_walkforward_holdout_trades.csv")
    return final_hours, robust_full, wf_second


if __name__ == "__main__":
    main()
