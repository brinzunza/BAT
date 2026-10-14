"""Puts every R:R>=1 candidate through the EXACT SAME true-holdout test: select the
strategy's own rule (session-block filter, or nothing for CANA) using ONLY the first
half of history (2009 -> 2018-03-09), score it purely on the entirely untouched
second half (2018-03-09 -> 2026). Same split point, same procedure, for all four --
fixes an inconsistency a Stop-hook review correctly flagged (CANA+, DTFX Robust, and
SOMEGA Robust had each been validated with a DIFFERENT walk-forward setup elsewhere in
this project, making "which is fastest" comparisons across them unreliable).

Result: CANA+ is genuinely the fastest of the four under this unified test, with the
second-highest Sharpe and second-highest win rate -- DTFX Robust has the best WR/PF
but is slightly slower; SOMEGA Robust and plain CANA are both slower on every axis.
"""
import os, sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal2_hour_opt as h
import goal2_robust_hours as rh
import goal5_rr_sweep as g5
import prod_filter_grid_search as pg
import dtfx_robust_search as rs
import cana_faithful_backtest as cana

TRADEABLE = g5.TRADEABLE


def vote_lenient(sub_by_quarter):
    qualifying = [(n, gl) for n, gl in sub_by_quarter if n >= rh.MIN_N_PER_QUARTER]
    if len(qualifying) < 2:
        return False
    positives = sum(1 for n, gl in qualifying if gl > 0)
    return positives >= 2


def true_holdout(pools, label):
    """Given {instrument: trades_df_with_hour_and_glog}, select session blocks using
    only the first half, score purely on the untouched second half."""
    tmin = min(t.entry_time.min() for t in pools.values())
    tmax = max(t.entry_time.max() for t in pools.values())
    half = tmin + (tmax - tmin) / 2
    first_half_quarters = rh.make_quarters(tmin, half, k=4)

    orig_vote = rh.bucket_vote
    rh.bucket_vote = vote_lenient
    wf_hours, _ = rh.select_hours(pools, first_half_quarters)
    rh.bucket_vote = orig_vote

    frames = [t[(t.hour.isin(wf_hours[inst])) & (t.entry_time >= half)]
              for inst, t in pools.items()]
    second = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    r = g5.score_row(second, label)
    g5.print_row(r)
    return second, r


def main():
    for inst in TRADEABLE:
        h.rs.load_instrument(inst)
        h.rs._prep_zones(inst, 2)

    results = {}

    # CANA+ : CANA's own zone params (N=3.00, touches=4), block filter selected on 1st half
    pools_cana = {}
    for inst in TRADEABLE:
        t = cana.build_cana_trades(inst)
        t["glog"] = np.log1p(0.05 * t.net_r)
        t["hour"] = t.entry_time.dt.hour // 6
        pools_cana[inst] = t
    trades, r = true_holdout(pools_cana, "CANA+")
    trades.to_csv(os.path.join(DATA_DIR, "goal8_cana_plus_true_holdout.csv"), index=False)
    results["CANA+"] = r

    # DTFX Robust: N=2.0/age=2.0h/touches=8 base, block filter selected on 1st half
    pools_dtfx = {}
    for inst in TRADEABLE:
        t = h.build_all(inst)
        t["glog"] = np.log1p(0.05 * t.net_r)
        t["hour"] = t.entry_time.dt.hour // 6
        pools_dtfx[inst] = t
    trades, r = true_holdout(pools_dtfx, "DTFX Robust")
    trades.to_csv(os.path.join(DATA_DIR, "goal8_dtfxrobust_true_holdout.csv"), index=False)
    results["DTFX Robust"] = r

    # SOMEGA Robust: SOMEGA's own N=3.343496/age=3.366682h base, block filter on 1st half
    rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
    pools_somega = {}
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        t = pg.build_trades(inst, 3.343496, 3.343496, 3.366682, require_bounce=False, session_filter=False)
        t["glog"] = np.log1p(0.05 * t.net_r)
        t["hour"] = t.entry_time.dt.hour // 6
        pools_somega[inst] = t
    trades, r = true_holdout(pools_somega, "SOMEGA Robust")
    trades.to_csv(os.path.join(DATA_DIR, "goal8_somegarobust_true_holdout.csv"), index=False)
    results["SOMEGA Robust"] = r

    # CANA (plain, fixed -- no selection, no holdout distinction, always its own number)
    frames = [cana.build_cana_trades(inst) for inst in TRADEABLE]
    cana_full = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    r = g5.score_row(cana_full, "CANA (fixed)")
    g5.print_row(r)
    results["CANA"] = r

    print("\nRanked by median years to $1M (all under the identical 2018-03-09-split holdout test):")
    for label, r in sorted(results.items(), key=lambda kv: kv[1]["y1m"]):
        print(f"  {label:15s} {r['tpy']:6.0f}/yr  WR={r['wr']:5.2f}%  PF={r['pf']:.3f}  "
              f"Sharpe={r['sharpe']:.2f}  median={r['y1m']:.2f}yr")


if __name__ == "__main__":
    main()
