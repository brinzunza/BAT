"""Honest improvement sweep for BEID (2026-09-09), using the FIXED state machine's real
PRE-2021 trades (research_results/beid_fixed_pre2021_trades.csv -- design sample, never
touched by POST-2026-04) and the project's own two-account, margin-and-concurrency-aware
MC (goal79 engine), not BEID's unconstrained single-curve model. Question: given the real
edge is far thinner than BEID.md claimed (PF ~1.14, not 1.32), is there ANY risk/touch/
zone-size combination that reaches $1M without near-certain ruin -- and if so, how does its
honest speed compare to BEID.md's claimed 29.3 months?
"""
import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from verify_beid_full_history_fixed import run_two_account_mc, PRE_END, POST_START

CSV = os.path.join(SCRIPT_DIR, "..", "research_results", "beid_fixed_pre2021_trades.csv")


def main():
    t0 = time.time()
    df = pd.read_csv(CSV)
    df["fill_time"] = pd.to_datetime(df["fill_time"])
    pre = df[df["fill_time"] < PRE_END].reset_index(drop=True)
    print(f"PRE-2021 trades loaded: {len(pre):,}\n")

    configs = [
        # (min_pip, touches, risk_pct, label)
        (10.0, 5, 0.06, "BEID.md claim: 10p/5touch/6%risk"),
        (10.0, 5, 0.03, "10p/5touch/3%risk"),
        (10.0, 5, 0.02, "10p/5touch/2%risk"),
        (10.0, 5, 0.01, "10p/5touch/1%risk"),
        (10.0, 1, 0.03, "10p/1touch(no rearm)/3%risk"),
        (10.0, 1, 0.02, "10p/1touch(no rearm)/2%risk"),
        (15.0, 5, 0.03, "15p/5touch/3%risk"),
        (15.0, 1, 0.02, "15p/1touch/2%risk"),
        (20.0, 1, 0.02, "20p/1touch/2%risk"),
        (20.0, 1, 0.03, "20p/1touch/3%risk"),
    ]

    print(f"{'Config':<32} | {'n':>7} | {'WR':>6} | {'PF':>6} | {'MedDays$1M':>10} | {'Prob$1M':>8} | {'MedMaxDD':>9} | {'Ruin%':>6}")
    print("-" * 105)
    for min_pip, touches, risk, label in configs:
        sub = pre[(pre["zone_pips"] >= min_pip) & (pre["touch_num"] <= touches)]
        n = len(sub)
        if n < 30:
            print(f"{label:<32} | {n:>7} | too few trades")
            continue
        wr = 100.0 * (sub["raw_result"] == "WIN").mean()
        gw = sub.loc[sub["net_pips"] > 0, "net_pips"].sum()
        gl = -sub.loc[sub["net_pips"] <= 0, "net_pips"].sum()
        pf = gw / gl if gl > 0 else float("inf")

        med_days, prob, med_dd, ruin = run_two_account_mc(sub, risk_pct=risk, n_sims=500)
        days_str = f"{med_days:.0f}d" if not np.isnan(med_days) else "never"
        print(f"{label:<32} | {n:>7,} | {wr:>5.1f}% | {pf:>6.3f} | {days_str:>10} | {prob:>7.1f}% | {med_dd:>8.1f}% | {ruin:>5.1f}%")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
