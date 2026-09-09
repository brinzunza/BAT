"""Regime-split re-validation of research_multi_touch_rearming.py's headline finding
(2026-09-08). That analysis pooled the full 2003-2026 dataset directly, which is exactly
what dtfx.md's own established methodology warns against: a CONFIRMED data-feed artifact
(goal108_regime_split_validation.py) inflates win rate/PF from 2021-01 through 2026-03 on
all 4 pairs. This script re-runs the SAME trade candidates (same zones, same touch-chain
logic -- reuses research_multi_touch_rearming.py's functions directly, no reimplementation)
but splits them into PRE-2021 (<2021-01-01, the trustworthy design sample) and POST-2026-04
(>=2026-04-01, the genuine never-touched holdout) before computing touch-cohort and cap-
sweep stats, exactly matching goal108's boundary convention.

Scope decision: Monte Carlo "days to $1M" is skipped here -- it's a full-history
compounding-speed metric, not the thing regime-splitting tests. The question this script
answers is narrower and more important first: does "win rate increases with touch depth"
survive on PRE-2021 alone and on the POST-2026-04 holdout alone, or was it substantially an
artifact of the same already-identified 2021-2026.03 data problem? WR/PF/EV per split
answers that directly and far more cheaply than re-running 300-sim MC races per split.
"""
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import research_multi_touch_rearming as rmt

PRE_END = pd.Timestamp("2021-01-01")     # goal108's exact boundary
POST_START = pd.Timestamp("2026-04-01")  # goal108's exact boundary

MODES = [
    ("10-Pip (1.0SL:1.0TP)", {"USDJPY": 10.0, "EURUSD": 10.0, "GBPUSD": 10.0, "EURGBP": 10.0}, 1.0, 1.0),
    ("PROD Hybrid (1.0SL:1.0TP)", {"USDJPY": 10.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0}, 1.0, 1.0),
    ("5-Pip (1.0SL:1.0TP)", {"USDJPY": 5.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0}, 1.0, 1.0),
]


def cohort_stats(df):
    n = len(df)
    if n == 0:
        return None
    wr = 100.0 * (df["net_r"] > 0).mean()
    avg_win = df.loc[df["net_r"] > 0, "net_r"].mean() if (df["net_r"] > 0).any() else 0.0
    gw = df.loc[df["net_r"] > 0, "net_r"].sum()
    gl = -df.loc[df["net_r"] <= 0, "net_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    ev = df["net_r"].mean()
    return dict(n=n, wr=wr, avg_win=avg_win, pf=pf, ev=ev)


def print_cohort_table(trades, label):
    total = len(trades)
    print(f"\n  --- Touch cohort, {label} (n={total:,}) ---")
    if total == 0:
        print("    (no trades in this window)")
        return
    for tn in range(1, 7):
        s = cohort_stats(trades[trades["touch_num"] == tn])
        if s is None:
            continue
        pct = 100.0 * s["n"] / total
        print(f"    Touch #{tn:<2} | n={s['n']:>7,} ({pct:>5.1f}%) | WR={s['wr']:>5.1f}% | "
              f"AvgWin=+{s['avg_win']:.2f}R | PF={s['pf']:>5.2f} | EV={s['ev']:>+6.3f}R")
    s7 = cohort_stats(trades[trades["touch_num"] >= 7])
    if s7:
        pct = 100.0 * s7["n"] / total
        print(f"    Touch #7+| n={s7['n']:>7,} ({pct:>5.1f}%) | WR={s7['wr']:>5.1f}% | "
              f"AvgWin=+{s7['avg_win']:.2f}R | PF={s7['pf']:>5.2f} | EV={s7['ev']:>+6.3f}R")


def print_cap_sweep(trades, label, trading_days):
    print(f"\n  --- Cap sweep (deterministic, no MC), {label} ---")
    if len(trades) == 0 or trading_days <= 0:
        print("    (no trades / no time span in this window)")
        return
    for cap in (1, 2, 3, 5, 10, 20):
        sub = trades[trades["touch_num"] <= cap]
        res, _, _, _, _ = rmt.g79.simulate_two_account(
            sub, risk_pct=rmt.RISK_PCT, start_per_side=100.0,
            max_leverage=50.0, cap=10_000_000.0, track_detail=False)
        acc = res[res["accepted"]]
        n_exec = len(acc)
        if n_exec == 0:
            print(f"    Cap {cap:<3} | Execs: 0")
            continue
        tpd = n_exec / trading_days
        wr = 100.0 * (acc["net_r"] > 0).mean()
        gw = acc.loc[acc["net_r"] > 0, "net_r"].sum()
        gl = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
        pf = gw / gl if gl > 0 else float("inf")
        desc = "Single-Touch" if cap == 1 else f"Max {cap} Touches"
        print(f"    Cap {desc:<14} | Execs: {n_exec:<7,} | T/Day: {tpd:>6.1f} | WR: {wr:>5.1f}% | PF: {pf:>5.2f}")


def main():
    cache = rmt.load_cache()
    all_min = min(d["times"].min() for d in cache.values())
    all_max = max(d["times"].max() for d in cache.values())
    pre_days = max(0.0, (min(PRE_END, all_max) - all_min).total_seconds() / 86400.0) * (5.0 / 7.0)
    post_days = max(0.0, (all_max - max(POST_START, all_min)).total_seconds() / 86400.0) * (5.0 / 7.0)
    print(f"Dataset span: {all_min} -> {all_max}")
    print(f"PRE-2021 window: {all_min} -> {PRE_END}  (~{pre_days:.0f} trading days)")
    print(f"POST-2026-04 window: {POST_START} -> {all_max}  (~{post_days:.0f} trading days, the genuine holdout)")
    print(f"SMOOTHED window (2021-01 to 2026-03, EXCLUDED from this validation, confirmed artifact): shown as 'n' only\n")

    for mode_name, min_pips, sl, tp in MODES:
        print("=" * 115)
        print(f"  REGIME-SPLIT VALIDATION: {mode_name}  (Protocol A: strict re-overshoot required)")
        print("=" * 115)
        trades_a = rmt.build_multi_touch_trades(cache, sl, tp, min_pips, require_re_overshoot=True,
                                                 max_touches=20, spread_pips=1.6)
        pre = trades_a[trades_a["entry_time"] < PRE_END]
        smoothed = trades_a[(trades_a["entry_time"] >= PRE_END) & (trades_a["entry_time"] < POST_START)]
        post = trades_a[trades_a["entry_time"] >= POST_START]
        print(f"\n  Split sizes: PRE-2021 n={len(pre):,}  SMOOTHED(excluded) n={len(smoothed):,}  POST-2026-04 n={len(post):,}")

        print_cohort_table(pre, "PRE-2021 (<2021-01-01, trustworthy design sample)")
        print_cohort_table(post, "POST-2026-04 (>=2026-04-01, genuine never-touched holdout)")

        print_cap_sweep(pre, "PRE-2021", pre_days)
        print_cap_sweep(post, "POST-2026-04 (holdout)", post_days)
        print()

    print("=" * 115)
    print("  Total runtime complete.")
    print("=" * 115)


if __name__ == "__main__":
    main()
