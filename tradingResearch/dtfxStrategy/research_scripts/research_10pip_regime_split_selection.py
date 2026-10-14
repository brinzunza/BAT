"""Select the best 10-pip-zone SL/TP config for PROD deployment (2026-09-08 request), the
right way this time: PRE-2021-only for selection, POST-2026-04 as a genuine untouched
holdout, per goal108's exact boundaries -- learned the hard way earlier this session (the
multi-touch rearming study's pooled full-history numbers overstated its effect ~3x because
45-55% of that pooled sample came from the confirmed 2021-2026.03 data-feed anomaly
window). Reuses compute_combined_comparison_metrics_FIXED.py's corrected full-spread
resolver -- same code, just regime-split before picking a winner instead of after.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import compute_combined_comparison_metrics_FIXED as fx

PRE_END = pd.Timestamp("2021-01-01")
POST_START = pd.Timestamp("2026-04-01")

CONFIGS_10P = [
    (1.0, 1.0, "1.0SL:1.0TP"),
    (1.0, 0.8, "1.0SL:0.8TP"),
    (1.25, 1.0, "1.25SL:1.0TP"),
    (1.0, 1.25, "1.0SL:1.25TP"),
    (1.25, 0.8, "1.25SL:0.8TP"),
    (1.5, 1.0, "1.5SL:1.0TP"),
    (2.0, 0.8, "2.0SL:0.8TP"),
    (3.0, 0.8, "3.0SL:0.8TP"),
]
MIN_ZONE_10P = 10.0


def split_stats(df):
    n = len(df)
    if n == 0:
        return dict(n=0, wr=float("nan"), pf=float("nan"), avg_win=float("nan"))
    wr = 100.0 * (df["net_r"] > 0).mean()
    avg_win = df.loc[df["net_r"] > 0, "net_r"].mean() if (df["net_r"] > 0).any() else 0.0
    gw = df.loc[df["net_r"] > 0, "net_r"].sum()
    gl = -df.loc[df["net_r"] <= 0, "net_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    return dict(n=n, wr=wr, avg_win=avg_win, pf=pf)


def main():
    t0 = time.time()
    cache = fx.g5.load_cache()
    all_min = min(d["times"].min() for d in cache.values())
    all_max = max(d["times"].max() for d in cache.values())
    pre_days = (min(PRE_END, all_max) - all_min).total_seconds() / 86400.0 * (5 / 7)
    post_days = max(0.0, (all_max - max(POST_START, all_min)).total_seconds() / 86400.0) * (5 / 7)
    print(f"Dataset span: {all_min} -> {all_max}")
    print(f"PRE-2021 (~{pre_days:.0f} trading days) / POST-2026-04 holdout (~{post_days:.0f} trading days)\n")

    print(f"{'Config':<14} | {'PRE n':>8} {'PRE WR':>7} {'PRE PF':>7} {'PRE T/D':>8} | "
          f"{'POST n':>7} {'POST WR':>8} {'POST PF':>8} {'POST T/D':>9}")
    print("-" * 100)

    rows = []
    for sl, tp, name in CONFIGS_10P:
        trades = fx.build_trades_correct(cache, MIN_ZONE_10P, sl, tp, spread_pips=1.6)
        pre = trades[trades["entry_time"] < PRE_END]
        post = trades[trades["entry_time"] >= POST_START]
        sp, sq = split_stats(pre), split_stats(post)
        pre_tpd = sp["n"] / pre_days if pre_days > 0 else float("nan")
        post_tpd = sq["n"] / post_days if post_days > 0 else float("nan")
        rows.append(dict(sl=sl, tp=tp, name=name, trades=trades, pre=sp, post=sq,
                          pre_tpd=pre_tpd, post_tpd=post_tpd))
        print(f"{name:<14} | {sp['n']:>8,} {sp['wr']:>6.1f}% {sp['pf']:>7.3f} {pre_tpd:>7.1f}/d | "
              f"{sq['n']:>7,} {sq['wr']:>7.1f}% {sq['pf']:>8.3f} {post_tpd:>8.1f}/d")

    best = max(rows, key=lambda r: r["pre"]["pf"])
    print(f"\n>>> Best PRE-2021 PF: {best['name']} (PF={best['pre']['pf']:.3f}, n={best['pre']['n']:,})")
    print(f">>> Same config on POST-2026-04 holdout: PF={best['post']['pf']:.3f}, WR={best['post']['wr']:.1f}%, n={best['post']['n']:,}")

    print(f"\n{'='*100}\nFull two-account + MC detail for the winning config: {best['name']}\n{'='*100}")
    for label, df in (("PRE-2021 (design)", best["trades"][best["trades"]["entry_time"] < PRE_END]),
                       ("POST-2026-04 (holdout)", best["trades"][best["trades"]["entry_time"] >= POST_START]),
                       ("FULL 2003-2026 (reference, includes contaminated window)", best["trades"])):
        if len(df) == 0:
            print(f"\n  {label}: no trades")
            continue
        res, eqb, eqs, dd, _ = fx.g5.g79.simulate_two_account(
            df, risk_pct=fx.RISK_PCT, start_per_side=fx.START_PER_SIDE,
            max_leverage=fx.MAX_LEVERAGE, cap=fx.ACCOUNT_CAP, track_detail=False)
        acc = res[res["accepted"]]
        n = len(acc)
        wr = 100.0 * (acc["net_r"] > 0).mean() if n else float("nan")
        gw = acc.loc[acc["net_r"] > 0, "net_r"].sum()
        gl = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
        pf = gw / gl if gl > 0 else float("inf")
        n_sims = 1000 if len(df) < 400_000 else 500
        med_days, med_dd = fx.run_mc_fixed(df, n_sims=n_sims, seed=42)
        print(f"\n  {label}")
        print(f"    two-account accepted: {n:,}  WR={wr:.2f}%  PF={pf:.3f}  deterministic maxDD={dd:.2f}%")
        print(f"    MC ({n_sims} sims): median days to $1M = {med_days:.1f}d   median max DD = {med_dd:.2f}%")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
