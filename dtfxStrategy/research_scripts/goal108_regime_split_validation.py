"""Direct user request (2026-09-07): "address the doubts and do all the tests needed to
find out the optimal robust strategy for live." Starting point: a data forensics finding
that changes everything downstream -- the 1-minute OHLC feed for all four pairs shows a
sharp, synchronized shift in bar microstructure starting 2021-01 and reverting 2026-04,
confirmed independently on EURUSD, GBPUSD, EURGBP, and USDJPY:

  % of near-zero-range (<0.15 pip) 1-minute bars:
    2009-2020 (pre):      ~1-4% typical, noisy natural quote microstructure
    2021-01 to 2026-03:   ~0.05-0.6%, a much "smoother" feed -- likely a different
                           vendor/processing pipeline, NOT a market regime change
    2026-04 onward:       reverts sharply back to ~2-12%, matching the pre-2021 character

This lines up exactly with every anomaly flagged earlier this session: the full-history WR
jump starting 2021 (goal99), and the within-2026 WR decay found in The 2026 Trial's
robustness thirds (the decay boundary sits almost exactly at the April 2026 reversion).
The conclusion: **the 2021-2025 outperformance is substantially a data artifact, not a
strategy edge or a market regime** -- and every parameter chosen anywhere in this project
using that window (including 10-pip zone height, 3-5% risk) needs to be re-derived using
only the trustworthy ("noisy", natural-microstructure) data.

This script re-runs the full parameter selection pipeline honestly:
  1. Split trades (from the SAME zones/entry-gate computed over full, continuous history,
     since trend/structure detection needs that continuity) into three regime buckets by
     entry time: PRE (<2021-01-01, trustworthy, ~11-17yr), SMOOTHED (2021-01-01 to
     2026-04-01, excluded from any selection), POST (>=2026-04-01, trustworthy but short --
     held out purely as a genuine walk-forward validation set, never used for selection).
  2. Sweep minimum zone height using ONLY the PRE bucket (variant A, fixed 1:1R).
  3. Sweep risk_pct under the real two-account architecture using ONLY the PRE bucket, at
     the zone height chosen in step 2.
  4. Validate the resulting frozen configuration against POST (genuine out-of-sample) and
     report SMOOTHED separately for comparison (not used for anything, shown to make the
     size of the artifact visible).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal79_two_account_portfolio as g79
import goal104_optimal_risk_two_account as g104
import goal99_dtfx_base_backtest as g99

PRE_END = pd.Timestamp("2021-01-01")
POST_START = pd.Timestamp("2026-04-01")
MIN_ZONE_SWEEP = [3, 5, 7, 10, 15, 20, 30]
RISK_SWEEP = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]


def bucket_stats(trades, lo=None, hi=None):
    t = trades
    if lo is not None:
        t = t[t["entry_time"] >= lo]
    if hi is not None:
        t = t[t["entry_time"] < hi]
    n = len(t)
    if n == 0:
        return dict(n=0, wr=float("nan"), pf=float("nan"))
    r = t["net_r"].values
    wr = 100.0 * (r > 0).mean()
    gl = -r[r <= 0].sum()
    pf = r[r > 0].sum() / gl if gl > 0 else float("inf")
    return dict(n=n, wr=wr, pf=pf)


def main():
    t0 = time.time()
    print("Loading + building zones once (full continuous history)...")
    cache = g104.load_and_build_zones()

    print(f"\n{'='*78}\nStep 1: minimum zone height, selected on PRE-2021 data ONLY\n{'='*78}")
    sweep_rows = []
    trades_by_height = {}
    for min_pips in MIN_ZONE_SWEEP:
        trades = g104.build_candidate_trades(cache, min_pips)
        trades_by_height[min_pips] = trades
        pre = bucket_stats(trades, hi=PRE_END)
        smoothed = bucket_stats(trades, lo=PRE_END, hi=POST_START)
        post = bucket_stats(trades, lo=POST_START)
        sweep_rows.append(dict(min_pips=min_pips, pre_n=pre["n"], pre_wr=pre["wr"], pre_pf=pre["pf"],
                                smoothed_n=smoothed["n"], smoothed_wr=smoothed["wr"], smoothed_pf=smoothed["pf"],
                                post_n=post["n"], post_wr=post["wr"], post_pf=post["pf"]))
        print(f"  {min_pips:>3}pips  PRE(<2021): n={pre['n']:>9,} WR={pre['wr']:>6.2f}% PF={pre['pf']:>6.3f}  |  "
              f"SMOOTHED(2021-26/03): n={smoothed['n']:>9,} WR={smoothed['wr']:>6.2f}% PF={smoothed['pf']:>6.3f}  |  "
              f"POST(>=26/04): n={post['n']:>6,} WR={post['wr']:>6.2f}% PF={post['pf']:>6.3f}")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(os.path.join(DATA_DIR, "goal108_regime_split_zone_sweep.csv"), index=False)

    best_row = sweep_df.loc[sweep_df["pre_pf"].idxmax()]
    CHOSEN_HEIGHT = int(best_row["min_pips"])
    print(f"\n>>> Best PRE-2021 PF at min_zone={CHOSEN_HEIGHT}pips (PF={best_row['pre_pf']:.3f}, n={int(best_row['pre_n']):,})")

    print(f"\n{'='*78}\nStep 2: risk per trade, two-account architecture, PRE-2021 data ONLY, min_zone={CHOSEN_HEIGHT}pips\n{'='*78}")
    trades_full = trades_by_height[CHOSEN_HEIGHT]
    trades_pre = trades_full[trades_full["entry_time"] < PRE_END].reset_index(drop=True)
    span_years = (trades_pre["exit_time"].max() - trades_pre["entry_time"].min()) / np.timedelta64(365, "D")

    risk_rows = []
    for risk_pct in RISK_SWEEP:
        g79.RISK_PCT = risk_pct
        result, eq_buy, eq_sell, max_dd, _ = g79.simulate_two_account(
            trades_pre, risk_pct=risk_pct, start_per_side=g79.START_PER_SIDE,
            max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=False)
        final_eq = eq_buy + eq_sell
        cagr = (final_eq / 200.0) ** (1.0 / span_years) - 1.0 if final_eq > g79.RUIN_FLOOR else -1.0
        accepted = int(result["accepted"].sum())
        risk_rows.append(dict(risk_pct=risk_pct, accepted=accepted, final_equity=final_eq,
                               max_dd_pct=max_dd, cagr=cagr))
        print(f"  risk={risk_pct:>5.0%}  accepted={accepted:>8,}  final=${final_eq:>16,.2f}  "
              f"maxDD={max_dd:>7.2f}%  CAGR={cagr:>8.2%}")

    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(os.path.join(DATA_DIR, "goal108_regime_split_risk_sweep.csv"), index=False)

    growth_optimal = risk_df.loc[risk_df["cagr"].idxmax()]
    CHOSEN_RISK = float(growth_optimal["risk_pct"])
    HALF_KELLY_RISK = min(RISK_SWEEP, key=lambda r: abs(r - CHOSEN_RISK / 2))
    print(f"\n>>> Growth-optimal risk on PRE-2021 data: {CHOSEN_RISK:.0%} (CAGR={growth_optimal['cagr']:.1%}, "
          f"maxDD={growth_optimal['max_dd_pct']:.1f}%)")
    print(f">>> Half-Kelly practical pick: {HALF_KELLY_RISK:.0%}")

    print(f"\n{'='*78}\nStep 3: walk-forward validation -- frozen config tested on POST-2026-04 (never used for selection)\n{'='*78}")
    trades_post = trades_full[trades_full["entry_time"] >= POST_START].reset_index(drop=True)
    trades_smoothed = trades_full[(trades_full["entry_time"] >= PRE_END) & (trades_full["entry_time"] < POST_START)].reset_index(drop=True)

    for label, chunk in [("SMOOTHED (2021-01 to 2026-03, excluded from selection, shown for comparison)", trades_smoothed),
                          ("POST (>=2026-04, genuine out-of-sample holdout)", trades_post)]:
        print(f"\n--- {label}, min_zone={CHOSEN_HEIGHT}pips ---")
        for risk_pct in [CHOSEN_RISK, HALF_KELLY_RISK]:
            g79.RISK_PCT = risk_pct
            result, eq_buy, eq_sell, max_dd, _ = g79.simulate_two_account(
                chunk, risk_pct=risk_pct, start_per_side=g79.START_PER_SIDE,
                max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=False)
            acc = result[result["accepted"]]
            wr = 100.0 * (acc["net_r"] > 0).mean() if len(acc) else float("nan")
            print(f"  risk={risk_pct:.0%}: n={len(acc):,} WR={wr:.2f}% final=${eq_buy+eq_sell:,.2f} maxDD={max_dd:.2f}%")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
