"""Direct user request (2026-09-07): "run the 5 pip min version and risk 3% per trade."

Full two-account backtest ($100 buy + $100 sell, 1 concurrent trade per side, 1.6-pip
spread on every entry, 50x leverage, $2M cap) at min_zone=5 pips, risk_pct=3% -- the same
methodology goal104/goal106/goal108 already established, run fresh for this exact combo
and reported against the honest PRE-2021 / POST-2026-04 split, plus the full-history run
for context.
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

MIN_ZONE_PIPS = 5.0
RISK_PCT = 0.03
PRE_END = pd.Timestamp("2021-01-01")
POST_START = pd.Timestamp("2026-04-01")


def period_profitability(trace):
    """% of calendar days/weeks/months/years where combined equity ended higher than it
    started, from the closed-trade equity trace."""
    if len(trace) == 0:
        return {}
    s = trace.set_index("close_time")["equity_combined"]
    out = {}
    for label, freq in [("day", "D"), ("week", "W"), ("month", "ME"), ("year", "YE")]:
        try:
            r = s.resample(freq).ohlc()
        except ValueError:
            r = s.resample("M" if freq == "ME" else "A").ohlc()
        r = r.dropna()
        if len(r) == 0:
            out[label] = float("nan")
            continue
        out[label] = 100.0 * (r["close"] > r["open"]).mean()
    return out


def run_segment(all_trades, label, lo=None, hi=None):
    t = all_trades
    if lo is not None:
        t = t[t["entry_time"] >= lo]
    if hi is not None:
        t = t[t["entry_time"] < hi]
    if len(t) == 0:
        print(f"\n--- {label}: no trades ---")
        return None
    result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
        t, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
        max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=True)
    acc = result[result["accepted"]]
    n = len(acc)
    wr = 100.0 * (acc["net_r"] > 0).mean() if n else float("nan")
    gross_win = acc.loc[acc["net_r"] > 0, "net_r"].sum()
    gross_loss = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    final = eq_buy + eq_sell
    span_days = (acc["entry_time"].max() - acc["entry_time"].min()).total_seconds() / 86400 if n else float("nan")
    span_years = span_days / 365.25 if span_days else float("nan")
    cagr = (final / 200.0) ** (1.0 / span_years) - 1.0 if (final > g79.RUIN_FLOOR and span_years > 0) else -1.0
    prof = period_profitability(trace)

    print(f"\n--- {label} ---")
    print(f"  span: {acc['entry_time'].min()} -> {acc['entry_time'].max()}  ({span_days:.0f} days)")
    print(f"  accepted trades: {n:,}  ({n/span_days:.2f}/day)   WR={wr:.2f}%  PF={pf:.3f}")
    print(f"  final balance: ${final:,.2f}  (buy=${eq_buy:,.2f} sell=${eq_sell:,.2f})   "
          f"maxDD={max_dd:.2f}%   CAGR={cagr:.1%}")
    print(f"  %% profitable -- day: {prof.get('day', float('nan')):.1f}%  "
          f"week: {prof.get('week', float('nan')):.1f}%  "
          f"month: {prof.get('month', float('nan')):.1f}%  "
          f"year: {prof.get('year', float('nan')):.1f}%")
    return dict(label=label, n=n, span_days=span_days, trades_per_day=n/span_days if span_days else float("nan"),
                wr=wr, pf=pf, final=final, eq_buy=eq_buy, eq_sell=eq_sell, max_dd=max_dd, cagr=cagr, **{
                    f"pct_profitable_{k}": v for k, v in prof.items()})


def main():
    t0 = time.time()
    print("Loading + building zones once (shared across both zone heights)...")
    cache = g104.load_and_build_zones()

    rows = []
    for zone_pips in (5.0, 10.0):
        global MIN_ZONE_PIPS
        MIN_ZONE_PIPS = zone_pips
        trades = g104.build_candidate_trades(cache, zone_pips)
        print(f"\n{'='*90}\nmin_zone={zone_pips:.0f}p, risk={RISK_PCT:.0%} -- "
              f"raw candidates (all 4 pairs, pre-concurrency): {len(trades):,}\n{'='*90}")
        rows.append(run_segment(trades, f"[{zone_pips:.0f}p] FULL HISTORY (2003/2009 -> 2026-09, all regimes mixed)"))
        rows.append(run_segment(trades, f"[{zone_pips:.0f}p] PRE-2021 (honest, trustworthy)", hi=PRE_END))
        rows.append(run_segment(trades, f"[{zone_pips:.0f}p] SMOOTHED 2021-01 to 2026-03 (data artifact window, comparison only)",
                                 lo=PRE_END, hi=POST_START))
        rows.append(run_segment(trades, f"[{zone_pips:.0f}p] POST-2026-04 (genuine walk-forward holdout)", lo=POST_START))
        rows.append(run_segment(trades, f"[{zone_pips:.0f}p] 2026 FRESH START (fresh $200, 2026-01-01 onward)",
                                 lo=pd.Timestamp("2026-01-01")))

    pd.DataFrame([r for r in rows if r]).to_csv(
        os.path.join(DATA_DIR, "goal112_5pip_3pct_report.csv"), index=False)
    print(f"\nSaved -> data/goal112_5pip_3pct_report.csv")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
