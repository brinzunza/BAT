"""Speed-King: ROLLING 3-month backtest, each window independently restarting at $200 --
2.0-pip spread, 3.00% risk, 50x leverage, $500,000 position-sizing cap (same assumptions
as goal34's Monte Carlo). "Rolling" here means stepped monthly with a 3-month window
(heavily overlapping), not calendar-quarter buckets -- this answers "if I'd started this
strategy fresh with $200 at ANY point in the last 16.9 years and run it for exactly 3
months, what would have happened," not just 4 fixed quarters per year.

Reuses goal20's exact margin-constrained model (position sizing/margin computed like
PRODLIVETRADING's broker layer, a signal only accepted if its margin fits inside
currently-available margin) and goal29's spread-rebuild helper (zone detection and win/loss
are spread-independent; only the net_r cost model changes).
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal20_speed_king_margin_report as g20
from goal29_speed_king_spread_sensitivity import build_raw, with_spread

rs, pg, TRADEABLE = g20.rs, g20.pg, g20.TRADEABLE

START = 200.0
SPREAD_PIPS = 2.0
RISK_PCT = 0.03
MAX_LEVERAGE = 50.0
CAP = 500_000.0
WINDOW_MONTHS = 3
STEP_MONTHS = 1
MIN_TRADES_FOR_STATS = 20


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    print("Building Speed-King's population at 2.0-pip spread...", flush=True)
    raw = build_raw()
    full = with_spread(raw, SPREAD_PIPS)
    data_start, data_end = full.entry_time.min(), full.entry_time.max()
    print(f"  {len(full):,} trades, {data_start} -> {data_end}\n")

    rows = []
    cursor = data_start
    while cursor + pd.DateOffset(months=WINDOW_MONTHS) <= data_end + pd.Timedelta(days=1):
        win_start = cursor
        win_end = min(cursor + pd.DateOffset(months=WINDOW_MONTHS), data_end)
        window = full[(full.entry_time >= win_start) & (full.entry_time <= win_end)]
        partial = win_end < cursor + pd.DateOffset(months=WINDOW_MONTHS)
        if len(window) >= MIN_TRADES_FOR_STATS:
            t, final_eq, max_dd_pct, accepted, _ = g20.simulate_margin_constrained(
                window, start=START, risk_pct=RISK_PCT, max_leverage=MAX_LEVERAGE,
                cap=CAP, track_detail=True)
            acc = t[t.accepted]
            if len(acc):
                wr = 100 * acc.is_win.mean()
                w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
                pf = w / l if l > 0 else float("inf")
                max_conc = int(t.concurrent_at_entry.max())
                accept_pct = 100 * len(acc) / len(t)
            else:
                wr = pf = float("nan"); max_conc = 0; accept_pct = float("nan")
            rows.append(dict(
                window_start=win_start.date(), window_end=win_end.date(),
                trades=len(t), accepted=len(acc), accept_pct=round(accept_pct, 1),
                win_rate=round(wr, 2), pf=round(pf, 3) if np.isfinite(pf) else pf,
                end_equity=round(final_eq, 2), return_pct=round(100 * (final_eq / START - 1), 2),
                max_dd_pct=round(max_dd_pct, 1), max_concurrent=max_conc, partial=partial,
            ))
        cursor += pd.DateOffset(months=STEP_MONTHS)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal35_rolling_3m_restart.csv"), index=False)

    complete = df[~df.partial]
    pct_profitable = 100 * (complete.return_pct > 0).mean()
    print("=" * 94)
    print(f"ROLLING {WINDOW_MONTHS}-MONTH BACKTEST, restart at $200 each window "
          f"(stepped every {STEP_MONTHS} month)")
    print(f"${START:.0f} start, {SPREAD_PIPS}-pip spread, {100*RISK_PCT:.1f}% risk, {MAX_LEVERAGE:.0f}x "
          f"leverage, ${CAP:,.0f} sizing cap")
    print("=" * 94)
    print(f"\n{len(df)} rolling windows spanned ({len(complete)} complete, "
          f"{len(df)-len(complete)} partial/truncated at the end of data)")
    print(f"\n%% profitable: {pct_profitable:.1f}%  ({int((complete.return_pct>0).sum())} up, "
          f"{int((complete.return_pct<=0).sum())} down)")
    print(f"Return: mean={complete.return_pct.mean():+.2f}%  median={complete.return_pct.median():+.2f}%")
    print(f"Best window:  {complete.loc[complete.return_pct.idxmax(),'window_start']} -> "
          f"{complete.loc[complete.return_pct.idxmax(),'window_end']} "
          f"({complete.return_pct.max():+.1f}%)")
    print(f"Worst window: {complete.loc[complete.return_pct.idxmin(),'window_start']} -> "
          f"{complete.loc[complete.return_pct.idxmin(),'window_end']} "
          f"({complete.return_pct.min():+.1f}%)")
    print(f"Max drawdown within a window: mean={complete.max_dd_pct.mean():.1f}%  "
          f"worst={complete.max_dd_pct.min():.1f}%")
    print(f"Mean accept rate: {complete.accept_pct.mean():.1f}%   "
          f"Mean max concurrent: {complete.max_concurrent.mean():.1f} (overall max: "
          f"{complete.max_concurrent.max()})")

    print(f"\nReturn percentiles across all {len(complete)} rolling windows:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  P{p}: {np.percentile(complete.return_pct, p):+.1f}%")

    print(f"\nWorst 10 windows by return:")
    for _, r in complete.nsmallest(10, "return_pct").iterrows():
        print(f"  {r.window_start} -> {r.window_end}: {r.return_pct:+.1f}%  (max DD {r.max_dd_pct:.1f}%)")

    print(f"\nFull table: data/goal35_rolling_3m_restart.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
