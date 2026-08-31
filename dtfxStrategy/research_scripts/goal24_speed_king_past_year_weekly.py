"""Speed-King backtest over the past year, week by week, at the "most robust" risk %
found by goal21's Monte Carlo sweep: 3.00% (50x leverage, margin-constrained) --
median 1.054yr to $1M, a genuine convex local optimum (not a fluke of one path; 2.5%
and 3.5% both lose to it from either direction), not the project's default 5%.

Uses the same margin-constrained model as goal20/21/23: $200 start, position sizing
and margin computed the same way PRODLIVETRADING's broker layer actually does it
(risk-based units, capped by 50x leverage), a signal only accepted if its margin fits
inside currently-available margin. Reuses goal20's full true-entry/exit trade stream
(C++-built) -- no need to rebuild it.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal20_speed_king_margin_report as g20

ROBUST_RISK_PCT = 0.03  # goal21_speed_king_risk_mc.py's MC-median-optimal risk level


def main():
    for inst in g20.TRADEABLE:
        g20.rs.load_instrument(inst)
        g20.rs._prep_zones(inst, g20.pg.MIN_RETRACEMENT)

    full = pd.read_csv(os.path.join(DATA_DIR, "goal20_speed_king_full_true_times.csv"),
                        parse_dates=["entry_time", "exit_time"])

    data_end = min(pd.to_datetime(g20.rs.load_instrument(i)[3].index.max()) for i in g20.TRADEABLE)
    window_start = data_end - pd.Timedelta(days=365)
    year = full[(full.entry_time >= window_start) & (full.entry_time <= data_end)].copy()

    print("=" * 92)
    print(f"Speed-King, past year, $200 start, {100*ROBUST_RISK_PCT:.2f}% risk (MC-robust optimum), "
          f"50x leverage, margin-constrained")
    print("=" * 92)
    print(f"Window: {window_start.date()} -> {data_end.date()}  ({len(year):,} signal candidates)\n")

    t, final_eq, max_dd, accepted, trace = g20.simulate_margin_constrained(
        year, start=g20.START, risk_pct=ROBUST_RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
        cap=g20.CAP, track_detail=True)

    acc = t[t.accepted]
    print(f"Signals: {len(t):,}   Accepted: {len(acc):,} ({100*len(acc)/len(t):.1f}%)   "
          f"Rejected (margin-blocked): {(~t.accepted).sum():,} ({100*(~t.accepted).mean():.1f}%)")
    print(f"Max concurrent positions reached: {t.concurrent_at_entry.max()}")
    wr_acc = 100 * acc.is_win.mean()
    w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    print(f"Accepted-trade stats: WR={wr_acc:.2f}%  PF={pf:.3f}")
    print(f"\nStarting equity: ${g20.START:,.2f}   Ending equity: ${final_eq:,.2f}   "
          f"Total return: {100*(final_eq/g20.START - 1):+.1f}%   Max drawdown: {max_dd:.1f}%\n")

    # ---------- week-by-week table, built from the realized CLOSE-event trace ----------
    trace["week"] = trace.close_time.dt.to_period("W")
    rows = []
    eq_before = g20.START
    for wk, sub in trace.groupby("week"):
        n_closed = len(sub)
        wins = (sub.net_r > 0).sum()
        wr = 100 * wins / n_closed if n_closed else float("nan")
        total_r = sub.net_r.sum()
        eq_end = sub.equity.iloc[-1]
        pct_return = 100 * (eq_end / eq_before - 1)
        rows.append(dict(week=str(wk), closed_trades=n_closed, win_rate=round(wr, 1),
                          total_r=round(total_r, 2), pct_return=round(pct_return, 2),
                          equity_end=round(eq_end, 2)))
        eq_before = eq_end

    week_df = pd.DataFrame(rows)
    print(week_df.to_string(index=False))
    week_df.to_csv(os.path.join(DATA_DIR, "goal24_speed_king_past_year_weekly.csv"), index=False)
    t.to_csv(os.path.join(DATA_DIR, "goal24_speed_king_past_year_trades.csv"), index=False)
    print(f"\nSaved: goal24_speed_king_past_year_weekly.csv, goal24_speed_king_past_year_trades.csv")


if __name__ == "__main__":
    main()
