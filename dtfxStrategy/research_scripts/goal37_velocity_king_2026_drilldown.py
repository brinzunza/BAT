"""Velocity-King, single continuous backtest over 2026 (Jan 1 -> data end, Aug 31), at
this deployment's exact settings (two $100 sub-accounts, 6% risk, 1.6-pip spread, 50x
leverage, max 1 open position per side). ONE continuous run, not independently reset per
period (that's the separate monthly-reset report) -- month/week/day tables below are all
groupings of the SAME equity curve, so a week's numbers reflect whatever equity the
account actually carried into it from the prior week, same as live trading would.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal30_apex_king_deployment as g30
import goal36_velocity_king_report as g36

RISK_PCT = 0.06
YEAR_START = pd.Timestamp("2026-01-01")


def main():
    print("Building Velocity-King TRUE holdout trade stream...")
    trades = g36.build_velocity_trades()
    data_end = trades.entry_time.max()
    window = trades[(trades.entry_time >= YEAR_START) & (trades.entry_time <= data_end)].copy()
    print(f"2026 window: {YEAR_START} -> {data_end}  ({len(window):,} signal candidates)")

    t, eq_buy, eq_sell, max_dd, trace = g30.simulate_two_account(window, RISK_PCT)
    acc = t[t.accepted]
    wr = 100 * acc.is_win.mean() if len(acc) else float("nan")
    w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    print(f"\nSignals: {len(t):,}  Accepted: {len(acc):,} ({100*len(acc)/len(t):.1f}%)  WR={wr:.2f}%  PF={pf:.3f}")
    print(f"Start $200 combined -> End ${eq_buy+eq_sell:,.2f} ({100*((eq_buy+eq_sell)/200-1):+.1f}%)  "
          f"maxDD={max_dd:.1f}%")

    if not len(trace):
        print("No closed trades in this window.")
        return

    # trace only carries close_time/equity (no net_r) -- attach it from acc, sorted into
    # the same close-time order trace itself was built in (heap-pop = ascending exit_time).
    trace["net_r"] = acc.sort_values("exit_time").net_r.values

    trace["day"] = trace.close_time.dt.date
    trace["week"] = trace.close_time.dt.to_period("W-SUN")  # week ending Sunday
    trace["month"] = trace.close_time.dt.to_period("M")

    daily = trace.groupby("day").agg(trades=("net_r", "size"),
                                      wr=("net_r", lambda x: 100 * (x > 0).mean()),
                                      total_r=("net_r", "sum"),
                                      equity_end=("equity_combined", "last"))
    weekly = trace.groupby("week").agg(trades=("net_r", "size"),
                                        wr=("net_r", lambda x: 100 * (x > 0).mean()),
                                        total_r=("net_r", "sum"),
                                        equity_end=("equity_combined", "last"))
    monthly = trace.groupby("month").agg(trades=("net_r", "size"),
                                          wr=("net_r", lambda x: 100 * (x > 0).mean()),
                                          total_r=("net_r", "sum"),
                                          equity_end=("equity_combined", "last"))

    daily.to_csv(os.path.join(DATA_DIR, "goal37_velocity_2026_daily.csv"))
    weekly.to_csv(os.path.join(DATA_DIR, "goal37_velocity_2026_weekly.csv"))
    monthly.to_csv(os.path.join(DATA_DIR, "goal37_velocity_2026_monthly.csv"))
    trace.to_csv(os.path.join(DATA_DIR, "goal37_velocity_2026_trace.csv"), index=False)

    print("\n=== MONTHLY ===")
    print(monthly.to_string())
    print("\n=== WEEKLY ===")
    print(weekly.to_string())
    print("\n=== DAILY ===")
    print(daily.to_string())

    print("\nSaved: goal37_velocity_2026_{daily,weekly,monthly,trace}.csv")


if __name__ == "__main__":
    main()
