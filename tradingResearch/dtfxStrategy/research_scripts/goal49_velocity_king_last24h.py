"""Velocity-King, last 24 hours of AVAILABLE data (data currently ends 2026-08-31 21:58 --
stale by however long it's been since then; LSE_API_KEY isn't available locally to refresh
it, see fetch_lse_update_tradeable.py). Same exact deployment model as goal36/goal37: two
independent $100 sub-accounts, 6% risk, max 1 open position per side, 50x leverage,
1.6-pip spread.
"""
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal30_apex_king_deployment as g30
import goal36_velocity_king_report as g36

RISK_PCT = 0.06


def main():
    print("Building Velocity-King TRUE trade stream (current local data)...")
    trades = g36.build_velocity_trades()
    data_end = trades.entry_time.max()
    window_start = data_end - pd.Timedelta(hours=24)
    window = trades[(trades.entry_time >= window_start) & (trades.entry_time <= data_end)].copy()
    print(f"Local data currently ends: {data_end}")
    print(f"Last-24h window (of available data): {window_start} -> {data_end}  "
          f"({len(window):,} signal candidates)")

    if len(window) < 5:
        print("Too few signals in this window to report meaningfully.")
        return

    t, eq_buy, eq_sell, max_dd, trace = g30.simulate_two_account(window, RISK_PCT)
    acc = t[t.accepted]
    wr = 100 * acc.is_win.mean() if len(acc) else float("nan")
    w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")

    print(f"\nSignals: {len(t):,}  Accepted: {len(acc):,} ({100*len(acc)/len(t):.1f}%)")
    print(f"Accepted WR={wr:.2f}%  PF={pf:.3f}")
    print(f"Start: $100 buy + $100 sell = $200 combined")
    print(f"End:   ${eq_buy:,.2f} buy + ${eq_sell:,.2f} sell = ${eq_buy+eq_sell:,.2f} combined "
          f"({100*((eq_buy+eq_sell)/200-1):+.1f}%)")
    print(f"Max drawdown (combined equity): {max_dd:.1f}%")

    print(f"\nPer-instrument breakdown:")
    for inst, sub in acc.groupby("instrument"):
        print(f"  {inst}: n={len(sub)}  WR={100*sub.is_win.mean():.1f}%")


if __name__ == "__main__":
    main()
