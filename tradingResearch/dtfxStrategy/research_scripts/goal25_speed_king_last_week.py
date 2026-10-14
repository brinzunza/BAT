"""Speed-King, last 7 days of data, $200 start, 3.00% risk (goal21's MC-robust optimum),
50x leverage, margin-constrained (goal20's model). Day-by-day breakdown -- a 7-day window
is too short for a meaningful week-by-week table.
"""
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal20_speed_king_margin_report as g20

ROBUST_RISK_PCT = 0.03


def main():
    for inst in g20.TRADEABLE:
        g20.rs.load_instrument(inst)
        g20.rs._prep_zones(inst, g20.pg.MIN_RETRACEMENT)

    full = pd.read_csv(os.path.join(DATA_DIR, "goal20_speed_king_full_true_times.csv"),
                        parse_dates=["entry_time", "exit_time"])
    data_end = full.entry_time.max()
    window_start = data_end - pd.Timedelta(days=7)
    week = full[(full.entry_time >= window_start) & (full.entry_time <= data_end)].copy()

    print(f"Window: {window_start} -> {data_end}  ({len(week):,} signal candidates)")

    t, final_eq, max_dd, accepted, trace = g20.simulate_margin_constrained(
        week, start=g20.START, risk_pct=ROBUST_RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
        cap=g20.CAP, track_detail=True)

    acc = t[t.accepted]
    print(f"Signals: {len(t):,}  Accepted: {len(acc):,} ({100*len(acc)/len(t):.1f}%)  "
          f"Rejected: {(~t.accepted).sum():,}")
    print(f"Max concurrent: {t.concurrent_at_entry.max()}")
    wr = 100 * acc.is_win.mean()
    w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    print(f"Accepted WR={wr:.2f}% PF={pf:.3f}")
    print(f"Start ${g20.START:.2f} -> End ${final_eq:.2f} ({100*(final_eq/g20.START-1):+.1f}%) "
          f"maxDD={max_dd:.1f}%\n")

    trace["day"] = trace.close_time.dt.date
    daily = trace.groupby("day").agg(
        trades=("net_r", "size"), wr=("net_r", lambda x: 100 * (x > 0).mean()),
        total_r=("net_r", "sum"), equity_end=("equity", "last"))
    print(daily)

    week.to_csv(os.path.join(DATA_DIR, "goal25_speed_king_last_week_trades.csv"), index=False)
    t.to_csv(os.path.join(DATA_DIR, "goal25_speed_king_last_week_margin.csv"), index=False)
    trace.to_csv(os.path.join(DATA_DIR, "goal25_speed_king_last_week_trace.csv"), index=False)
    print("\nSaved: goal25_speed_king_last_week_{trades,margin,trace}.csv")


if __name__ == "__main__":
    main()
