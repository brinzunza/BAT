"""Speed-King: past-month backtest, 2026-so-far weekly backtest, and Monte Carlo
drawdown/speed-to-$1M -- all at goal21's MC-robust risk level (3.00%, not the
project's default 5%), on the freshest data (LSE vault, current through 2026-08-31).

Past-month and 2026 backtests use the margin-constrained model (goal20/goal21):
$200 start, position sizing/margin computed like PRODLIVETRADING's broker layer,
50x leverage, a signal only accepted if its margin fits inside currently-available
margin. The Monte Carlo uses this project's standard methodology (run_mc_capped /
dtfx_engine.run_mc_capped, single sequential stream, $2M position-sizing cap) at the
same 3.00% risk -- the same "median years to $1M" metric used everywhere else in
this project, not margin-constrained (that would need goal21's much more expensive
block-bootstrap-of-real-timing approach; kept as the standard, fast methodology here
since that's what "time to $1M" means throughout the rest of this file).
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal20_speed_king_margin_report as g20

ROBUST_RISK_PCT = 0.03


def part1_past_month(full, data_end):
    print("=" * 92)
    print(f"PART 1: Past month, $200 start, {100*ROBUST_RISK_PCT:.2f}% risk (MC-robust), "
          f"50x leverage, margin-constrained")
    print("=" * 92)
    window_start = data_end - pd.Timedelta(days=30)
    month = full[(full.entry_time >= window_start) & (full.entry_time <= data_end)].copy()
    print(f"Window: {window_start} -> {data_end}  ({len(month):,} signal candidates)")

    t, final_eq, max_dd, accepted, trace = g20.simulate_margin_constrained(
        month, start=g20.START, risk_pct=ROBUST_RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
        cap=g20.CAP, track_detail=True)
    acc = t[t.accepted]
    wr = 100 * acc.is_win.mean()
    w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")

    print(f"\nSignals: {len(t):,}   Accepted: {len(acc):,} ({100*len(acc)/len(t):.1f}%)   "
          f"Rejected: {(~t.accepted).sum():,} ({100*(~t.accepted).mean():.1f}%)")
    print(f"Max concurrent positions: {t.concurrent_at_entry.max()}")
    print(f"Accepted-trade WR={wr:.2f}%  PF={pf:.3f}")
    print(f"Start ${g20.START:,.2f} -> End ${final_eq:,.2f} ({100*(final_eq/g20.START-1):+.1f}%)  "
          f"Max drawdown={max_dd:.1f}%")

    t.to_csv(os.path.join(DATA_DIR, "goal26_speed_king_past_month_margin.csv"), index=False)
    return t, trace


def part2_2026_weekly(full):
    print("\n" + "=" * 92)
    print(f"PART 2: 2026 so far, week by week, $200 start, {100*ROBUST_RISK_PCT:.2f}% risk "
          f"(MC-robust), 50x leverage, margin-constrained")
    print("=" * 92)
    year = full[full.entry_time.dt.year == 2026].copy()
    print(f"{len(year):,} signal candidates in 2026 so far")

    t, final_eq, max_dd, accepted, trace = g20.simulate_margin_constrained(
        year, start=g20.START, risk_pct=ROBUST_RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
        cap=g20.CAP, track_detail=True)
    acc = t[t.accepted]
    print(f"\nSignals: {len(t):,}   Accepted: {len(acc):,} ({100*len(acc)/len(t):.1f}%)   "
          f"Max concurrent: {t.concurrent_at_entry.max()}")
    print(f"Start ${g20.START:,.2f} -> End ${final_eq:,.2f} ({100*(final_eq/g20.START-1):+.1f}%)  "
          f"Max drawdown={max_dd:.1f}%\n")

    trace["week"] = trace.close_time.dt.to_period("W")
    rows = []
    eq_before = g20.START
    for wk, sub in trace.groupby("week"):
        n_closed = len(sub)
        wr = 100 * (sub.net_r > 0).sum() / n_closed if n_closed else float("nan")
        total_r = sub.net_r.sum()
        eq_end = sub.equity.iloc[-1]
        pct_return = 100 * (eq_end / eq_before - 1)
        rows.append(dict(week=str(wk), closed_trades=n_closed, win_rate=round(wr, 1),
                          total_r=round(total_r, 2), pct_return=round(pct_return, 2),
                          equity_end=round(eq_end, 2)))
        eq_before = eq_end
    week_df = pd.DataFrame(rows)
    print(week_df.to_string(index=False))
    week_df.to_csv(os.path.join(DATA_DIR, "goal26_speed_king_2026_weekly.csv"), index=False)
    t.to_csv(os.path.join(DATA_DIR, "goal26_speed_king_2026_margin.csv"), index=False)


def part3_monte_carlo(full):
    print("\n" + "=" * 92)
    print(f"PART 3: Monte Carlo (25,000 sims, standard sequential-stream methodology, "
          f"$2M cap, {100*ROBUST_RISK_PCT:.2f}% risk) -- drawdown and speed to $1M/$10M")
    print("=" * 92)
    net_r = full.net_r.values.astype(np.float64)
    years = (full.entry_time.max() - full.entry_time.min()).days / 365.25
    tpy = len(full) / years
    t1_raw, t10_raw, max_dd, ruined = dtfx_engine.run_mc_capped(
        net_r, 25_000, g20.BLOCK, g20.MAX_TRADES, 26, start=g20.START, risk_frac=ROBUST_RISK_PCT,
        cap=g20.CAP, target1=g20.T1, target2=g20.T10, ruin_floor=g20.RUIN_FLOOR)
    t1_hit = t1_raw[t1_raw > 0]; t10_hit = t10_raw[t10_raw > 0]
    y1 = t1_hit / tpy; y10 = t10_hit / tpy
    dd_pct = max_dd * 100

    print(f"Trade universe: {len(full):,} trades ({tpy:,.0f}/yr), {years:.1f}yr history")
    print(f"\nTime to $1M: quickest={y1.min():.3f}yr  slowest={y1.max():.3f}yr  "
          f"avg={y1.mean():.3f}yr  median={np.median(y1):.3f}yr")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(y1, p):.3f} yrs")
    print(f"\nTime to $10M ($2M cap): median={np.median(y10):.3f}yr "
          f"(n={len(y10):,} of 25,000 reached it, {100*len(y10)/25000:.1f}%)")
    print(f"\nMax drawdown: median={np.median(dd_pct):.1f}%  mean={dd_pct.mean():.1f}%  "
          f"worst={dd_pct.min():.1f}%  best={dd_pct.max():.1f}%")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(dd_pct, p):.1f}%")
    print(f"\n%% reaching $1M: {100*len(y1)/25000:.2f}%   %% ruined: {100*ruined.mean():.3f}%")

    out = pd.DataFrame({
        "years_to_1m": np.where(t1_raw > 0, t1_raw / tpy, np.nan),
        "years_to_10m": np.where(t10_raw > 0, t10_raw / tpy, np.nan),
        "max_drawdown_pct": dd_pct, "ruined": ruined,
    })
    out.to_csv(os.path.join(DATA_DIR, "goal26_speed_king_mc_25000.csv"), index=False)


def main():
    for inst in g20.TRADEABLE:
        g20.rs.load_instrument(inst)
        g20.rs._prep_zones(inst, g20.pg.MIN_RETRACEMENT)

    full = pd.read_csv(os.path.join(DATA_DIR, "goal20_speed_king_full_true_times.csv"),
                        parse_dates=["entry_time", "exit_time"])
    data_end = full.entry_time.max()
    print(f"Loaded {len(full):,} Speed-King trades, {full.entry_time.min()} -> {data_end}\n")

    part1_past_month(full, data_end)
    part2_2026_weekly(full)
    part3_monte_carlo(full)


if __name__ == "__main__":
    main()
