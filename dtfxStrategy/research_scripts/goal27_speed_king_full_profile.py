"""Speed-King (N=7.0, R:R=0.20, age_h=0.175, unconditional re-arm, touches=4):
full statistical profile + month-by-month restart backtest, at the user's own
parameters (3.00% risk -- goal21's MC-robust optimum, already the live .env value --
50x leverage margin realism, 1.6-pip spread).

Reuses goal20's exact margin-constrained model unchanged: position sizing and
per-position margin computed the same way PRODLIVETRADING's broker layer actually
does it (broker/oanda_client.py::position_size -- risk-based units, capped by
MAX_LEVERAGE=50x), and a new signal is only ACCEPTED if its required margin fits
inside currently-available margin (equity minus margin already committed to
still-open positions). Spread is already baked into every trade's net_r via
prod_filter_grid_search's 1.6-pip cost model (SPREAD_PIPS_TYPICAL), charged on
stop-outs only (matches the live hybrid-stop design) -- not a separate knob here.

PART 1 -- full-history profile (single continuous compounding stream, $200 start,
3% risk, margin-constrained, project's usual $2M position-sizing cap for long-run
numerical sanity):
  - % of days / weeks / months / years with a net-positive realized return (periods
    with zero trade closes are excluded from this %, not counted as losing -- flagged
    separately)
  - mean and median realized return per trade (both raw R-multiple and % of account
    equity at entry -- these differ whenever the leverage cap binds and shrinks a
    trade below its full 3% risk target)
  - mean and median realized return per active day / week / month / year
  - drawdown profile: the realized single-path underwater curve (max/mean/median
    depth, distinct episodes >10% and their duration/recovery time) PLUS the
    forward-looking Monte Carlo max-drawdown distribution (25,000 sims) at the same
    3% risk, for context beyond this one historical path

PART 2 -- month-by-month, EACH CALENDAR MONTH INDEPENDENTLY RESTARTS AT $200 (not
continuous compounding): only that month's own signal candidates feed the margin
walk, exactly like goal20's single "past month" report repeated for every calendar
month with enough trade candidates, as far back as the data allows (EURUSD/GBPUSD/
EURGBP from 2009-09, USDJPY joins 2018-01). This isolates each month's raw quality,
independent of what equity level a real compounding account would have carried into
it.
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

import dtfx_engine
import goal20_speed_king_margin_report as g20

RISK_PCT = 0.03  # goal21's MC-robust optimum; also the live .env value as of Speed-King's deploy
MIN_TRADES_FOR_STATS = 20  # this project's standing convention (see goal19/goal20's report_window)


def load_full():
    for inst in g20.TRADEABLE:
        g20.rs.load_instrument(inst)
        g20.rs._prep_zones(inst, g20.pg.MIN_RETRACEMENT)
    path = os.path.join(DATA_DIR, "goal20_speed_king_full_true_times.csv")
    if os.path.exists(path):
        full = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    else:
        full = g20.build_speed_king_full()
        full.to_csv(path, index=False)
    return full


# ---------------------------------------------------------------------------
# PART 1: full-history continuous-compounding profile
# ---------------------------------------------------------------------------

def period_stats(trace, freq, label):
    """trace has columns close_time, equity (post-trade, after each accepted position
    closes). Resampling to `freq` with .last().ffill() is exactly correct for a period
    return: equity genuinely doesn't move during a period with zero closes, so a
    forward-filled flat value IS the right marker, not a gap to be interpolated."""
    s = trace.set_index("close_time")["equity"].sort_index()
    resampled = s.resample(freq).last().ffill()
    first_idx = resampled.index[0] - pd.tseries.frequencies.to_offset(freq)
    resampled = pd.concat([pd.Series([g20.START], index=[first_idx]), resampled])
    rets = resampled.pct_change().dropna() * 100
    counts = s.resample(freq).count().reindex(rets.index, fill_value=0)
    active = rets[counts > 0]
    flat = int((counts == 0).sum())
    return dict(
        label=label, n_periods_active=len(active), n_periods_flat=flat,
        pct_profitable=100 * (active > 0).mean() if len(active) else float("nan"),
        mean_return_pct=active.mean() if len(active) else float("nan"),
        median_return_pct=active.median() if len(active) else float("nan"),
        best_pct=active.max() if len(active) else float("nan"),
        worst_pct=active.min() if len(active) else float("nan"),
    )


def drawdown_episodes(daily_equity, min_depth_pct=10.0):
    """daily_equity: a daily-frequency, forward-filled equity Series. Returns a list of
    episodes where the underwater drawdown from the running peak exceeded min_depth_pct,
    each with (start=peak date, trough date, depth%, recovery date or None if ongoing)."""
    running_max = daily_equity.cummax()
    dd = (daily_equity - running_max) / running_max * 100
    episodes = []
    in_dd = False
    ep_start = ep_trough_date = None
    ep_trough = 0.0
    for date, d in dd.items():
        if d < -1e-9:
            if not in_dd:
                in_dd = True
                ep_start = date
                ep_trough = 0.0
                ep_trough_date = date
            if d < ep_trough:
                ep_trough = d
                ep_trough_date = date
        else:
            if in_dd:
                if ep_trough <= -min_depth_pct:
                    episodes.append(dict(peak_date=ep_start, trough_date=ep_trough_date,
                                          depth_pct=ep_trough, recovered_date=date,
                                          duration_days=(date - ep_start).days))
                in_dd = False
    if in_dd and ep_trough <= -min_depth_pct:
        episodes.append(dict(peak_date=ep_start, trough_date=ep_trough_date,
                              depth_pct=ep_trough, recovered_date=None,
                              duration_days=(daily_equity.index[-1] - ep_start).days))
    return episodes, dd


def part1_full_history_profile(full):
    print("=" * 92)
    print(f"PART 1: Full-history profile -- $200 start, {100*RISK_PCT:.2f}% risk, 50x leverage,")
    print(f"        margin-constrained, single continuous compounding stream, $2M sizing cap")
    print("=" * 92)

    t, final_eq, max_dd_pct, accepted, trace = g20.simulate_margin_constrained(
        full, start=g20.START, risk_pct=RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
        cap=g20.CAP, track_detail=True)
    acc = t[t.accepted].copy()

    print(f"\nSignal candidates: {len(t):,}   Accepted (margin allowed): {len(acc):,} "
          f"({100*len(acc)/len(t):.1f}%)   Rejected: {(~t.accepted).sum():,}")
    wr = 100 * acc.is_win.mean()
    w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    print(f"Accepted-trade stats: WR={wr:.2f}%  PF={pf:.3f}")
    print(f"Start ${g20.START:,.2f} -> End ${final_eq:,.2f}  ({(final_eq/g20.START)-1:+,.1%} over "
          f"{(full.entry_time.max()-full.entry_time.min()).days/365.25:.1f} years)")

    # per-trade returns: R-multiple (pure edge) vs % of equity at entry (real dollars,
    # accounting for whenever the leverage cap shrinks a trade below full risk_pct).
    # Re-derive risk_dollars the same way simulate_margin_constrained did internally
    # (it isn't one of the columns returned in `t`, only equity/margin at entry are).
    risk_dollars = []
    for row in acc.itertuples():
        _, r_dollars, _ = g20.position_size_and_margin(
            row.instrument, min(row.equity_at_entry, g20.CAP), row.entry_price, row.stop_risk,
            RISK_PCT, g20.MAX_LEVERAGE, None)
        risk_dollars.append(r_dollars)
    acc["risk_dollars"] = risk_dollars
    acc["pnl_dollars"] = acc.risk_dollars * acc.net_r
    acc["trade_return_pct"] = 100 * acc.pnl_dollars / acc.equity_at_entry

    print(f"\nPer-trade returns (n={len(acc):,} accepted trades):")
    print(f"  Net R-multiple:      mean={acc.net_r.mean():+.3f}R   median={acc.net_r.median():+.3f}R")
    print(f"  % of equity at entry: mean={acc.trade_return_pct.mean():+.3f}%   "
          f"median={acc.trade_return_pct.median():+.3f}%")
    print(f"  Fraction of trades sized below full {100*RISK_PCT:.1f}% risk (leverage-capped): "
          f"{100*(acc.risk_dollars < acc.equity_at_entry*RISK_PCT*0.999).mean():.1f}%")

    print(f"\nPeriod returns (active periods only -- a period with zero trade closes is excluded,")
    print(f" not counted as a loss; its count is shown separately):")
    period_rows = []
    for freq, lbl in [("D", "Day"), ("W", "Week"), ("M", "Month"), ("Y", "Year")]:
        r = period_stats(trace, freq, lbl)
        period_rows.append(r)
        print(f"  {lbl:6s}: {r['pct_profitable']:5.1f}% profitable  "
              f"(n_active={r['n_periods_active']:,}, n_flat={r['n_periods_flat']:,})   "
              f"mean={r['mean_return_pct']:+7.2f}%   median={r['median_return_pct']:+7.2f}%   "
              f"best={r['best_pct']:+8.2f}%   worst={r['worst_pct']:+8.2f}%")
    pd.DataFrame(period_rows).to_csv(os.path.join(DATA_DIR, "goal27_period_stats.csv"), index=False)

    # drawdown profile: realized single-path (daily-resampled) + forward MC at same risk level
    daily_eq = trace.set_index("close_time")["equity"].resample("D").last().ffill()
    daily_eq = pd.concat([pd.Series([g20.START], index=[daily_eq.index[0] - pd.Timedelta(days=1)]),
                           daily_eq])
    episodes, dd_curve = drawdown_episodes(daily_eq, min_depth_pct=10.0)
    print(f"\nDrawdown profile (realized single historical path, daily marks):")
    print(f"  Max drawdown ever reached: {dd_curve.min():.1f}%")
    print(f"  Mean/median of the underwater curve (all days, including 0% at new highs): "
          f"mean={dd_curve.mean():.2f}%  median={dd_curve.median():.2f}%")
    print(f"  Distinct drawdown episodes exceeding -10%: {len(episodes)}")
    recovered = [e for e in episodes if e["recovered_date"] is not None]
    if recovered:
        durs = [e["duration_days"] for e in recovered]
        print(f"    Of these, {len(recovered)} fully recovered -- time-to-recover (peak to new high): "
              f"mean={np.mean(durs):.0f}d  median={np.median(durs):.0f}d  worst={max(durs)}d")
    ongoing = [e for e in episodes if e["recovered_date"] is None]
    if ongoing:
        print(f"    {len(ongoing)} episode(s) still open as of the end of data (most recent: "
              f"{ongoing[-1]['depth_pct']:.1f}%, {ongoing[-1]['duration_days']}d and counting)")
    print(f"\n  Worst 5 episodes by depth:")
    for e in sorted(episodes, key=lambda e: e["depth_pct"])[:5]:
        rec = e["recovered_date"].date() if e["recovered_date"] is not None else "ongoing"
        print(f"    {e['peak_date'].date()} -> trough {e['trough_date'].date()} "
              f"({e['depth_pct']:.1f}%) -> recovered {rec}  [{e['duration_days']}d]")

    print(f"\nForward-looking Monte Carlo max-drawdown distribution (25,000 sims, same "
          f"{100*RISK_PCT:.1f}% risk, $2M sizing cap, standard sequential-stream methodology):")
    net_r_all = full.net_r.values.astype(np.float64)
    from goal10_cana_nt_mc_capped import BLOCK, MAX_TRADES, CAP, T1, T10, RUIN_FLOOR
    t1_raw, t10_raw, mc_dd, ruined = dtfx_engine.run_mc_capped(
        net_r_all, 25_000, BLOCK, MAX_TRADES, 21, start=g20.START, risk_frac=RISK_PCT,
        cap=CAP, target1=T1, target2=T10, ruin_floor=RUIN_FLOOR)
    mc_dd_pct = mc_dd * 100
    for p in [10, 25, 50, 75, 90]:
        print(f"    P{p}: {np.percentile(mc_dd_pct, p):.1f}%")
    print(f"    Worst of 25,000: {mc_dd_pct.min():.1f}%   %% ruined: {100*ruined.mean():.3f}%")

    return dict(acc=acc, trace=trace, period_rows=period_rows, episodes=episodes,
                dd_curve=dd_curve, mc_dd_pct=mc_dd_pct, final_eq=final_eq)


# ---------------------------------------------------------------------------
# PART 2: month-by-month, each month restarts at $200
# ---------------------------------------------------------------------------

def part2_monthly_restart(full):
    print("\n" + "=" * 92)
    print(f"PART 2: Month-by-month, EACH MONTH RESTARTS AT $200 -- {100*RISK_PCT:.2f}% risk, "
          f"50x leverage,")
    print(f"        margin-constrained, 1.6-pip spread (already in net_r). As far back as the")
    print(f"        data allows (EURUSD/GBPUSD/EURGBP from 2009-09; USDJPY joins 2018-01).")
    print("=" * 92)

    data_end = min(pd.to_datetime(g20.rs.load_instrument(i)[3].index.max()) for i in g20.TRADEABLE)
    months = pd.period_range(full.entry_time.min().to_period("M"), data_end.to_period("M"), freq="M")

    rows = []
    for m in months:
        m_start = m.start_time
        m_end = min(m.end_time, data_end)
        window = full[(full.entry_time >= m_start) & (full.entry_time <= m_end)].copy()
        partial = m_end < m.end_time
        if len(window) < MIN_TRADES_FOR_STATS:
            rows.append(dict(month=str(m), trades=len(window), insufficient=True, partial=partial))
            continue

        t, final_eq, max_dd_pct, accepted, _ = g20.simulate_margin_constrained(
            window, start=g20.START, risk_pct=RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
            cap=None, track_detail=True)
        acc = t[t.accepted]
        if len(acc):
            wr = 100 * acc.is_win.mean()
            w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
            pf = w / l if l > 0 else float("inf")
            max_conc = int(t.concurrent_at_entry.max())
        else:
            wr = pf = float("nan"); max_conc = 0
        rows.append(dict(
            month=str(m), trades=len(t), accepted=len(acc),
            accept_pct=round(100*len(acc)/len(t), 1) if len(t) else float("nan"),
            win_rate=round(wr, 2), pf=round(pf, 3) if np.isfinite(pf) else pf,
            start_equity=g20.START, end_equity=round(final_eq, 2),
            return_pct=round(100*(final_eq/g20.START - 1), 2),
            max_dd_pct=round(max_dd_pct, 1), max_concurrent=max_conc,
            insufficient=False, partial=partial,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal27_monthly_restart.csv"), index=False)

    valid = df[~df.insufficient].copy()
    print(f"\n{len(df)} calendar months spanned; {len(valid)} with >= {MIN_TRADES_FOR_STATS} "
          f"trade candidates (the rest -- mostly the very first partial month(s) -- are marked "
          f"insufficient in the CSV, not silently dropped).")

    complete = valid[~valid.partial]  # exclude the current in-progress month from the headline stats
    pct_profitable = 100 * (complete.return_pct > 0).mean()
    print(f"\nOf {len(complete)} COMPLETE months: {pct_profitable:.1f}% profitable "
          f"({int((complete.return_pct > 0).sum())} up, {int((complete.return_pct <= 0).sum())} down)")
    print(f"  Monthly return (restart-at-$200): mean={complete.return_pct.mean():+.2f}%  "
          f"median={complete.return_pct.median():+.2f}%")
    print(f"  Best month:  {complete.loc[complete.return_pct.idxmax(), 'month']} "
          f"({complete.return_pct.max():+.1f}%)")
    print(f"  Worst month: {complete.loc[complete.return_pct.idxmin(), 'month']} "
          f"({complete.return_pct.min():+.1f}%)")
    print(f"  Mean max-drawdown within a month: {complete.max_dd_pct.mean():.1f}%   "
          f"worst single month: {complete.max_dd_pct.min():.1f}%")
    print(f"  Mean accept rate (margin-allowed / signal candidates): {complete.accept_pct.mean():.1f}%")
    print(f"  Mean max concurrent positions reached per month: {complete.max_concurrent.mean():.1f}  "
          f"(overall max: {complete.max_concurrent.max()})")

    print(f"\nFull table written to data/goal27_monthly_restart.csv ({len(df)} rows).")
    print(df.to_string(index=False))
    return df


def main():
    print("Loading Speed-King full trade log (true entry/exit times, C++ engine)...", flush=True)
    full = load_full()
    print(f"  {len(full):,} candidate signals, {full.entry_time.min()} -> {full.entry_time.max()}\n")

    p1 = part1_full_history_profile(full)
    p2 = part2_monthly_restart(full)


if __name__ == "__main__":
    main()
