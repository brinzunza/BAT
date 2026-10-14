"""Speed-King (N=7.0, R:R=0.20, age_h=0.175, unconditional re-arm, touches=4) report:

1. Past-month backtest with REALISTIC margin-based concurrency: $200 start, 5% risk
   per trade, 1.6-pip spread, position sizing and per-position margin computed the
   same way PRODLIVETRADING's broker layer actually does it
   (broker/oanda_client.py::position_size -- risk-based units, capped by
   MAX_LEVERAGE=50), and a new trade signal is only ACCEPTED if its required margin
   fits inside currently-available margin (equity minus margin already committed to
   still-open positions). This is a materially different (and more realistic)
   concurrency model than every other backtest in this project, which treats trades
   as a single sequential, unlimited-concurrency stream.

2. A standard block-bootstrap Monte Carlo (this project's usual methodology --
   run_mc_capped, single sequential stream, NOT margin-constrained) on the full
   multi-year trade history, reporting drawdown and speed to $1M/$10M.

3. Weekly-returns-with-equity backtest table for 2026 (standard, unconstrained
   sequential compounding, matching how every other "backtest over <year>" report in
   this project has been built).
"""
import os
import sys
import heapq
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal5_rr_sweep as g5
from goal10_cana_nt_mc_capped import BLOCK, MAX_TRADES, CAP, T1, T10, RUIN_FLOOR

rs, pg, TRADEABLE = g5.rs, g5.pg, g5.TRADEABLE

SK_N, SK_RR, SK_AGE_H, SK_BOUNCE, SK_TOUCHES = 7.0, 0.20, 0.175, False, 4

START = 200.0
RISK_PCT = 0.05
MAX_LEVERAGE = 50.0
GBP_USD_APPROX = 1.36  # matches PRODLIVETRADING/broker/oanda_client.py's EURGBP sizing approximation


def build_speed_king_full():
    """Full-history Speed-King trades with true entry_time, exit_time, entry price (mid).
    Uses the C++ engine's simulate_generic_ex directly (dtfx_engine) -- same zone-detection
    and bar-walk as everywhere else in this project, plus the exit/resolution bar this
    margin/concurrency simulation needs to know when a position actually frees its margin."""
    frames = []
    for inst in TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (SK_N * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(SK_AGE_H * 3600 * 1e9), SK_N, SK_N * SK_RR, SK_BOUNCE,
            pg.PROBE_DEPTH, 1.0, SK_TOUCHES, pg.MAX_WINDOW_BARS)
        pip = pg.PIP_SIZE[inst]
        spread_price = pg.SPREAD_PIPS_TYPICAL * pip
        r_gross = np.where(is_win, SK_RR, -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        frames.append(pd.DataFrame({
            "instrument": inst,
            "entry_time": pd.to_datetime(times[entry_idx]),
            "exit_time": pd.to_datetime(times[exit_idx]),
            "entry_price": entry_mid,
            "stop_risk": stop_risk,
            "net_r": net_r,
            "is_win": is_win,
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def position_size_and_margin(instrument, equity, entry_price, stop_risk, risk_pct=RISK_PCT,
                              max_leverage=MAX_LEVERAGE, cap=None):
    """Exact port of PRODLIVETRADING/broker/oanda_client.py::position_size, plus the
    margin (notional / max_leverage) that position would occupy. `cap`, when given, is
    this project's usual $2M position-sizing-cap modeling convention (risk off
    min(equity, cap), not off an arbitrarily large equity) -- needed to keep multi-year
    sweeps numerically sane; the real bot has no such cap, so leave it None to match
    live behavior exactly (used for the single past-month report)."""
    risk_base = min(equity, cap) if cap is not None else equity
    risk_dollars = risk_base * risk_pct
    price_diff = stop_risk
    if instrument == "USDJPY":
        risk_per_unit_usd = price_diff / entry_price
        risk_based_units = risk_dollars / risk_per_unit_usd
        max_notional = risk_base * max_leverage
        leverage_capped_units = max_notional
        notional_fn = lambda u: u  # 1 unit = $1 notional
    elif instrument == "EURGBP":
        risk_per_unit_usd = price_diff * GBP_USD_APPROX
        risk_based_units = risk_dollars / risk_per_unit_usd
        max_notional = risk_base * max_leverage
        eur_usd_approx = entry_price * GBP_USD_APPROX
        leverage_capped_units = max_notional / eur_usd_approx
        notional_fn = lambda u: u * eur_usd_approx
    else:
        risk_per_unit_usd = price_diff
        risk_based_units = risk_dollars / risk_per_unit_usd
        max_notional = risk_base * max_leverage
        leverage_capped_units = max_notional / entry_price
        notional_fn = lambda u: u * entry_price

    units = max(min(risk_based_units, leverage_capped_units), 1.0)
    actual_risk_dollars = units * risk_per_unit_usd
    notional = notional_fn(units)
    margin = notional / max_leverage
    return units, actual_risk_dollars, margin


def simulate_margin_constrained(trades, start=START, risk_pct=RISK_PCT, max_leverage=MAX_LEVERAGE,
                                 cap=None, track_detail=True):
    """Event-driven walk: opens a position only if its required margin fits inside
    currently-available margin (equity - margin already committed to open positions).
    Equity used for sizing is REALIZED equity only (closed trades), matching
    oanda_client.py's own stated design -- an open position's floating P&L never
    inflates or shrinks the next trade's size before it's actually realized.

    Open positions are tracked in a min-heap keyed by exit_time (O(log k) close/open,
    k = concurrency, not O(n)) and inputs are pulled once into plain numpy/list arrays
    (not pandas .iterrows()) -- needed to make a multi-risk sweep over the full
    ~250k-trade, 17-year history run in seconds rather than minutes per risk level.
    `track_detail=False` skips the per-trade output columns (only needed for the
    single-run report, not a sweep) for a further speedup."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    entry_time = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_time = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values

    equity = start
    margin_used = 0.0
    heap = []  # (exit_time_ns, risk_dollars, margin, net_r)

    accepted = np.zeros(n, dtype=bool)
    equity_at_accept = np.full(n, np.nan) if track_detail else None
    margin_at_accept = np.full(n, np.nan) if track_detail else None
    concurrent_at_accept = np.zeros(n, dtype=np.int64) if track_detail else None
    close_times, close_equities, close_net_r = ([], [], []) if track_detail else (None, None, None)

    peak_equity = start
    max_dd_pct = 0.0

    for i in range(n):
        et = entry_time[i]
        while heap and heap[0][0] <= et:
            xt, r_dollars, margin, r = heapq.heappop(heap)
            equity += r_dollars * r
            margin_used -= margin
            if equity > peak_equity:
                peak_equity = equity
            dd = 100 * (equity - peak_equity) / peak_equity
            if dd < max_dd_pct:
                max_dd_pct = dd
            if track_detail:
                close_times.append(xt)
                close_equities.append(equity)
                close_net_r.append(r)
        margin_used = max(margin_used, 0.0)  # guard against float drift

        units, risk_dollars, margin = position_size_and_margin(
            instrument[i], equity, entry_price[i], stop_risk[i], risk_pct, max_leverage, cap)

        # Available-margin capacity must use the SAME capped base as position sizing --
        # otherwise once real equity grows past `cap`, positions stay small (correctly)
        # but get checked against an unbounded margin pool, making everything trivially
        # "affordable" and defeating the entire point of a margin constraint.
        available_capacity = min(equity, cap) if cap is not None else equity
        if margin_used + margin <= available_capacity:
            accepted[i] = True
            if track_detail:
                equity_at_accept[i] = equity
                margin_at_accept[i] = margin
                concurrent_at_accept[i] = len(heap) + 1
            heapq.heappush(heap, (exit_time[i], risk_dollars, margin, net_r[i]))
            margin_used += margin

    # flush remaining open positions at the end, in exit-time order (heap pops in order)
    while heap:
        xt, r_dollars, margin, r = heapq.heappop(heap)
        equity += r_dollars * r
        if equity > peak_equity:
            peak_equity = equity
        dd = 100 * (equity - peak_equity) / peak_equity
        if dd < max_dd_pct:
            max_dd_pct = dd
        if track_detail:
            close_times.append(xt)
            close_equities.append(equity)
            close_net_r.append(r)

    if track_detail:
        trace = pd.DataFrame({
            "close_time": pd.to_datetime(np.array(close_times, dtype="int64")),
            "equity": close_equities, "net_r": close_net_r,
        })
        t["accepted"] = accepted
        t["equity_at_entry"] = equity_at_accept
        t["margin_at_entry"] = margin_at_accept
        t["concurrent_at_entry"] = concurrent_at_accept
        return t, equity, max_dd_pct, accepted, trace
    return None, equity, max_dd_pct, accepted, None


def part1_past_month_margin_backtest(full):
    print("=" * 90)
    print("PART 1: Past-month backtest, $200 start, 5% risk, margin-based concurrency (50x leverage)")
    print("=" * 90)
    data_end = min(pd.to_datetime(rs.load_instrument(i)[3].index.max()) for i in TRADEABLE)
    window_start = data_end - pd.Timedelta(days=30)
    month = full[(full.entry_time >= window_start) & (full.entry_time <= data_end)].copy()
    print(f"Window: {window_start} -> {data_end}  ({len(month):,} signal candidates)")

    t, final_equity, max_dd_pct, _, _ = simulate_margin_constrained(month)
    acc = t[t.accepted]
    rej = t[~t.accepted]

    print(f"\nSignals: {len(t):,}   Accepted (margin allowed): {len(acc):,} ({100*len(acc)/len(t):.1f}%)"
          f"   Rejected (margin-blocked): {len(rej):,} ({100*len(rej)/len(t):.1f}%)")
    print(f"Max concurrent positions actually reached: {t.concurrent_at_entry.max()}")
    print(f"Median margin used per accepted position: ${acc.margin_at_entry.median():,.2f} "
          f"({100*acc.margin_at_entry.median()/acc.equity_at_entry.median():.1f}% of equity at entry)")

    if len(acc):
        wr = 100 * acc.is_win.mean()
        w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        print(f"\nAccepted-trade stats: WR={wr:.2f}%  PF={pf:.3f}")

    print(f"\nStarting equity: ${START:,.2f}")
    print(f"Ending equity:   ${final_equity:,.2f}")
    print(f"Total return:    {100*(final_equity/START - 1):+.1f}%")
    print(f"Max drawdown over the month: {max_dd_pct:.1f}%")

    t.to_csv(os.path.join(DATA_DIR, "goal20_speed_king_margin_month.csv"), index=False)
    return t


def part2_monte_carlo(full):
    print("\n" + "=" * 90)
    print("PART 2: Monte Carlo (25,000 sims, $2M position-sizing cap, standard sequential-stream")
    print("        methodology -- NOT margin/concurrency constrained, same as every other MC")
    print("        in this project) -- drawdown and speed to $1M / $10M")
    print("=" * 90)
    net_r = full.net_r.values.astype(np.float64)
    years = (full.entry_time.max() - full.entry_time.min()).days / 365.25
    tpy = len(full) / years
    t1_raw, t10_raw, max_dd, ruined = dtfx_engine.run_mc_capped(
        net_r, 25_000, BLOCK, MAX_TRADES, 20, start=START, risk_frac=RISK_PCT,
        cap=CAP, target1=T1, target2=T10, ruin_floor=RUIN_FLOOR)
    t1_hit = t1_raw[t1_raw > 0]; t10_hit = t10_raw[t10_raw > 0]
    y1 = t1_hit / tpy; y10 = t10_hit / tpy
    dd_pct = max_dd * 100

    print(f"Trade universe: {len(full):,} trades ({tpy:,.0f}/yr), {years:.1f}yr history")
    print(f"\nTime to $1M: quickest={y1.min():.3f}yr  slowest={y1.max():.3f}yr  "
          f"avg={y1.mean():.3f}yr  median={np.median(y1):.3f}yr")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(y1, p):.3f} yrs")
    print(f"\nTime to $10M (capped at $2M position sizing): median={np.median(y10):.3f}yr "
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
    out.to_csv(os.path.join(DATA_DIR, "goal20_speed_king_mc_25000.csv"), index=False)


def part3_2026_weekly(full):
    print("\n" + "=" * 90)
    print("PART 3: 2026 backtest, weekly returns + $200-start equity evolution")
    print("        (standard sequential-stream compounding with the project's usual $2M")
    print("        position-sizing cap -- risk = 5% of min(equity, $2M) -- NOT margin/")
    print("        concurrency constrained; see Part 1 for the realistic-margin version)")
    print("=" * 90)
    t = full[full.entry_time.dt.year == 2026].sort_values("entry_time").reset_index(drop=True)
    eq = START
    equities_before, equities_after = [], []
    for r in t.net_r.values:
        equities_before.append(eq)
        risk_dollars = min(eq, CAP) * RISK_PCT
        eq += risk_dollars * r
        eq = max(eq, RUIN_FLOOR)
        equities_after.append(eq)
    t["equity_before"] = equities_before
    t["equity"] = equities_after
    t["week"] = t.entry_time.dt.to_period("W")

    rows = []
    for wk, sub in t.groupby("week"):
        wr = 100 * sub.is_win.mean()
        total_r = sub.net_r.sum()
        pct_return = 100 * (sub.equity.iloc[-1] / sub.equity_before.iloc[0] - 1)
        rows.append(dict(week=str(wk), trades=len(sub), win_rate=round(wr, 1),
                          total_r=round(total_r, 2), pct_return=round(pct_return, 2),
                          equity_end=round(sub.equity.iloc[-1], 2)))
    week_df = pd.DataFrame(rows)
    print(week_df.to_string(index=False))
    print(f"\n2026 so far: {len(t):,} trades, start ${START:,.2f} -> end ${eq:,.2f} "
          f"({100*(eq/START - 1):+.1f}%)")
    week_df.to_csv(os.path.join(DATA_DIR, "goal20_speed_king_2026_weekly.csv"), index=False)


RISK_GRID = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175,
             0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]


def part4_risk_sweep(full):
    print("\n" + "=" * 90)
    print("PART 4: risk_pct sweep (50x leverage, margin-constrained) -- what maximizes returns?")
    print("=" * 90)
    print("Two windows, same margin-constrained model as Part 1:")
    print("  (a) FULL 16.9yr history, 251k trades -- the robust answer (with the project's usual")
    print("      $2M position-sizing cap, needed to keep 17 years of compounding numerically sane;")
    print("      applied consistently to BOTH position sizing and the margin-availability check)")
    print("  (b) the same past-month window as Part 1 -- for direct comparison, no cap needed")
    print("      (equity never gets remotely close to $2M in one month at any of these risk levels)")

    data_end = min(pd.to_datetime(rs.load_instrument(i)[3].index.max()) for i in TRADEABLE)
    window_start = data_end - pd.Timedelta(days=30)
    month = full[(full.entry_time >= window_start) & (full.entry_time <= data_end)].copy()

    rows = []
    for risk_pct in RISK_GRID:
        _, eq_full, dd_full, acc_full, _ = simulate_margin_constrained(
            full, risk_pct=risk_pct, cap=CAP, track_detail=False)
        _, eq_month, dd_month, acc_month, _ = simulate_margin_constrained(
            month, risk_pct=risk_pct, cap=None, track_detail=False)
        rows.append(dict(
            risk_pct=risk_pct,
            full_accept_pct=round(100 * acc_full.mean(), 1),
            full_final_equity=eq_full,
            full_max_dd_pct=round(dd_full, 1),
            month_accept_pct=round(100 * acc_month.mean(), 1),
            month_return_pct=round(100 * (eq_month / START - 1), 1),
            month_max_dd_pct=round(dd_month, 1),
        ))

    df = pd.DataFrame(rows)
    best_full = df.loc[df.full_final_equity.idxmax()]
    best_month = df.loc[df.month_return_pct.idxmax()]

    print(f"\n{'risk%':>6} | {'FULL 16.9yr: accept%':>20} {'final equity':>16} {'max DD':>8} | "
          f"{'PAST MONTH: accept%':>20} {'return%':>10} {'max DD':>8}")
    for r in rows:
        marker_f = " <-- max" if r["risk_pct"] == best_full.risk_pct else ""
        marker_m = " <-- max" if r["risk_pct"] == best_month.risk_pct else ""
        print(f"{100*r['risk_pct']:5.2f}% | {r['full_accept_pct']:19.1f}% "
              f"${r['full_final_equity']:>14,.0f} {r['full_max_dd_pct']:7.1f}%{marker_f:8s} | "
              f"{r['month_accept_pct']:19.1f}% {r['month_return_pct']:9.1f}%{marker_m:8s} "
              f"{r['month_max_dd_pct']:7.1f}%")

    print(f"\nMaximizes 16.9yr final equity: risk={100*best_full.risk_pct:.2f}% "
          f"(${best_full.full_final_equity:,.0f}, accept={best_full.full_accept_pct:.1f}%, "
          f"maxDD={best_full.full_max_dd_pct:.1f}%)")
    print(f"Maximizes past-month return:   risk={100*best_month.risk_pct:.2f}% "
          f"({best_month.month_return_pct:+.1f}%, accept={best_month.month_accept_pct:.1f}%, "
          f"maxDD={best_month.month_max_dd_pct:.1f}%)")

    df.to_csv(os.path.join(DATA_DIR, "goal20_speed_king_risk_sweep.csv"), index=False)
    return df


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    print("Building Speed-King full-history trades (true entry/exit times)...", flush=True)
    full = build_speed_king_full()
    full.to_csv(os.path.join(DATA_DIR, "goal20_speed_king_full_true_times.csv"), index=False)
    print(f"  {len(full):,} trades, {full.entry_time.min()} -> {full.entry_time.max()}\n")

    part1_past_month_margin_backtest(full)
    part2_monte_carlo(full)
    part3_2026_weekly(full)
    part4_risk_sweep(full)


if __name__ == "__main__":
    main()
