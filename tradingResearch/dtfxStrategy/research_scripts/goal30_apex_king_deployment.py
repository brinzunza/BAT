"""Apex-King deployment package: everything needed to replace Speed-King in
PRODLIVETRADING with a genuinely accurate model of the real live architecture --
TWO SEPARATE $100 sub-accounts (buy-only, sell-only, never pooled), sized the same way
the fixed bot.py actually does it (sizing_base = min(buy_equity, sell_equity), but the
accept/reject margin check uses each side's OWN real capacity), max 1 open position per
side, 1.6-pip spread, 50x leverage, $500,000 account cap (this run's ceiling, not the
project's usual $2M -- per this request).

This is a step up in fidelity from goal28/goal29's single-shared-pool approximation:
those modeled ONE combined equity pool with a 1-per-side position-count cap layered on;
here equity, margin, and P&L are tracked separately per side, exactly mirroring two
real, independent broker sub-accounts -- growth on the buy side doesn't inflate sell-side
capacity or vice versa.

Sections: (1) risk_pct sweep to find Apex-King's own MC-robust optimum under this exact
setup, (2) full-history backtest at that risk, (3) monthly-reset growth table, (4) last
calendar month backtest, (5) Monte Carlo: time to $1M and $10M.
"""
import os
import sys
import heapq
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal20_speed_king_margin_report as g20
import goal29_rerank_1_per_side as g29

position_size_and_margin = g20.position_size_and_margin
YEAR_NS = 365.25 * 86400 * 1e9

START_PER_SIDE = 100.0
MAX_LEVERAGE = 50.0
ACCOUNT_CAP = 500_000.0   # this request's ceiling (not the project's usual $2M)
BLOCK = g20.BLOCK
RUIN_FLOOR = g20.RUIN_FLOOR
RISK_GRID = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]

APEX_CFG = g29.STRATEGIES["Apex-King"]


def build_apex_trades():
    for inst in g29.TRADEABLE:
        g29.rs.load_instrument(inst)
        g29.rs._prep_zones(inst, g29.pg.MIN_RETRACEMENT)
    return g29.get_true_trades("Apex-King", APEX_CFG)


def simulate_two_account(trades, risk_pct, start_per_side=START_PER_SIDE,
                          max_leverage=MAX_LEVERAGE, cap=ACCOUNT_CAP, track_detail=True):
    """Event-driven walk with two genuinely separate sub-account equity/margin pools.
    Sizing base = min(buy_equity, sell_equity) (matches bot.py's fixed formula exactly);
    the accept/reject check uses each side's OWN real capacity (matches
    account_margin_available() being per-sub-account). Max 1 open position per side."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    entry_time = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_time = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values
    is_buy = t.is_buy.values

    equity = {True: start_per_side, False: start_per_side}
    margin_used = {True: 0.0, False: 0.0}
    open_count = {True: 0, False: 0}
    heap = []

    peak_combined = start_per_side * 2
    max_dd_pct = 0.0
    accepted = np.zeros(n, dtype=bool)
    close_times, close_buy, close_sell = [], [], []

    for i in range(n):
        et = entry_time[i]
        while heap and heap[0][0] <= et:
            xt, r_dollars, margin, r, side = heapq.heappop(heap)
            equity[side] += r_dollars * r
            margin_used[side] -= margin
            open_count[side] -= 1
            combined = equity[True] + equity[False]
            if combined > peak_combined:
                peak_combined = combined
            dd = 100 * (combined - peak_combined) / peak_combined
            if dd < max_dd_pct:
                max_dd_pct = dd
            if track_detail:
                close_times.append(xt); close_buy.append(equity[True]); close_sell.append(equity[False])
        margin_used[True] = max(margin_used[True], 0.0)
        margin_used[False] = max(margin_used[False], 0.0)

        side = bool(is_buy[i])
        sizing_base = min(equity[True], equity[False])
        _, risk_dollars, margin = position_size_and_margin(
            instrument[i], sizing_base, entry_price[i], stop_risk[i], risk_pct, max_leverage, cap)
        avail = min(equity[side], cap) - margin_used[side]

        if margin <= avail and open_count[side] < 1:
            accepted[i] = True
            heapq.heappush(heap, (exit_time[i], risk_dollars, margin, net_r[i], side))
            margin_used[side] += margin
            open_count[side] += 1

    while heap:
        xt, r_dollars, margin, r, side = heapq.heappop(heap)
        equity[side] += r_dollars * r
        combined = equity[True] + equity[False]
        if combined > peak_combined:
            peak_combined = combined
        dd = 100 * (combined - peak_combined) / peak_combined
        if dd < max_dd_pct:
            max_dd_pct = dd
        if track_detail:
            close_times.append(xt); close_buy.append(equity[True]); close_sell.append(equity[False])

    trace = pd.DataFrame({"close_time": pd.to_datetime(np.array(close_times, dtype="int64")),
                           "equity_buy": close_buy, "equity_sell": close_sell})
    if len(trace):
        trace["equity_combined"] = trace.equity_buy + trace.equity_sell
    t = t.copy()
    t["accepted"] = accepted
    return t, equity[True], equity[False], max_dd_pct, trace


def mc_path_two_account(entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r, is_buy,
                         risk_pct, rng, block=BLOCK, max_trades=200_000,
                         start_per_side=START_PER_SIDE, max_leverage=MAX_LEVERAGE, cap=ACCOUNT_CAP,
                         target1=1_000_000.0, target2=10_000_000.0, ruin_floor=RUIN_FLOOR):
    n = len(entry_ns)
    equity = {True: start_per_side, False: start_per_side}
    margin_used = {True: 0.0, False: 0.0}
    open_count = {True: 0, False: 0}
    heap = []
    clock = 0
    t1, t10 = -1.0, -1.0
    peak_combined = start_per_side * 2
    dd_worst = 0.0
    ruined = False
    processed = 0
    accepted_n = 0

    while processed < max_trades and t10 < 0 and not ruined:
        start_idx = rng.integers(0, n)
        end_idx = min(start_idx + block, n)
        offset = clock - entry_ns[start_idx]
        block_exit_max = clock

        for idx in range(start_idx, end_idx):
            et = entry_ns[idx] + offset
            xt = exit_ns[idx] + offset
            if xt > block_exit_max:
                block_exit_max = xt

            while heap and heap[0][0] <= et:
                _, r_dollars, margin, r, side = heapq.heappop(heap)
                equity[side] += r_dollars * r
                margin_used[side] -= margin
                open_count[side] -= 1
                combined = equity[True] + equity[False]
                if combined > peak_combined:
                    peak_combined = combined
                dd = 100 * (combined - peak_combined) / peak_combined
                if dd < dd_worst:
                    dd_worst = dd
                if combined <= ruin_floor:
                    ruined = True
            margin_used[True] = max(margin_used[True], 0.0)
            margin_used[False] = max(margin_used[False], 0.0)
            if ruined:
                break

            side = bool(is_buy[idx])
            sizing_base = min(equity[True], equity[False])
            _, risk_dollars, margin = position_size_and_margin(
                instrument[idx], sizing_base, entry_price[idx], stop_risk[idx], risk_pct,
                max_leverage, cap)
            avail = min(equity[side], cap) - margin_used[side]

            if margin <= avail and open_count[side] < 1:
                heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx], side))
                margin_used[side] += margin
                open_count[side] += 1
                accepted_n += 1

            processed += 1
            combined = equity[True] + equity[False]
            if t1 < 0 and combined >= target1:
                t1 = et / YEAR_NS
            if t1 >= 0 and t10 < 0 and combined >= target2:
                t10 = et / YEAR_NS
                break
            if processed >= max_trades:
                break

        if ruined or t10 >= 0:
            break
        clock = block_exit_max

    return t1, t10, dd_worst, ruined, processed, accepted_n


def run_mc(trades, risk_pct, n_paths, seed, block=BLOCK, max_trades=200_000, **kwargs):
    t = trades.sort_values("entry_time").reset_index(drop=True)
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    instrument = t.instrument.values
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    is_buy = t.is_buy.values

    rng = np.random.default_rng(seed)
    t1s = np.empty(n_paths); t10s = np.empty(n_paths); dds = np.empty(n_paths)
    ruins = np.zeros(n_paths, dtype=bool); accs = np.empty(n_paths)
    for p in range(n_paths):
        t1, t10, dd, ruined, processed, accepted_n = mc_path_two_account(
            entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r, is_buy,
            risk_pct, rng, block=block, max_trades=max_trades, **kwargs)
        t1s[p] = t1; t10s[p] = t10; dds[p] = dd; ruins[p] = ruined
        accs[p] = accepted_n / processed if processed else np.nan
    return t1s, t10s, dds, ruins, accs


def section_risk_sweep(trades):
    print("\n" + "=" * 90 + "\nSECTION 1: Risk-pct sweep (two $100 accounts, 1/side cap, $500k account cap)\n" + "=" * 90)
    rows = []
    for risk_pct in RISK_GRID:
        t0 = time.perf_counter()
        t1s, t10s, dds, ruins, accs = run_mc(trades, risk_pct, n_paths=80, seed=3000 + int(risk_pct * 10000),
                                              max_trades=250_000)
        reached = t1s >= 0
        med_yrs = np.median(t1s[reached]) if reached.any() else float("inf")
        med_dd = np.median(dds)
        pct_ruin = 100 * ruins.mean()
        dt = time.perf_counter() - t0
        rows.append(dict(risk_pct=risk_pct, median_yrs_1m=med_yrs, median_dd=med_dd,
                          pct_ruin=pct_ruin, pct_reached=100 * reached.mean()))
        print(f"  risk={100*risk_pct:5.1f}%  yrs->1M(median)={med_yrs:6.3f}  medDD={med_dd:6.1f}%  "
              f"ruin={pct_ruin:5.2f}%  reached={100*reached.mean():5.1f}%  [{dt:4.1f}s]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal30_apex_risk_sweep.csv"), index=False)
    # MC-robust optimum: among risk levels with 0% ruin, fastest median years-to-$1M
    safe = df[df.pct_ruin == 0.0]
    pick_from = safe if len(safe) else df
    best = pick_from.loc[pick_from.median_yrs_1m.idxmin()]
    print(f"\n  MC-robust optimum: {100*best.risk_pct:.1f}% risk "
          f"(median {best.median_yrs_1m:.3f} yrs to $1M, {best.pct_ruin:.2f}% ruin, medDD {best.median_dd:.1f}%)")
    return float(best.risk_pct)


def section_historical_backtest(trades, risk_pct):
    print("\n" + "=" * 90 + f"\nSECTION 2: Full historical backtest at {100*risk_pct:.1f}% risk\n" + "=" * 90)
    t, eq_buy, eq_sell, max_dd, trace = simulate_two_account(trades, risk_pct)
    acc = t[t.accepted]
    wr = 100 * acc.is_win.mean()
    w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    print(f"Window: {t.entry_time.min()} -> {t.entry_time.max()}  ({years:.1f} years)")
    print(f"Signals: {len(t):,}  Accepted: {len(acc):,} ({100*len(acc)/len(t):.1f}%)")
    print(f"Accepted WR={wr:.2f}%  PF={pf:.3f}")
    print(f"Start: $100 buy + $100 sell = $200 combined")
    print(f"End:   ${eq_buy:,.2f} buy + ${eq_sell:,.2f} sell = ${eq_buy+eq_sell:,.2f} combined "
          f"({100*((eq_buy+eq_sell)/200-1):+.1f}%)")
    print(f"Max drawdown (combined equity): {max_dd:.1f}%")
    trace.to_csv(os.path.join(DATA_DIR, "goal30_apex_historical_trace.csv"), index=False)
    t.to_csv(os.path.join(DATA_DIR, "goal30_apex_historical_trades.csv"), index=False)
    return t, trace


def section_monthly_reset(trades, risk_pct):
    print("\n" + "=" * 90 + f"\nSECTION 3: Monthly growth, account reset to $100/$100 each month\n" + "=" * 90)
    t = trades.copy()
    t["month"] = t.entry_time.dt.to_period("M")
    rows = []
    for month, sub in t.groupby("month"):
        if len(sub) < 5:
            continue
        _, eq_buy, eq_sell, max_dd, _ = simulate_two_account(sub, risk_pct, track_detail=False)
        combined = eq_buy + eq_sell
        rows.append(dict(month=str(month), signals=len(sub), end_buy=eq_buy, end_sell=eq_sell,
                          end_combined=combined, return_pct=100 * (combined / 200 - 1), max_dd_pct=max_dd))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal30_apex_monthly_reset.csv"), index=False)
    print(df.to_string(index=False))
    print(f"\nMedian monthly return: {df.return_pct.median():+.1f}%   "
          f"Mean: {df.return_pct.mean():+.1f}%   "
          f"Win months: {100*(df.return_pct > 0).mean():.0f}%   "
          f"Worst month: {df.return_pct.min():+.1f}%   Best month: {df.return_pct.max():+.1f}%")
    return df


def section_last_month(trades, risk_pct):
    print("\n" + "=" * 90 + f"\nSECTION 4: Last calendar month only\n" + "=" * 90)
    data_end = trades.entry_time.max()
    last_month_start = (data_end.replace(day=1))
    sub = trades[(trades.entry_time >= last_month_start) & (trades.entry_time <= data_end)].copy()
    print(f"Window: {last_month_start.date()} -> {data_end}  ({len(sub):,} signal candidates)")
    t, eq_buy, eq_sell, max_dd, trace = simulate_two_account(sub, risk_pct)
    acc = t[t.accepted]
    if len(acc):
        wr = 100 * acc.is_win.mean()
        print(f"Accepted: {len(acc):,}/{len(t):,} ({100*len(acc)/len(t):.1f}%)  WR={wr:.2f}%")
    print(f"Start $200 combined -> End ${eq_buy+eq_sell:,.2f} ({100*((eq_buy+eq_sell)/200-1):+.1f}%)  maxDD={max_dd:.1f}%")

    if len(trace):
        trace["day"] = trace.close_time.dt.date
        daily = trace.groupby("day").agg(trades=("equity_combined", "size"),
                                          equity_end=("equity_combined", "last"))
        print("\nDay-by-day (combined equity):")
        print(daily.to_string())
    trace.to_csv(os.path.join(DATA_DIR, "goal30_apex_last_month_trace.csv"), index=False)
    return t, trace


def section_monte_carlo(trades, risk_pct, n_paths=500):
    print("\n" + "=" * 90 +
          f"\nSECTION 5: Monte Carlo -- time to $1M and $10M ({n_paths} paths, "
          f"$500k account cap, {100*risk_pct:.1f}% risk)\n" + "=" * 90)
    t0 = time.perf_counter()
    t1s, t10s, dds, ruins, accs = run_mc(trades, risk_pct, n_paths=n_paths, seed=4242, max_trades=400_000)
    dt = time.perf_counter() - t0
    reached1 = t1s >= 0
    reached10 = t10s >= 0
    print(f"Reached $1M:  {100*reached1.mean():.1f}% of paths")
    if reached1.any():
        y = t1s[reached1]
        print(f"  Years to $1M: P5={np.percentile(y,5):.2f}  P50={np.percentile(y,50):.2f}  "
              f"P95={np.percentile(y,95):.2f}")
    print(f"Reached $10M: {100*reached10.mean():.1f}% of paths")
    if reached10.any():
        y = t10s[reached10]
        print(f"  Years to $10M: P5={np.percentile(y,5):.2f}  P50={np.percentile(y,50):.2f}  "
              f"P95={np.percentile(y,95):.2f}")
    print(f"Ruined: {100*ruins.mean():.2f}%   Median max drawdown: {np.median(dds):.1f}%   "
          f"Median concurrency-accept rate: {100*np.nanmedian(accs):.1f}%   [{dt:.1f}s]")


def main():
    print("Building Apex-King TRUE holdout trade stream (with side)...")
    trades = build_apex_trades()
    print(f"  {len(trades):,} signal candidates, {trades.entry_time.min()} -> {trades.entry_time.max()}")

    best_risk = section_risk_sweep(trades)
    section_historical_backtest(trades, best_risk)
    section_monthly_reset(trades, best_risk)
    section_last_month(trades, best_risk)
    section_monte_carlo(trades, best_risk)


if __name__ == "__main__":
    main()
