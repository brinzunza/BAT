#!/usr/bin/env python3
"""Goal 42: Comprehensive Two-Subaccount Backtest for Titan-76.

Architecture:
- Two separate $100 sub-accounts (buy-only, sell-only, never pooled)
- Position sizing: 6% risk of min(buy_equity, sell_equity)
- 50x max leverage
- 1.6-pip fixed spread across EURUSD, GBPUSD, EURGBP, USDJPY
- Max 1 open position per side (max 1 buy across all pairs, max 1 sell across all pairs)
- Target: $1,000,000 ceiling

Sections evaluated:
1. Full Historical Backtest (2009-2026 and Holdout 2018-2026)
2. Monthly-Reset Backtest (Every month resets to $100 Buy / $100 Sell)
3. Last Calendar Month Backtest (August 2026 exact outcome)
4. Monte Carlo Simulation to $1,000,000 (50,000 block-bootstrap paths)
"""
import os
import sys
import time
import heapq
import numpy as np
import pandas as pd

os.environ["MPLCONFIGDIR"] = "/tmp"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal5_rr_sweep as g5
import goal20_speed_king_margin_report as g20

position_size_and_margin = g20.position_size_and_margin
BLOCK = g20.BLOCK
RUIN_FLOOR = g20.RUIN_FLOOR
YEAR_NS = 365.25 * 86400 * 1e9

SPLIT = pd.Timestamp("2018-03-09")
START_PER_SIDE = 100.0
RISK_PCT = 0.06
MAX_LEVERAGE = 50.0
ACCOUNT_CAP = 1_000_000.0
SPREAD_PIPS = 1.6
N_MC_SIMS = 50000

TITAN_CFG = {
    "ret": 1,
    "n": 2.0,
    "rr": 0.76,
    "age_h": 0.20,
    "touches": 6,
    "probe": 0.75,
    "exc": 1.0,
    "bounce": False
}


def build_titan_trades():
    print("Pre-caching and generating Titan-76 raw trade stream with 1.6-pip spread...", flush=True)
    t0 = time.time()
    ret = TITAN_CFG["ret"]
    n = TITAN_CFG["n"]
    rr = TITAN_CFG["rr"]
    age_h = TITAN_CFG["age_h"]
    touches = TITAN_CFG["touches"]
    probe = TITAN_CFG["probe"]
    exc = TITAN_CFG["exc"]
    bounce = TITAN_CFG["bounce"]

    min_fill_age_ns = int(age_h * 3600 * 1e9)
    target_mult = n * rr
    frames = []

    for inst in g5.TRADEABLE:
        z = g5.rs._prep_zones(inst, ret)
        times, lows, highs, df = g5.rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")

        pip_size = g5.pg.PIP_SIZE[inst]
        spread_price = SPREAD_PIPS * pip_size

        for side_is_buy in (True, False):
            mask = z["is_buy"] == side_is_buy
            top = np.ascontiguousarray(z["top"][mask], dtype=np.float64)
            bottom = np.ascontiguousarray(z["bottom"][mask], dtype=np.float64)
            mid = np.ascontiguousarray(z["mid"][mask], dtype=np.float64)
            is_buy_arr = np.ascontiguousarray(z["is_buy"][mask], dtype=np.bool_)
            start_idx = np.ascontiguousarray(z["start_idx"][mask], dtype=np.int64)
            c_ns = np.ascontiguousarray(created_ns[mask], dtype=np.int64)

            half_width = (top - bottom) / 2.0
            risk_pct = (n * half_width) / mid
            ok = risk_pct >= g5.pg.MIN_RISK_PCT

            if not np.any(ok):
                continue

            touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
                top[ok], bottom[ok], mid[ok], is_buy_arr[ok], start_idx[ok], c_ns[ok],
                times_ns, lows, highs, min_fill_age_ns, n, target_mult, bounce, probe, exc,
                touches, g5.pg.MAX_WINDOW_BARS
            )

            if len(entry_idx) == 0:
                continue

            r_gross = np.where(is_win, rr, -1.0)
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
                "is_buy": side_is_buy,
                "touch_num": touch_num,
            }))

    all_trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    print(f"Generated {len(all_trades):,} raw candidate trades in {time.time()-t0:.2f}s.\n", flush=True)
    return all_trades


def simulate_two_account(trades, risk_pct=RISK_PCT, start_per_side=START_PER_SIDE,
                          max_leverage=MAX_LEVERAGE, cap=ACCOUNT_CAP, track_detail=True):
    """Event-driven walk with two separate sub-accounts (buy-only, sell-only).
    Sizing base = min(buy_equity, sell_equity).
    Max 1 open position per side. Max leverage 50x."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    if n == 0:
        return t, start_per_side, start_per_side, 0.0, pd.DataFrame()

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

    peak_combined = start_per_side * 2.0
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
            dd = 100.0 * (combined - peak_combined) / peak_combined
            if dd < max_dd_pct:
                max_dd_pct = dd
            if track_detail:
                close_times.append(xt)
                close_buy.append(equity[True])
                close_sell.append(equity[False])

        margin_used[True] = max(margin_used[True], 0.0)
        margin_used[False] = max(margin_used[False], 0.0)

        side = bool(is_buy[i])
        sizing_base = min(equity[True], equity[False])
        if sizing_base <= RUIN_FLOOR:
            continue

        _, risk_dollars, margin = position_size_and_margin(
            instrument[i], sizing_base, entry_price[i], stop_risk[i], risk_pct, max_leverage, cap
        )
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
        dd = 100.0 * (combined - peak_combined) / peak_combined
        if dd < max_dd_pct:
            max_dd_pct = dd
        if track_detail:
            close_times.append(xt)
            close_buy.append(equity[True])
            close_sell.append(equity[False])

    trace = pd.DataFrame({
        "close_time": pd.to_datetime(np.array(close_times, dtype="int64")),
        "equity_buy": close_buy,
        "equity_sell": close_sell
    })
    if len(trace):
        trace["equity_combined"] = trace.equity_buy + trace.equity_sell

    t = t.copy()
    t["accepted"] = accepted
    return t, equity[True], equity[False], max_dd_pct, trace


def run_monte_carlo_two_account(trades, n_sims=N_MC_SIMS, target=1_000_000.0):
    print(f"Running {n_sims:,} two-account Monte Carlo block-bootstrap simulations...", flush=True)
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values
    is_buy = t.is_buy.values

    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    tpy = n / years

    rng = np.random.default_rng(42)
    days_to_1m = []
    trades_to_1m = []
    max_dds = []
    ruined_count = 0

    t0 = time.time()
    for sim in range(n_sims):
        equity = {True: START_PER_SIDE, False: START_PER_SIDE}
        margin_used = {True: 0.0, False: 0.0}
        open_count = {True: 0, False: 0}
        heap = []
        clock = 0
        peak = START_PER_SIDE * 2.0
        worst_dd = 0.0
        ruined = False
        hit_target = False
        t_hit_days = -1.0
        trades_count = 0

        while trades_count < 100_000 and not hit_target and not ruined:
            start_idx = rng.integers(0, n)
            end_idx = min(start_idx + BLOCK, n)
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
                    comb = equity[True] + equity[False]
                    if comb > peak:
                        peak = comb
                    dd = 100.0 * (comb - peak) / peak
                    if dd < worst_dd:
                        worst_dd = dd
                    if comb <= RUIN_FLOOR or equity[True] <= 1.0 or equity[False] <= 1.0:
                        ruined = True

                if ruined:
                    break

                side = bool(is_buy[idx])
                sizing_base = min(equity[True], equity[False])
                _, risk_dollars, margin = position_size_and_margin(
                    instrument[idx], sizing_base, entry_price[idx], stop_risk[idx], RISK_PCT,
                    MAX_LEVERAGE, ACCOUNT_CAP
                )
                avail = min(equity[side], ACCOUNT_CAP) - margin_used[side]

                if margin <= avail and open_count[side] < 1:
                    heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx], side))
                    margin_used[side] += margin
                    open_count[side] += 1
                    trades_count += 1

                comb = equity[True] + equity[False]
                if comb >= target:
                    hit_target = True
                    t_hit_days = (et / YEAR_NS) * 365.25
                    break

            clock = block_exit_max

        if ruined:
            ruined_count += 1
        if hit_target:
            days_to_1m.append(t_hit_days)
            trades_to_1m.append(trades_count)
        max_dds.append(worst_dd)

    days_arr = np.array(days_to_1m)
    trades_arr = np.array(trades_to_1m)
    dds_arr = np.array(max_dds)

    print(f"Completed {n_sims:,} MC simulations in {time.time()-t0:.2f}s.\n", flush=True)
    return days_arr, trades_arr, dds_arr, ruined_count / n_sims


def main():
    print("=" * 115)
    print("  TITAN-76 DUAL SUB-ACCOUNT BACKTEST (BUY-ONLY & SELL-ONLY, NEVER POOLED)")
    print("  Position Sizing: 6% of min(buy_equity, sell_equity) | 50x Leverage | 1.6-Pip Spread | Max 1 Position/Side")
    print("=" * 115)

    all_trades = build_titan_trades()
    holdout_trades = all_trades[all_trades.entry_time >= SPLIT].reset_index(drop=True)

    # -------------------------------------------------------------
    # SECTION 1: FULL HISTORICAL & TRUE HOLDOUT BACKTEST
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  SECTION 1: HISTORICAL BACKTEST & TRUE HOLDOUT PERFORMANCE")
    print("=" * 90)

    # 1. Full 16.9-year backtest
    t_hist, b_eq_hist, s_eq_hist, dd_hist, trace_hist = simulate_two_account(all_trades)
    comb_hist = b_eq_hist + s_eq_hist
    acc_hist = t_hist.accepted.sum()
    acc_pct_hist = 100.0 * acc_hist / len(t_hist)
    win_acc_hist = 100.0 * (t_hist[t_hist.accepted].net_r > 0).mean()

    print(f"--- Full History (2009 - August 2026, 16.9 Years) ---")
    print(f"  Raw Trades Generated:      {len(all_trades):,}")
    print(f"  Accepted Trades (1/side):  {acc_hist:,} ({acc_pct_hist:.2f}% accepted)")
    print(f"  Accepted Win Rate:         {win_acc_hist:.2f}%")
    print(f"  Final Buy Sub-Account:     ${b_eq_hist:,.2f}")
    print(f"  Final Sell Sub-Account:    ${s_eq_hist:,.2f}")
    print(f"  Final Combined Equity:     ${comb_hist:,.2f} (Started with $200.00)")
    print(f"  Max Historical Drawdown:   {dd_hist:.2f}%\n")

    # 2. True Holdout (2018 - 2026)
    t_hld, b_eq_hld, s_eq_hld, dd_hld, trace_hld = simulate_two_account(holdout_trades)
    comb_hld = b_eq_hld + s_eq_hld
    acc_hld = t_hld.accepted.sum()
    acc_pct_hld = 100.0 * acc_hld / len(t_hld)
    win_acc_hld = 100.0 * (t_hld[t_hld.accepted].net_r > 0).mean()

    print(f"--- True Out-of-Sample Holdout (March 2018 - August 2026, 8.48 Years) ---")
    print(f"  Raw Holdout Trades:        {len(holdout_trades):,}")
    print(f"  Accepted Trades (1/side):  {acc_hld:,} ({acc_pct_hld:.2f}% accepted)")
    print(f"  Accepted Win Rate:         {win_acc_hld:.2f}%")
    print(f"  Final Buy Sub-Account:     ${b_eq_hld:,.2f}")
    print(f"  Final Sell Sub-Account:    ${s_eq_hld:,.2f}")
    print(f"  Final Combined Equity:     ${comb_hld:,.2f} (Started with $200.00)")
    print(f"  Max Holdout Drawdown:      {dd_hld:.2f}%\n")

    # -------------------------------------------------------------
    # SECTION 2: MONTHLY-RESET BACKTEST (DISTRIBUTION OF MONTHS)
    # -------------------------------------------------------------
    print("=" * 90)
    print("  SECTION 2: MONTHLY-RESET PERFORMANCE TABLE (Start $100 Buy / $100 Sell each month)")
    print("=" * 90)

    all_trades["month_year"] = all_trades.entry_time.dt.to_period("M")
    monthly_records = []

    for period, mdf in all_trades.groupby("month_year"):
        _, m_buy, m_sell, m_dd, _ = simulate_two_account(mdf, track_detail=False)
        m_comb = m_buy + m_sell
        m_gain_pct = 100.0 * (m_comb - 200.0) / 200.0
        monthly_records.append({
            "Month": str(period),
            "Raw Trades": len(mdf),
            "Accepted": int(mdf.shape[0]),
            "Final Buy ($)": round(m_buy, 2),
            "Final Sell ($)": round(m_sell, 2),
            "Combined ($)": round(m_comb, 2),
            "Return (%)": round(m_gain_pct, 2),
            "Max DD (%)": round(m_dd, 2)
        })

    df_monthly = pd.DataFrame(monthly_records)
    pos_months = (df_monthly["Return (%)"] > 0).sum()
    total_months = len(df_monthly)
    med_monthly_return = df_monthly["Return (%)"].median()
    mean_monthly_return = df_monthly["Return (%)"].mean()
    p10_ret = df_monthly["Return (%)"].quantile(0.10)
    p90_ret = df_monthly["Return (%)"].quantile(0.90)

    print(f"Total Calendar Months Tested:  {total_months}")
    print(f"Profitable Months:             {pos_months} / {total_months} ({100.0*pos_months/total_months:.1f}%)")
    print(f"Median Monthly Return:         +{med_monthly_return:,.1f}%")
    print(f"Mean Monthly Return:           +{mean_monthly_return:,.1f}%")
    print(f"P10 - P90 Monthly Return:      {p10_ret:,.1f}% to +{p90_ret:,.1f}%\n")

    print("Sample Recent Months (Past 12 Months):")
    print(df_monthly.tail(12).to_string(index=False))

    # -------------------------------------------------------------
    # SECTION 3: LAST CALENDAR MONTH BACKTEST (AUGUST 2026)
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  SECTION 3: LAST CALENDAR MONTH BACKTEST (AUGUST 2026)")
    print("=" * 90)

    last_month_trades = all_trades[all_trades.entry_time >= "2026-08-01"].reset_index(drop=True)
    t_last, b_last, s_last, dd_last, trace_last = simulate_two_account(last_month_trades)
    comb_last = b_last + s_last
    ret_last_pct = 100.0 * (comb_last - 200.0) / 200.0
    acc_last = t_last.accepted.sum()
    wr_last = 100.0 * (t_last[t_last.accepted].net_r > 0).mean() if acc_last > 0 else 0.0

    print(f"Period:                        2026-08-01 00:00 to 2026-08-31 21:59")
    print(f"Starting Capital:              $100.00 Buy Sub-Account + $100.00 Sell Sub-Account ($200.00 Combined)")
    print(f"Raw Signals Generated:         {len(last_month_trades):,}")
    print(f"Accepted Trades Executed:      {acc_last:,} ({100.0*acc_last/len(last_month_trades):.1f}% accepted)")
    print(f"Win Rate:                      {wr_last:.2f}%")
    print(f"Ending Buy Sub-Account:        ${b_last:,.2f}")
    print(f"Ending Sell Sub-Account:       ${s_last:,.2f}")
    print(f"Ending Combined Equity:        ${comb_last:,.2f}")
    print(f"Net Gain for August 2026:      +{ret_last_pct:,.2f}% (+${comb_last - 200.0:,.2f})")
    print(f"Max Intra-Month Drawdown:      {dd_last:.2f}%\n")

    # -------------------------------------------------------------
    # SECTION 4: MONTE CARLO SIMULATION TO $1,000,000
    # -------------------------------------------------------------
    print("=" * 90)
    print("  SECTION 4: 50,000-SIMULATION TWO-SUBACCOUNT MONTE CARLO TO $1,000,000")
    print("=" * 90)

    days_to_1m, trades_to_1m, max_dds, ruin_rate = run_monte_carlo_two_account(holdout_trades, n_sims=N_MC_SIMS, target=1_000_000.0)

    p10_d = np.percentile(days_to_1m, 10) if len(days_to_1m) else float("nan")
    p25_d = np.percentile(days_to_1m, 25) if len(days_to_1m) else float("nan")
    p50_d = np.median(days_to_1m) if len(days_to_1m) else float("nan")
    p75_d = np.percentile(days_to_1m, 75) if len(days_to_1m) else float("nan")
    p90_d = np.percentile(days_to_1m, 90) if len(days_to_1m) else float("nan")

    p50_tr = np.median(trades_to_1m) if len(trades_to_1m) else float("nan")
    p10_tr = np.percentile(trades_to_1m, 10) if len(trades_to_1m) else float("nan")
    p90_tr = np.percentile(trades_to_1m, 90) if len(trades_to_1m) else float("nan")

    p50_dd = np.median(max_dds)
    p10_dd = np.percentile(max_dds, 10)
    p90_dd = np.percentile(max_dds, 90)

    print(f"Start:                         $100 Buy / $100 Sell ($200 Combined)")
    print(f"Target:                        $1,000,000.00")
    print(f"Risk per Trade:                6.0% of min(buy_equity, sell_equity)")
    print(f"Probability of Ruin:           {ruin_rate:.2%}")
    print(f"Success Rate to $1M:           {100.0 * len(days_to_1m) / N_MC_SIMS:.2f}%\n")

    print(f"--- Calendar Days to $1,000,000 Distribution ---")
    print(f"  P10 (Fastest 10%):           {p10_d:.1f} days")
    print(f"  P25:                         {p25_d:.1f} days")
    print(f"  P50 (Median):                {p50_d:.1f} days ({p50_d/365.25:.3f} years)")
    print(f"  P75:                         {p75_d:.1f} days")
    print(f"  P90 (Slowest 10%):           {p90_d:.1f} days\n")

    print(f"--- Trades Executed to $1,000,000 ---")
    print(f"  Median Trades to $1M:        {p50_tr:,.0f} trades")
    print(f"  P10 - P90 Trades:            {p10_tr:,.0f} to {p90_tr:,.0f} trades\n")

    print(f"--- Peak-to-Trough Drawdown Distribution ---")
    print(f"  P10 Drawdown:                {p90_dd:.1f}%")
    print(f"  Median Drawdown (P50):       {p50_dd:.1f}%")
    print(f"  P90 Drawdown (Worst 10%):    {p10_dd:.1f}%\n")


if __name__ == "__main__":
    main()
