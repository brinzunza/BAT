"""Zone Maturity Backtest & Research:
Investigating whether Zone Age (Maturity) before midpoint entry improves edge, win rate, profit factor, and compounding velocity.

Age Definitions:
- total_zone_age = touch_bar - z_created (time in minutes from zone formation to 50% midpoint touch entry)
- overshoot_age = overshoot_bar - z_created (time from formation to breakout overshoot)
- retracement_age = touch_bar - overshoot_bar (time from overshoot to 50% midpoint pullback)

Tests:
1. Age Bucket Breakdown: Win rate, Profit factor, Net R across fine-grained age bins (0-15m, 15-30m, 30-60m, 1-2h, 2-4h, 4-8h, 8-12h, 12-24h, >24h).
2. Min Age Sweep: min_age in [0, 5, 10, 15, 30, 45, 60, 90, 120, 240, 480, 720] min.
3. Max Age Sweep: max_age in [15, 30, 45, 60, 90, 120, 240, 480, 720, 1440, 2880, 4320] min.
4. 2D Maturity Window Sweep: (min_age, max_age) combinations.
5. Evaluated across 10-Pip zones, 5-Pip zones, and PROD Hybrid config under Full Symmetric Spread (1.6p and 2.5p).
"""
import os
import sys
import time
import heapq
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal95_every_swing_zones as g95
import goal79_two_account_portfolio as g79

INSTRUMENTS = {
    "EURUSD": ("lse_eurusd_1min_full.csv", 1e-4),
    "GBPUSD": ("lse_gbpusd_1min_full.csv", 1e-4),
    "EURGBP": ("lse_eurgbp_1min_full.csv", 1e-4),
    "USDJPY": ("lse_usdjpy_1min_2018on.csv", 1e-2),
}
MIN_STOP_PIPS = 3.0
MAX_SEARCH_WINDOW = 4320  # up to 3 days (72h) to allow wide max_age testing
RISK_PCT = 0.03


@njit(cache=True)
def resolve_trades_with_age(
    z_created, z_top, z_bottom, z_is_buy, high, low,
    spread_price, min_stop_price, max_window, min_zone_height,
    sl_mult, tp_mult, min_age, max_age
):
    n_zones = len(z_created)
    n_bars = len(high)

    entry_idx = np.full(n_zones, -1, dtype=np.int64)
    exit_idx = np.full(n_zones, -1, dtype=np.int64)
    outcome = np.zeros(n_zones, dtype=np.int8)  # 0: unres, 1: win, 2: loss
    net_r = np.zeros(n_zones, dtype=np.float64)
    entry_price_out = np.zeros(n_zones, dtype=np.float64)
    stop_risk_out = np.zeros(n_zones, dtype=np.float64)
    zone_age_out = np.zeros(n_zones, dtype=np.int64)
    overshoot_age_out = np.zeros(n_zones, dtype=np.int64)
    retrace_age_out = np.zeros(n_zones, dtype=np.int64)

    half_spread = spread_price / 2.0

    for zi in range(n_zones):
        top = z_top[zi]
        bottom = z_bottom[zi]
        height = top - bottom
        if height < min_zone_height:
            continue

        h = height / 2.0
        mid = (top + bottom) / 2.0
        c_bar = z_created[zi]
        is_buy = z_is_buy[zi]

        # 1. Wait for overshoot within max_window
        overshot = False
        overshoot_bar = -1
        w_end = min(c_bar + max_window, n_bars)

        for bi in range(c_bar, w_end):
            if is_buy:
                if high[bi] > top:
                    overshot = True
                    overshoot_bar = bi
                    break
            else:
                if low[bi] < bottom:
                    overshot = True
                    overshoot_bar = bi
                    break

        if not overshot:
            continue

        # 2. Wait for 50% midpoint touch
        touched = False
        touch_bar = -1
        w_end2 = min(overshoot_bar + max_window, n_bars)

        for bi in range(overshoot_bar, w_end2):
            if is_buy:
                if low[bi] <= bottom:
                    break
                if low[bi] <= mid:
                    touched = True
                    touch_bar = bi
                    break
            else:
                if high[bi] >= top:
                    break
                if high[bi] >= mid:
                    touched = True
                    touch_bar = bi
                    break

        if not touched:
            continue

        # Age filters
        total_age = touch_bar - c_bar
        if total_age < min_age or total_age > max_age:
            continue

        # 3. Setup Price Levels with FULL SYMMETRIC SPREAD
        if is_buy:
            entry_p = mid + half_spread
            raw_stop = mid - sl_mult * h
            stop_p = raw_stop - half_spread
            raw_target = mid + tp_mult * h
            target_p = raw_target - half_spread
            risk = entry_p - stop_p
            if risk < min_stop_price:
                continue
            win_r_val = (target_p - entry_p) / risk if risk > 0 else 0.0
        else:
            entry_p = mid - half_spread
            raw_stop = mid + sl_mult * h
            stop_p = raw_stop + half_spread
            raw_target = mid - tp_mult * h
            target_p = raw_target + half_spread
            risk = stop_p - entry_p
            if risk < min_stop_price:
                continue
            win_r_val = (entry_p - target_p) / risk if risk > 0 else 0.0

        # 4. Resolve Exit
        w_end3 = min(touch_bar + max_window, n_bars)
        for bi in range(touch_bar, w_end3):
            if is_buy:
                if low[bi] <= stop_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 2
                    net_r[zi] = -1.0
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    zone_age_out[zi] = total_age
                    overshoot_age_out[zi] = overshoot_bar - c_bar
                    retrace_age_out[zi] = touch_bar - overshoot_bar
                    break
                elif high[bi] >= target_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 1
                    net_r[zi] = win_r_val
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    zone_age_out[zi] = total_age
                    overshoot_age_out[zi] = overshoot_bar - c_bar
                    retrace_age_out[zi] = touch_bar - overshoot_bar
                    break
            else:
                if high[bi] >= stop_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 2
                    net_r[zi] = -1.0
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    zone_age_out[zi] = total_age
                    overshoot_age_out[zi] = overshoot_bar - c_bar
                    retrace_age_out[zi] = touch_bar - overshoot_bar
                    break
                elif low[bi] <= target_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 1
                    net_r[zi] = win_r_val
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    zone_age_out[zi] = total_age
                    overshoot_age_out[zi] = overshoot_bar - c_bar
                    retrace_age_out[zi] = touch_bar - overshoot_bar
                    break

    return (entry_idx, exit_idx, outcome, net_r, entry_price_out, stop_risk_out,
            zone_age_out, overshoot_age_out, retrace_age_out)


def load_cache():
    cache = {}
    for inst, (fname, pip) in INSTRUMENTS.items():
        path = os.path.join(DATA_DIR, fname)
        fresh_path = f"/tmp/dtfx_fresh_data/{inst}_merged.csv"
        if os.path.exists(fresh_path):
            path = fresh_path

        df = pd.read_csv(path, parse_dates=["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df = df.drop_duplicates(subset="date").set_index("date").sort_index()

        times = df.index
        high_ = df["high"].values.astype(np.float64)
        low_ = df["low"].values.astype(np.float64)
        close_ = df["close"].values.astype(np.float64)

        confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high_, low_, close_)
        (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
         ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)

        cache[inst] = dict(
            pip=pip, times=times, high=high_, low=low_, close=close_,
            z_created=z_created, z_top=z_top, z_bottom=z_bottom, z_is_buy=z_is_buy
        )
    return cache


def build_trades(cache, sl_mult, tp_mult, min_pips_dict, min_age=0, max_age=4320, spread_pips=1.6):
    frames = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = spread_pips * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = min_pips_dict[inst] * pip

        (entry_idx, exit_idx, outcome, net_r, entry_price, stop_risk,
         zone_age, overshoot_age, retrace_age) = resolve_trades_with_age(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            spread_price, min_stop_price, MAX_SEARCH_WINDOW, min_zone_height,
            sl_mult, tp_mult, min_age, max_age
        )

        resolved = outcome > 0
        ei = entry_idx[resolved]
        xi = exit_idx[resolved]
        is_buy = d["z_is_buy"][resolved]
        bars_held = xi - ei

        frames.append(pd.DataFrame({
            "instrument": inst,
            "symbol": inst,
            "entry_time": d["times"][ei],
            "exit_time": d["times"][xi],
            "entry_price": entry_price[resolved],
            "stop_risk": stop_risk[resolved],
            "net_r": net_r[resolved],
            "is_buy": is_buy,
            "bars_held": bars_held,
            "zone_age_min": zone_age[resolved],
            "overshoot_age_min": overshoot_age[resolved],
            "retrace_age_min": retrace_age[resolved],
        }))
    trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return trades


def run_mc_fixed(trades, n_sims=300, target=1_000_000.0, seed=42):
    rng = np.random.default_rng(seed)
    n = len(trades)
    if n == 0:
        return 999.0, 0.0

    entry_ns = trades.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = trades.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = trades.entry_price.values
    stop_risk = trades.stop_risk.values
    net_r = trades.net_r.values
    instrument = trades.instrument.values
    is_buy = trades.is_buy.values

    BLOCK = 50
    START_PER_SIDE = 100.0
    MAX_LEVERAGE = 50.0
    ACCOUNT_CAP = 10_000_000.0
    RUIN_FLOOR = 20.0

    days_to_target = []
    max_dds = []

    for sim_i in range(n_sims):
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

        while trades_count < 50_000 and not hit_target and not ruined:
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
                _, risk_dollars, margin = g79.position_size_and_margin(
                    instrument[idx], sizing_base, entry_price[idx], stop_risk[idx], RISK_PCT,
                    MAX_LEVERAGE, ACCOUNT_CAP)
                avail = min(equity[side], ACCOUNT_CAP) - margin_used[side]

                if margin <= avail and open_count[side] < 1:
                    heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx], side))
                    margin_used[side] += margin
                    open_count[side] += 1
                    trades_count += 1

                comb = equity[True] + equity[False]
                if comb >= target:
                    hit_target = True
                    t_hit_days = et / 86400e9
                    break

            clock = block_exit_max

        if hit_target:
            days_to_target.append(t_hit_days)
        max_dds.append(worst_dd)

    med_days = float(np.median(days_to_target)) if len(days_to_target) else 999.0
    med_dd = float(np.median(max_dds))
    return med_days, med_dd


def evaluate_set(trades_all, trading_days, spread_pips=1.6):
    res, _, _, _, _ = g79.simulate_two_account(
        trades_all, risk_pct=RISK_PCT, start_per_side=100.0,
        max_leverage=50.0, cap=10_000_000.0, track_detail=False
    )
    acc = res[res["accepted"]]
    n_exec = len(acc)
    if n_exec == 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 999.0, 0.0
    tpd = n_exec / trading_days
    wr = 100.0 * (acc["net_r"] > 0).mean()
    avg_win = acc.loc[acc["net_r"] > 0, "net_r"].mean() if (acc["net_r"] > 0).any() else 0.0
    gw = acc.loc[acc["net_r"] > 0, "net_r"].sum()
    gl = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
    pf = gw / gl if gl > 0 else 0.0
    med_days, med_dd = run_mc_fixed(trades_all, n_sims=300)
    return n_exec, tpd, wr, avg_win, pf, med_days, med_dd


def main():
    print("==========================================================================")
    print("  RESEARCH: IMPACT OF ZONE MATURITY (AGE) ON TRADING EDGE")
    print("==========================================================================")
    cache = load_cache()
    all_times = [d["times"].min() for d in cache.values()] + [d["times"].max() for d in cache.values()]
    total_days = (max(all_times) - min(all_times)).total_seconds() / 86400.0
    trading_days = total_days * (5.0 / 7.0)

    # Test baseline on 10-Pip (1.0SL : 1.0TP) and PROD Hybrid (1.0SL : 1.0TP)
    modes = [
        ("10-Pip (1.0SL : 1.0TP)", {"USDJPY": 10.0, "EURUSD": 10.0, "GBPUSD": 10.0, "EURGBP": 10.0}, 1.0, 1.0),
        ("PROD Hybrid (1.0SL : 1.0TP)", {"USDJPY": 10.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0}, 1.0, 1.0),
        ("5-Pip (1.0SL : 1.0TP)", {"USDJPY": 5.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0}, 1.0, 1.0),
    ]

    for mode_name, min_pips, sl, tp in modes:
        print(f"\n" + "=" * 110)
        print(f"  ANALYSIS FOR: {mode_name}")
        print("=" * 110)

        # 1. Pull unfiltered all-age trades (up to 72h max)
        trades_full = build_trades(cache, sl, tp, min_pips, min_age=0, max_age=4320, spread_pips=1.6)
        
        # --- SECTION 1: AGE BUCKET BREAKDOWN ---
        print("\n--- 1. Zone Age at Entry (Maturity) Bucket Breakdown ---")
        bins = [0, 15, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320]
        labels = ["0-15m", "15-30m", "30-60m", "1-2h", "2-4h", "4-8h", "8-12h", "12-24h", "24-48h", "48-72h"]
        trades_full["age_bucket"] = pd.cut(trades_full["zone_age_min"], bins=bins, labels=labels, right=False)

        bucket_stats = []
        for l in labels:
            sub = trades_full[trades_full["age_bucket"] == l]
            cnt = len(sub)
            if cnt == 0:
                continue
            wr = 100.0 * (sub["net_r"] > 0).mean()
            avg_win = sub.loc[sub["net_r"] > 0, "net_r"].mean() if (sub["net_r"] > 0).any() else 0.0
            gw = sub.loc[sub["net_r"] > 0, "net_r"].sum()
            gl = -sub.loc[sub["net_r"] <= 0, "net_r"].sum()
            pf = gw / gl if gl > 0 else 0.0
            ev = sub["net_r"].mean()
            pct_of_trades = (cnt / len(trades_full)) * 100.0
            bucket_stats.append({
                "Bucket": l, "Trades": cnt, "% of Total": pct_of_trades,
                "WR": wr, "AvgWin(R)": avg_win, "PF": pf, "EV(R)": ev
            })
            print(f"  Age: {l:<8} | Trades: {cnt:<6} ({pct_of_trades:>5.1f}%) | WR: {wr:>5.1f}% | AvgWin: +{avg_win:.2f}R | PF: {pf:>4.2f} | EV: {ev:>+5.3f}R")

        # --- SECTION 2: MIN AGE FILTER SWEEP ---
        print("\n--- 2. Minimum Age Filter Sweep (Zone must be at least N min old) ---")
        min_ages = [0, 5, 10, 15, 30, 45, 60, 90, 120, 240, 480]
        for ma in min_ages:
            sub = trades_full[trades_full["zone_age_min"] >= ma]
            n_exec, tpd, wr, avg_win, pf, med_days, med_dd = evaluate_set(sub, trading_days)
            print(f"  Min Age: {ma:>3}m | Execs: {n_exec:<6} | T/Day: {tpd:>5.1f} | WR: {wr:>5.1f}% | AvgWin: +{avg_win:.2f}R | PF: {pf:>4.2f} | MC $1M: {med_days:>5.1f}d | MC DD: {med_dd:>5.1f}%")

        # --- SECTION 3: MAX AGE FILTER SWEEP ---
        print("\n--- 3. Maximum Age Filter Sweep (Zone must enter within N min of formation) ---")
        max_ages = [15, 30, 60, 120, 240, 480, 720, 1440, 2880, 4320]
        for ma in max_ages:
            sub = trades_full[trades_full["zone_age_min"] <= ma]
            n_exec, tpd, wr, avg_win, pf, med_days, med_dd = evaluate_set(sub, trading_days)
            print(f"  Max Age: {ma:>4}m ({ma/60:>4.1f}h) | Execs: {n_exec:<6} | T/Day: {tpd:>5.1f} | WR: {wr:>5.1f}% | AvgWin: +{avg_win:.2f}R | PF: {pf:>4.2f} | MC $1M: {med_days:>5.1f}d | MC DD: {med_dd:>5.1f}%")

        # --- SECTION 4: 2D MATURITY WINDOW SWEEP (Min Age x Max Age) ---
        print("\n--- 4. 2D Maturity Window Sweep (Min Age x Max Age) ---")
        windows = [
            (0, 1440, "0m - 24h (Standard Baseline)"),
            (0, 60, "0m - 1h (Ultra Fresh)"),
            (0, 120, "0m - 2h (Fresh)"),
            (0, 240, "0m - 4h (Active Session)"),
            (0, 480, "0m - 8h (Full Session)"),
            (15, 1440, "15m - 24h (Skip Immediate Flash-touches)"),
            (30, 1440, "30m - 24h (Matured 30m+)"),
            (60, 1440, "60m - 24h (Matured 1h+)"),
            (15, 240, "15m - 4h (Prime Sweet Spot)"),
            (30, 480, "30m - 8h (Matured Session)"),
        ]
        for min_a, max_a, desc in windows:
            sub = trades_full[(trades_full["zone_age_min"] >= min_a) & (trades_full["zone_age_min"] <= max_a)]
            n_exec, tpd, wr, avg_win, pf, med_days, med_dd = evaluate_set(sub, trading_days)
            print(f"  Window: {desc:<38} | Execs: {n_exec:<6} | T/Day: {tpd:>5.1f} | WR: {wr:>5.1f}% | PF: {pf:>4.2f} | MC $1M: {med_days:>5.1f}d | MC DD: {med_dd:>5.1f}%")


if __name__ == "__main__":
    main()
