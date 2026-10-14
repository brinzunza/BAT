"""Multi-Touch Zone Rearming Research & Backtest:
Investigating whether rearming a zone after a winning trade (allowing it to be traded again and again until a loss or expiration) benefits the strategy.

Mechanisms Tested:
1. Touch-by-Touch Cohort Analysis:
   - Touch 1 (Initial) vs Touch 2 vs Touch 3 vs Touch 4 vs Touch 5+
   - Measures: Trades count, Win Rate, Avg Win R, Profit Factor, Net Expectancy R.
   - Answers: Does edge increase, stay constant, or decay with repeated touches?

2. Rearm Protocols:
   - Protocol A (Strict Re-overshoot): After a win, price must overshoot the far edge again before re-entering on 50% midpoint touch.
   - Protocol B (Direct Re-touch): After a win, any subsequent return to 50% midpoint triggers entry immediately.

3. Max Rearms Sweep:
   - Cap = 1 (Standard Single-Touch Baseline)
   - Cap = 2 (Allow 1 rearm)
   - Cap = 3 (Allow 2 rearms)
   - Cap = 5
   - Cap = Unlimited (Until Stop Loss or 24h expiration)

4. Evaluated under Full Symmetric Spread (1.6p and 2.5p) and Fixed Margin Monte Carlo.
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
MAX_WINDOW = 1440
RISK_PCT = 0.03


@njit(cache=True)
def resolve_multi_touch_trades(
    z_created, z_top, z_bottom, z_is_buy, high, low,
    spread_price, min_stop_price, max_window, min_zone_height,
    sl_mult, tp_mult, require_re_overshoot, max_touches_cap
):
    n_zones = len(z_created)
    n_bars = len(high)

    # Preallocate trade arrays (assuming up to 10 trades per zone max)
    max_trades_est = n_zones * 10
    out_entry_bar = np.full(max_trades_est, -1, dtype=np.int64)
    out_exit_bar = np.full(max_trades_est, -1, dtype=np.int64)
    out_outcome = np.zeros(max_trades_est, dtype=np.int8) # 1: win, 2: loss
    out_net_r = np.zeros(max_trades_est, dtype=np.float64)
    out_entry_price = np.zeros(max_trades_est, dtype=np.float64)
    out_stop_risk = np.zeros(max_trades_est, dtype=np.float64)
    out_touch_num = np.zeros(max_trades_est, dtype=np.int16)
    out_zone_idx = np.zeros(max_trades_est, dtype=np.int64)

    trade_count = 0
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
        zone_expiry_bar = min(c_bar + max_window, n_bars)

        # Setup price levels with full symmetric spread
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

        # Start multi-touch cycle
        curr_bar = c_bar
        touch_i = 0
        needs_overshoot = True

        while curr_bar < zone_expiry_bar and touch_i < max_touches_cap:
            # 1. Check for overshoot if required
            if needs_overshoot:
                overshot = False
                overshoot_bar = -1
                for bi in range(curr_bar, zone_expiry_bar):
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
                    break
                search_touch_from = overshoot_bar
            else:
                search_touch_from = curr_bar

            # 2. Wait for 50% midpoint touch
            touched = False
            touch_bar = -1
            for bi in range(search_touch_from, zone_expiry_bar):
                if is_buy:
                    if low[bi] <= bottom:
                        # pierced through zone before touching mid
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
                break

            # 3. Resolve trade outcome
            resolved = False
            exit_bar = -1
            trade_outcome = 0
            trade_net_r = 0.0

            for bi in range(touch_bar, zone_expiry_bar):
                if is_buy:
                    if low[bi] <= stop_p:
                        exit_bar = bi
                        trade_outcome = 2
                        trade_net_r = -1.0
                        resolved = True
                        break
                    elif high[bi] >= target_p:
                        exit_bar = bi
                        trade_outcome = 1
                        trade_net_r = win_r_val
                        resolved = True
                        break
                else:
                    if high[bi] >= stop_p:
                        exit_bar = bi
                        trade_outcome = 2
                        trade_net_r = -1.0
                        resolved = True
                        break
                    elif low[bi] <= target_p:
                        exit_bar = bi
                        trade_outcome = 1
                        trade_net_r = win_r_val
                        resolved = True
                        break

            if not resolved:
                # Trade remained unresolved before zone expired
                break

            # Record trade
            touch_i += 1
            if trade_count < max_trades_est:
                out_entry_bar[trade_count] = touch_bar
                out_exit_bar[trade_count] = exit_bar
                out_outcome[trade_count] = trade_outcome
                out_net_r[trade_count] = trade_net_r
                out_entry_price[trade_count] = entry_p
                out_stop_risk[trade_count] = risk
                out_touch_num[trade_count] = touch_i
                out_zone_idx[trade_count] = zi
                trade_count += 1

            # Rearm logic:
            if trade_outcome == 2:
                # Loss -> zone broken, stop rearming
                break
            else:
                # Win -> rearm!
                curr_bar = exit_bar + 1
                needs_overshoot = require_re_overshoot

    return (out_entry_bar[:trade_count], out_exit_bar[:trade_count], out_outcome[:trade_count],
            out_net_r[:trade_count], out_entry_price[:trade_count], out_stop_risk[:trade_count],
            out_touch_num[:trade_count], out_zone_idx[:trade_count])


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


def build_multi_touch_trades(cache, sl_mult, tp_mult, min_pips_dict, require_re_overshoot=True, max_touches=10, spread_pips=1.6):
    frames = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = spread_pips * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = min_pips_dict[inst] * pip

        (entry_bar, exit_bar, outcome, net_r, entry_price, stop_risk,
         touch_num, zone_idx) = resolve_multi_touch_trades(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            spread_price, min_stop_price, MAX_WINDOW, min_zone_height,
            sl_mult, tp_mult, require_re_overshoot, max_touches
        )

        resolved = outcome > 0
        ei = entry_bar[resolved]
        xi = exit_bar[resolved]
        is_buy = d["z_is_buy"][zone_idx[resolved]]
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
            "touch_num": touch_num[resolved],
            "zone_idx": zone_idx[resolved],
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


def evaluate_set(trades_all, trading_days):
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
    print("  RESEARCH: MULTI-TOUCH ZONE REARMING BACKTEST")
    print("==========================================================================")
    cache = load_cache()
    all_times = [d["times"].min() for d in cache.values()] + [d["times"].max() for d in cache.values()]
    total_days = (max(all_times) - min(all_times)).total_seconds() / 86400.0
    trading_days = total_days * (5.0 / 7.0)

    modes = [
        ("10-Pip (1.0SL : 1.0TP)", {"USDJPY": 10.0, "EURUSD": 10.0, "GBPUSD": 10.0, "EURGBP": 10.0}, 1.0, 1.0),
        ("PROD Hybrid (1.0SL : 1.0TP)", {"USDJPY": 10.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0}, 1.0, 1.0),
        ("5-Pip (1.0SL : 1.0TP)", {"USDJPY": 5.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0}, 1.0, 1.0),
    ]

    for mode_name, min_pips, sl, tp in modes:
        print(f"\n" + "=" * 115)
        print(f"  ANALYSIS FOR: {mode_name}")
        print("=" * 115)

        # 1. Pull full multi-touch trades (Protocol A: require re-overshoot)
        trades_proto_a = build_multi_touch_trades(cache, sl, tp, min_pips, require_re_overshoot=True, max_touches=20, spread_pips=1.6)

        print("\n--- 1. Touch Cohort Analysis (Protocol A: Strict Re-overshoot) ---")
        for tn in range(1, 7):
            sub = trades_proto_a[trades_proto_a["touch_num"] == tn]
            cnt = len(sub)
            if cnt == 0:
                continue
            wr = 100.0 * (sub["net_r"] > 0).mean()
            avg_win = sub.loc[sub["net_r"] > 0, "net_r"].mean() if (sub["net_r"] > 0).any() else 0.0
            gw = sub.loc[sub["net_r"] > 0, "net_r"].sum()
            gl = -sub.loc[sub["net_r"] <= 0, "net_r"].sum()
            pf = gw / gl if gl > 0 else 0.0
            ev = sub["net_r"].mean()
            pct = (cnt / len(trades_proto_a)) * 100.0
            print(f"  Touch #{tn:<2} | Trades: {cnt:<6} ({pct:>5.1f}%) | WR: {wr:>5.1f}% | AvgWin: +{avg_win:.2f}R | PF: {pf:>4.2f} | EV: {ev:>+5.3f}R")

        sub_high = trades_proto_a[trades_proto_a["touch_num"] >= 7]
        if len(sub_high):
            wr = 100.0 * (sub_high["net_r"] > 0).mean()
            avg_win = sub_high.loc[sub_high["net_r"] > 0, "net_r"].mean()
            gw = sub_high.loc[sub_high["net_r"] > 0, "net_r"].sum()
            gl = -sub_high.loc[sub_high["net_r"] <= 0, "net_r"].sum()
            pf = gw / gl if gl > 0 else 0.0
            ev = sub_high["net_r"].mean()
            print(f"  Touch #7+ | Trades: {len(sub_high):<6} ({(len(sub_high)/len(trades_proto_a))*100:>5.1f}%) | WR: {wr:>5.1f}% | AvgWin: +{avg_win:.2f}R | PF: {pf:>4.2f} | EV: {ev:>+5.3f}R")

        # 2. Protocol A: Max Touches Cap Sweep
        print("\n--- 2. Protocol A (Require Re-Overshoot): Max Touches Cap Sweep ---")
        caps = [1, 2, 3, 5, 10, 20]
        for cap in caps:
            sub = trades_proto_a[trades_proto_a["touch_num"] <= cap]
            n_exec, tpd, wr, avg_win, pf, med_days, med_dd = evaluate_set(sub, trading_days)
            desc = "Single-Touch Baseline" if cap == 1 else f"Max {cap} Touches"
            print(f"  Cap: {desc:<22} | Execs: {n_exec:<6} | T/Day: {tpd:>5.1f} | WR: {wr:>5.1f}% | AvgWin: +{avg_win:.2f}R | PF: {pf:>4.2f} | MC $1M: {med_days:>5.1f}d | MC DD: {med_dd:>5.1f}%")

        # 3. Pull full multi-touch trades (Protocol B: Direct Re-touch without re-overshoot)
        trades_proto_b = build_multi_touch_trades(cache, sl, tp, min_pips, require_re_overshoot=False, max_touches=20, spread_pips=1.6)

        print("\n--- 3. Touch Cohort Analysis (Protocol B: Direct Retouch / No Re-overshoot) ---")
        for tn in range(1, 7):
            sub = trades_proto_b[trades_proto_b["touch_num"] == tn]
            cnt = len(sub)
            if cnt == 0:
                continue
            wr = 100.0 * (sub["net_r"] > 0).mean()
            avg_win = sub.loc[sub["net_r"] > 0, "net_r"].mean() if (sub["net_r"] > 0).any() else 0.0
            gw = sub.loc[sub["net_r"] > 0, "net_r"].sum()
            gl = -sub.loc[sub["net_r"] <= 0, "net_r"].sum()
            pf = gw / gl if gl > 0 else 0.0
            ev = sub["net_r"].mean()
            pct = (cnt / len(trades_proto_b)) * 100.0
            print(f"  Touch #{tn:<2} | Trades: {cnt:<6} ({pct:>5.1f}%) | WR: {wr:>5.1f}% | AvgWin: +{avg_win:.2f}R | PF: {pf:>4.2f} | EV: {ev:>+5.3f}R")

        print("\n--- 4. Protocol B (Direct Retouch): Max Touches Cap Sweep ---")
        for cap in caps:
            sub = trades_proto_b[trades_proto_b["touch_num"] <= cap]
            n_exec, tpd, wr, avg_win, pf, med_days, med_dd = evaluate_set(sub, trading_days)
            desc = "Single-Touch Baseline" if cap == 1 else f"Max {cap} Touches"
            print(f"  Cap: {desc:<22} | Execs: {n_exec:<6} | T/Day: {tpd:>5.1f} | WR: {wr:>5.1f}% | AvgWin: +{avg_win:.2f}R | PF: {pf:>4.2f} | MC $1M: {med_days:>5.1f}d | MC DD: {med_dd:>5.1f}%")


if __name__ == "__main__":
    main()
