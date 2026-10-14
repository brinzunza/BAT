"""2D Grid Optimization & Monte Carlo Race to $1,000,000 for 10-PIP MINIMUM ZONES:
Finding the Optimal (Stop Loss Multiplier, Take Profit Multiplier) Combination for Maximum Spread Robustness and Fastest Race Velocity.

Parameters:
- MIN_ZONE_PIPS: 10.0
- Grid:
  - Stop Multiplier (M_sl): [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
  - Target Multiplier (M_tp): [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

Metrics:
1. Spread Robustness: Performance under standard 1.6-pip spread vs stressed 2.5-pip spread.
2. Speed to $1M: Days and trades needed to compound $200 -> $1,000,000.
3. Safety: Maximum Drawdown & Ruin Rate.
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
import goal99_dtfx_base_backtest as g99
import goal79_two_account_portfolio as g79

INSTRUMENTS = {
    "EURUSD": ("lse_eurusd_1min_full.csv", 1e-4),
    "GBPUSD": ("lse_gbpusd_1min_full.csv", 1e-4),
    "EURGBP": ("lse_eurgbp_1min_full.csv", 1e-4),
    "USDJPY": ("lse_usdjpy_1min_2018on.csv", 1e-2),
}
MIN_ZONE_PIPS = 10.0
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440
RISK_PCT = 0.03

SL_GRID = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
TP_GRID = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


@njit(cache=True)
def resolve_trades_grid(
    z_created, z_top, z_bottom, z_is_buy, high, low,
    spread_price, min_stop_price, max_window, min_zone_height,
    sl_mult, tp_mult
):
    n_zones = len(z_created)
    n_bars = len(high)

    entry_idx = np.full(n_zones, -1, dtype=np.int64)
    exit_idx = np.full(n_zones, -1, dtype=np.int64)
    outcome = np.zeros(n_zones, dtype=np.int8)  # 0: unres, 1: win, 2: loss
    net_r = np.zeros(n_zones, dtype=np.float64)
    entry_price_out = np.zeros(n_zones, dtype=np.float64)
    stop_risk_out = np.zeros(n_zones, dtype=np.float64)

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

        # 1. Wait for overshoot
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

        # 3. Setup Price Levels with (sl_mult, tp_mult)
        entry_p = mid
        if is_buy:
            raw_stop = mid - sl_mult * h
            stop_p = raw_stop - half_spread
            raw_target = mid + tp_mult * h
            target_p = raw_target - half_spread
            risk = entry_p - stop_p
            if risk < min_stop_price:
                continue
            win_r_val = (target_p - entry_p) / risk if risk > 0 else 0.0
        else:
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
                    break
                elif high[bi] >= target_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 1
                    net_r[zi] = win_r_val
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
            else:
                if high[bi] >= stop_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 2
                    net_r[zi] = -1.0
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
                elif low[bi] <= target_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 1
                    net_r[zi] = win_r_val
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break

    return entry_idx, exit_idx, outcome, net_r, entry_price_out, stop_risk_out


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


def build_trades_for_pair(cache, sl_mult, tp_mult, spread_pips=1.6):
    frames = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = spread_pips * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = MIN_ZONE_PIPS * pip

        entry_idx, exit_idx, outcome, net_r, entry_price, stop_risk = resolve_trades_grid(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            spread_price, min_stop_price, MAX_WINDOW, min_zone_height, sl_mult, tp_mult
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
        }))
    trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return trades


def run_mc_fast(trades, n_sims=300, target=1_000_000.0, seed=42):
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
                    margin_used[side] -= margin
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


def main():
    print("==========================================================================")
    print("  2D GRID SEARCH FOR 10-PIP ZONES: SL MULTIPLIER x TP MULTIPLIER")
    print("  Finding the Fastest Race to $1M & Most Spread-Resilient Configuration")
    print("==========================================================================")
    cache = load_cache()

    grid_results = []
    print("\nScanning 2D Grid (SL Multiplier vs TP Multiplier) for MIN_ZONE_PIPS = 10.0...")

    for sl in SL_GRID:
        for tp in TP_GRID:
            t0 = time.time()
            # 1. Standard Spread 1.6 pips
            trades_std = build_trades_for_pair(cache, sl, tp, spread_pips=1.6)
            
            # Deterministic simulation
            res_std, eq_b, eq_s, max_dd, _ = g79.simulate_two_account(
                trades_std, risk_pct=RISK_PCT, start_per_side=100.0,
                max_leverage=50.0, cap=10_000_000.0, track_detail=False
            )
            acc = res_std[res_std["accepted"]]
            n_exec = len(acc)
            wr = 100.0 * (acc["net_r"] > 0).mean() if n_exec else 0.0
            avg_win = acc.loc[acc["net_r"] > 0, "net_r"].mean() if (acc["net_r"] > 0).any() else 0.0
            gw = acc.loc[acc["net_r"] > 0, "net_r"].sum()
            gl = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
            pf_std = gw / gl if gl > 0 else float("inf")
            avg_dur = acc["bars_held"].mean() if n_exec else 0.0

            # 2. Stressed Spread 2.5 pips (Stress test)
            trades_stress = build_trades_for_pair(cache, sl, tp, spread_pips=2.5)
            res_stress, _, _, _, _ = g79.simulate_two_account(
                trades_stress, risk_pct=RISK_PCT, start_per_side=100.0,
                max_leverage=50.0, cap=10_000_000.0, track_detail=False
            )
            acc_str = res_stress[res_stress["accepted"]]
            gw_str = acc_str.loc[acc_str["net_r"] > 0, "net_r"].sum()
            gl_str = -acc_str.loc[acc_str["net_r"] <= 0, "net_r"].sum()
            pf_stress = gw_str / gl_str if gl_str > 0 else float("inf")
            
            # Spread Degradation %: how much PF drops when spread widens from 1.6p -> 2.5p
            pf_drop_pct = ((pf_std - pf_stress) / pf_std) * 100.0 if pf_std > 0 else 100.0

            # 3. Monte Carlo Race to $1M (300 paths for fast ranking)
            med_days_1m, med_dd_mc = run_mc_fast(trades_std, n_sims=300, target=1_000_000.0)

            res_entry = {
                "sl": sl,
                "tp": tp,
                "ratio": f"{sl}SL : {tp}TP",
                "n_exec": n_exec,
                "wr": wr,
                "avg_win_r": avg_win,
                "pf_1.6p": pf_std,
                "pf_2.5p": pf_stress,
                "pf_drop_pct": pf_drop_pct,
                "med_days_1m": med_days_1m,
                "med_dd_mc": med_dd_mc,
                "avg_dur": avg_dur,
            }
            grid_results.append(res_entry)

            print(f"  SL={sl:<4} TP={tp:<4} -> Execs: {n_exec:<5} | WR: {wr:>5.1f}% | AvgWin: +{avg_win:>4.2f}R | PF(1.6p): {pf_std:>5.2f} | PF(2.5p): {pf_stress:>5.2f} | Days to $1M: {med_days_1m:>5.1f}d | MC DD: {med_dd_mc:>5.1f}% ({time.time()-t0:.1f}s)")

    df_grid = pd.DataFrame(grid_results)

    # Sort by Fastest to $1M
    df_fastest = df_grid.sort_values("med_days_1m").reset_index(drop=True)

    print("\n\n" + "="*125)
    print("  TOP 10 FASTEST CONFIGURATIONS TO $1,000,000 (10-PIP ZONES)")
    print("="*125)
    print(f"{'Rank':<5} | {'Config':<15} | {'Days to $1M':<12} | {'Win Rate':<9} | {'Avg Win (R)':<12} | {'PF (1.6p)':<10} | {'PF (2.5p)':<10} | {'Spread Resilience':<18} | {'MC Max DD':<10}")
    print("-" * 125)
    for i, r in df_fastest.head(10).iterrows():
        resilience = f"-{r['pf_drop_pct']:.1f}% drop"
        print(f"#{i+1:<4} | {r['ratio']:<15} | {r['med_days_1m']:>6.1f} days  | {r['wr']:>6.2f}%   | {r['avg_win_r']:>+8.2f} R    | {r['pf_1.6p']:>8.2f}  | {r['pf_2.5p']:>8.2f}  | {resilience:<18} | {r['med_dd_mc']:>6.1f}%")

    # Sort by Most Spread-Resilient (Lowest PF drop under stress) among high-PF configs
    high_pf = df_grid[df_grid["pf_1.6p"] >= 1.8].sort_values("pf_drop_pct").reset_index(drop=True)
    print("\n\n" + "="*125)
    print("  TOP 5 MOST SPREAD-ROBUST CONFIGURATIONS (10-PIP ZONES)")
    print("="*125)
    print(f"{'Rank':<5} | {'Config':<15} | {'Days to $1M':<12} | {'Win Rate':<9} | {'Avg Win (R)':<12} | {'PF (1.6p)':<10} | {'PF (2.5p)':<10} | {'Spread Resilience':<18} | {'MC Max DD':<10}")
    print("-" * 125)
    for i, r in high_pf.head(5).iterrows():
        resilience = f"-{r['pf_drop_pct']:.1f}% drop"
        print(f"#{i+1:<4} | {r['ratio']:<15} | {r['med_days_1m']:>6.1f} days  | {r['wr']:>6.2f}%   | {r['avg_win_r']:>+8.2f} R    | {r['pf_1.6p']:>8.2f}  | {r['pf_2.5p']:>8.2f}  | {resilience:<18} | {r['med_dd_mc']:>6.1f}%")
    print("="*125)

    # Save to CSV for reference
    out_csv = os.path.join(SCRIPT_DIR, "grid_10pip_results.csv")
    df_grid.to_csv(out_csv, index=False)
    print(f"\nSaved full results to {out_csv}")


if __name__ == "__main__":
    main()
