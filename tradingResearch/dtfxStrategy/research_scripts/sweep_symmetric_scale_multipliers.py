"""Comprehensive Sweep of Symmetric Stop & Target Multipliers (1:1R, 2:2R, 3:3R ... 10:10R)
User Request:
- Floor: 5.0 min pip zone
- Stop & Target scaled symmetrically:
  - 1:1R (1x SL, 1x TP - Standard Zone Edge)
  - 1.5:1.5R (1.5x SL, 1.5x TP)
  - 2:2R (2x SL, 2x TP)
  - 3:3R (3x SL, 3x TP)
  - 4:4R (4x SL, 4x TP)
  - 5:5R (5x SL, 5x TP)
  - 6:6R (6x SL, 6x TP)
  - 7:7R (7x SL, 7x TP)
  - 8:8R (8x SL, 8x TP)
  - 9:9R (9x SL, 9x TP)
  - 10:10R (10x SL, 10x TP)
- Model: Two-account sub-account execution ($200 starting capital, 3% risk per trade, 1-per-side concurrency, 1.6-pip spread, 50x leverage).
"""
import os
import sys
import time
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
SPREAD_PIPS = 1.6
MIN_STOP_PIPS = 3.0
MIN_ZONE_PIPS = 5.0
MAX_WINDOW = 1440
RISK_PCT = 0.03

SCALE_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


@njit(cache=True)
def resolve_trades_symmetric_scale(
    z_created, z_top, z_bottom, z_is_buy, high, low,
    spread_price, min_stop_price, max_window, min_zone_height, scale_k
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

        # 3. Setup Price Levels with Symmetric Scale k
        entry_p = mid
        if is_buy:
            raw_stop = mid - scale_k * h
            stop_p = raw_stop - half_spread
            raw_target = mid + scale_k * h
            target_p = raw_target - half_spread
            risk = entry_p - stop_p
            if risk < min_stop_price:
                continue
            win_r_val = (target_p - entry_p) / risk if risk > 0 else 0.0
        else:
            raw_stop = mid + scale_k * h
            stop_p = raw_stop + half_spread
            raw_target = mid - scale_k * h
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


def build_trades_for_scale(cache, scale_k):
    frames = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = MIN_ZONE_PIPS * pip

        entry_idx, exit_idx, outcome, net_r, entry_price, stop_risk = resolve_trades_symmetric_scale(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            spread_price, min_stop_price, MAX_WINDOW, min_zone_height, scale_k
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


def evaluate_scale(trades, scale_k):
    result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
        trades, risk_pct=RISK_PCT, start_per_side=100.0,
        max_leverage=50.0, cap=10_000_000.0, track_detail=True
    )
    acc = result[result["accepted"]].copy().reset_index(drop=True)
    n_exec = len(acc)
    wins = acc[acc["net_r"] > 0]
    losses = acc[acc["net_r"] <= 0]
    wr = 100.0 * len(wins) / n_exec if n_exec else 0.0
    avg_win_r = wins["net_r"].mean() if len(wins) else 0.0
    gw = wins["net_r"].sum()
    gl = -losses["net_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    final_eq = eq_buy + eq_sell
    avg_dur = acc["bars_held"].mean() if n_exec else 0.0

    # 2026 YTD breakdown
    acc_2026 = acc[acc["entry_time"] >= pd.Timestamp("2026-01-01")]
    wins_26 = acc_2026[acc_2026["net_r"] > 0]
    losses_26 = acc_2026[acc_2026["net_r"] <= 0]
    wr_2026 = 100.0 * len(wins_26) / len(acc_2026) if len(acc_2026) else 0.0
    gw_26 = wins_26["net_r"].sum()
    gl_26 = -losses_26["net_r"].sum()
    pf_2026 = gw_26 / gl_26 if gl_26 > 0 else float("inf")

    # Past 30 Days
    latest_t = trades["entry_time"].max()
    month_start = latest_t - pd.Timedelta(days=30)
    month_trades = trades[trades["entry_time"] >= month_start].copy().reset_index(drop=True)
    res_m, eq_b_m, eq_s_m, dd_m, _ = g79.simulate_two_account(
        month_trades, risk_pct=RISK_PCT, start_per_side=100.0,
        max_leverage=50.0, cap=10_000_000.0, track_detail=True
    )
    acc_m = res_m[res_m["accepted"]].copy().reset_index(drop=True)
    wins_m = acc_m[acc_m["net_r"] > 0]
    losses_m = acc_m[acc_m["net_r"] <= 0]
    wr_m = 100.0 * len(wins_m) / len(acc_m) if len(acc_m) else 0.0
    gw_m = wins_m["net_r"].sum()
    gl_m = -losses_m["net_r"].sum()
    pf_m = gw_m / gl_m if gl_m > 0 else float("inf")
    final_eq_m = eq_b_m + eq_s_m
    ret_m = (final_eq_m / 200.0 - 1.0) * 100.0

    return {
        "scale_k": scale_k,
        "n_exec": n_exec,
        "wr": wr,
        "avg_win_r": avg_win_r,
        "pf": pf,
        "max_dd": max_dd * 100.0,
        "avg_dur": avg_dur,
        "n_2026": len(acc_2026),
        "wr_2026": wr_2026,
        "pf_2026": pf_2026,
        "n_30d": len(acc_m),
        "wr_30d": wr_m,
        "pf_30d": pf_m,
        "final_30d": final_eq_m,
        "ret_30d": ret_m,
    }


def main():
    print("==========================================================================")
    print("  SYMMETRIC SCALE SWEEP: 1:1R, 2:2R, 3:3R ... 10:10R")
    print("  Floor: 5.0 Pips | Risk: 3% | Concurrency: 1-per-side | Spread: 1.6 pips")
    print("==========================================================================")
    print("Loading data & precomputing swings...")
    cache = load_cache()

    results = []
    for k in SCALE_MULTIPLIERS:
        t0 = time.time()
        trades = build_trades_for_scale(cache, k)
        res = evaluate_scale(trades, k)
        results.append(res)
        label = f"{k:g}:{k:g}R"
        print(f"  Scale {label:<8} -> Executed: {res['n_exec']:>9,} | WR: {res['wr']:>5.1f}% | Avg Win: +{res['avg_win_r']:>4.2f}R | PF: {res['pf']:>5.2f} | 2026 PF: {res['pf_2026']:>5.2f} | 30D Ret: {res['ret_30d']:>+6.1f}% ({time.time()-t0:.1f}s)")

    # Print Final Markdown Table
    print("\n\n" + "="*115)
    print("  FINAL SYMMETRIC SCALE SWEEP COMPARISON TABLE")
    print("="*115)
    print(f"{'Symmetric Scale':<18} | {'Trades':<10} | {'Win Rate':<8} | {'Avg Win (R)':<12} | {'Multi-Yr PF':<12} | {'2026 WR':<8} | {'2026 PF':<8} | {'30D Ret':<8} | {'Avg Dur':<8}")
    print("-" * 115)
    for r in results:
        label = f"{r['scale_k']:g} : {r['scale_k']:g} R"
        print(f"{label:<18} | {r['n_exec']:<10,} | {r['wr']:>6.2f}% | {r['avg_win_r']:>+8.2f} R    | {r['pf']:>9.3f}    | {r['wr_2026']:>6.2f}% | {r['pf_2026']:>6.2f} | {r['ret_30d']:>+6.1f}% | {r['avg_dur']:>5.1f}m")
    print("="*115)


if __name__ == "__main__":
    main()
