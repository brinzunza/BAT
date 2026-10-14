"""Backtest & Comparison: True 1:1 Net R (Post-Spread Target Adjustment & Market Entry)
User Request:
1. Market order entry at 50% zone touch (enters on Ask for Buy, Bid for Sell).
2. Risk R = Entry Price to Stop Price (including spread friction).
3. Target Price is pushed out by the spread so that Net Win = +1.0R and Net Loss = -1.0R (True 1:1 Net R).
4. Run comparison against the standard baseline for:
   - Past 30 Days (Fresh $200 account growth)
   - 2026 YTD
   - Full Multi-Year History
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
ZONE_PIPS_MAP = {
    "USDJPY": 10.0,
    "EURUSD": 5.0,
    "GBPUSD": 5.0,
    "EURGBP": 5.0,
}
SPREAD_PIPS = 1.6
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440
RISK_PCT = 0.03


@njit(cache=True)
def resolve_true_1r_market_entry(
    z_created, z_top, z_bottom, z_is_buy, high, low,
    spread_price, min_stop_price, max_window, min_zone_height
):
    """Resolve DTFX zones with:
    - Market entry on 50% touch (Buy on Ask = mid + spread/2, Sell on Bid = mid - spread/2)
    - Stop placed at opposite zone edge (Buy stop at bottom - spread/2, Sell stop at top + spread/2)
    - Effective Risk R = abs(Entry - Stop)
    - Target placed at Entry +/- R so that Win = +1.0R Net and Loss = -1.0R Net
    """
    n_zones = len(z_created)
    n_bars = len(high)

    entry_idx = np.full(n_zones, -1, dtype=np.int64)
    exit_idx = np.full(n_zones, -1, dtype=np.int64)
    outcome = np.zeros(n_zones, dtype=np.int8)  # 0: unresolved, 1: win (+1R), 2: loss (-1R)
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

        # 2. Wait for 50% midpoint touch after overshoot
        touched = False
        touch_bar = -1
        w_end2 = min(overshoot_bar + max_window, n_bars)

        for bi in range(overshoot_bar, w_end2):
            if is_buy:
                # Invalidation if price breaches bottom before tagging mid
                if low[bi] <= bottom:
                    break
                if low[bi] <= mid:
                    touched = True
                    touch_bar = bi
                    break
            else:
                # Invalidation if price breaches top before tagging mid
                if high[bi] >= top:
                    break
                if high[bi] >= mid:
                    touched = True
                    touch_bar = bi
                    break

        if not touched:
            continue

        # 3. Market Entry & Price Levels
        if is_buy:
            entry_p = mid + half_spread          # Market Buy fill on Ask
            stop_p = bottom - half_spread        # Stop exit on Bid
            risk = entry_p - stop_p              # Full Risk in price
            if risk < min_stop_price:
                continue
            # Target is placed so that exit Bid - entry Ask = risk
            target_bid = entry_p + risk
            target_mid = target_bid + half_spread  # Mid chart price for target
        else:
            entry_p = mid - half_spread          # Market Sell fill on Bid
            stop_p = top + half_spread           # Stop exit on Ask
            risk = stop_p - entry_p              # Full Risk in price
            if risk < min_stop_price:
                continue
            # Target is placed so that entry Bid - exit Ask = risk
            target_ask = entry_p - risk
            target_mid = target_ask - half_spread  # Mid chart price for target

        # 4. Resolve Exit
        w_end3 = min(touch_bar + max_window, n_bars)
        for bi in range(touch_bar, w_end3):
            if is_buy:
                # Check stop (Bid <= stop_p => Low <= bottom)
                if low[bi] <= bottom:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 2
                    net_r[zi] = -1.0
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
                # Check target (Bid >= target_bid => High >= target_mid)
                elif high[bi] >= target_mid:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 1
                    net_r[zi] = 1.0
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
            else:
                # Check stop (Ask >= stop_p => High >= top)
                if high[bi] >= top:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 2
                    net_r[zi] = -1.0
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
                # Check target (Ask <= target_ask => Low <= target_mid)
                elif low[bi] <= target_mid:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 1
                    net_r[zi] = 1.0
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
        n_bars = len(df)

        confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high_, low_, close_)
        (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
         ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)
        last_high_of_bar, last_low_of_bar = g99.build_last_confirmed(confirm_idx, is_high, price, n_bars)

        cache[inst] = dict(
            pip=pip, times=times, high=high_, low=low_, close=close_,
            z_created=z_created, z_top=z_top, z_bottom=z_bottom, z_is_buy=z_is_buy,
            last_high_of_bar=last_high_of_bar, last_low_of_bar=last_low_of_bar
        )
    return cache


def build_trades_true_1r(cache):
    frames = []
    for inst, d in cache.items():
        min_zone_pips = ZONE_PIPS_MAP[inst]
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = min_zone_pips * pip

        entry_idx, exit_idx, outcome, net_r, entry_price, stop_risk = resolve_true_1r_market_entry(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            spread_price, min_stop_price, MAX_WINDOW, min_zone_height
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


def build_trades_standard_baseline(cache):
    frames = []
    for inst, d in cache.items():
        min_zone_pips = ZONE_PIPS_MAP[inst]
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = min_zone_pips * pip

        entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            d["last_high_of_bar"], d["last_low_of_bar"], spread_price, min_stop_price, MAX_WINDOW,
            False, min_zone_height, True
        )

        resolved = outcome > 0
        ei = entry_idx[resolved]
        xi = exit_idx[resolved]
        is_buy = d["z_is_buy"][resolved]
        raw_stop = np.where(is_buy, d["z_bottom"][resolved], d["z_top"][resolved])
        half_spread = spread_price / 2.0
        stop_price = np.where(is_buy, raw_stop - half_spread, raw_stop + half_spread)
        stop_risk = np.abs(entry_price[resolved] - stop_price)
        bars_held = xi - ei

        frames.append(pd.DataFrame({
            "instrument": inst,
            "symbol": inst,
            "entry_time": d["times"][ei],
            "exit_time": d["times"][xi],
            "entry_price": entry_price[resolved],
            "stop_risk": stop_risk,
            "net_r": net_r[resolved],
            "is_buy": is_buy,
            "bars_held": bars_held,
        }))
    trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return trades


def evaluate(trades, label=""):
    print(f"\n{'='*75}\nEvaluating: {label}\n{'='*75}")
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
    avg_loss_r = losses["net_r"].mean() if len(losses) else -1.0
    gw = wins["net_r"].sum()
    gl = -losses["net_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    final_eq = eq_buy + eq_sell

    # 2026 YTD
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
    res_m, eq_b_m, eq_s_m, dd_m, trace_m = g79.simulate_two_account(
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
    ret_m_pct = (final_eq_m / 200.0 - 1.0) * 100.0

    print(f"Total History Trades: {n_exec:,} | WR: {wr:.2f}% | Avg Win: +{avg_win_r:.2f}R | PF: {pf:.3f}")
    print(f"2026 YTD Trades:      {len(acc_2026):,} | 2026 WR: {wr_2026:.2f}% | 2026 PF: {pf_2026:.3f}")
    print(f"Past 30 Days Trades:  {len(acc_m):,} | 30D WR: {wr_m:.2f}% | 30D PF: {pf_m:.3f}")
    print(f"Past 30 Days Balance: $200.00 -> ${final_eq_m:,.2f} ({ret_m_pct:+.2f}%)")

    return {
        "label": label,
        "n_exec": n_exec,
        "wr": wr,
        "avg_win_r": avg_win_r,
        "pf": pf,
        "n_2026": len(acc_2026),
        "wr_2026": wr_2026,
        "pf_2026": pf_2026,
        "n_30d": len(acc_m),
        "wr_30d": wr_m,
        "pf_30d": pf_m,
        "final_30d": final_eq_m,
        "ret_30d": ret_m_pct,
        "trace_30d": trace_m,
    }


def main():
    print("Loading data & precomputing swings...")
    cache = load_cache()

    print("\nBuilding trades for Standard Baseline (unadjusted target)...")
    trades_base = build_trades_standard_baseline(cache)

    print("Building trades for True 1:1 Net R (spread-adjusted target + market entry)...")
    trades_1r = build_trades_true_1r(cache)

    res_base = evaluate(trades_base, "Baseline (Raw Zone Target, Avg Win ~+0.43R)")
    res_1r = evaluate(trades_1r, "True 1:1 Net R (Spread-Adjusted Target, Exact +1.00R / -1.00R)")

    print("\n\n" + "="*95)
    print("  HEAD-TO-HEAD COMPARISON SUMMARY")
    print("="*95)
    print(f"{'Metric':<35} | {'Standard Baseline':<25} | {'True 1:1 Net R (New)':<25}")
    print("-" * 95)
    print(f"{'Avg Win Post-Spread':<35} | {res_base['avg_win_r']:>+7.2f} R {'':<16} | {res_1r['avg_win_r']:>+7.2f} R")
    print(f"{'Avg Loss Post-Spread':<35} | {'-1.00 R':<25} | {'-1.00 R':<25}")
    print(f"{'Effective Win:Loss Ratio':<35} | {'1 : 0.43':<25} | {'1 : 1.00 (True 1:1)':<25}")
    print(f"{'Required Breakeven Win Rate':<35} | {'~70.0%':<25} | {'50.0%':<25}")
    print("-" * 95)
    print(f"{'Past 30 Days Trades':<35} | {res_base['n_30d']:<25,} | {res_1r['n_30d']:<25,}")
    print(f"{'Past 30 Days Win Rate':<35} | {res_base['wr_30d']:>6.2f}% {'':<17} | {res_1r['wr_30d']:>6.2f}%")
    print(f"{'Past 30 Days Profit Factor':<35} | {res_base['pf_30d']:>6.3f} {'':<17} | {res_1r['pf_30d']:>6.3f}")
    print(f"{'Past 30 Days Growth ($200)':<35} | ${res_base['final_30d']:>8.2f} ({res_base['ret_30d']:>+5.1f}%) {'':<4} | ${res_1r['final_30d']:>8.2f} ({res_1r['ret_30d']:>+5.1f}%)")
    print("-" * 95)
    print(f"{'2026 YTD Win Rate':<35} | {res_base['wr_2026']:>6.2f}% {'':<17} | {res_1r['wr_2026']:>6.2f}%")
    print(f"{'2026 YTD Profit Factor':<35} | {res_base['pf_2026']:>6.3f} {'':<17} | {res_1r['pf_2026']:>6.3f}")
    print("-" * 95)
    print(f"{'Multi-Year Win Rate':<35} | {res_base['wr']:>6.2f}% {'':<17} | {res_1r['wr']:>6.2f}%")
    print(f"{'Multi-Year Profit Factor':<35} | {res_base['pf']:>6.3f} {'':<17} | {res_1r['pf']:>6.3f}")
    print("="*95)


if __name__ == "__main__":
    main()
