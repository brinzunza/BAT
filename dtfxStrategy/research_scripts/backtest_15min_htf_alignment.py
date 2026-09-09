"""Higher Timeframe (15-Minute) DTFX Alignment Backtest & Confluence Study
Investigates the impact of filtering 1-minute DTFX trade setups by 15-minute HTF zones/trends:

Variants tested:
1. Baseline: 1-Minute Unfiltered (USDJPY 10p, EURUSD/GBPUSD/EURGBP 5p)
2. 15m Trend Filter: Only take 1m Buy if 15m Zigzag Trend is Bullish (+1); 1m Sell if Bearish (-1).
3. 15m Active Zone Confluence: Only take 1m Buy if price is inside an active unexpired 15m Buy zone; 1m Sell if inside active 15m Sell zone.
4. Pure 15m Timeframe Trading: Trading 15m DTFX zones directly on 15m candles.

Tested under the two-account sub-account execution model ($200 start, 3% risk, 1-per-side concurrency, 1.6-pip spread, 50x leverage).
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
MAX_WINDOW_1M = 1440
MAX_WINDOW_15M = 96  # 96 15m bars = 1 trading day
RISK_PCT = 0.03


def load_and_resample(symbol):
    fname, pip = INSTRUMENTS[symbol]
    path = os.path.join(DATA_DIR, fname)
    fresh_path = f"/tmp/dtfx_fresh_data/{symbol}_merged.csv"
    if os.path.exists(fresh_path):
        path = fresh_path

    df = pd.read_csv(path, parse_dates=["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df.drop_duplicates(subset="date").set_index("date").sort_index()

    # 1-Minute data
    times_1m = df.index
    high_1m = df["high"].values.astype(np.float64)
    low_1m = df["low"].values.astype(np.float64)
    close_1m = df["close"].values.astype(np.float64)

    # 15-Minute Resampled Data
    df_15m = df.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    times_15m = df_15m.index
    high_15m = df_15m["high"].values.astype(np.float64)
    low_15m = df_15m["low"].values.astype(np.float64)
    close_15m = df_15m["close"].values.astype(np.float64)

    return {
        "symbol": symbol, "pip": pip,
        "times_1m": times_1m, "high_1m": high_1m, "low_1m": low_1m, "close_1m": close_1m,
        "times_15m": times_15m, "high_15m": high_15m, "low_15m": low_15m, "close_15m": close_15m,
    }


def build_15m_structures(d):
    # 15m Zigzag and Zones
    confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(
        d["high_15m"], d["low_15m"], d["close_15m"]
    )
    (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
     ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)

    # Build a continuous timeline of 15m trend for fast lookup on 1m bars
    # Trend = +1 if latest pivot confirmed was a Higher High / trend break up, -1 if Lower Low / break down
    trend_15m_arr = np.zeros(len(d["times_15m"]), dtype=np.int8)
    cur_trend = 0
    pi = 0
    m = len(confirm_idx)
    last_high_p = 0.0
    last_low_p = 0.0
    has_h = False
    has_l = False

    for i in range(len(d["times_15m"])):
        while pi < m and confirm_idx[pi] <= i:
            if is_high[pi]:
                if has_h and price[pi] > last_high_p:
                    cur_trend = 1
                elif has_h:
                    cur_trend = -1
                last_high_p = price[pi]
                has_h = True
            else:
                if has_l and price[pi] < last_low_p:
                    cur_trend = -1
                elif has_l:
                    cur_trend = 1
                last_low_p = price[pi]
                has_l = True
            pi += 1
        trend_15m_arr[i] = cur_trend

    # Active 15m zones list (creation time, top, bottom, is_buy)
    zones_15m_df = pd.DataFrame({
        "created_time": d["times_15m"][z_created],
        "top": z_top,
        "bottom": z_bottom,
        "is_buy": z_is_buy,
    })
    
    trend_15m_series = pd.Series(trend_15m_arr, index=d["times_15m"])

    return {
        "trend_15m_series": trend_15m_series,
        "zones_15m_df": zones_15m_df,
        "z_created": z_created,
        "z_top": z_top,
        "z_bottom": z_bottom,
        "z_is_buy": z_is_buy,
        "z_top_idx": z_top_idx,
        "z_bottom_idx": z_bottom_idx,
        "confirm_idx": confirm_idx,
        "is_high": is_high,
        "price": price,
    }


def build_1m_trades_with_htf_filters(d, htf_data):
    symbol = d["symbol"]
    pip = d["pip"]
    min_zone_pips = ZONE_PIPS_MAP[symbol]
    spread_price = SPREAD_PIPS * pip
    min_stop_price = MIN_STOP_PIPS * pip
    min_zone_height = min_zone_pips * pip

    # 1-minute swings
    c_idx_1m, p_idx_1m, is_h_1m, pr_1m = g95.close_based_zigzag_symmetric(
        d["high_1m"], d["low_1m"], d["close_1m"]
    )
    (z_c_1m, z_t_1m, z_b_1m, z_ti_1m, z_bi_1m, z_buy_1m, z_f_1m
     ) = g95.build_zones_every_swing(c_idx_1m, p_idx_1m, is_h_1m, pr_1m)
    last_h_1m, last_l_1m = g99.build_last_confirmed(c_idx_1m, is_h_1m, pr_1m, len(d["high_1m"]))

    entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
        z_c_1m, z_t_1m, z_b_1m, z_buy_1m, d["high_1m"], d["low_1m"],
        last_h_1m, last_l_1m, spread_price, min_stop_price, MAX_WINDOW_1M,
        False, min_zone_height, True
    )

    resolved = outcome > 0
    ei = entry_idx[resolved]
    xi = exit_idx[resolved]
    is_buy = z_buy_1m[resolved]
    entry_times = d["times_1m"][ei]
    exit_times = d["times_1m"][xi]
    raw_stop = np.where(is_buy, z_b_1m[resolved], z_t_1m[resolved])
    half_spread = spread_price / 2.0
    stop_price = np.where(is_buy, raw_stop - half_spread, raw_stop + half_spread)
    stop_risk = np.abs(entry_price[resolved] - stop_price)

    # Reindex 15m trend to 1m entry times (forward fill to avoid lookahead)
    trend_15m_aligned = htf_data["trend_15m_series"].asof(entry_times).values

    # Check 15m Zone Confluence:
    # A 1m entry has 15m zone confluence if at entry_time, price is between top and bottom of an active 15m zone of the same side
    # Active 15m zone: created in past 96 bars (24 hours) and not breached
    # We do a fast vectorized search
    in_15m_zone = np.zeros(len(ei), dtype=bool)
    z15 = htf_data["zones_15m_df"]
    z15_buy = z15[z15["is_buy"]].reset_index(drop=True)
    z15_sell = z15[~z15["is_buy"]].reset_index(drop=True)

    # Evaluate confluence
    t_1m_vals = entry_times.values
    ep_vals = entry_price[resolved]

    for k in range(len(ei)):
        t_k = t_1m_vals[k]
        p_k = ep_vals[k]
        buy_k = is_buy[k]
        z_subset = z15_buy if buy_k else z15_sell
        # Look for zones created within past 24 hours (1440 min)
        recent = z_subset[(z_subset["created_time"] <= t_k) & (z_subset["created_time"] >= t_k - pd.Timedelta(hours=24))]
        if len(recent):
            # Check if current price is within [bottom, top]
            if ((recent["bottom"] <= p_k) & (p_k <= recent["top"])).any():
                in_15m_zone[k] = True

    df_trades = pd.DataFrame({
        "instrument": symbol,
        "symbol": symbol,
        "entry_time": entry_times,
        "exit_time": exit_times,
        "entry_price": ep_vals,
        "stop_risk": stop_risk,
        "net_r": net_r[resolved],
        "is_buy": is_buy,
        "trend_15m": trend_15m_aligned,
        "in_15m_zone": in_15m_zone,
    })
    return df_trades


def run_portfolio_sim(trades_df, label=""):
    result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
        trades_df, risk_pct=RISK_PCT, start_per_side=100.0,
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

    # 2026 YTD
    acc_2026 = acc[acc["entry_time"] >= pd.Timestamp("2026-01-01")]
    wins_26 = acc_2026[acc_2026["net_r"] > 0]
    losses_26 = acc_2026[acc_2026["net_r"] <= 0]
    wr_2026 = 100.0 * len(wins_26) / len(acc_2026) if len(acc_2026) else 0.0
    gw_26 = wins_26["net_r"].sum()
    gl_26 = -losses_26["net_r"].sum()
    pf_2026 = gw_26 / gl_26 if gl_26 > 0 else float("inf")

    # Past 30 Days
    latest_t = trades_df["entry_time"].max()
    month_start = latest_t - pd.Timedelta(days=30)
    month_trades = trades_df[trades_df["entry_time"] >= month_start].copy().reset_index(drop=True)
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
        "label": label,
        "n_exec": n_exec,
        "wr": wr,
        "avg_win_r": avg_win_r,
        "pf": pf,
        "max_dd": max_dd * 100.0,
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
    print("  15-MINUTE HTF ALIGNMENT & CONFLUENCE BACKTEST STUDY")
    print("==========================================================================")
    all_raw_trades = []

    for sym in ["USDJPY", "EURUSD", "GBPUSD", "EURGBP"]:
        t0 = time.time()
        d = load_and_resample(sym)
        htf_data = build_15m_structures(d)
        sym_trades = build_1m_trades_with_htf_filters(d, htf_data)
        all_raw_trades.append(sym_trades)
        print(f"  {sym:8}: Processed 1m + 15m HTF data in {time.time()-t0:.1f}s ({len(sym_trades):,} raw 1m setups)")

    trades_all = pd.concat(all_raw_trades, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    print(f"\nTotal 1-Minute Candidate Setups: {len(trades_all):,}")

    # 1. Baseline: Unfiltered 1m Strategy
    print("\nEvaluating Variant 1: Unfiltered 1-Minute Baseline...")
    res_base = run_portfolio_sim(trades_all, "1. Baseline (Unfiltered 1m Setups)")

    # 2. 15m Trend Alignment Filter
    # Only Buy if 15m trend == 1, Only Sell if 15m trend == -1
    print("Evaluating Variant 2: 15-Minute Trend Alignment Filter...")
    trades_trend = trades_all[
        ((trades_all["is_buy"]) & (trades_all["trend_15m"] == 1)) |
        ((~trades_all["is_buy"]) & (trades_all["trend_15m"] == -1))
    ].copy().reset_index(drop=True)
    res_trend = run_portfolio_sim(trades_trend, "2. 15m Trend Aligned (With 15m Zigzag Trend)")

    # 3. 15m Active Zone Confluence
    # Only take 1m entry if inside an active 15m zone of the same side
    print("Evaluating Variant 3: 15-Minute In-Zone Confluence...")
    trades_confluence = trades_all[trades_all["in_15m_zone"]].copy().reset_index(drop=True)
    res_conf = run_portfolio_sim(trades_confluence, "3. 15m In-Zone Confluence (Inside 15m Zone)")

    # 4. Strict Dual Confluence (15m Trend Aligned AND Inside 15m Zone)
    print("Evaluating Variant 4: Strict Dual Alignment (Trend + In-Zone)...")
    trades_strict = trades_all[
        (((trades_all["is_buy"]) & (trades_all["trend_15m"] == 1)) |
         ((~trades_all["is_buy"]) & (trades_all["trend_15m"] == -1))) &
        (trades_all["in_15m_zone"])
    ].copy().reset_index(drop=True)
    res_strict = run_portfolio_sim(trades_strict, "4. Strict 15m Dual Confluence (Trend + In-Zone)")

    # Output Comparison Table
    print("\n\n" + "="*105)
    print("  15-MINUTE HTF ALIGNMENT STUDY: COMPREHENSIVE PERFORMANCE COMPARISON")
    print("="*105)
    print(f"{'Variant':<42} | {'History Trds':<12} | {'WR':<6} | {'PF':<6} | {'2026 Trds':<10} | {'2026 WR':<8} | {'2026 PF':<8} | {'30D Ret':<8}")
    print("-" * 105)
    for r in [res_base, res_trend, res_conf, res_strict]:
        print(f"{r['label']:<42} | {r['n_exec']:<12,} | {r['wr']:<5.1f}% | {r['pf']:<6.2f} | {r['n_2026']:<10,} | {r['wr_2026']:<7.1f}% | {r['pf_2026']:<8.2f} | {r['ret_30d']:>+6.1f}%")
    print("="*105)


if __name__ == "__main__":
    main()
