"""Backtest DTFX with No Minimum Zone Floor (MIN_ZONE_PIPS = 0.0)
Tests:
1. No Zone Floor (min_zone_pips = 0.0, min_stop_pips = 3.0)
2. Completely Unfiltered Micro-Zones (min_zone_pips = 0.0, min_stop_pips = 0.0)
3. Comparison against:
   - Live Setup (USDJPY 10p, Others 5p)
   - Global 5-pip Floor
   - Global 10-pip Floor

Under the realistic two-account live architecture ($200 starting capital, 3% risk per trade, 1-per-side concurrency, 1.6-pip spread, 50:1 leverage).
"""
import os
import sys
import time
import numpy as np
import pandas as pd

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
MAX_WINDOW = 1440
RISK_PCT = 0.03


def load_and_build_cache():
    cache = {}
    for inst, (fname, pip) in INSTRUMENTS.items():
        t1 = time.time()
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            fresh_path = f"/tmp/dtfx_fresh_data/{inst}_merged.csv"
            if os.path.exists(fresh_path):
                path = fresh_path
            else:
                raise FileNotFoundError(f"Cannot find {inst} at {path}")
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
        print(f"  {inst:8}: loaded ({len(z_created):,} raw swings)")
    return cache


def build_trades(cache, min_zone_pips, min_stop_pips):
    frames = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = min_stop_pips * pip
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

        frames.append(pd.DataFrame({
            "instrument": inst,
            "symbol": inst,
            "entry_time": d["times"][ei],
            "exit_time": d["times"][xi],
            "entry_price": entry_price[resolved],
            "stop_risk": stop_risk,
            "net_r": net_r[resolved],
            "is_buy": is_buy,
        }))
    trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return trades


def evaluate(trades, label=""):
    print(f"\n{'='*75}\nEvaluating: {label}\nCandidate Trades: {len(trades):,}\n{'='*75}")
    t0 = time.time()
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

    print(f"Executed Trades:       {n_exec:,}")
    print(f"Overall Win Rate:      {wr:.2f}%")
    print(f"Avg Win Net R:         +{avg_win_r:.3f} R")
    print(f"Profit Factor:         {pf:.3f}")
    print(f"2026 YTD Executed:     {len(acc_2026):,} | WR: {wr_2026:.2f}% | PF: {pf_2026:.3f}")
    print(f"Past 30 Days Growth:   $200.00 -> ${final_eq_m:,.2f} ({ret_m:+.2f}%) | WR: {wr_m:.2f}% | PF: {pf_m:.3f}")

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
        "ret_30d": ret_m,
    }


def main():
    print("==========================================================================")
    print("  DTFX NO MINIMUM PIP ZONE BACKTEST & COMPARISON")
    print("==========================================================================")
    cache = load_and_build_cache()

    # 1. Completely Unfiltered (min_zone = 0, min_stop = 0)
    print("\nBuilding trades for: No Floors (min_zone=0p, min_stop=0p)...")
    t_zero_zero = build_trades(cache, min_zone_pips=0.0, min_stop_pips=0.0)
    res_zero_zero = evaluate(t_zero_zero, "1. Pure Zero Floor (min_zone=0p, min_stop=0p)")

    # 2. No Zone Floor with 3-pip Stop Floor (min_zone = 0, min_stop = 3)
    print("\nBuilding trades for: No Zone Floor (min_zone=0p, min_stop=3p)...")
    t_zero_stop3 = build_trades(cache, min_zone_pips=0.0, min_stop_pips=3.0)
    res_zero_stop3 = evaluate(t_zero_stop3, "2. No Zone Floor (min_zone=0p, min_stop=3p)")

    # 3. Global 5.0-Pip Floor
    print("\nBuilding trades for: Global 5.0-Pip Floor...")
    t_5p = build_trades(cache, min_zone_pips=5.0, min_stop_pips=3.0)
    res_5p = evaluate(t_5p, "3. Global 5.0-Pip Floor")

    # 4. Global 10.0-Pip Floor
    print("\nBuilding trades for: Global 10.0-Pip Floor...")
    t_10p = build_trades(cache, min_zone_pips=10.0, min_stop_pips=3.0)
    res_10p = evaluate(t_10p, "4. Global 10.0-Pip Floor")

    # 5. Live Setup (USDJPY 10p, Others 5p)
    print("\nBuilding trades for: Live Setup (USDJPY 10p, Others 5p)...")
    frames = []
    for inst, d in cache.items():
        min_z = 10.0 if inst == "USDJPY" else 5.0
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = 3.0 * pip
        min_zone_height = min_z * pip
        entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            d["last_high_of_bar"], d["last_low_of_bar"], spread_price, min_stop_price, MAX_WINDOW,
            False, min_zone_height, True
        )
        resolved = outcome > 0
        ei = entry_idx[resolved]; xi = exit_idx[resolved]; is_buy = d["z_is_buy"][resolved]
        raw_stop = np.where(is_buy, d["z_bottom"][resolved], d["z_top"][resolved])
        half_spread = spread_price / 2.0
        stop_price = np.where(is_buy, raw_stop - half_spread, raw_stop + half_spread)
        stop_risk = np.abs(entry_price[resolved] - stop_price)
        frames.append(pd.DataFrame({
            "instrument": inst, "symbol": inst, "entry_time": d["times"][ei],
            "exit_time": d["times"][xi], "entry_price": entry_price[resolved],
            "stop_risk": stop_risk, "net_r": net_r[resolved], "is_buy": is_buy
        }))
    t_live = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    res_live = evaluate(t_live, "5. Live Setup (USDJPY 10p, Others 5p)")

    # Print Final Summary Table
    print("\n\n" + "="*115)
    print("  COMPREHENSIVE BACKTEST SUMMARY: MIN ZONE FLOOR COMPARISON")
    print("="*115)
    print(f"{'Configuration':<42} | {'Trades':<10} | {'WR':<6} | {'Avg Win':<8} | {'PF':<6} | {'2026 Trds':<10} | {'2026 PF':<8} | {'30D Ret':<8}")
    print("-" * 115)
    for r in [res_zero_zero, res_zero_stop3, res_5p, res_10p, res_live]:
        print(f"{r['label']:<42} | {r['n_exec']:<10,} | {r['wr']:<5.1f}% | {r['avg_win_r']:>+6.2f}R | {r['pf']:<6.2f} | {r['n_2026']:<10,} | {r['pf_2026']:<8.2f} | {r['ret_30d']:>+6.1f}%")
    print("="*115)


if __name__ == "__main__":
    main()
