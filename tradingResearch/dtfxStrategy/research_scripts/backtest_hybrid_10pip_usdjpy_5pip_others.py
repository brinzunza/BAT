"""Backtest script for the new per-symbol DTFX setup:
- USDJPY: 10.0 pips
- EURUSD: 5.0 pips
- GBPUSD: 5.0 pips
- EURGBP: 5.0 pips

Compared against:
- Global 5.0-pip baseline
- Global 10.0-pip baseline

Uses the exact live two-account architecture ($100 Long, $100 Short, 1-per-side max concurrency, 3% risk, 50x leverage, 1.6-pip spread).
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
MIN_STOP_PIPS = 3.0
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
        print(f"  {inst:8}: loaded + zones built ({len(z_created):,} raw swings) in {time.time()-t1:.1f}s")
    return cache


def build_candidate_trades_for_setup(cache, zone_pips_map, symmetric_exit_spread=True):
    frames = []
    for inst, d in cache.items():
        min_zone_pips = zone_pips_map[inst]
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = min_zone_pips * pip

        entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            d["last_high_of_bar"], d["last_low_of_bar"], spread_price, min_stop_price, MAX_WINDOW,
            False, min_zone_height, symmetric_exit_spread
        )

        resolved = outcome > 0
        ei = entry_idx[resolved]
        xi = exit_idx[resolved]
        is_buy = d["z_is_buy"][resolved]
        raw_stop = np.where(is_buy, d["z_bottom"][resolved], d["z_top"][resolved])
        if symmetric_exit_spread:
            half_spread = spread_price / 2.0
            stop_price = np.where(is_buy, raw_stop - half_spread, raw_stop + half_spread)
        else:
            stop_price = raw_stop
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


def evaluate_setup(cache, zone_pips_map, label=""):
    print(f"\n{'='*75}\nEvaluating: {label}\nConfig: {zone_pips_map}\n{'='*75}")
    t0 = time.time()
    trades = build_candidate_trades_for_setup(cache, zone_pips_map)
    print(f"Total candidate trades across portfolio: {len(trades):,}")

    result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
        trades, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
        max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=True
    )
    acc = result[result["accepted"]].copy()
    n_exec = len(acc)
    wr = 100.0 * (acc["net_r"] > 0).mean() if n_exec else 0.0
    wins = acc[acc["net_r"] > 0]
    losses = acc[acc["net_r"] <= 0]
    gross_win_r = wins["net_r"].sum()
    gross_loss_r = -losses["net_r"].sum()
    pf = gross_win_r / gross_loss_r if gross_loss_r > 0 else float("inf")
    final_equity = eq_buy + eq_sell

    # 2026 YTD breakdown
    acc_2026 = acc[acc["entry_time"] >= pd.Timestamp("2026-01-01")]
    wr_2026 = 100.0 * (acc_2026["net_r"] > 0).mean() if len(acc_2026) else 0.0
    gw_2026 = acc_2026.loc[acc_2026["net_r"] > 0, "net_r"].sum()
    gl_2026 = -acc_2026.loc[acc_2026["net_r"] <= 0, "net_r"].sum()
    pf_2026 = gw_2026 / gl_2026 if gl_2026 > 0 else float("inf")

    trace_df = pd.DataFrame(trace)
    if len(trace_df):
        trace_df["time"] = pd.to_datetime(trace_df["close_time"])
        trace_2026 = trace_df[trace_df["time"] >= pd.Timestamp("2026-01-01")]
        if len(trace_2026):
            eq_start_2026 = trace_2026["equity_combined"].iloc[0]
            eq_end_2026 = trace_2026["equity_combined"].iloc[-1]
            peak_2026 = trace_2026["equity_combined"].cummax()
            dd_2026 = ((peak_2026 - trace_2026["equity_combined"]) / peak_2026).max() * 100.0
        else:
            dd_2026 = 0.0
    else:
        dd_2026 = 0.0

    print(f"\n--- Results for {label} ---")
    print(f"Executed Trades:       {n_exec:,}")
    print(f"Overall Win Rate:      {wr:.2f}%")
    print(f"Overall Profit Factor: {pf:.3f}")
    print(f"Max Drawdown (All):    {max_dd*100:.2f}%")
    print(f"Final Equity ($200):   ${final_equity:,.2f}")
    print(f"\n--- 2026 YTD Performance ---")
    print(f"2026 Executed Trades:  {len(acc_2026):,}")
    print(f"2026 Win Rate:         {wr_2026:.2f}%")
    print(f"2026 Profit Factor:    {pf_2026:.3f}")
    print(f"2026 Max Drawdown:     {dd_2026:.2f}%")
    
    print("\n--- Executed Trades Breakdown by Symbol ---")
    for sym in zone_pips_map.keys():
        s_trades = acc[acc["symbol"] == sym]
        s_wr = 100.0 * (s_trades["net_r"] > 0).mean() if len(s_trades) else 0.0
        s_gw = s_trades.loc[s_trades["net_r"] > 0, "net_r"].sum()
        s_gl = -s_trades.loc[s_trades["net_r"] <= 0, "net_r"].sum()
        s_pf = s_gw / s_gl if s_gl > 0 else float("inf")
        print(f"  {sym:8}: {len(s_trades):>6,} trades | WR: {s_wr:>5.1f}% | PF: {s_pf:>5.2f}")

    elapsed = time.time() - t0
    print(f"Evaluation finished in {elapsed:.1f}s")
    return {
        "label": label,
        "n_exec": n_exec,
        "wr": wr,
        "pf": pf,
        "max_dd": max_dd * 100.0,
        "final_equity": final_equity,
        "n_2026": len(acc_2026),
        "wr_2026": wr_2026,
        "pf_2026": pf_2026,
        "dd_2026": dd_2026,
    }


def main():
    print("==========================================================================")
    print("  DTFX STRATEGY BACKTEST: MULTI-ASSET COMPARISON")
    print("==========================================================================")
    print("Loading data & precomputing swing structures...")
    cache = load_and_build_cache()

    # 1. Global 5.0 Pip Baseline
    res_5pip = evaluate_setup(cache, {
        "USDJPY": 5.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0
    }, label="Global 5.0-Pip Floor")

    # 2. Global 10.0 Pip Baseline
    res_10pip = evaluate_setup(cache, {
        "USDJPY": 10.0, "EURUSD": 10.0, "GBPUSD": 10.0, "EURGBP": 10.0
    }, label="Global 10.0-Pip Floor")

    # 3. New Setup: USDJPY 10.0 Pip, All Others 5.0 Pip
    res_new = evaluate_setup(cache, {
        "USDJPY": 10.0, "EURUSD": 5.0, "GBPUSD": 5.0, "EURGBP": 5.0
    }, label="New Setup (USDJPY 10-pip, Others 5-pip)")

    # Print Comparison Table
    print("\n\n" + "="*95)
    print("  FINAL MULTI-YEAR & 2026 YTD COMPARISON SUMMARY")
    print("="*95)
    print(f"{'Configuration':<38} | {'Total Trades':<12} | {'WR':<6} | {'PF':<6} | {'Max DD':<7} | {'2026 Trds':<10} | {'2026 PF':<8} | {'2026 DD':<8}")
    print("-" * 95)
    for r in [res_5pip, res_10pip, res_new]:
        print(f"{r['label']:<38} | {r['n_exec']:<12,} | {r['wr']:<5.1f}% | {r['pf']:<6.2f} | {r['max_dd']:<6.1f}% | {r['n_2026']:<10,} | {r['pf_2026']:<8.2f} | {r['dd_2026']:<7.1f}%")
    print("="*95)


if __name__ == "__main__":
    main()
