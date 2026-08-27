"""Vendor cross-check: does the OANDA-vs-LSE EUR/USD 1-minute data disagree enough to flip
strategy outcomes for 2019+?

Context: a prior session lost the original lse_eurusd_1min_full.csv (cached only in an OS
temp scratchpad -- see ../CHECKPOINT_NOTES.md) and re-fetched EUR/USD from OANDA as a
substitute, keeping the old filename. That substitute showed a stark, unexplained
profitability sign-inversion for 2019-2025 versus the previously-established (LSE-sourced)
baseline. This script re-fetches the ACTUAL original vendor's data (see fetch_lse_eurusd.py)
and re-runs the exact same live-config strategy on it, to see whether the anomaly disappears.

Uses the real, git-committed find_pivots_strict and run_structure_fsm (not a reconstruction)
from this repo, plus the 2x-multiplier / risk-floor rule that is what's actually live in
PRODLIVETRADING (strategy/engine.py: N_MULTIPLIER=2, MIN_RISK_PCT=0.00025, min fill age 1h).
"""
import os
import numpy as np
import pandas as pd

from dtfx_eurusd_backtest_strict import find_pivots_strict, MIN_RETRACEMENT
from dtfx_eurusd_backtest import run_structure_fsm

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lse_eurusd_1min_full.csv")
N_MULTIPLIER = 2.0
MIN_RISK_PCT = 0.00025
MIN_FILL_AGE_HOURS = 1.0
FILL_SEARCH_FAST_BARS = 60 * 24 * 5     # 5-day fast pass, unbounded fallback if needed
MAX_BARS_TO_RESOLVE = 60 * 24 * 30      # 30-day cap on post-entry resolution


def _first_true(mask):
    idx = np.flatnonzero(mask)
    return idx[0] if len(idx) else len(mask)


def simulate_zone(zone, times, lows, highs):
    is_buy = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy else zone["top"]
    half_width = (zone["top"] - zone["bottom"]) / 2
    risk = N_MULTIPLIER * half_width

    start_idx = np.searchsorted(times, np.datetime64(zone["created_at"]), side="right")
    if start_idx >= len(times):
        return None

    fast_end = min(start_idx + FILL_SEARCH_FAST_BARS, len(times))
    sub_lows, sub_highs = lows[start_idx:fast_end], highs[start_idx:fast_end]
    breach_idx = _first_true(sub_lows <= invalidation_level if is_buy else sub_highs >= invalidation_level)
    mid_idx = _first_true(sub_lows <= zone["mid"] if is_buy else sub_highs >= zone["mid"])

    if mid_idx == len(sub_lows) and breach_idx == len(sub_lows) and fast_end < len(times):
        sub_lows, sub_highs = lows[start_idx:], highs[start_idx:]
        breach_idx = _first_true(sub_lows <= invalidation_level if is_buy else sub_highs >= invalidation_level)
        mid_idx = _first_true(sub_lows <= zone["mid"] if is_buy else sub_highs >= zone["mid"])

    if breach_idx <= mid_idx or mid_idx == len(sub_lows):
        return None

    entry_idx_abs = start_idx + mid_idx
    entry_time = pd.Timestamp(times[entry_idx_abs])
    after_start = entry_idx_abs + 1
    after_end = min(after_start + MAX_BARS_TO_RESOLVE, len(times))
    if after_start >= len(times):
        return None

    stop_level = zone["mid"] - risk if is_buy else zone["mid"] + risk
    target_level = zone["mid"] + risk if is_buy else zone["mid"] - risk

    a_lows, a_highs = lows[after_start:after_end], highs[after_start:after_end]
    sl_idx = _first_true(a_lows <= stop_level if is_buy else a_highs >= stop_level)
    tp_idx = _first_true(a_highs >= target_level if is_buy else a_lows <= target_level)
    if sl_idx == len(a_lows) and tp_idx == len(a_lows):
        return None

    status = "stopped_out" if sl_idx <= tp_idx else "take_profit"
    risk_pct_of_price = risk / zone["mid"]
    fill_hours = (entry_time - zone["created_at"]).total_seconds() / 3600
    return dict(entry_time=entry_time, status=status,
                risk_pct_of_price=risk_pct_of_price, fill_hours=fill_hours)


def main():
    print(f"Loading {DATA_PATH} ...", flush=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[["open", "high", "low", "close"]]
    print(f"Loaded {len(df):,} 1-minute bars, {df.index.min()} to {df.index.max()}", flush=True)

    zigzag = find_pivots_strict(df, MIN_RETRACEMENT)
    _, zones = run_structure_fsm(zigzag)
    print(f"{len(zones):,} zones (MIN_RETRACEMENT={MIN_RETRACEMENT})", flush=True)

    times, lows, highs = df.index.values, df["low"].values, df["high"].values
    rows = []
    for z in zones:
        tr = simulate_zone(z, times, lows, highs)
        if tr is None:
            continue
        if tr["risk_pct_of_price"] < MIN_RISK_PCT:
            continue
        if tr["fill_hours"] < MIN_FILL_AGE_HOURS:
            continue
        r = 1.0 if tr["status"] == "take_profit" else -1.0
        rows.append(dict(entry_time=tr["entry_time"], year=tr["entry_time"].year, r=r,
                          status=tr["status"]))

    trades = pd.DataFrame(rows)
    print(f"{len(trades):,} trades after risk-floor + fill-age filters\n", flush=True)

    yearly = trades.groupby("year").agg(
        trades=("r", "count"),
        win_rate=("r", lambda s: (s > 0).mean()),
        avg_r=("r", "mean"),
        sum_r=("r", "sum"),
    ).reset_index()
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print(yearly.to_string(index=False))

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "dtfx_lse_vendor_check_yearly.csv")
    yearly.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
