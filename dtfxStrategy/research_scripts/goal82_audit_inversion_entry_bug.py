"""Audit of goal77/goal79/goal80/goal81's "invert" mechanism (the STRATEGY_RANKINGS.md #1
"Shallow-Entry Fade" claiming PF up to 3.72 and 100% of months profitable, full 2009-2026
history): entry_price's spread-shift branches on `is_buy` (the ORIGINAL, un-traded zone
direction) rather than `trade_is_buy` (the direction actually being traded once inverted).

Worked example (height=20, spread=1.6, is_buy=True non-deep case, entry_level=mid=100):
  entry_price = 100 + 1.6 = 101.6 (shifted TOWARD top, per the buy-side spread convention)
  invert=False (buy):  stop=90,  target=110 -> eff_risk=11.6, win_r=8.4/11.6  = 0.724
  invert=True  (sell): stop=110, target=90  -> eff_risk=8.4,  win_r=11.6/8.4 = 1.381

Same fill, same entry price -- but because entry was shifted per the BUY side's spread
convention while the trade actually taken is a SELL, the inverted trade's stop distance
shrinks and reward distance grows simultaneously, nearly DOUBLING win_r as a pure artifact.
A real inverted execution should reprice entry for ITS OWN (sell-side) spread cost, which
moves entry the OPPOSITE way (worse for the seller, not better).

This script reruns goal81's exact 30%-entry variant with entry_price correctly repriced for
the ACTUALLY-TRADED direction, everything else (zone detection, touch level, stop/target
edges, two-account sizing/concurrency, Monte Carlo) byte-for-byte unchanged, to measure how
much of the headline result survives.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats as sps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import goal77_dynamic_zigzag_backtest as g77
import goal79_two_account_portfolio as g79
import goal81_shallow_entry_two_account as g81

rs, pg = g77.rs, g77.pg
simulate_two_account = g79.simulate_two_account
run_monte_carlo_two_account = g79.run_monte_carlo_two_account
period_stats = g79.period_stats
report_variant = g81.report_variant
thirds_robustness = g81.thirds_robustness

INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"]
MAX_WINDOW = 1440
MIN_STOP_PIPS = 3.0


@njit(cache=True)
def _resolve_frac_ex_fixed(z_created, z_top, z_bottom, z_is_buy, high, low, max_window,
                            min_stop_price, spread_price, frac, invert):
    """Identical to goal81._resolve_frac_ex EXCEPT entry_price's spread-shift now uses
    trade_is_buy (the direction actually traded) instead of is_buy (the zone's own,
    possibly-un-traded, approach direction) -- the one line changed is marked below."""
    n_bars = len(high)
    m = len(z_created)
    out_outcome = np.zeros(m, dtype=np.int8)
    out_net_r = np.zeros(m, dtype=np.float64)
    out_entry_price = np.zeros(m, dtype=np.float64)
    out_stop_risk = np.zeros(m, dtype=np.float64)
    out_touch = np.full(m, -1, dtype=np.int64)
    out_exit = np.full(m, -1, dtype=np.int64)

    for zi in range(m):
        top, bottom, is_buy = z_top[zi], z_bottom[zi], z_is_buy[zi]
        height = top - bottom
        entry_level = (top - frac * height) if is_buy else (bottom + frac * height)
        trade_is_buy = (not is_buy) if invert else is_buy
        stop_price = bottom if trade_is_buy else top
        target_price = top if trade_is_buy else bottom
        if abs(entry_level - stop_price) < min_stop_price:
            continue
        c = z_created[zi]
        end = c + max_window
        if end > n_bars:
            end = n_bars
        touch_idx = -1
        for j in range(c + 1, end):
            touched = (low[j] <= entry_level) if is_buy else (high[j] >= entry_level)
            if touched:
                touch_idx = j
                break
        if touch_idx == -1:
            continue
        # --- FIX: reprice for the direction actually traded (trade_is_buy), not the
        # zone's own approach direction (is_buy). A real sell fill costs the seller the
        # spread by giving a WORSE (lower) price, symmetric to how a real buy fill costs
        # the buyer a worse (higher) price -- whichever direction is actually executed.
        if trade_is_buy:
            deep = low[touch_idx] <= entry_level - spread_price
            entry_price = entry_level if deep else entry_level + spread_price
        else:
            deep = high[touch_idx] >= entry_level + spread_price
            entry_price = entry_level if deep else entry_level - spread_price
        eff_risk = abs(entry_price - stop_price)
        if eff_risk <= 0:
            continue
        win_r = abs(target_price - entry_price) / eff_risk

        outcome, exit_idx = 0, -1
        for k in range(touch_idx, end):
            hit_sl = (low[k] <= stop_price) if trade_is_buy else (high[k] >= stop_price)
            hit_tp = (high[k] >= target_price) if trade_is_buy else (low[k] <= target_price)
            if hit_sl:
                outcome, exit_idx = 2, k; break
            elif hit_tp:
                outcome, exit_idx = 1, k; break
        if outcome == 0:
            continue
        out_touch[zi] = touch_idx
        out_exit[zi] = exit_idx
        out_outcome[zi] = outcome
        out_net_r[zi] = win_r if outcome == 1 else -1.0
        out_entry_price[zi] = entry_price
        out_stop_risk[zi] = eff_risk

    return out_touch, out_exit, out_outcome, out_net_r, out_entry_price, out_stop_risk


def build_frac_trades_fixed(frac):
    frames = []
    for inst in INSTRUMENTS:
        times, lows, highs, df = rs.load_instrument(inst)
        high_ = df["high"].values.astype(np.float64)
        low_ = df["low"].values.astype(np.float64)
        pip = pg.PIP_SIZE[inst]
        spread_price = pg.SPREAD_PIPS_TYPICAL * pip

        confirm_idx, pivot_idx, is_high, price = g77._zigzag_pivots(high_, low_)
        z_created, z_top, z_bottom, z_is_buy, z_is_flip = g77._build_zones(confirm_idx, is_high, price)
        touch, exit_, outcome, net_r, entry_price, stop_risk = _resolve_frac_ex_fixed(
            z_created, z_top, z_bottom, z_is_buy, high_, low_, MAX_WINDOW,
            MIN_STOP_PIPS * pip, spread_price, frac, True)

        resolved = outcome > 0
        entry_idx = touch[resolved]
        exit_idx = exit_[resolved]
        frames.append(pd.DataFrame({
            "instrument": inst,
            "entry_time": times[entry_idx],
            "exit_time": times[exit_idx],
            "entry_price": entry_price[resolved],
            "stop_risk": stop_risk[resolved],
            "net_r": net_r[resolved],
            "is_buy": ~z_is_buy[resolved],
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


if __name__ == "__main__":
    t0 = time.time()
    print("Re-running goal81's 30% and 50% variants with entry_price correctly repriced for")
    print("the ACTUALLY-TRADED (inverted) direction instead of the zone's own approach direction.\n")

    for frac, label in [(0.5, "FIXED: Baseline (50% entry)"), (0.3, "FIXED: Shallow entry (30%)")]:
        print("\n" + "#" * 100)
        buggy = g81.build_frac_trades(frac)
        fixed = build_frac_trades_fixed(frac)
        print(f"frac={frac}: buggy n={len(buggy):,}  fixed n={len(fixed):,}")

        res_buggy = report_variant(f"BUGGY (original) frac={frac}", buggy, n_mc_sims=1500)
        res_fixed = report_variant(f"{label}", fixed, n_mc_sims=1500)
        if res_fixed is not None:
            thirds_robustness(res_fixed["accepted"], label)

    print(f"\n(total: {time.time()-t0:.0f}s)")


def extended_frac_sweep():
    print("\n\n" + "=" * 100)
    print("EXTENDED: full corrected frac sweep (20/25/30/35/40%)")
    print("=" * 100)
    for frac in [0.2, 0.25, 0.3, 0.35, 0.4]:
        fixed = build_frac_trades_fixed(frac)
        res = report_variant(f"FIXED frac={frac}", fixed, n_mc_sims=1500)
        if res is not None:
            thirds_robustness(res["accepted"], f"FIXED frac={frac}")


if __name__ == "__main__" and os.environ.get("RUN_EXTENDED_SWEEP"):
    extended_frac_sweep()
