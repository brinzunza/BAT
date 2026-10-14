"""Corrected spread-cost convention for the Shallow-Entry Fade strategy (direct user
request, 2026-09-22/23): entry, stop, and target must stay at the exact raw zone/technical
levels -- no repricing of any kind -- and exactly ONE spread's worth of cost is removed
from each trade's R-multiple AFTER the raw (spread-free) win/loss result is computed,
applied uniformly whether the trade won or lost.

This replaces goal82_audit_inversion_entry_bug.py's spread handling, which repriced
entry_price by up to one spread depending on whether the touch bar was "deep" enough,
then computed win_r off that shifted entry. That model has two problems the user's
specification avoids by construction:
  1. It folds cost into the definition of "1R" itself (eff_risk = |entry - stop| uses the
     REPRICED entry), rather than charging a fixed cost against a fixed-technical-level
     result.
  2. Because a stop-out's net_r is tautologically -1.0*eff_risk regardless of how entry
     was repriced, losing trades never actually pay any of that cost -- only the reward
     side ever shrinks. That's an asymmetric, non-obvious way for a spread cost to show
     up, and it understates how much a real stop-out costs (you paid to cross the spread
     to get IN, whether the trade then wins or loses).

Because entry is never shifted here, the original entry-price bug this project found in
goal82 (repricing using the zone's un-traded approach direction `is_buy` instead of the
actually-traded direction `trade_is_buy`) cannot recur -- there is no repricing branch to
get the direction of wrong in the first place.

Zone detection and the stop/target exit-resolution loop (including the conservative
stop-first tie-break on the fill bar) are byte-for-byte identical to goal77/goal82;
only the entry-price/cost step changes.

Verified impact on the 30%-entry, 4-pair, 2009-2026 full history (see this session's
comparison, goal85 vs goal82's fixed model): PF 1.366 -> 1.188, monthly consistency
100% (204/204) -> 98.5% (201/204), yearly consistency unchanged at 100% (17/17), all
three independent thirds of history remain individually profitable (PF 1.21/1.33/1.12).
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import goal77_dynamic_zigzag_backtest as g77

rs, pg = g77.rs, g77.pg

INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"]
MAX_WINDOW = 1440
MIN_STOP_PIPS = 3.0


@njit(cache=True)
def _resolve_frac_flatcost(z_created, z_top, z_bottom, z_is_buy, high, low, max_window,
                            min_stop_price, spread_price, frac, invert):
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

        # entry/stop/target: exactly the raw zone/technical levels -- no repricing.
        entry_price = entry_level
        eff_risk = abs(entry_price - stop_price)
        if eff_risk <= 0:
            continue
        raw_win_r = abs(target_price - entry_price) / eff_risk
        cost_in_r = spread_price / eff_risk  # exactly 1.0x spread, expressed in R units

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
        raw_r = raw_win_r if outcome == 1 else -1.0
        net_r = raw_r - cost_in_r  # 1.0 spread removed AFTER the raw result, both sides

        out_touch[zi] = touch_idx
        out_exit[zi] = exit_idx
        out_outcome[zi] = outcome
        out_net_r[zi] = net_r
        out_entry_price[zi] = entry_price
        out_stop_risk[zi] = eff_risk

    return out_touch, out_exit, out_outcome, out_net_r, out_entry_price, out_stop_risk


def build_frac_trades_flatcost(frac):
    frames = []
    for inst in INSTRUMENTS:
        times, lows, highs, df = rs.load_instrument(inst)
        high_ = df["high"].values.astype(np.float64)
        low_ = df["low"].values.astype(np.float64)
        pip = pg.PIP_SIZE[inst]
        spread_price = pg.SPREAD_PIPS_TYPICAL * pip

        confirm_idx, pivot_idx, is_high, price = g77._zigzag_pivots(high_, low_)
        z_created, z_top, z_bottom, z_is_buy, z_is_flip = g77._build_zones(confirm_idx, is_high, price)
        touch, exit_, outcome, net_r, entry_price, stop_risk = _resolve_frac_flatcost(
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
