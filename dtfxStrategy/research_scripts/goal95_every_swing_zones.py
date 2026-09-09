"""Direct user correction (2026-09-06): goal92's zone builder was missing "a ton of
zones" because its against-trend side only locks in on a genuine structure break (a
close beyond the last CONFIRMED opposite extreme) -- every intermediate wiggle during a
longer retracement gets silently absorbed into one running candidate instead of each
becoming its own swing. Corrected instruction: every swing should produce a zone.

Two changes from goal92, both explicit:
  1. Pivot detection goes back to symmetric -- BOTH sides lock in on the same simple
     rule (one candle's close reverses against its predecessor's), not just the
     with-trend side. This is the same close-based trigger already hand-validated in
     goal92, just applied to both directions instead of only one.
  2. Zone building goes back to "every consecutive swing pair is a zone" -- this is
     goal77's ORIGINAL _build_zones logic (pullback zone when a new pivot fails to
     exceed the last pivot of its own type; flip zone at the trend change), which an
     earlier correction this session replaced with a structure-confirmation gate. That
     gate is intentionally NOT applied here, per this message's explicit instruction --
     noted since it directly reverses that earlier correction, not silently dropped.

Also per this message: entry reverts to 50% (from the 30% pick), and direction reverts
to NON-inverted -- buy the zone in an uptrend context, sell it in a downtrend context,
no fade. This is the original, most basic form of the rule, meant to be verified
correct before any of the refinements (structure confirmation, shallower entry,
inversion) go back on top of it.
"""
import os
import sys

import numpy as np
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import goal77_dynamic_zigzag_backtest as g77
import goal81_shallow_entry_two_account as g81

rs, pg = g77.rs, g77.pg
resolve_zones_at_frac = g81._resolve_frac_ex


@njit(cache=True)
def close_based_zigzag_symmetric(high, low, close):
    """Every swing confirms the same way, on both sides: locked in the instant one
    candle's close falls on the opposite side of its immediate predecessor's close.
    Same trigger as goal92, just no longer gated to only the with-trend side."""
    n = len(close)
    out_confirm = np.empty(n, dtype=np.int64)
    out_pivot = np.empty(n, dtype=np.int64)
    out_is_high = np.empty(n, dtype=np.bool_)
    out_price = np.empty(n, dtype=np.float64)
    count = 0

    tracking_high = True
    pot, pot_idx = high[0], 0

    for i in range(1, n):
        if tracking_high:
            if high[i] > pot:
                pot, pot_idx = high[i], i
            if close[i] < close[i - 1]:
                out_confirm[count] = i; out_pivot[count] = pot_idx
                out_is_high[count] = True; out_price[count] = pot
                count += 1
                tracking_high = False
                pot, pot_idx = low[i], i
        else:
            if low[i] < pot:
                pot, pot_idx = low[i], i
            if close[i] > close[i - 1]:
                out_confirm[count] = i; out_pivot[count] = pot_idx
                out_is_high[count] = False; out_price[count] = pot
                count += 1
                tracking_high = True
                pot, pot_idx = high[i], i

    return out_confirm[:count], out_pivot[:count], out_is_high[:count], out_price[:count]


@njit(cache=True)
def build_zones_every_swing(confirm_idx, pivot_idx, is_high, price):
    """goal77._build_zones' original logic exactly (every consecutive swing pair
    becomes a zone -- pullback when a new pivot fails to exceed the last pivot of its
    own type, flip at the trend change), extended only to also track which bar each
    edge's price actually came from, for the chart."""
    m = len(confirm_idx)
    z_created = np.empty(m, dtype=np.int64)
    z_top = np.empty(m, dtype=np.float64)
    z_bottom = np.empty(m, dtype=np.float64)
    z_top_idx = np.empty(m, dtype=np.int64)
    z_bottom_idx = np.empty(m, dtype=np.int64)
    z_is_buy = np.empty(m, dtype=np.bool_)
    z_is_flip = np.empty(m, dtype=np.bool_)
    zc = 0

    has_high, has_low = False, False
    last_high_price, last_low_price = 0.0, 0.0
    last_high_idx, last_low_idx = -1, -1
    trend = 0

    for k in range(m):
        ci = confirm_idx[k]
        pi = pivot_idx[k]
        p = price[k]
        if is_high[k]:
            if has_high:
                if p > last_high_price:
                    if trend == -1 and has_low:
                        z_created[zc] = ci; z_top[zc] = last_high_price; z_bottom[zc] = last_low_price
                        z_top_idx[zc] = last_high_idx; z_bottom_idx[zc] = last_low_idx
                        z_is_buy[zc] = True; z_is_flip[zc] = True
                        zc += 1
                    trend = 1
                else:
                    if has_low:
                        z_created[zc] = ci; z_top[zc] = p; z_bottom[zc] = last_low_price
                        z_top_idx[zc] = pi; z_bottom_idx[zc] = last_low_idx
                        z_is_buy[zc] = False; z_is_flip[zc] = False
                        zc += 1
                    trend = -1
            last_high_price = p; last_high_idx = pi; has_high = True
        else:
            if has_low:
                if p < last_low_price:
                    if trend == 1 and has_high:
                        z_created[zc] = ci; z_top[zc] = last_high_price; z_bottom[zc] = last_low_price
                        z_top_idx[zc] = last_high_idx; z_bottom_idx[zc] = last_low_idx
                        z_is_buy[zc] = False; z_is_flip[zc] = True
                        zc += 1
                    trend = -1
                else:
                    if has_high:
                        z_created[zc] = ci; z_top[zc] = last_high_price; z_bottom[zc] = p
                        z_top_idx[zc] = last_high_idx; z_bottom_idx[zc] = pi
                        z_is_buy[zc] = True; z_is_flip[zc] = False
                        zc += 1
                    trend = 1
            last_low_price = p; last_low_idx = pi; has_low = True

    return (z_created[:zc], z_top[:zc], z_bottom[:zc], z_top_idx[:zc], z_bottom_idx[:zc],
            z_is_buy[:zc], z_is_flip[:zc])


if __name__ == "__main__":
    # same 12-bar synthetic trace used to validate goal92, run through this version too
    close = np.array([10,12,14,13,15,17,16,19,18,20,9,11], dtype=np.float64)
    high  = np.array([10,12,14,13.5,15,17,16.5,19,18.5,20,9.5,11.5], dtype=np.float64)
    low   = np.array([10,12,13.5,13,14.5,17,16,18.5,18,20,9,10.5], dtype=np.float64)

    confirm_idx, pivot_idx, is_high, price = close_based_zigzag_symmetric(high, low, close)
    print("Swings (every one, both sides symmetric):")
    for i in range(len(confirm_idx)):
        print(f"  bar {pivot_idx[i]}: {'HIGH' if is_high[i] else 'low'} = {price[i]}  (confirmed@bar{confirm_idx[i]})")

    (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
     ) = build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)
    print(f"\nZones (every swing pair, {len(z_created)} total):")
    for i in range(len(z_created)):
        kind = "FLIP" if z_is_flip[i] else "pullback"
        side = "BUY" if z_is_buy[i] else "SELL"
        print(f"  created@bar{z_created[i]}: {side} {kind}  top={z_top[i]} (bar {z_top_idx[i]})  "
              f"bottom={z_bottom[i]} (bar {z_bottom_idx[i]})")

    print(f"\nOriginal goal92 (structure-gated) found only 4 zones on this same data --")
    print(f"this version finds {len(z_created)}.")
