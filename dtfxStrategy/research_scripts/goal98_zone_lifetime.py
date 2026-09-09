"""Direct user request (2026-09-06): apply the corrected zone-lifetime rule from the
synthetic "Reading the Swing" diagram -- a zone's box runs from its own defining swing
until it is invalidated, where "invalidated" means (1) price first leaves the zone behind,
trading beyond its far edge in the direction the zone's own formation implied, (2) price
then swings back and genuinely tags the zone's 50% level, and (3) price then trades out of
the zone entirely through either edge -- to real 1-minute market data, using intrabar
high/low (not close) for every touch, since real fills happen on wicks.

This reuses goal95's zone set unchanged (close-based-symmetric pivots, every swing is a
zone, no structure gate) and adds one new function on top: resolve_zone_lifetime. Its
outputs double as a trade outcome under a stricter, more patient fill discipline than this
project's usual _resolve_frac_ex (which fills on the very first touch of the level,
without requiring the zone be "left behind" first) -- exiting through the target-side edge
after the tag counts as a win, exiting through the stop-side edge counts as a loss, exactly
mirroring the stop/target convention used everywhere else in this project.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def resolve_zone_lifetime(z_created, z_top, z_bottom, z_is_buy, high, low):
    """For each zone: find the first bar (at or after creation) where price trades beyond
    the zone's far edge in its own continuation direction (buy: high > top; sell: low <
    bottom) -- flip zones, built from an already-dormant leg, often already satisfy this on
    their own creation bar. From there, find the first bar where price swings back and
    tags the zone's 50% level (buy: low <= mid; sell: high >= mid). From there, find the
    first bar where price exits the zone through either edge -- the target-side edge is a
    win (the continuation resumed), the stop-side edge is a loss (the level failed).

    Returns (overshoot_idx, tag_idx, exit_idx, outcome) parallel arrays, each -1 / 0 where
    that stage never happens before the data runs out (the zone just stays open at
    whichever stage it reached)."""
    m = len(z_created)
    n_bars = len(high)
    out_overshoot = np.full(m, -1, dtype=np.int64)
    out_tag = np.full(m, -1, dtype=np.int64)
    out_exit = np.full(m, -1, dtype=np.int64)
    out_outcome = np.zeros(m, dtype=np.int8)  # 0 = open/unresolved, 1 = win, 2 = loss

    for zi in range(m):
        top = z_top[zi]; bottom = z_bottom[zi]; is_buy = z_is_buy[zi]
        mid = (top + bottom) / 2.0
        c = z_created[zi]

        overshoot_idx = -1
        if is_buy:
            if high[c] > top:
                overshoot_idx = c
        else:
            if low[c] < bottom:
                overshoot_idx = c
        if overshoot_idx == -1:
            for j in range(c + 1, n_bars):
                if is_buy:
                    if high[j] > top:
                        overshoot_idx = j; break
                else:
                    if low[j] < bottom:
                        overshoot_idx = j; break
        if overshoot_idx == -1:
            continue

        tag_idx = -1
        for j in range(overshoot_idx, n_bars):
            if is_buy:
                if low[j] <= mid:
                    tag_idx = j; break
            else:
                if high[j] >= mid:
                    tag_idx = j; break
        if tag_idx == -1:
            out_overshoot[zi] = overshoot_idx
            continue

        exit_idx = -1; outcome = 0
        for j in range(tag_idx, n_bars):
            if is_buy:
                hit_stop = low[j] < bottom
                hit_target = high[j] > top
            else:
                hit_stop = high[j] > top
                hit_target = low[j] < bottom
            if hit_stop:
                exit_idx = j; outcome = 2; break
            elif hit_target:
                exit_idx = j; outcome = 1; break
        out_overshoot[zi] = overshoot_idx
        out_tag[zi] = tag_idx
        if exit_idx != -1:
            out_exit[zi] = exit_idx
            out_outcome[zi] = outcome

    return out_overshoot, out_tag, out_exit, out_outcome
