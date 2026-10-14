"""Direct user bug report (2026-09-06): goal77._build_zones is wrong. It updates
last_high_price/last_low_price and flips `trend` on EVERY pivot the dynamic zigzag hands
it -- including ones that never actually broke the prior confirmed structure (a lower
high in an uptrend, a higher low in a downtrend). That treats every minor wiggle as a
trend flip and a zone, which is why the interactive chart showed hundreds of zones over
one week.

The user's own restated rule: "in an uptrend when price is going up, it needs to have a
retracement and then go back up, once it makes a higher high than the previous highest
point, then it becomes confirmed" -- i.e. a retracement pivot is PENDING until price
subsequently exceeds the prior CONFIRMED extreme (not just any pivot value). This is
exactly what PRODLIVETRADING/strategy/structure_fsm.py already implements correctly
(pending_high_idx/pending_low_idx only promote to confirmed on a genuine structure
break) -- that file was read earlier this session but goal77's zone builder never
adopted its pending/confirmed distinction. This port fixes that, keeping the dynamic
zigzag pivot detector (no fixed lookback, no retracement threshold) unchanged, and adds
the flip-zone concept structure_fsm.py doesn't need (it's specific to this session's
fade-the-pullback strategy) using the same "only a genuine break promotes state" rule.
"""
import os
import sys

import numpy as np
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import goal77_dynamic_zigzag_backtest as g77

_zigzag_pivots = g77._zigzag_pivots


@njit(cache=True)
def build_zones_correct(confirm_idx, pivot_idx, is_high, price):
    """Structure-based zone builder: a pivot only promotes confirmed_high/confirmed_low
    (and only then can create a zone) when it GENUINELY breaks the prior confirmed
    extreme on either side -- exactly mirroring structure_fsm.py's pending/confirmed
    state machine. A pivot that doesn't break structure is held as `pending` only and
    changes nothing else -- no trend flip, no zone, no reference update.

    Pullback zone: price breaks its OWN trend's confirmed extreme further (e.g. a new
    high exceeding confirmed_high in an uptrend) -> the pending opposite pivot since the
    last confirmed extreme is promoted, and the zone is [old confirmed extreme, that
    newly-confirmed pivot].
    Flip zone: price breaks the confirmed extreme on the OPPOSITE side of the current
    trend (e.g. a new low below confirmed_low while trend is still 'up') -> the trend
    reverses, and the zone is the last established leg of the OLD trend
    [confirmed_high, confirmed_low], unchanged from before the break.
    """
    m = len(confirm_idx)
    z_created = np.empty(m, dtype=np.int64)
    z_top = np.empty(m, dtype=np.float64)
    z_bottom = np.empty(m, dtype=np.float64)
    z_top_idx = np.empty(m, dtype=np.int64)
    z_bottom_idx = np.empty(m, dtype=np.int64)
    z_is_buy = np.empty(m, dtype=np.bool_)
    z_is_flip = np.empty(m, dtype=np.bool_)
    zc = 0

    has_ch, has_cl = False, False           # confirmed high / low
    ch, cl = 0.0, 0.0
    ch_idx, cl_idx = -1, -1
    has_ph, has_pl = False, False           # pending high / low (since the last confirmed extreme)
    ph, pl = 0.0, 0.0
    ph_idx, pl_idx = -1, -1
    trend = 0                               # 0 unknown, 1 up, -1 down

    for k in range(m):
        ci = confirm_idx[k]
        pi = pivot_idx[k]
        v = price[k]

        if is_high[k]:
            if not has_ch:
                ch, ch_idx, has_ch = v, pi, True
            elif trend == 1:
                if v > ch:
                    # genuine higher high -- promotes any pending low into a confirmed
                    # pullback zone [old confirmed high .. that pending low]
                    if has_pl:
                        z_created[zc] = ci; z_top[zc] = ch; z_bottom[zc] = pl
                        z_top_idx[zc] = ch_idx; z_bottom_idx[zc] = pl_idx
                        z_is_buy[zc] = True; z_is_flip[zc] = False
                        zc += 1
                        cl, cl_idx, has_cl = pl, pl_idx, True
                        has_pl = False
                    ch, ch_idx = v, pi
                else:
                    ph, ph_idx, has_ph = v, pi, True   # lower high -- PENDING only
            else:  # trend == -1 (or 0, treat as down-side reference) and this high breaks it
                if v > ch:
                    # breaks the downtrend's confirmed high -> FLIP UP.
                    # zone = the old downtrend's last leg, [confirmed_high .. confirmed_low],
                    # unchanged -- exactly the dormant leg becoming the new trend's first zone.
                    if has_cl:
                        z_created[zc] = ci; z_top[zc] = ch; z_bottom[zc] = cl
                        z_top_idx[zc] = ch_idx; z_bottom_idx[zc] = cl_idx
                        z_is_buy[zc] = True; z_is_flip[zc] = True
                        zc += 1
                    ch, ch_idx = v, pi
                    trend = 1
                    has_ph = False
                else:
                    ph, ph_idx, has_ph = v, pi, True
        else:
            if not has_cl:
                cl, cl_idx, has_cl = v, pi, True
            elif trend == -1:
                if v < cl:
                    if has_ph:
                        z_created[zc] = ci; z_top[zc] = ph; z_bottom[zc] = cl
                        z_top_idx[zc] = ph_idx; z_bottom_idx[zc] = cl_idx
                        z_is_buy[zc] = False; z_is_flip[zc] = False
                        zc += 1
                        ch, ch_idx, has_ch = ph, ph_idx, True
                        has_ph = False
                    cl, cl_idx = v, pi
                else:
                    pl, pl_idx, has_pl = v, pi, True
            else:  # trend == 1 (or 0) and this low breaks it
                if v < cl:
                    if has_ch:
                        z_created[zc] = ci; z_top[zc] = ch; z_bottom[zc] = cl
                        z_top_idx[zc] = ch_idx; z_bottom_idx[zc] = cl_idx
                        z_is_buy[zc] = False; z_is_flip[zc] = True
                        zc += 1
                    cl, cl_idx = v, pi
                    trend = -1
                    has_pl = False
                else:
                    pl, pl_idx, has_pl = v, pi, True

    return (z_created[:zc], z_top[:zc], z_bottom[:zc], z_top_idx[:zc], z_bottom_idx[:zc],
            z_is_buy[:zc], z_is_flip[:zc])


if __name__ == "__main__":
    # sanity check: run on one instrument and compare zone COUNT against the old
    # (buggy) builder over the same pivot sequence, on a short recent window
    import pandas as pd
    rs, pg = g77.rs, g77.pg
    times, lows, highs, df = rs.load_instrument("USDJPY")
    times = pd.DatetimeIndex(times)
    mask = times >= pd.Timestamp("2026-07-15")
    sub_df = df.loc[mask]
    high_ = sub_df["high"].values.astype(np.float64)
    low_ = sub_df["low"].values.astype(np.float64)

    confirm_idx, pivot_idx, is_high, price = _zigzag_pivots(high_, low_)
    print(f"raw dynamic-zigzag pivots: {len(confirm_idx):,}")

    z_created_old, z_top_old, z_bottom_old, z_is_buy_old, z_is_flip_old = g77._build_zones(
        confirm_idx, is_high, price)
    print(f"OLD (buggy) zone builder: {len(z_created_old):,} zones "
          f"(pullback {int((~z_is_flip_old).sum())}, flip {int(z_is_flip_old.sum())})")

    (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
     ) = build_zones_correct(confirm_idx, pivot_idx, is_high, price)
    print(f"NEW (corrected) zone builder: {len(z_created):,} zones "
          f"(pullback {int((~z_is_flip).sum())}, flip {int(z_is_flip.sum())})")
