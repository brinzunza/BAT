"""Direct user specification (2026-09-06), replacing goal89's structure-break fix with a
more precise, close-based mechanism:

  Uptrend = higher highs + higher lows. Track a CONFIRMED higher-high (ch) and CONFIRMED
  higher-low (cl), plus whichever POTENTIAL extreme is currently accumulating.

  While price is making new highs (the WITH-trend extreme), the potential high just
  extends -- locked in as confirmed the instant a single candle's CLOSE ends up below
  its predecessor's close (a simple 1-bar momentum flip, no minimum retracement). That
  lock-in switches tracking to the potential low (now the AGAINST-trend retracement).

  The retracement (potential low) is NOT locked in by a simple close reversal -- it only
  resolves one of two ways, both requiring a genuine CLOSE beyond a level (never just a
  wick):
    - closes below the last CONFIRMED low -> structure breaks the other way -> FLIP.
      Sell zone = [confirmed_high, confirmed_low] (the leg that was extending the old
      uptrend, now dormant -- unchanged, per this session's established flip-zone rule).
    - closes back above the last CONFIRMED high -> the uptrend continues. The
      retracement is promoted to confirmed_low, a new potential high starts fresh from
      here, and a pullback zone = [confirmed_high, the just-promoted confirmed_low] is
      created.

  Everything mirrors symmetrically for a downtrend (potential low is the WITH-trend
  extreme there, locked in by a single up-close; potential high is the AGAINST-trend
  bounce, structure-governed).

  Two explicit interpretation calls made where the request was ambiguous, flagged rather
  than silently assumed:
    1. The flip-side break ("price moves below the last confirmed higher low") is
       treated as CLOSE-based too, for consistency with the explicit "closed... not just
       a wick" requirement on the continuation side -- not stated for the flip side, but
       applying it there too avoids an asymmetric, wick-sensitive flip trigger sitting
       next to a close-based continuation trigger.
    2. "the potential higher low becomes the confirmed higher high" is implemented as
       "becomes the confirmed higher LOW" -- the literal wording appears to be a typo,
       since a low becoming a "high" contradicts the rest of the mechanism and the
       parallel construction of every other sentence in the spec.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def build_zones_close_based(open_, high, low, close):
    n = len(close)
    z_created = np.empty(n, dtype=np.int64)
    z_top = np.empty(n, dtype=np.float64)
    z_bottom = np.empty(n, dtype=np.float64)
    z_top_idx = np.empty(n, dtype=np.int64)
    z_bottom_idx = np.empty(n, dtype=np.int64)
    z_is_buy = np.empty(n, dtype=np.bool_)
    z_is_flip = np.empty(n, dtype=np.bool_)
    zc = 0

    # every locked-in potential extreme (for visualization -- shows the "candidate" trail
    # even where no zone was created), separate from the zone-worthy confirmed extremes
    piv_idx = np.empty(n, dtype=np.int64)
    piv_price = np.empty(n, dtype=np.float64)
    piv_is_high = np.empty(n, dtype=np.bool_)
    pc = 0

    trend = 1              # 1 = up, -1 = down (arbitrary bootstrap assumption)
    tracking_high = True   # which potential extreme is currently accumulating
    pot, pot_idx = high[0], 0

    has_ch, has_cl = False, False
    ch, ch_idx = 0.0, -1
    cl, cl_idx = 0.0, -1

    for i in range(1, n):
        if tracking_high:
            if high[i] > pot:
                pot, pot_idx = high[i], i

            if trend == 1:
                # WITH-trend (up): simple single-candle close reversal locks it in
                if close[i] < close[i - 1]:
                    piv_idx[pc] = pot_idx; piv_price[pc] = pot; piv_is_high[pc] = True; pc += 1
                    ch, ch_idx, has_ch = pot, pot_idx, True
                    tracking_high = False
                    pot, pot_idx = low[i], i
            else:
                # AGAINST-trend (a bounce during a downtrend): structure-governed
                if has_ch and close[i] > ch:
                    # FLIP UP: the old downtrend's last leg becomes the new zone
                    if has_ch and has_cl:
                        z_created[zc] = i; z_top[zc] = ch; z_bottom[zc] = cl
                        z_top_idx[zc] = ch_idx; z_bottom_idx[zc] = cl_idx
                        z_is_buy[zc] = True; z_is_flip[zc] = True
                        zc += 1
                    piv_idx[pc] = pot_idx; piv_price[pc] = pot; piv_is_high[pc] = True; pc += 1
                    ch, ch_idx = pot, pot_idx
                    trend = 1
                    pot, pot_idx = high[i], i   # fresh with-trend high tracking
                elif has_cl and close[i] < cl:
                    # downtrend continues: this bounce is confirmed, pullback (sell) zone
                    if has_cl:
                        z_created[zc] = i; z_top[zc] = pot; z_bottom[zc] = cl
                        z_top_idx[zc] = pot_idx; z_bottom_idx[zc] = cl_idx
                        z_is_buy[zc] = False; z_is_flip[zc] = False
                        zc += 1
                    piv_idx[pc] = pot_idx; piv_price[pc] = pot; piv_is_high[pc] = True; pc += 1
                    ch, ch_idx, has_ch = pot, pot_idx, True
                    tracking_high = False       # switch to with-trend (down) = low
                    pot, pot_idx = low[i], i
        else:
            if low[i] < pot:
                pot, pot_idx = low[i], i

            if trend == -1:
                if close[i] > close[i - 1]:
                    piv_idx[pc] = pot_idx; piv_price[pc] = pot; piv_is_high[pc] = False; pc += 1
                    cl, cl_idx, has_cl = pot, pot_idx, True
                    tracking_high = True
                    pot, pot_idx = high[i], i
            else:
                if has_cl and close[i] < cl:
                    # FLIP DOWN
                    if has_ch and has_cl:
                        z_created[zc] = i; z_top[zc] = ch; z_bottom[zc] = cl
                        z_top_idx[zc] = ch_idx; z_bottom_idx[zc] = cl_idx
                        z_is_buy[zc] = False; z_is_flip[zc] = True
                        zc += 1
                    piv_idx[pc] = pot_idx; piv_price[pc] = pot; piv_is_high[pc] = False; pc += 1
                    cl, cl_idx = pot, pot_idx
                    trend = -1
                    pot, pot_idx = low[i], i
                elif has_ch and close[i] > ch:
                    # uptrend continues: this retracement is confirmed, pullback (buy) zone
                    if has_ch:
                        z_created[zc] = i; z_top[zc] = ch; z_bottom[zc] = pot
                        z_top_idx[zc] = ch_idx; z_bottom_idx[zc] = pot_idx
                        z_is_buy[zc] = True; z_is_flip[zc] = False
                        zc += 1
                    piv_idx[pc] = pot_idx; piv_price[pc] = pot; piv_is_high[pc] = False; pc += 1
                    cl, cl_idx, has_cl = pot, pot_idx, True
                    tracking_high = True        # switch to with-trend (up) = high
                    pot, pot_idx = high[i], i

    return (z_created[:zc], z_top[:zc], z_bottom[:zc], z_top_idx[:zc], z_bottom_idx[:zc],
            z_is_buy[:zc], z_is_flip[:zc], piv_idx[:pc], piv_price[:pc], piv_is_high[:pc])


if __name__ == "__main__":
    # ---- hand-traceable synthetic validation before touching real data ----
    # Bars index:  0    1    2    3    4    5    6    7    8    9   10   11
    # close:      10   12   14   13   15   17   16   19   18   20    9   11
    # Story: rises 10->14 (bar2 peak), bar3 closes down (13<14) -> locks confirmed_high=14
    #        (at bar2). Tracks potential low from bar3 (low=13-ish). Rises again through
    #        bar5=17, bar6 closes down (16<17) -- wait, we need the retracement to resolve
    #        via STRUCTURE not a simple reversal, so build bars so it genuinely closes
    #        above ch(14) at some point while retracing, then continues to a new high,
    #        confirms it, retraces again, then finally closes below a confirmed low to flip.
    n = 12
    close = np.array([10,12,14,13,15,17,16,19,18,20,9,11], dtype=np.float64)
    high  = np.array([10,12,14,13.5,15,17,16.5,19,18.5,20,9.5,11.5], dtype=np.float64)
    low   = np.array([10,12,13.5,13,14.5,17,16,18.5,18,20,9,10.5], dtype=np.float64)
    openp = close.copy()

    (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip,
     piv_idx, piv_price, piv_is_high) = build_zones_close_based(openp, high, low, close)

    print("Locked-in potentials (trail):")
    for i in range(len(piv_idx)):
        print(f"  bar {piv_idx[i]}: {'HIGH' if piv_is_high[i] else 'low'} = {piv_price[i]}")
    print("\nZones:")
    for i in range(len(z_created)):
        kind = "FLIP" if z_is_flip[i] else "pullback"
        side = "BUY" if z_is_buy[i] else "SELL"
        print(f"  created@bar{z_created[i]}: {side} {kind}  top={z_top[i]} (bar {z_top_idx[i]})  "
              f"bottom={z_bottom[i]} (bar {z_bottom_idx[i]})")
