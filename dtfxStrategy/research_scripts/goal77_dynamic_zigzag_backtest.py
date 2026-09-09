"""Direct user request (2026-09-05): scale the corrected zone rule + dynamic zigzag swing
confirmation (no fixed lookback window -- a candidate high/low confirms the instant price
fails to extend it, per the user's own correction) up from a 24h illustrative check to a
real backtest: 4 symbols (EURUSD, GBPUSD, USDJPY, EURGBP), each over a trailing 1-year
window and then full history, on 1-minute bars.

Full history is ~6.3M 1-minute bars per symbol, and this swing rule confirms on roughly
half of all bars (687 pivots from 1435 bars in the earlier 24h check) -- a pure-Python
per-bar loop plus a bounded per-zone resolution scan would be far too slow at that scale,
so both stages are numba-JIT'd here, matching this project's own established convention
for large sequential backtests (prod_filter_grid_search.py's _simulate_generic_numba,
goal76's SBS sequence scan).

Cost model: same realistic-fill convention as the rest of this project -- a touch that
reaches the full spread past the zone's own mid gets a genuinely free entry; a shallow
touch gets repriced one spread away (no double-charging on the exit side, per this
session's own corrected rule).
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29

rs, pg = g29.rs, g29.pg


@njit(cache=True)
def _zigzag_pivots(high, low):
    """Dynamic zigzag: a candidate high/low confirms the instant a bar fails to extend it.
    No fixed lookback window, no retracement threshold. Returns parallel arrays (confirm_idx,
    pivot_idx, is_high, price), chronological by confirm_idx."""
    n = len(high)
    out_confirm = np.empty(n, dtype=np.int64)
    out_pivot = np.empty(n, dtype=np.int64)
    out_is_high = np.empty(n, dtype=np.bool_)
    out_price = np.empty(n, dtype=np.float64)
    count = 0

    direction_up = True
    cand_price = high[0]
    cand_idx = 0
    for i in range(1, n):
        if direction_up:
            if high[i] > cand_price:
                cand_price = high[i]; cand_idx = i
            else:
                out_confirm[count] = i; out_pivot[count] = cand_idx
                out_is_high[count] = True; out_price[count] = cand_price
                count += 1
                direction_up = False
                cand_price = low[i]; cand_idx = i
        else:
            if low[i] < cand_price:
                cand_price = low[i]; cand_idx = i
            else:
                out_confirm[count] = i; out_pivot[count] = cand_idx
                out_is_high[count] = False; out_price[count] = cand_price
                count += 1
                direction_up = True
                cand_price = high[i]; cand_idx = i

    return out_confirm[:count], out_pivot[:count], out_is_high[:count], out_price[:count]


@njit(cache=True)
def _build_zones(confirm_idx, is_high, price):
    """Corrected zone rule: only pullback legs (matching the current trend direction) are
    live zones. Advancing legs are structure only, UNTIL trend inverts, at which point the
    dormant advancing leg becomes the new trend's first zone (kind=flip)."""
    m = len(confirm_idx)
    z_created = np.empty(m, dtype=np.int64)
    z_top = np.empty(m, dtype=np.float64)
    z_bottom = np.empty(m, dtype=np.float64)
    z_is_buy = np.empty(m, dtype=np.bool_)
    z_is_flip = np.empty(m, dtype=np.bool_)
    zc = 0

    has_high, has_low = False, False
    last_high_price, last_low_price = 0.0, 0.0
    trend = 0  # 0=unknown, 1=up, -1=down

    for k in range(m):
        ci = confirm_idx[k]
        p = price[k]
        if is_high[k]:
            if has_high:
                if p > last_high_price:
                    if trend == -1 and has_low:
                        z_created[zc] = ci; z_top[zc] = last_high_price; z_bottom[zc] = last_low_price
                        z_is_buy[zc] = True; z_is_flip[zc] = True
                        zc += 1
                    trend = 1
                else:
                    if has_low:
                        z_created[zc] = ci; z_top[zc] = p; z_bottom[zc] = last_low_price
                        z_is_buy[zc] = False; z_is_flip[zc] = False
                        zc += 1
                    trend = -1
            last_high_price = p; has_high = True
        else:
            if has_low:
                if p < last_low_price:
                    if trend == 1 and has_high:
                        z_created[zc] = ci; z_top[zc] = last_high_price; z_bottom[zc] = last_low_price
                        z_is_buy[zc] = False; z_is_flip[zc] = True
                        zc += 1
                    trend = -1
                else:
                    if has_high:
                        z_created[zc] = ci; z_top[zc] = last_high_price; z_bottom[zc] = p
                        z_is_buy[zc] = True; z_is_flip[zc] = False
                        zc += 1
                    trend = 1
            last_low_price = p; has_low = True

    return z_created[:zc], z_top[:zc], z_bottom[:zc], z_is_buy[:zc], z_is_flip[:zc]


@njit(cache=True)
def _resolve_zones(z_created, z_top, z_bottom, z_is_buy, high, low, max_window,
                    min_stop_price, spread_price, invert=False):
    """z_is_buy controls the zone's TRUE price context -- which side price approaches the
    zone from, hence the touch condition and entry-repricing direction. This must NOT be
    flipped to "invert" the trade: doing so changes which bars even count as a touch (a
    buy zone is approached from above, low<=mid; a sell zone from below, high>=mid), not
    just which side wins, producing a different and not-really-meaningful experiment
    (caught after the first version of this script gave a non-complementary win rate,
    which is what gave the bug away).

    invert=True instead keeps touch detection and entry price exactly as in the original
    (same zone context, same fill), and only swaps which boundary counts as stop vs
    target for the OUTCOME -- a true "take the other side of the same fill" inversion,
    which must produce an exactly complementary win/loss on every resolved trade."""
    n_bars = len(high)
    m = len(z_created)
    out_touch = np.full(m, -1, dtype=np.int64)
    out_exit = np.full(m, -1, dtype=np.int64)
    out_outcome = np.zeros(m, dtype=np.int8)  # 0=none/no_touch, 1=win, 2=loss
    out_net_r = np.zeros(m, dtype=np.float64)

    for zi in range(m):
        top, bottom, is_buy = z_top[zi], z_bottom[zi], z_is_buy[zi]
        if (top - bottom) / 2.0 < min_stop_price:
            continue
        mid = (top + bottom) / 2.0
        c = z_created[zi]
        end = c + max_window
        if end > n_bars:
            end = n_bars
        touch_idx = -1
        for j in range(c + 1, end):
            touched = (low[j] <= mid) if is_buy else (high[j] >= mid)
            if touched:
                touch_idx = j
                break
        if touch_idx == -1:
            continue
        if is_buy:
            deep = low[touch_idx] <= mid - spread_price
            entry_price = mid if deep else mid + spread_price
        else:
            deep = high[touch_idx] >= mid + spread_price
            entry_price = mid if deep else mid - spread_price

        trade_is_buy = (not is_buy) if invert else is_buy
        stop_price = bottom if trade_is_buy else top
        target_price = top if trade_is_buy else bottom
        eff_risk = abs(entry_price - stop_price)
        if eff_risk <= 0:
            continue
        win_r = abs(target_price - entry_price) / eff_risk

        outcome, exit_idx = 0, -1
        for k in range(touch_idx, end):
            hit_sl = (low[k] <= stop_price) if trade_is_buy else (high[k] >= stop_price)
            hit_tp = (high[k] >= target_price) if trade_is_buy else (low[k] <= target_price)
            if hit_sl:
                outcome, exit_idx = 2, k
                break
            elif hit_tp:
                outcome, exit_idx = 1, k
                break
        if outcome == 0:
            continue
        out_touch[zi] = touch_idx
        out_exit[zi] = exit_idx
        out_outcome[zi] = outcome
        out_net_r[zi] = win_r if outcome == 1 else -1.0

    return out_touch, out_exit, out_outcome, out_net_r


def run_backtest(inst, start=None, max_window=1440, min_stop_pips=3.0, invert=False):
    times, lows, highs, df = rs.load_instrument(inst)
    if start is not None:
        mask = times >= np.datetime64(start)
        times, df = times[mask], df.iloc[mask].reset_index(drop=True)
    high_ = df["high"].values.astype(np.float64)
    low_ = df["low"].values.astype(np.float64)
    pip = pg.PIP_SIZE[inst]
    spread_price = pg.SPREAD_PIPS_TYPICAL * pip

    confirm_idx, pivot_idx, is_high, price = _zigzag_pivots(high_, low_)
    z_created, z_top, z_bottom, z_is_buy, z_is_flip = _build_zones(confirm_idx, is_high, price)
    # z_is_buy is always the zone's TRUE context (touch condition + entry repricing) --
    # invert is handled inside _resolve_zones by swapping stop/target for the SAME fill,
    # not by flipping is_buy here (see that function's docstring for why the latter is wrong).
    touch, exit_, outcome, net_r = _resolve_zones(
        z_created, z_top, z_bottom, z_is_buy, high_, low_, max_window,
        min_stop_pips * pip, spread_price, invert=invert)

    resolved = outcome > 0
    n_pivots, n_zones, n_resolved = len(confirm_idx), len(z_created), int(resolved.sum())
    if n_resolved < 5:
        return dict(inst=inst, n_bars=len(df), n_pivots=n_pivots, n_zones=n_zones,
                     n_resolved=n_resolved, pf=None, wr=None, freq=None,
                     entry_times=None, net_r=None)

    entry_idx = touch[resolved]
    entry_times = times[entry_idx]
    net_r_r = net_r[resolved]
    years = max((entry_times.max() - entry_times.min()) / np.timedelta64(365, "D"), 1e-6)
    wr = 100.0 * (net_r_r > 0).mean()
    gw = net_r_r[net_r_r > 0].sum()
    gl = -net_r_r[net_r_r <= 0].sum()
    pf = gw / gl if gl > 0 else float("inf")
    is_flip_r = z_is_flip[resolved]
    return dict(inst=inst, n_bars=len(df), n_pivots=n_pivots, n_zones=n_zones,
                n_resolved=n_resolved, pf=pf, wr=wr, freq=n_resolved / years,
                entry_times=entry_times, net_r=net_r_r, is_flip=is_flip_r)


if __name__ == "__main__":
    import time
    t0 = time.time()
    INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"]

    print("=== Trailing 1 year, 1-minute bars ===")
    one_year_results = {}
    for inst in INSTRUMENTS:
        times, _, _, _ = rs.load_instrument(inst)
        start = times.max() - np.timedelta64(365, "D")
        r = run_backtest(inst, start=start)
        one_year_results[inst] = r
        if r["n_resolved"] < 5:
            print(f"  {inst}: n_bars={r['n_bars']} pivots={r['n_pivots']} zones={r['n_zones']} "
                  f"resolved={r['n_resolved']} (too few) ({time.time()-t0:.0f}s)")
            continue
        flip_r = r["net_r"][r["is_flip"]]
        pull_r = r["net_r"][~r["is_flip"]]
        print(f"  {inst}: bars={r['n_bars']} pivots={r['n_pivots']} zones={r['n_zones']} "
              f"resolved={r['n_resolved']} freq={r['freq']:.0f}/yr WR={r['wr']:.1f}% PF={r['pf']:.3f} "
              f"(flip n={len(flip_r)} sum={flip_r.sum():.1f}R | pullback n={len(pull_r)} sum={pull_r.sum():.1f}R) "
              f"({time.time()-t0:.0f}s)")

    print("\n=== Full history, 1-minute bars ===")
    full_results = {}
    for inst in INSTRUMENTS:
        r = run_backtest(inst, start=None)
        full_results[inst] = r
        if r["n_resolved"] < 5:
            print(f"  {inst}: n_bars={r['n_bars']} pivots={r['n_pivots']} zones={r['n_zones']} "
                  f"resolved={r['n_resolved']} (too few) ({time.time()-t0:.0f}s)")
            continue
        flip_r = r["net_r"][r["is_flip"]]
        pull_r = r["net_r"][~r["is_flip"]]
        print(f"  {inst}: bars={r['n_bars']} pivots={r['n_pivots']} zones={r['n_zones']} "
              f"resolved={r['n_resolved']} freq={r['freq']:.0f}/yr WR={r['wr']:.1f}% PF={r['pf']:.3f} "
              f"(flip n={len(flip_r)} sum={flip_r.sum():.1f}R | pullback n={len(pull_r)} sum={pull_r.sum():.1f}R) "
              f"({time.time()-t0:.0f}s)")

    print("\n=== INVERTED positions: every buy -> sell, every sell -> buy (1 year) ===")
    for inst in INSTRUMENTS:
        times, _, _, _ = rs.load_instrument(inst)
        start = times.max() - np.timedelta64(365, "D")
        r = run_backtest(inst, start=start, invert=True)
        if r["n_resolved"] < 5:
            print(f"  {inst}: resolved={r['n_resolved']} (too few)")
            continue
        flip_r = r["net_r"][r["is_flip"]]
        pull_r = r["net_r"][~r["is_flip"]]
        print(f"  {inst}: resolved={r['n_resolved']} freq={r['freq']:.0f}/yr WR={r['wr']:.1f}% PF={r['pf']:.3f} "
              f"(flip n={len(flip_r)} sum={flip_r.sum():.1f}R | pullback n={len(pull_r)} sum={pull_r.sum():.1f}R) "
              f"({time.time()-t0:.0f}s)")

    print("\n=== INVERTED positions: every buy -> sell, every sell -> buy (full history) ===")
    for inst in INSTRUMENTS:
        r = run_backtest(inst, start=None, invert=True)
        if r["n_resolved"] < 5:
            print(f"  {inst}: resolved={r['n_resolved']} (too few)")
            continue
        flip_r = r["net_r"][r["is_flip"]]
        pull_r = r["net_r"][~r["is_flip"]]
        print(f"  {inst}: resolved={r['n_resolved']} freq={r['freq']:.0f}/yr WR={r['wr']:.1f}% PF={r['pf']:.3f} "
              f"(flip n={len(flip_r)} sum={flip_r.sum():.1f}R | pullback n={len(pull_r)} sum={pull_r.sum():.1f}R) "
              f"({time.time()-t0:.0f}s)")

    print("\n=== Last 24 hours, 1-minute bars: original vs inverted ===")
    for inst in INSTRUMENTS:
        times, _, _, _ = rs.load_instrument(inst)
        start = times.max() - np.timedelta64(24, "h")
        r_o = run_backtest(inst, start=start, invert=False)
        r_i = run_backtest(inst, start=start, invert=True)
        print(f"  {inst}: bars={r_o['n_bars']} pivots={r_o['n_pivots']} zones={r_o['n_zones']}")
        if r_o["n_resolved"] < 5:
            print(f"    original: resolved={r_o['n_resolved']} (too few)")
        else:
            print(f"    original: resolved={r_o['n_resolved']} freq={r_o['freq']:.0f}/yr "
                  f"WR={r_o['wr']:.1f}% PF={r_o['pf']:.3f}")
        if r_i["n_resolved"] < 5:
            print(f"    inverted: resolved={r_i['n_resolved']} (too few)")
        else:
            print(f"    inverted: resolved={r_i['n_resolved']} freq={r_i['freq']:.0f}/yr "
                  f"WR={r_i['wr']:.1f}% PF={r_i['pf']:.3f}")

    print(f"\n(total: {time.time()-t0:.0f}s)")
