"""Direct user request (2026-09-06): two new research questions on top of the validated
fade-the-pullback rule (goal77/goal79):

  (A) Entry depth: is 50% actually the best retracement level, or would a shallower
      (30%) or deeper (70%) entry do better? frac=0 means "no retracement required, entry
      right at the zone's near edge"; frac=1 means "full fill, entry at the zone's far
      edge"; frac=0.5 reproduces the existing 50%-entry rule exactly.

  (B) "Fade the breakout": the user's own idea -- in an uptrend, once a (buy-context)
      zone's BOTTOM is taken out entirely (a full fill, the classical invalidation), use
      that break itself as a fresh BUY entry (mirrored: a sell-context zone's TOP taken
      out becomes a fresh SELL entry) -- betting on a reversal of the breakout, not a
      continuation. Stop is placed one zone-height beyond the entry; two target
      conventions are compared: a plain 1:1 (relative to entry) vs. the zone's own
      opposite-edge price level (these coincide when the entry fill is exactly at the
      edge, and diverge by the spread when it isn't -- see write-up).

Reuses goal77's exact zigzag/zone-building code unchanged (only entry/stop/target
placement varies); both experiments run full history, all four pairs, same 1.6-pip
modeled spread and $3-pip minimum stop floor as everywhere else in this project.
"""
import os
import sys
import time

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
def _resolve_frac(z_created, z_top, z_bottom, z_is_buy, high, low, max_window,
                   min_stop_price, min_height_price, spread_price, frac, invert):
    """Same rule as goal77._resolve_zones, generalized from a fixed 50% entry to an
    arbitrary retracement fraction `frac` (0 = shallowest, at the near edge; 1 = deepest,
    at the far edge; 0.5 reproduces the original mid-entry exactly).

    Two separate floors, deliberately not conflated: `min_stop_price` rejects a trade
    whose ACTUAL risk at this entry depth is unrealistically tiny (frac near 0 or 1);
    `min_height_price`, held FIXED across every frac in a sweep, controls which zones
    qualify at all -- without it, a shallow-entry sweep silently restricts itself to
    only the widest zones (a selection effect), making "shallower entry" and "only
    trade wide zones" indistinguishable."""
    n_bars = len(high)
    m = len(z_created)
    out_outcome = np.zeros(m, dtype=np.int8)
    out_net_r = np.zeros(m, dtype=np.float64)

    for zi in range(m):
        top, bottom, is_buy = z_top[zi], z_bottom[zi], z_is_buy[zi]
        height = top - bottom
        if height < min_height_price:
            continue
        entry_level = (top - frac * height) if is_buy else (bottom + frac * height)
        trade_is_buy = (not is_buy) if invert else is_buy
        stop_price = bottom if trade_is_buy else top
        target_price = top if trade_is_buy else bottom
        # nominal (pre-spread) risk at this entry depth -- NOT a fixed height/2, since
        # frac != 0.5 makes entry-to-stop shorter than entry-to-target (or vice versa)
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
        if is_buy:
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
                outcome = 2; break
            elif hit_tp:
                outcome = 1; break
        if outcome == 0:
            continue
        out_outcome[zi] = outcome
        out_net_r[zi] = win_r if outcome == 1 else -1.0

    return out_outcome, out_net_r


@njit(cache=True)
def _resolve_breakout_fade(z_created, z_top, z_bottom, z_is_buy, high, low, max_window,
                            min_stop_price, spread_price, use_zone_level_target):
    """A zone's far edge (the classical invalidation point -- bottom for a buy-context
    zone, top for a sell-context zone) getting taken out becomes a fresh entry in the
    SAME direction as the zone's own context (buy-zone break -> buy; sell-zone break ->
    sell) -- a bet that the breakout itself reverses. Stop is one zone-height beyond the
    entry; target is either a plain 1:1 (use_zone_level_target=False) or the zone's own
    opposite edge (True) -- these differ only by how much the entry fill slipped past
    the exact edge."""
    n_bars = len(high)
    m = len(z_created)
    out_outcome = np.zeros(m, dtype=np.int8)
    out_net_r = np.zeros(m, dtype=np.float64)

    for zi in range(m):
        top, bottom, is_buy = z_top[zi], z_bottom[zi], z_is_buy[zi]
        height = top - bottom
        if height / 2.0 < min_stop_price:
            continue
        c = z_created[zi]
        end = c + max_window
        if end > n_bars:
            end = n_bars

        break_level = bottom if is_buy else top
        touch_idx = -1
        for j in range(c + 1, end):
            touched = (low[j] <= break_level) if is_buy else (high[j] >= break_level)
            if touched:
                touch_idx = j
                break
        if touch_idx == -1:
            continue

        if is_buy:
            deep = low[touch_idx] <= break_level - spread_price
            entry_price = break_level if deep else break_level + spread_price
            stop_price = entry_price - height
            target_price = top if use_zone_level_target else entry_price + height
        else:
            deep = high[touch_idx] >= break_level + spread_price
            entry_price = break_level if deep else break_level - spread_price
            stop_price = entry_price + height
            target_price = bottom if use_zone_level_target else entry_price - height

        eff_risk = abs(entry_price - stop_price)
        if eff_risk <= 0:
            continue

        outcome, exit_idx = 0, -1
        for k in range(touch_idx, end):
            if is_buy:
                hit_sl = low[k] <= stop_price
                hit_tp = high[k] >= target_price
            else:
                hit_sl = high[k] >= stop_price
                hit_tp = low[k] <= target_price
            if hit_sl:
                outcome = 2; break
            elif hit_tp:
                outcome = 1; break
        if outcome == 0:
            continue
        win_r = abs(target_price - entry_price) / eff_risk
        out_outcome[zi] = outcome
        out_net_r[zi] = win_r if outcome == 1 else -1.0

    return out_outcome, out_net_r


def _zones_for(inst):
    times, lows, highs, df = rs.load_instrument(inst)
    high_ = df["high"].values.astype(np.float64)
    low_ = df["low"].values.astype(np.float64)
    confirm_idx, pivot_idx, is_high, price = g77._zigzag_pivots(high_, low_)
    z_created, z_top, z_bottom, z_is_buy, z_is_flip = g77._build_zones(confirm_idx, is_high, price)
    pip = pg.PIP_SIZE[inst]
    spread_price = pg.SPREAD_PIPS_TYPICAL * pip
    return z_created, z_top, z_bottom, z_is_buy, high_, low_, pip, spread_price, times


def summarize(outcome, net_r, times, touch_proxy_created_idx):
    resolved = outcome > 0
    n = int(resolved.sum())
    if n < 20:
        return dict(n=n, wr=None, pf=None, freq=None)
    r = net_r[resolved]
    wr = 100.0 * (r > 0).mean()
    gw, gl = r[r > 0].sum(), -r[r <= 0].sum()
    pf = gw / gl if gl > 0 else float("inf")
    return dict(n=n, wr=wr, pf=pf, avg_win=r[r > 0].mean() if (r > 0).any() else float("nan"))


def experiment_a(min_height_pips=0.0, title_suffix=""):
    print("=" * 100)
    print(f"EXPERIMENT A -- entry depth sweep (inverted, full history, all 4 pairs){title_suffix}")
    print("=" * 100)
    fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    zone_cache = {inst: _zones_for(inst) for inst in INSTRUMENTS}

    rows = []
    for frac in fracs:
        agg_r = []
        per_symbol = {}
        for inst in INSTRUMENTS:
            z_created, z_top, z_bottom, z_is_buy, high_, low_, pip, spread_price, times = zone_cache[inst]
            min_height_price = min_height_pips * pip
            outcome, net_r = _resolve_frac(z_created, z_top, z_bottom, z_is_buy, high_, low_,
                                            MAX_WINDOW, MIN_STOP_PIPS * pip, min_height_price,
                                            spread_price, frac, True)
            resolved = outcome > 0
            r = net_r[resolved]
            agg_r.append(r)
            gw, gl = r[r > 0].sum(), -r[r <= 0].sum()
            per_symbol[inst] = dict(n=len(r), wr=100 * (r > 0).mean(), pf=gw / gl if gl > 0 else float("inf"))
        all_r = np.concatenate(agg_r)
        gw, gl = all_r[all_r > 0].sum(), -all_r[all_r <= 0].sum()
        wr = 100 * (all_r > 0).mean()
        pf = gw / gl if gl > 0 else float("inf")
        avg_win = all_r[all_r > 0].mean() if (all_r > 0).any() else float("nan")
        rows.append((frac, len(all_r), wr, pf, avg_win))
        print(f"\nfrac={frac:.1f}: n={len(all_r):,}  WR={wr:.2f}%  PF={pf:.3f}  avg_win={avg_win:.3f}R")
        for inst, s in per_symbol.items():
            print(f"    {inst}: n={s['n']:,} WR={s['wr']:.1f}% PF={s['pf']:.3f}")

    print(f"\n--- summary: portfolio PF by entry depth{title_suffix} ---")
    for frac, n, wr, pf, avg_win in rows:
        print(f"  {int(frac*100):>3d}% entry: n={n:>8,}  WR={wr:5.2f}%  PF={pf:.3f}  avg_win={avg_win:.3f}R")
    return rows, zone_cache


def experiment_a_robustness(zone_cache, fracs_to_check):
    """Basic split-history sanity check on the most promising fracs: first half vs
    second half of EACH symbol's own bar history individually (not just the aggregate) --
    a shallow-entry edge that only exists in one half would be a red flag, not a finding."""
    print("\n--- robustness: first half vs second half of each symbol's own history ---")
    for frac in fracs_to_check:
        halves = {"first half": [], "second half": []}
        for inst in INSTRUMENTS:
            z_created, z_top, z_bottom, z_is_buy, high_, low_, pip, spread_price, times = zone_cache[inst]
            n_bars = len(high_)
            mid_bar = n_bars // 2
            outcome, net_r = _resolve_frac(z_created, z_top, z_bottom, z_is_buy, high_, low_,
                                            MAX_WINDOW, MIN_STOP_PIPS * pip, 0.0, spread_price, frac, True)
            resolved = outcome > 0
            created_resolved = z_created[resolved]
            r = net_r[resolved]
            halves["first half"].append(r[created_resolved < mid_bar])
            halves["second half"].append(r[created_resolved >= mid_bar])
        print(f"  frac={frac:.1f}:", end=" ")
        for label, chunks in halves.items():
            rr = np.concatenate(chunks)
            gw, gl = rr[rr > 0].sum(), -rr[rr <= 0].sum()
            pf = gw / gl if gl > 0 else float("inf")
            wr = 100 * (rr > 0).mean()
            print(f"{label}: n={len(rr):,} WR={wr:.1f}% PF={pf:.3f}  |", end=" ")
        print()


def experiment_b():
    print("\n" + "=" * 100)
    print("EXPERIMENT B -- fade the breakout: zone's far edge taken out -> entry in the zone's own")
    print("direction, stop 1 zone-height beyond entry, TP = 1:1 vs TP = zone's opposite edge")
    print("=" * 100)
    zone_cache = {inst: _zones_for(inst) for inst in INSTRUMENTS}

    for mode, use_level in [("1:1 (relative to entry)", False), ("zone level (opposite edge)", True)]:
        agg_r = []
        per_symbol = {}
        for inst in INSTRUMENTS:
            z_created, z_top, z_bottom, z_is_buy, high_, low_, pip, spread_price, times = zone_cache[inst]
            outcome, net_r = _resolve_breakout_fade(z_created, z_top, z_bottom, z_is_buy, high_, low_,
                                                     MAX_WINDOW, MIN_STOP_PIPS * pip, spread_price, use_level)
            resolved = outcome > 0
            r = net_r[resolved]
            agg_r.append(r)
            gw, gl = r[r > 0].sum(), -r[r <= 0].sum()
            per_symbol[inst] = dict(n=len(r), wr=100 * (r > 0).mean(), pf=gw / gl if gl > 0 else float("inf"))
        all_r = np.concatenate(agg_r)
        gw, gl = all_r[all_r > 0].sum(), -all_r[all_r <= 0].sum()
        wr = 100 * (all_r > 0).mean()
        pf = gw / gl if gl > 0 else float("inf")
        print(f"\n[{mode}]  n={len(all_r):,}  WR={wr:.2f}%  PF={pf:.3f}")
        for inst, s in per_symbol.items():
            print(f"    {inst}: n={s['n']:,} WR={s['wr']:.1f}% PF={s['pf']:.3f}")


if __name__ == "__main__":
    t0 = time.time()
    rows, zone_cache = experiment_a(min_height_pips=0.0, title_suffix=" [natural population per frac]")
    experiment_a_robustness(zone_cache, [0.1, 0.2, 0.3, 0.4])
    print()
    experiment_a(min_height_pips=30.0, title_suffix=" [CONTROL: fixed population, height >= 30 pips for every frac]")
    experiment_b()
    print(f"\n(total: {time.time()-t0:.0f}s)")
