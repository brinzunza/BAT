"""The canonical backtest of the base strategy defined in `../dtfx.md`, run against the
FULL EURUSD 1-minute history (2009-09-25 through 2026-09-03, ~6.34M bars).

Reuses goal95's swing/zone construction unchanged. Implements the two target conventions
from dtfx.md #7 on top of goal98's overshoot-then-tag entry gate (#6), with realistic
entry-side spread (1.6 pips, depth-dependent, #8) and a 3-pip minimum stop:

  (A) fixed 1:1R          -- target = the zone's own far edge (top for buy, bottom for sell)
  (B) trailing last-confirmed-extreme -- target = max(top, live last-confirmed-high) for a
      buy / min(bottom, live last-confirmed-low) for a sell, re-read every bar the trade is
      open, so it can only ever match or beat (A), never be worse.

A practical MAX_WINDOW (1440 bars = 1 trading day) bounds the overshoot->tag->exit scan so
a handful of pathological zones can't force a scan to the end of a 6.34M-bar series; zones
that don't fully resolve within that window are left out of the trade list entirely (same
convention as `_resolve_frac_ex` elsewhere in this project).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal95_every_swing_zones as g95

PIP = 1e-4                 # EURUSD
SPREAD_PIPS = 1.6
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440           # 1 trading day of 1-minute bars


@njit(cache=True)
def build_last_confirmed(confirm_idx, is_high, price, n_bars):
    """Forward-filled 'most recently confirmed swing high/low as of bar i' -- the plain
    running value from goal95's build_zones_every_swing (last_high_price/last_low_price),
    exposed per-bar so a trade held open can re-read it live. NOT a running max -- it can
    go up or down as swings alternate, exactly matching dtfx.md #4."""
    last_high = np.full(n_bars, np.nan)
    last_low = np.full(n_bars, np.nan)
    cur_high = 0.0; cur_low = 0.0
    has_high = False; has_low = False
    pi = 0
    m = len(confirm_idx)
    for i in range(n_bars):
        while pi < m and confirm_idx[pi] <= i:
            if is_high[pi]:
                cur_high = price[pi]; has_high = True
            else:
                cur_low = price[pi]; has_low = True
            pi += 1
        if has_high:
            last_high[i] = cur_high
        if has_low:
            last_low[i] = cur_low
    return last_high, last_low


@njit(cache=True)
def build_confirmed_hh_ll(confirm_idx, is_high, price, n_bars):
    """Unlike build_last_confirmed (any high/low pivot, unconditional), this only updates
    on a GENUINE higher-high (a high pivot that exceeds the immediately preceding high
    pivot -- the same 'p > last_high_price' condition build_zones_every_swing uses to
    decide trend==1) or genuine lower-low (symmetric). Holds its value through any
    subsequent lower-high/higher-low or trend flip, until the next genuine HH/LL. This is
    what dtfx.md's user-specified variant C target means by 'last confirmed higher high' /
    'last confirmed lower low' -- not just any confirmed swing of that type."""
    last_hh = np.full(n_bars, np.nan)
    last_ll = np.full(n_bars, np.nan)
    cur_hh = 0.0; cur_ll = 0.0
    has_hh = False; has_ll = False
    prev_high = 0.0; has_prev_high = False
    prev_low = 0.0; has_prev_low = False
    pi = 0
    m = len(confirm_idx)
    for i in range(n_bars):
        while pi < m and confirm_idx[pi] <= i:
            p = price[pi]
            if is_high[pi]:
                if has_prev_high and p > prev_high:
                    cur_hh = p; has_hh = True
                prev_high = p; has_prev_high = True
            else:
                if has_prev_low and p < prev_low:
                    cur_ll = p; has_ll = True
                prev_low = p; has_prev_low = True
            pi += 1
        if has_hh:
            last_hh[i] = cur_hh
        if has_ll:
            last_ll[i] = cur_ll
    return last_hh, last_ll


@njit(cache=True)
def resolve_fixed_structural_target(z_created, z_top, z_bottom, z_is_buy, high, low,
                                     last_hh_of_bar, last_ll_of_bar,
                                     spread_price, min_stop_price, max_window,
                                     min_zone_height):
    """Variant C: same overshoot-then-tag entry gate and same stop as resolve_base_strategy,
    but the target is the last CONFIRMED HIGHER HIGH (buy) / LOWER LOW (sell) as of the
    entry bar specifically -- read once at entry and then held fixed for the trade's
    duration, not re-read bar by bar. Fixing it avoids the failure mode of variant B (a
    target that keeps sliding further away for as long as the trend keeps extending,
    which it structurally cannot chase down before pulling back to the stop instead).

    min_zone_height (price units, e.g. min_pips*pip): a RAW zone-height floor, checked
    before any spread adjustment -- distinct from min_stop_price, which floors the
    post-spread effective risk. Defaults to 0.0 (no additional filter beyond min_stop_price),
    preserving prior behavior; set >0 for the min-zone-size robustness sweep."""
    m = len(z_created)
    n_bars = len(high)
    out_entry_idx = np.full(m, -1, dtype=np.int64)
    out_exit_idx = np.full(m, -1, dtype=np.int64)
    out_outcome = np.zeros(m, dtype=np.int8)
    out_net_r = np.zeros(m, dtype=np.float64)
    out_rr = np.zeros(m, dtype=np.float64)   # the R:R actually on offer for this trade
    out_entry_price = np.zeros(m, dtype=np.float64)
    out_eff_risk = np.zeros(m, dtype=np.float64)

    for zi in range(m):
        top = z_top[zi]; bottom = z_bottom[zi]; is_buy = z_is_buy[zi]
        height = top - bottom
        if height <= 0 or height < min_zone_height:
            continue
        mid = (top + bottom) / 2.0
        c = z_created[zi]
        end = c + max_window
        if end > n_bars:
            end = n_bars

        overshoot_idx = -1
        if is_buy:
            if high[c] > top:
                overshoot_idx = c
        else:
            if low[c] < bottom:
                overshoot_idx = c
        if overshoot_idx == -1:
            for j in range(c + 1, end):
                if is_buy:
                    if high[j] > top:
                        overshoot_idx = j; break
                else:
                    if low[j] < bottom:
                        overshoot_idx = j; break
        if overshoot_idx == -1:
            continue

        tag_idx = -1
        for j in range(overshoot_idx, end):
            if is_buy:
                if low[j] <= mid:
                    tag_idx = j; break
            else:
                if high[j] >= mid:
                    tag_idx = j; break
        if tag_idx == -1:
            continue

        if is_buy:
            deep = low[tag_idx] <= mid - spread_price
            entry_price = mid if deep else mid + spread_price
        else:
            deep = high[tag_idx] >= mid + spread_price
            entry_price = mid if deep else mid - spread_price

        stop_price = bottom if is_buy else top
        eff_risk = abs(entry_price - stop_price)
        if eff_risk < min_stop_price:
            continue

        # target fixed once, at entry
        if is_buy:
            live = last_hh_of_bar[tag_idx]
            target_price = top if (np.isnan(live) or live < top) else live
        else:
            live = last_ll_of_bar[tag_idx]
            target_price = bottom if (np.isnan(live) or live > bottom) else live
        rr_on_offer = abs(target_price - entry_price) / eff_risk

        exit_idx = -1; outcome = 0; net_r = 0.0
        for j in range(tag_idx, end):
            if is_buy:
                hit_stop = low[j] < stop_price
                hit_target = high[j] > target_price
            else:
                hit_stop = high[j] > stop_price
                hit_target = low[j] < target_price
            if hit_stop:
                exit_idx = j; outcome = 2; net_r = -1.0
                break
            elif hit_target:
                exit_idx = j; outcome = 1; net_r = rr_on_offer
                break
        if exit_idx == -1:
            continue

        out_entry_idx[zi] = tag_idx
        out_exit_idx[zi] = exit_idx
        out_outcome[zi] = outcome
        out_net_r[zi] = net_r
        out_rr[zi] = rr_on_offer
        out_entry_price[zi] = entry_price
        out_eff_risk[zi] = eff_risk

    return out_entry_idx, out_exit_idx, out_outcome, out_net_r, out_rr, out_entry_price, out_eff_risk


@njit(cache=True)
def resolve_base_strategy(z_created, z_top, z_bottom, z_is_buy, high, low,
                           last_high_of_bar, last_low_of_bar,
                           spread_price, min_stop_price, max_window, use_dynamic_target,
                           min_zone_height, symmetric_exit_spread=False):
    """dtfx.md #6 (overshoot -> tag gate) + #7 (stop always the opposite edge; target is
    either the fixed far edge (use_dynamic_target=False) or max/min'd against the live
    last-confirmed-extreme (True)) + #8 (entry-side spread only, depth-dependent).

    min_zone_height (price units): a RAW zone-height floor checked before any spread
    adjustment, distinct from min_stop_price (which floors post-spread effective risk).
    Pass 0.0 to disable (matches prior behavior).

    symmetric_exit_spread (direct instruction, 2026-09-07): matches PRODLIVETRADING's
    marketable-limit order convention (bot.py::_place_trade) -- rather than the original
    "0 or a full spread, depending on how deep the touch went" entry-only cost model below,
    every price is offset by half the spread, in the direction that costs the position: entry
    mid+/-spread/2, and BOTH exits (stop and target) also worse by spread/2. Verified
    algebraically to make a full round trip cost exactly one spread total (half on entry,
    half on exit), split the way real marketable limit orders on both legs actually would,
    rather than charging the whole thing on entry and nothing on exit. Default is False so
    every existing call site (and every number already published in dtfx.md) is completely
    unaffected -- this only activates where a caller explicitly opts in."""
    m = len(z_created)
    n_bars = len(high)
    out_entry_idx = np.full(m, -1, dtype=np.int64)
    out_exit_idx = np.full(m, -1, dtype=np.int64)
    out_outcome = np.zeros(m, dtype=np.int8)      # 0 = excluded/unresolved, 1 = win, 2 = loss
    out_net_r = np.zeros(m, dtype=np.float64)
    out_entry_price = np.zeros(m, dtype=np.float64)

    for zi in range(m):
        top = z_top[zi]; bottom = z_bottom[zi]; is_buy = z_is_buy[zi]
        height = top - bottom
        if height <= 0 or height < min_zone_height:
            continue
        mid = (top + bottom) / 2.0
        c = z_created[zi]
        end = c + max_window
        if end > n_bars:
            end = n_bars

        # stage 1: overshoot (dtfx.md #6.1)
        overshoot_idx = -1
        if is_buy:
            if high[c] > top:
                overshoot_idx = c
        else:
            if low[c] < bottom:
                overshoot_idx = c
        if overshoot_idx == -1:
            for j in range(c + 1, end):
                if is_buy:
                    if high[j] > top:
                        overshoot_idx = j; break
                else:
                    if low[j] < bottom:
                        overshoot_idx = j; break
        if overshoot_idx == -1:
            continue

        # stage 2: tag (dtfx.md #6.2), with realistic entry-side fill (#8)
        tag_idx = -1
        for j in range(overshoot_idx, end):
            if is_buy:
                if low[j] <= mid:
                    tag_idx = j; break
            else:
                if high[j] >= mid:
                    tag_idx = j; break
        if tag_idx == -1:
            continue

        half_spread = spread_price / 2.0
        if symmetric_exit_spread:
            entry_price = mid + half_spread if is_buy else mid - half_spread
        else:
            if is_buy:
                deep = low[tag_idx] <= mid - spread_price
                entry_price = mid if deep else mid + spread_price
            else:
                deep = high[tag_idx] >= mid + spread_price
                entry_price = mid if deep else mid - spread_price

        raw_stop_price = bottom if is_buy else top
        if not symmetric_exit_spread:
            stop_price = raw_stop_price
        elif is_buy:
            stop_price = raw_stop_price - half_spread
        else:
            stop_price = raw_stop_price + half_spread
        eff_risk = abs(entry_price - stop_price)
        if eff_risk < min_stop_price:
            continue

        # stage 3: exit (dtfx.md #7)
        exit_idx = -1; outcome = 0; net_r = 0.0
        for j in range(tag_idx, end):
            if use_dynamic_target:
                if is_buy:
                    live = last_high_of_bar[j]
                    raw_target = top if (np.isnan(live) or live < top) else live
                else:
                    live = last_low_of_bar[j]
                    raw_target = bottom if (np.isnan(live) or live > bottom) else live
            else:
                raw_target = top if is_buy else bottom
            if symmetric_exit_spread:
                target_price = raw_target - half_spread if is_buy else raw_target + half_spread
            else:
                target_price = raw_target

            if is_buy:
                hit_stop = low[j] < stop_price
                hit_target = high[j] > target_price
            else:
                hit_stop = high[j] > stop_price
                hit_target = low[j] < target_price

            if hit_stop:
                exit_idx = j; outcome = 2; net_r = -1.0
                break
            elif hit_target:
                exit_idx = j; outcome = 1
                net_r = abs(target_price - entry_price) / eff_risk
                break
        if exit_idx == -1:
            continue

        out_entry_idx[zi] = tag_idx
        out_exit_idx[zi] = exit_idx
        out_outcome[zi] = outcome
        out_net_r[zi] = net_r
        out_entry_price[zi] = entry_price

    return out_entry_idx, out_exit_idx, out_outcome, out_net_r, out_entry_price


def summarize(label, times, outcome, net_r, entry_idx):
    resolved = outcome > 0
    n = int(resolved.sum())
    if n == 0:
        print(f"{label}: no resolved trades")
        return None
    r = net_r[resolved]
    wins = (outcome[resolved] == 1)
    wr = 100.0 * wins.sum() / n
    gross_win = r[r > 0].sum()
    gross_loss = -r[r <= 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    avg_r = r.mean()
    total_r = r.sum()
    years = (times.iloc[-1] - times.iloc[0]).days / 365.25
    tpy = n / years if years > 0 else float("nan")
    print(f"\n=== {label} ===")
    print(f"  trades resolved:     {n:,}  ({tpy:,.0f}/yr over {years:.1f}yr)")
    print(f"  win rate:            {wr:.2f}%")
    print(f"  avg R per trade:     {avg_r:+.4f}")
    print(f"  total R (sum):       {total_r:+,.1f}")
    print(f"  profit factor:       {pf:.3f}")
    return dict(n=n, wr=wr, avg_r=avg_r, total_r=total_r, pf=pf, tpy=tpy, years=years)


def main():
    t0 = time.time()
    path = os.path.join(DATA_DIR, "lse_eurusd_1min_full.csv")
    print(f"Loading {path} ...")
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded {len(df):,} bars, {df['date'].iloc[0]} .. {df['date'].iloc[-1]}  ({time.time()-t0:.1f}s)")

    times = df["date"]
    high_ = df["high"].values.astype(np.float64)
    low_ = df["low"].values.astype(np.float64)
    close_ = df["close"].values.astype(np.float64)
    n_bars = len(df)

    t1 = time.time()
    confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high_, low_, close_)
    print(f"Swings: {len(confirm_idx):,} ({100*len(confirm_idx)/n_bars:.1f}% of bars)  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
     ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)
    print(f"Zones: {len(z_created):,}  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    last_high_of_bar, last_low_of_bar = build_last_confirmed(confirm_idx, is_high, price, n_bars)
    print(f"Running last-confirmed extremes built  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    last_hh_of_bar, last_ll_of_bar = build_confirmed_hh_ll(confirm_idx, is_high, price, n_bars)
    print(f"Running confirmed-HH/LL (genuine higher-high / lower-low only) built  ({time.time()-t1:.1f}s)")

    spread_price = SPREAD_PIPS * PIP
    min_stop_price = MIN_STOP_PIPS * PIP

    t1 = time.time()
    entry_idx_a, exit_idx_a, outcome_a, net_r_a, entry_price_a = resolve_base_strategy(
        z_created, z_top, z_bottom, z_is_buy, high_, low_,
        last_high_of_bar, last_low_of_bar, spread_price, min_stop_price, MAX_WINDOW, False, 0.0)
    print(f"Variant A (fixed 1:1R) resolved  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    entry_idx_b, exit_idx_b, outcome_b, net_r_b, entry_price_b = resolve_base_strategy(
        z_created, z_top, z_bottom, z_is_buy, high_, low_,
        last_high_of_bar, last_low_of_bar, spread_price, min_stop_price, MAX_WINDOW, True, 0.0)
    print(f"Variant B (trailing last-confirmed extreme) resolved  ({time.time()-t1:.1f}s)")

    t1 = time.time()
    entry_idx_c, exit_idx_c, outcome_c, net_r_c, rr_c, entry_price_c, eff_risk_c = resolve_fixed_structural_target(
        z_created, z_top, z_bottom, z_is_buy, high_, low_,
        last_hh_of_bar, last_ll_of_bar, spread_price, min_stop_price, MAX_WINDOW, 0.0)
    print(f"Variant C (fixed, last confirmed HH/LL at entry) resolved  ({time.time()-t1:.1f}s)")

    stats_a = summarize("Variant A -- fixed 1:1R", times, outcome_a, net_r_a, entry_idx_a)
    stats_b = summarize("Variant B -- trailing last-confirmed extreme", times, outcome_b, net_r_b, entry_idx_b)
    stats_c = summarize("Variant C -- fixed last confirmed HH/LL at entry", times, outcome_c, net_r_c, entry_idx_c)
    resolved_c = outcome_c > 0
    if resolved_c.sum() > 0:
        print(f"  avg R:R on offer:    {rr_c[resolved_c].mean():.3f}   (median {np.median(rr_c[resolved_c]):.3f})")

    # per-year breakdown for both, for a robustness look
    def per_year(label, outcome, net_r, entry_idx):
        resolved = outcome > 0
        idx = entry_idx[resolved]
        yrs = times.iloc[idx].dt.year.values
        r = net_r[resolved]
        out = pd.DataFrame({"year": yrs, "r": r, "win": (outcome[resolved] == 1)})
        g = out.groupby("year").agg(n=("r", "size"), wr=("win", "mean"), total_r=("r", "sum"))
        g["wr"] = (g["wr"] * 100).round(1)
        g["total_r"] = g["total_r"].round(1)
        print(f"\n--- {label}: per-year ---")
        print(g.to_string())
        return g

    yearly_a = per_year("Variant A", outcome_a, net_r_a, entry_idx_a)
    yearly_b = per_year("Variant B", outcome_b, net_r_b, entry_idx_b)
    yearly_c = per_year("Variant C", outcome_c, net_r_c, entry_idx_c)

    yearly_a.to_csv(os.path.join(DATA_DIR, "goal99_variant_a_yearly.csv"))
    yearly_b.to_csv(os.path.join(DATA_DIR, "goal99_variant_b_yearly.csv"))
    yearly_c.to_csv(os.path.join(DATA_DIR, "goal99_variant_c_yearly.csv"))

    with open(os.path.join(DATA_DIR, "goal99_dtfx_base_backtest_summary.txt"), "w") as f:
        f.write(f"EURUSD full history ({times.iloc[0]} .. {times.iloc[-1]}), {n_bars:,} bars\n")
        f.write(f"Swings: {len(confirm_idx):,}   Zones: {len(z_created):,}\n\n")
        for label, s in [("Variant A -- fixed 1:1R", stats_a),
                          ("Variant B -- trailing last-confirmed extreme", stats_b),
                          ("Variant C -- fixed last confirmed HH/LL at entry", stats_c)]:
            f.write(f"{label}\n")
            if s is None:
                f.write("  no resolved trades\n\n")
                continue
            f.write(f"  trades resolved: {s['n']:,} ({s['tpy']:,.0f}/yr over {s['years']:.1f}yr)\n")
            f.write(f"  win rate: {s['wr']:.2f}%\n")
            f.write(f"  avg R per trade: {s['avg_r']:+.4f}\n")
            f.write(f"  total R (sum): {s['total_r']:+,.1f}\n")
            f.write(f"  profit factor: {s['pf']:.3f}\n\n")
        if resolved_c.sum() > 0:
            f.write(f"Variant C avg R:R on offer: {rr_c[resolved_c].mean():.3f} (median {np.median(rr_c[resolved_c]):.3f})\n")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
