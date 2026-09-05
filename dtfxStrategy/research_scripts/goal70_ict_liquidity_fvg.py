"""/goal (2026-09-04): "dtfx in general is based on trend following (higher highs higher
lows, lower highs lower lows) and instituational trading (institutions leave large orders
behind since they cannot be filled immediately. But it doesn't take liquidity sweeps,
momentum, and fair value gap (imbalance). backtest with these ideas until you find a
winning strategy (ideally stay away from R:R that is too low)."

New signal construction, not a re-tuning of the existing DTFX zone-fade engine -- this
project's zone-fade thesis (large resting institutional orders creating a support/resistance
zone) already IS a trend-structure concept (PRODLIVETRADING/strategy/structure_fsm.py's own
confirmed-high/confirmed-low FSM is literally a formalized HH/HL, LH/LL tracker), but it has
never combined that with the specific ICT/SMC constructs named here:

- **Liquidity sweep**: a bar's WICK exceeds a recent swing high/low (triggering the resting
  stop-orders assumed to cluster just beyond it) but its CLOSE snaps back inside -- a false
  breakout / stop-hunt, structurally different from DTFX's own swing confirmation (which
  requires a CLOSE beyond the level, not a wick-then-reject).
- **Fair Value Gap (FVG / imbalance)**: the classic 3-candle inefficiency -- a bullish FVG
  at bar i exists when low[i+1] > high[i-1] (price never traded through that band), a
  bearish FVG when high[i+1] < low[i-1]. Marks a genuinely fast, one-sided impulse.
- **Momentum**: the size of the FVG itself (in ATR-normalized terms) and the impulse
  candle's own body-to-range ratio, as a filter for "was this a real institutional push or
  just noise."

Strategy: after a liquidity sweep of a recent swing point, look for a same-direction FVG
forming within the next few bars (continuation confirmation) -- entry on price returning to
that FVG zone, stop beyond the sweep's own extreme, target at a real R:R (not the <=0.3
extreme this session's meta-labeling era converged on) -- explicitly tested at R:R 1.0-3.0
per this goal's own instruction to stay away from too-low R:R.
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import prod_filter_grid_search as pg

rs = g29.rs


def detect_fvg(high, low):
    """Bullish FVG at i: low[i+1] > high[i-1] -> gap [high[i-1], low[i+1]].
    Bearish FVG at i: high[i+1] < low[i-1] -> gap [high[i+1], low[i-1]].
    Returns per-bar (is_bull_fvg, bull_gap_lo, bull_gap_hi, is_bear_fvg, bear_gap_lo, bear_gap_hi),
    indexed at the MIDDLE bar i of the 3-candle pattern (the impulse candle)."""
    n = len(high)
    is_bull = np.zeros(n, dtype=bool)
    is_bear = np.zeros(n, dtype=bool)
    bull_lo = np.full(n, np.nan)
    bull_hi = np.full(n, np.nan)
    bear_lo = np.full(n, np.nan)
    bear_hi = np.full(n, np.nan)

    h_im1, l_ip1 = high[:-2], low[2:]
    bull_mask = l_ip1 > h_im1
    is_bull[1:-1] = bull_mask
    bull_lo[1:-1] = np.where(bull_mask, h_im1, np.nan)
    bull_hi[1:-1] = np.where(bull_mask, l_ip1, np.nan)

    l_im1, h_ip1 = low[:-2], high[2:]
    bear_mask = h_ip1 < l_im1
    is_bear[1:-1] = bear_mask
    bear_lo[1:-1] = np.where(bear_mask, h_ip1, np.nan)
    bear_hi[1:-1] = np.where(bear_mask, l_im1, np.nan)

    return is_bull, bull_lo, bull_hi, is_bear, bear_lo, bear_hi


def detect_swings(high, low, lookback=15):
    """A bar i is a confirmed swing high if its high is the max over [i-lookback, i+lookback]
    (confirmed only once the `lookback` bars AFTER it are known -- causal once confirmed,
    zero lookahead at the time it becomes usable). Symmetric for swing lows. Vectorized via
    pandas rolling max/min (centered window) -- a Python per-bar loop over millions of 1-min
    bars would take far too long."""
    h = pd.Series(high)
    l = pd.Series(low)
    window = 2 * lookback + 1
    roll_max = h.rolling(window, center=True, min_periods=window).max()
    roll_min = l.rolling(window, center=True, min_periods=window).min()
    is_swing_high = (h == roll_max).values & ~roll_max.isna().values
    is_swing_low = (l == roll_min).values & ~roll_min.isna().values
    return is_swing_high, is_swing_low


def confirmed_swing_levels(is_swing, price, lookback, n):
    """Forward-filled 'most recent CONFIRMED swing price known as of bar i' -- a swing at
    index j is only knowable starting at j+lookback (once the centered rolling window that
    validated it has fully passed), so the swing's own effect is shifted forward by
    `lookback` bars before forward-filling. Zero lookahead: at any bar i, only swings whose
    confirmation index <= i are visible."""
    confirmed_at = np.where(is_swing)[0] + lookback
    values = price[is_swing]
    valid = confirmed_at < n
    confirmed_at, values = confirmed_at[valid], values[valid]
    out = np.full(n, np.nan)
    out[confirmed_at] = values
    return pd.Series(out).ffill().values


def equal_level_clusters(is_swing, price, lookback, n, eq_tol_pips, pip, max_lookback_swings=20):
    """Equal-highs/equal-lows: a confirmed swing is part of a liquidity-pool CLUSTER when
    it sits within `eq_tol_pips` of an earlier swing (checked against the last
    `max_lookback_swings` prior swings, not the whole history -- an ICT equal-high/low pool
    is a LOCAL phenomenon, not a claim that price never revisited that level in 10 years).
    Returns a forward-filled 'most recent confirmed equal-level price known as of bar i'
    series that only updates on an actual cluster (rarer, and by construction a stronger
    liquidity concentration than any single isolated swing)."""
    swing_idx = np.where(is_swing)[0]
    swing_val = price[swing_idx]
    confirmed_at = swing_idx + lookback
    valid = confirmed_at < n
    confirmed_at, swing_val = confirmed_at[valid], swing_val[valid]

    cluster_confirmed_at = []
    cluster_price = []
    tol = eq_tol_pips * pip
    for i in range(1, len(swing_val)):
        lo = max(0, i - max_lookback_swings)
        prior = swing_val[lo:i]
        diffs = np.abs(prior - swing_val[i])
        j = np.argmin(diffs)
        if diffs[j] <= tol:
            cluster_confirmed_at.append(confirmed_at[i])
            cluster_price.append((prior[j] + swing_val[i]) / 2.0)

    out = np.full(n, np.nan)
    if cluster_confirmed_at:
        out[np.array(cluster_confirmed_at)] = np.array(cluster_price)
    return pd.Series(out).ffill().values


def detect_liquidity_sweeps(open_, high, low, close, swing_lookback=15):
    """Bullish sweep at bar i: low[i] wicks below the most recent CONFIRMED swing low (the
    resting sell-stop cluster assumed to sit just beyond it) but close[i] snaps back above
    it -- a false breakdown / stop-hunt, structurally different from DTFX's own CLOSE-based
    swing confirmation. Bearish sweep is the mirror case against the most recent confirmed
    swing high."""
    n = len(close)
    is_swing_high, is_swing_low = detect_swings(high, low, swing_lookback)
    swing_low_level = confirmed_swing_levels(is_swing_low, low, swing_lookback, n)
    swing_high_level = confirmed_swing_levels(is_swing_high, high, swing_lookback, n)

    prior_swing_low = np.roll(swing_low_level, 1)
    prior_swing_high = np.roll(swing_high_level, 1)
    prior_swing_low[0] = np.nan
    prior_swing_high[0] = np.nan

    bullish_sweep = (low < prior_swing_low) & (close > prior_swing_low)
    bearish_sweep = (high > prior_swing_high) & (close < prior_swing_high)

    # De-duplicate: price can sit below/above the SAME forward-filled swing level for
    # several consecutive bars, each independently satisfying the sweep condition -- keep
    # only the FIRST bar of each such streak (the actual stop-hunt moment), not every bar
    # of what's really one ongoing event.
    bullish_sweep = bullish_sweep & ~np.roll(bullish_sweep, 1)
    bearish_sweep = bearish_sweep & ~np.roll(bearish_sweep, 1)
    bullish_sweep[0] = False
    bearish_sweep[0] = False
    return bullish_sweep, bearish_sweep, swing_low_level, swing_high_level


def detect_equal_level_sweeps(high, low, close, swing_lookback, eq_tol_pips, pip,
                               max_lookback_swings=20):
    """Same sweep mechanic as detect_liquidity_sweeps, but against equal-highs/equal-lows
    liquidity POOLS (2+ swings clustering within eq_tol_pips of each other) instead of any
    single isolated swing -- a stronger, more genuinely institutional liquidity
    concentration, and by construction a rarer, more selective sweep."""
    n = len(close)
    is_swing_high, is_swing_low = detect_swings(high, low, swing_lookback)
    eql_level = equal_level_clusters(is_swing_low, low, swing_lookback, n, eq_tol_pips, pip,
                                      max_lookback_swings)
    eqh_level = equal_level_clusters(is_swing_high, high, swing_lookback, n, eq_tol_pips, pip,
                                      max_lookback_swings)

    prior_eql = np.roll(eql_level, 1)
    prior_eqh = np.roll(eqh_level, 1)
    prior_eql[0] = np.nan
    prior_eqh[0] = np.nan

    bullish_sweep = (low < prior_eql) & (close > prior_eql)
    bearish_sweep = (high > prior_eqh) & (close < prior_eqh)
    bullish_sweep = bullish_sweep & ~np.roll(bullish_sweep, 1)
    bearish_sweep = bearish_sweep & ~np.roll(bearish_sweep, 1)
    bullish_sweep[0] = False
    bearish_sweep[0] = False
    return bullish_sweep, bearish_sweep, eql_level, eqh_level


def trend_state(high, low, swing_lookback=60):
    """Continuous HH/HL, LH/LL trend state (mirrors structure_fsm's own confirmed-high/
    confirmed-low tracker, built independently here so it can be looked up at any bar, not
    just at zone-creation events): +1 (up) once the last two confirmed swing highs AND the
    last two confirmed swing lows are both ascending, -1 (down) when both are descending,
    0 otherwise. Forward-filled from each swing's own confirmation bar -- zero lookahead."""
    n = len(high)
    is_swing_high, is_swing_low = detect_swings(high, low, swing_lookback)
    sh_conf = np.where(is_swing_high)[0] + swing_lookback
    sh_val = high[np.where(is_swing_high)[0]]
    valid = sh_conf < n
    sh_conf, sh_val = sh_conf[valid], sh_val[valid]
    sl_conf = np.where(is_swing_low)[0] + swing_lookback
    sl_val = low[np.where(is_swing_low)[0]]
    valid = sl_conf < n
    sl_conf, sl_val = sl_conf[valid], sl_val[valid]

    high_trend_up = np.full(len(sh_conf), np.nan)
    for i in range(1, len(sh_conf)):
        high_trend_up[i] = sh_val[i] > sh_val[i - 1]
    low_trend_up = np.full(len(sl_conf), np.nan)
    for i in range(1, len(sl_conf)):
        low_trend_up[i] = sl_val[i] > sl_val[i - 1]

    hti = np.full(n, np.nan)
    hti[sh_conf] = high_trend_up
    hti_ff = pd.Series(hti).ffill().values
    lti = np.full(n, np.nan)
    lti[sl_conf] = low_trend_up
    lti_ff = pd.Series(lti).ffill().values

    state = np.zeros(n, dtype=np.int8)
    state[(hti_ff == 1) & (lti_ff == 1)] = 1
    state[(hti_ff == 0) & (lti_ff == 0)] = -1
    return state


def build_ict_touches(inst, rr, swing_lookback=15, fvg_lookahead=10, stop_buffer_pips=2.0,
                       min_fvg_atr_ratio=None, atr_window=200, min_stop_pips=5.0,
                       require_trend_alignment=False, trend_swing_lookback=60,
                       target_mode="fixed_rr", min_liquidity_rr=1.0, max_liquidity_rr=6.0,
                       require_rejection_candle=False, sweep_mode="single_swing",
                       eq_tol_pips=3.0, eq_max_lookback_swings=20):
    """target_mode='liquidity': targets the nearest already-known OPPOSING swing level (the
    next liquidity pool a sweep-then-continuation move is classically expected to run to)
    instead of a fixed R:R multiple -- falls back to `rr` when that level implies an R:R
    outside [min_liquidity_rr, max_liquidity_rr] (too close to be worth it, or implausibly
    far to trust as a real target)."""
    """Core signal: after a liquidity sweep, look forward up to `fvg_lookahead` bars for a
    same-direction Fair Value Gap (momentum/institutional-push confirmation) -- entry when
    price returns to touch that FVG zone, stop beyond the sweep's own extreme (plus a small
    buffer), target at a fixed R:R multiple of the resulting risk. Applies this project's
    own realistic-fill cost model (spread charged on every trade, fill-depth requirement for
    a genuinely free entry) so results are directly comparable to every other candidate this
    session. `min_fvg_atr_ratio`, if given, filters to FVGs at least this many ATRs wide --
    the momentum/quality filter this goal explicitly asked for. `require_trend_alignment`
    keeps only bullish setups while the broader HH/HL structure is already up (and bearish
    while it's down) -- the explicit trend-following gate this goal's own wording asked for,
    on top of DTFX's own swing-confirmation logic that every sweep/FVG here builds on."""
    times, lows, highs, df = rs.load_instrument(inst)
    open_, high_, low_, close_ = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(close_)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    pip = pg.PIP_SIZE[inst]
    spread_price = pg.SPREAD_PIPS_TYPICAL * pip

    if sweep_mode == "equal_levels":
        bullish_sweep, bearish_sweep, swing_low_level, swing_high_level = detect_equal_level_sweeps(
            high_, low_, close_, swing_lookback, eq_tol_pips, pip, max_lookback_swings=eq_max_lookback_swings)
    else:
        bullish_sweep, bearish_sweep, swing_low_level, swing_high_level = detect_liquidity_sweeps(
            open_, high_, low_, close_, swing_lookback)
    is_bull_fvg, bull_lo, bull_hi, is_bear_fvg, bear_lo, bear_hi = detect_fvg(high_, low_)

    if require_trend_alignment:
        trend = trend_state(high_, low_, trend_swing_lookback)
        bullish_sweep = bullish_sweep & (trend == 1)
        bearish_sweep = bearish_sweep & (trend == -1)

    if min_fvg_atr_ratio is not None:
        tr = np.maximum(high_[1:] - low_[1:], np.abs(high_[1:] - close_[:-1]))
        tr = np.maximum(tr, np.abs(low_[1:] - close_[:-1]))
        atr = pd.Series(np.concatenate([[np.nan], tr])).rolling(atr_window).mean().values
        bull_gap_w = bull_hi - bull_lo
        bear_gap_w = bear_hi - bear_lo
        is_bull_fvg = is_bull_fvg & (bull_gap_w >= min_fvg_atr_ratio * atr)
        is_bear_fvg = is_bear_fvg & (bear_gap_w >= min_fvg_atr_ratio * atr)

    sweep_idx_bull = np.where(bullish_sweep)[0]
    sweep_idx_bear = np.where(bearish_sweep)[0]
    fvg_idx_bull = np.where(is_bull_fvg)[0]
    fvg_idx_bear = np.where(is_bear_fvg)[0]

    def build_side(sweep_idx, fvg_idx, gap_lo, gap_hi, sweep_extreme, opposing_level, side_is_buy):
        rows = []
        fj = 0
        for si in sweep_idx:
            while fj < len(fvg_idx) and fvg_idx[fj] <= si:
                fj += 1
            k = fj
            while k < len(fvg_idx) and fvg_idx[k] <= si + fvg_lookahead:
                fi = fvg_idx[k]
                mid = (gap_lo[fi] + gap_hi[fi]) / 2.0
                stop_extreme = sweep_extreme[si]
                if side_is_buy:
                    stop_price = stop_extreme - stop_buffer_pips * pip
                    stop_risk = mid - stop_price
                else:
                    stop_price = stop_extreme + stop_buffer_pips * pip
                    stop_risk = stop_price - mid
                if stop_risk <= 0 or stop_risk < min_stop_pips * pip:
                    k += 1
                    continue
                if target_mode == "liquidity":
                    opp = opposing_level[fi]
                    implied_rr = ((opp - mid) / stop_risk) if side_is_buy else ((mid - opp) / stop_risk)
                    use_rr = implied_rr if (not np.isnan(opp) and min_liquidity_rr <= implied_rr <= max_liquidity_rr) else rr
                    target_price = mid + stop_risk * use_rr if side_is_buy else mid - stop_risk * use_rr
                else:
                    target_price = mid + stop_risk * rr if side_is_buy else mid - stop_risk * rr
                entry_search_start = fi + 1
                max_window = 60 * 24 * 5
                end = min(entry_search_start + max_window, n)
                entered = False
                for j in range(entry_search_start, end):
                    if side_is_buy:
                        touched = low_[j] <= mid
                    else:
                        touched = high_[j] >= mid
                    if touched and require_rejection_candle:
                        bar_range = high_[j] - low_[j]
                        if bar_range <= 0:
                            touched = False
                        else:
                            close_pos = (close_[j] - low_[j]) / bar_range
                            touched = close_pos >= 0.5 if side_is_buy else close_pos <= 0.5
                    if touched:
                        deep_enough = (low_[j] <= mid - spread_price) if side_is_buy else (high_[j] >= mid + spread_price)
                        realistic_entry = mid if deep_enough else (mid + spread_price if side_is_buy else mid - spread_price)
                        eff_risk = abs(realistic_entry - stop_price)
                        for m in range(j, end):
                            hit_stop = (low_[m] <= stop_price) if side_is_buy else (high_[m] >= stop_price)
                            hit_target = (high_[m] >= target_price) if side_is_buy else (low_[m] <= target_price)
                            if hit_stop and hit_target:
                                is_win = False
                                exit_idx = m
                                break
                            elif hit_stop:
                                is_win = False
                                exit_idx = m
                                break
                            elif hit_target:
                                is_win = True
                                exit_idx = m
                                break
                        else:
                            entered = False
                            break
                        win_gross_r = (rr if deep_enough else (target_price - realistic_entry) / eff_risk) if side_is_buy else \
                                      (rr if deep_enough else (realistic_entry - target_price) / eff_risk)
                        r_gross = win_gross_r if is_win else -1.0
                        cost_in_r = 0.0 if not deep_enough else spread_price / eff_risk
                        net_r = r_gross - cost_in_r
                        rows.append(dict(instrument=inst, is_buy=side_is_buy, sweep_idx=si,
                                          fvg_idx=fi, entry_idx=j, exit_idx=exit_idx,
                                          entry_time=times[j], exit_time=times[exit_idx],
                                          entry_price=mid, stop_risk=stop_risk,
                                          is_win=is_win, net_r=net_r, shallow_touch=not deep_enough,
                                          fvg_width_pips=(gap_hi[fi] - gap_lo[fi]) / pip,
                                          bars_sweep_to_fvg=fi - si,
                                          bars_fvg_to_entry=j - fi,
                                          sweep_created_at=times[si], fvg_created_at=times[fi]))
                        entered = True
                        break
                k += 1
                if entered:
                    break
        return rows

    rows_buy = build_side(sweep_idx_bull, fvg_idx_bull, bull_lo, bull_hi, swing_low_level,
                           swing_high_level, True)
    rows_sell = build_side(sweep_idx_bear, fvg_idx_bear, bear_lo, bear_hi, swing_high_level,
                            swing_low_level, False)
    t = pd.DataFrame(rows_buy + rows_sell)
    if not t.empty:
        # A given FVG can be "claimed" by more than one preceding sweep within lookahead --
        # the resulting trade (same fvg_idx -> same mid/entry-search) is economically
        # identical regardless of which sweep flagged it tradeable, so keep only the first.
        t = t.drop_duplicates(subset=["is_buy", "fvg_idx"]).sort_values("entry_time").reset_index(drop=True)
    return t


if __name__ == "__main__":
    import time
    t0 = time.time()
    times, lows, highs, df = rs.load_instrument("EURUSD")
    print(f"loaded {len(df):,} bars in {time.time()-t0:.0f}s")

    open_, high_, low_, close_ = df["open"].values, df["high"].values, df["low"].values, df["close"].values

    t0 = time.time()
    is_bull_fvg, bull_lo, bull_hi, is_bear_fvg, bear_lo, bear_hi = detect_fvg(high_, low_)
    print(f"FVG detection: {is_bull_fvg.sum():,} bullish, {is_bear_fvg.sum():,} bearish "
          f"({time.time()-t0:.0f}s)")

    t0 = time.time()
    bullish_sweep, bearish_sweep, _, _ = detect_liquidity_sweeps(open_, high_, low_, close_)
    print(f"liquidity sweeps: {bullish_sweep.sum():,} bullish, {bearish_sweep.sum():,} bearish "
          f"({time.time()-t0:.0f}s)")
