"""Direct user request (2026-09-05): "do the same with this one" -- a third YouTube video
transcript (Stoic Trader, "SBS: swing breakout sequence") describing a standalone 5-step
liquidity-trap pattern, structurally different from the two prior videos (goal74/75, which
were HTF-trend + zone-retracement strategies). This one is explicitly NOT trend-dependent
("I wouldn't force that [HTF] bias onto the sequence, I would follow the sequence") -- it's
a self-contained reversal sequence claimed to repeat fractally on every time frame:

1. Breakout: price breaks a swing high/low (a genuine structural break; a wick counts, no
   candle close required).
2. First pullback: price retraces back into the broken swing's own range -- "the first
   stop" (early breakout traders get squeezed here first).
3. New high (or equal highs): price pushes on to a genuine new extreme beyond the
   breakout's own high/low, trapping continuation/breakout buyers.
4. Liquidation of #2: price reverses hard and sweeps back through the #2 pullback level --
   this is where the breakout-buyer traps from step 3 (who didn't take profit) get stopped
   out, and where sellers who bought the #2 dip in isolation also get run.
5. Reversal (the actual entry): AFTER the sweep at #4, a double-bottom/double-top reversal
   forms -- entry via a stop order at the level that reversal breaks, not on the raw sweep
   itself.
6. Target: back toward (explicitly NOT through) the #3 high -- "I don't even set it towards
   the high... because sometimes we have a big pullback right before the high gets taken."

Genuinely new versus every prior DTFX-based strategy this session: (a) a specific 5-stage
SEQUENTIAL state machine per zone (touch -> new extreme -> re-break -> reversal), not a
single filter/gate composed with the existing zone-fade touch builder, and (b) a STOP
order entry (fills by trading through a trigger price) rather than a resting LIMIT order at
a zone's mid -- mechanically different cost model: a stop order always crosses the spread
once triggered (there's no "genuinely free deep fill" the way a passive limit order can
get), so cost is charged via entry-side repricing on every trade, unconditionally.

Implemented as a numba-JIT'd per-zone scan (this project's own established pattern for
large sequential state-machine backtests, e.g. prod_filter_grid_search.py's
_simulate_generic_numba) since a pure-Python loop over ~140K zones x up to a ~1440-bar
window per zone would be too slow.
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
import goal30_apex_king_deployment as g30
import goal51_single_symbol_closest_price_backtest as g51

rs, pg = g29.rs, g29.pg


@njit(cache=True)
def _scan_sbs_sequences(high_, low_, top, bottom, is_buy, created_idx, max_window_bars,
                         reversal_lookback, stop_buffer_price, target_buffer_price,
                         spread_price):
    n_bars = len(high_)
    n_zones = len(top)
    out_is_buy = np.zeros(n_zones, dtype=np.bool_)
    out_entry_idx = np.zeros(n_zones, dtype=np.int64)
    out_entry_price = np.zeros(n_zones, dtype=np.float64)
    out_stop_price = np.zeros(n_zones, dtype=np.float64)
    out_target_price = np.zeros(n_zones, dtype=np.float64)
    out_exit_idx = np.zeros(n_zones, dtype=np.int64)
    out_is_win = np.zeros(n_zones, dtype=np.bool_)
    out_zone_idx = np.zeros(n_zones, dtype=np.int64)
    count = 0

    for zi in range(n_zones):
        c_idx = created_idx[zi]
        if c_idx >= n_bars:
            continue
        end = c_idx + max_window_bars
        if end > n_bars:
            end = n_bars
        zt, zb, zbuy = top[zi], bottom[zi], is_buy[zi]
        zmid = (zt + zb) / 2.0

        touch_idx = -1
        for j in range(c_idx, end):
            if zbuy:
                if low_[j] <= zmid:
                    touch_idx = j
                    break
            else:
                if high_[j] >= zmid:
                    touch_idx = j
                    break
        if touch_idx == -1:
            continue

        new_extreme_idx = -1
        for j in range(touch_idx, end):
            if zbuy:
                if high_[j] > zt:
                    new_extreme_idx = j
                    break
            else:
                if low_[j] < zb:
                    new_extreme_idx = j
                    break
        if new_extreme_idx == -1:
            continue

        run_extreme = high_[new_extreme_idx] if zbuy else low_[new_extreme_idx]
        break_idx = -1
        j = new_extreme_idx + 1
        while j < end:
            if zbuy:
                if high_[j] > run_extreme:
                    run_extreme = high_[j]
                if low_[j] < zb:
                    break_idx = j
                    break
            else:
                if low_[j] < run_extreme:
                    run_extreme = low_[j]
                if high_[j] > zt:
                    break_idx = j
                    break
            j += 1
        if break_idx == -1:
            continue

        window_end = break_idx + reversal_lookback
        if window_end > end:
            window_end = end
        if window_end <= break_idx:
            continue

        # STEP 5, causal streaming version: at each new bar, check whether it triggers entry
        # using ONLY the reference level established from bars strictly BEFORE it, then
        # afterward update the running low/reference for future iterations. The original
        # version pre-scanned the whole window for its GLOBAL minimum before placing the
        # entry trigger -- correct-looking, but it requires knowing which of many future
        # bars ends up being the true lowest before deciding where "the" reversal point is,
        # which no real-time trader could know in advance. This version never uses a bar's
        # own extremes to trigger its own entry decision -- only bars already in the past.
        if zbuy:
            running_low, pre_high = low_[break_idx], high_[break_idx]
            entry_idx = -1
            for k in range(break_idx + 1, window_end):
                if high_[k] >= pre_high:
                    entry_idx = k
                    break
                if low_[k] < running_low:
                    running_low, pre_high = low_[k], high_[k]
                elif high_[k] > pre_high:
                    pre_high = high_[k]
            if entry_idx == -1:
                continue
            local_low = running_low
            entry_price = pre_high + spread_price
            stop_price = local_low - stop_buffer_price
            target_price = run_extreme - target_buffer_price
            if target_price <= entry_price or stop_price >= entry_price:
                continue
        else:
            running_high, pre_low = high_[break_idx], low_[break_idx]
            entry_idx = -1
            for k in range(break_idx + 1, window_end):
                if low_[k] <= pre_low:
                    entry_idx = k
                    break
                if high_[k] > running_high:
                    running_high, pre_low = high_[k], low_[k]
                elif low_[k] < pre_low:
                    pre_low = low_[k]
            if entry_idx == -1:
                continue
            local_high = running_high
            entry_price = pre_low - spread_price
            stop_price = local_high + stop_buffer_price
            target_price = run_extreme + target_buffer_price
            if target_price >= entry_price or stop_price <= entry_price:
                continue

        trade_end = entry_idx + max_window_bars
        if trade_end > n_bars:
            trade_end = n_bars
        exit_idx, is_win, found = -1, False, False
        for m in range(entry_idx, trade_end):
            if zbuy:
                hit_stop = low_[m] <= stop_price
                hit_target = high_[m] >= target_price
            else:
                hit_stop = high_[m] >= stop_price
                hit_target = low_[m] <= target_price
            if hit_stop:
                exit_idx, is_win, found = m, False, True
                break
            elif hit_target:
                exit_idx, is_win, found = m, True, True
                break
        if not found:
            continue

        out_is_buy[count] = zbuy
        out_entry_idx[count] = entry_idx
        out_entry_price[count] = entry_price
        out_stop_price[count] = stop_price
        out_target_price[count] = target_price
        out_exit_idx[count] = exit_idx
        out_is_win[count] = is_win
        out_zone_idx[count] = zi
        count += 1

    return (out_is_buy[:count], out_entry_idx[:count], out_entry_price[:count],
            out_stop_price[:count], out_target_price[:count], out_exit_idx[:count],
            out_is_win[:count], out_zone_idx[:count])


def build_sbs_touches(inst, max_window_bars=1440, reversal_lookback=30, stop_buffer_pips=2.0,
                       target_buffer_pips=3.0, min_stop_pips=5.0, min_target_r=0.5):
    times, lows, highs, df = rs.load_instrument(inst)
    high_, low_ = df["high"].values.astype(np.float64), df["low"].values.astype(np.float64)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
    top, bottom, is_buy = z["top"], z["bottom"], z["is_buy"]
    created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
    created_idx = np.searchsorted(times_ns, created_ns)

    pip = pg.PIP_SIZE[inst]
    spread_price = pg.SPREAD_PIPS_TYPICAL * pip

    (out_is_buy, out_entry_idx, out_entry_price, out_stop_price, out_target_price,
     out_exit_idx, out_is_win, out_zone_idx) = _scan_sbs_sequences(
        high_, low_, top, bottom, is_buy, created_idx, max_window_bars, reversal_lookback,
        stop_buffer_pips * pip, target_buffer_pips * pip, spread_price)

    if len(out_is_buy) == 0:
        return pd.DataFrame(columns=["instrument", "is_buy", "created_at", "entry_time",
                                      "exit_time", "entry_price", "stop_risk", "net_r", "is_win"])

    effective_risk = np.abs(out_entry_price - out_stop_price)
    win_gross_r_all = np.abs(out_target_price - out_entry_price) / effective_risk
    # min_target_r guards against a real artifact found on GBPUSD: 12.8% of raw sequences
    # had run_extreme sitting almost exactly at pre_high (median target_r=0.11 among them --
    # target barely above entry, essentially free), producing a 99.98% win rate among that
    # subset purely from geometry, not genuine continuation. Excluding them barely moved
    # overall PF (1.735 -> 1.652 on GBPUSD raw) -- the core edge isn't propped up by this,
    # but it's a real degenerate-target class worth removing on principle, not just this once.
    keep = ((effective_risk / pip) >= min_stop_pips) & (win_gross_r_all >= min_target_r)
    out_is_buy, out_entry_idx = out_is_buy[keep], out_entry_idx[keep]
    out_entry_price, out_stop_price = out_entry_price[keep], out_stop_price[keep]
    out_target_price, out_exit_idx = out_target_price[keep], out_exit_idx[keep]
    out_is_win, out_zone_idx = out_is_win[keep], out_zone_idx[keep]
    effective_risk = effective_risk[keep]
    win_gross_r = win_gross_r_all[keep]

    # Stop-order entry: always crosses the spread once triggered (already baked into
    # out_entry_price above -- no separate cost_in_r, matching this project's own corrected
    # rule that a repriced entry is never double-charged on the exit side too.
    net_r = np.where(out_is_win, win_gross_r, -1.0)

    return pd.DataFrame({
        "instrument": inst, "is_buy": out_is_buy,
        "created_at": pd.to_datetime(times[created_idx[out_zone_idx]]),
        "entry_time": pd.to_datetime(times[out_entry_idx]),
        "exit_time": pd.to_datetime(times[out_exit_idx]),
        "entry_price": out_entry_price, "stop_risk": effective_risk,
        "net_r": net_r, "is_win": out_is_win, "target_r": win_gross_r,
        # each zone fires at most once in this scan (no re-arm/multi-touch concept, unlike
        # the zone-fade builder) -- touch_num=1 uniformly is a correct no-op for
        # resolve_closest_price's (entry_price, created_at) re-arm-ordering group-by.
        "touch_num": 1,
    })


def evaluate(inst, **kwargs):
    t = build_sbs_touches(inst, **kwargs)
    if len(t) < 15:
        return None
    price_at = g51._price_lookup(inst)
    age_h = 0.02  # stop orders trigger fast; a short fill-age gate is appropriate here
    resolved = g51.resolve_closest_price(t, age_h, price_at)
    if len(resolved) < 15:
        return None
    acc_t, _, _, _, _ = g30.simulate_two_account(resolved, 0.06)
    acc = acc_t[acc_t.accepted].sort_values("entry_time").reset_index(drop=True)
    if len(acc) < 15:
        return None
    years = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25
    wr = 100 * (acc.net_r > 0).mean()
    gw, gl = acc[acc.net_r > 0].net_r.sum(), -acc[acc.net_r <= 0].net_r.sum()
    pf = gw / gl if gl > 0 else float("inf")
    return dict(acc=acc, n=len(acc), freq=len(acc) / years, wr=wr, pf=pf)


def thirds_pf(acc):
    if len(acc) < 45:
        return (None, None, None)
    start, end = acc.entry_time.min(), acc.entry_time.max()
    span = end - start
    b1, b2 = start + span / 3, start + 2 * span / 3
    windows = [acc[acc.entry_time < b1], acc[(acc.entry_time >= b1) & (acc.entry_time < b2)],
               acc[acc.entry_time >= b2]]

    def pf_of(x):
        if len(x) < 10:
            return None
        gw, gl = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return gw / gl if gl > 0 else float("inf")
    return tuple(pf_of(w) for w in windows)


if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=== Raw sequence yield (GBPUSD, before any acceptance pipeline) ===")
    t = build_sbs_touches("GBPUSD")
    print(f"  n={len(t)} in {time.time()-t0:.1f}s, WR={100*t.is_win.mean():.1f}%, "
          f"mean target_r={t.target_r.mean():.2f}, "
          f"raw PF={(t[t.net_r>0].net_r.sum() / -t[t.net_r<=0].net_r.sum()):.3f}")

    print("\n=== True-accepted result per symbol ===")
    all_acc = []
    for inst in ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]:
        r = evaluate(inst)
        if r is None:
            print(f"  {inst}: too few")
            continue
        p1, p2, p3 = thirds_pf(r["acc"])
        print(f"  {inst}: n={r['n']} freq={r['freq']:.1f}/yr WR={r['wr']:.1f}% PF={r['pf']:.3f} "
              f"thirds={p1}/{p2}/{p3} ({time.time()-t0:.0f}s)")
        all_acc.append(r["acc"])

    if all_acc:
        pooled = pd.concat(all_acc, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
        years = (pooled.entry_time.max() - pooled.entry_time.min()).days / 365.25
        wr = 100 * (pooled.net_r > 0).mean()
        gw, gl = pooled[pooled.net_r > 0].net_r.sum(), -pooled[pooled.net_r <= 0].net_r.sum()
        pf = gw / gl if gl > 0 else float("inf")
        print(f"POOLED (4 symbols): n={len(pooled)} freq={len(pooled)/years:.1f}/yr WR={wr:.1f}% PF={pf:.3f}")
        split = pd.Timestamp("2018-01-01")
        train, hold = pooled[pooled.entry_time < split], pooled[pooled.entry_time >= split]

        def pf_of(x):
            if len(x) < 5:
                return None, len(x)
            gwx, glx = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
            return (gwx / glx if glx > 0 else float("inf")), len(x)
        print(f"  train (pre-2018): PF={pf_of(train)[0]} n={pf_of(train)[1]}")
        print(f"  holdout (2018+): PF={pf_of(hold)[0]} n={pf_of(hold)[1]}")

    print(f"\n(total: {time.time()-t0:.0f}s)")
