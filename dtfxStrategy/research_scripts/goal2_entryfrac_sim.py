"""Generalizes prod_filter_grid_search's _simulate_generic to an arbitrary entry
retracement depth (entry_frac), instead of the fixed 50% (mid). entry_frac=0.5 exactly
reproduces the existing mid-entry behavior; entry_frac=0.3 ("30% tap") triggers on a
SHALLOWER pullback -- price only needs to retrace 30% of the way into the zone from the
near edge before the resting limit fires, which should trigger more often (a shallower
level is touched sooner/more frequently than a deeper 50% retracement) at the cost of a
closer, tighter, more failure-prone entry relative to the zone's own structure.

Stop/target are still N_MULTIPLIER * zone_half_width from the entry level, symmetric
(1:1 R:R preserved), independent of entry_frac -- only WHERE inside the zone the order
rests is changed, not the risk-sizing convention.
"""
import os, sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
import prod_filter_grid_search as pg
from dtfx_somega_params import SESSION_HOUR_FILTER

INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]


@njit(cache=True)
def _simulate_entryfrac(zone_top, zone_bottom, zone_is_buy, zone_start_idx, zone_created_ns,
                         times_ns, lows, highs, min_fill_age_ns, entry_frac, stop_mult,
                         target_mult, require_bounce, probe_depth, exc_mult, max_touches, max_window):
    n = len(zone_top)
    n_times = len(lows)
    max_out = n * max_touches
    out_touch = np.zeros(max_out, dtype=np.int64)
    out_entry_idx = np.full(max_out, -1, dtype=np.int64)
    out_is_win = np.zeros(max_out, dtype=np.bool_)
    out_stop_risk = np.zeros(max_out, dtype=np.float64)
    out_count = 0

    for i in range(n):
        top, bottom = zone_top[i], zone_bottom[i]
        is_buy = zone_is_buy[i]
        zone_width = top - bottom
        half_width = zone_width / 2.0
        # entry_level: for a buy zone, entry_frac of the way DOWN from top toward bottom
        entry_level = (top - entry_frac * zone_width) if is_buy else (bottom + entry_frac * zone_width)
        stop_risk = stop_mult * half_width
        target_risk = target_mult * half_width
        invalidation = bottom if is_buy else top
        probe_lvl = (top - probe_depth * zone_width) if is_buy else (bottom + probe_depth * zone_width)
        exc_target = (top + exc_mult * zone_width) if is_buy else (bottom - exc_mult * zone_width)
        stop_level = entry_level - stop_risk if is_buy else entry_level + stop_risk
        target_level = entry_level + target_risk if is_buy else entry_level - target_risk
        created_ns = zone_created_ns[i]

        j = zone_start_idx[i]
        if j >= n_times:
            continue
        window_end = min(j + max_window, n_times)

        zone_dead = False
        in_trade = False
        touch_count = 0
        in_probe = False
        had_bounce = False
        cur_entry_j = -1  # bar index of the CURRENT touch's actual fill, not its resolution

        while j < window_end and touch_count < max_touches:
            lo = lows[j]
            hi = highs[j]

            if not zone_dead:
                breached = (lo <= invalidation) if is_buy else (hi >= invalidation)
                if breached:
                    zone_dead = True
                elif require_bounce:
                    probed = (lo <= probe_lvl) if is_buy else (hi >= probe_lvl)
                    if probed:
                        in_probe = True
                    if in_probe:
                        bounced = (hi >= exc_target) if is_buy else (lo <= exc_target)
                        if bounced:
                            had_bounce = True

            if in_trade:
                tp_hit = (hi >= target_level) if is_buy else (lo <= target_level)
                sl_hit = (lo <= stop_level) if is_buy else (hi >= stop_level)
                if tp_hit or sl_hit:
                    is_win = tp_hit and not sl_hit
                    out_touch[out_count] = touch_count
                    out_entry_idx[out_count] = cur_entry_j
                    out_is_win[out_count] = is_win
                    out_stop_risk[out_count] = stop_risk
                    out_count += 1
                    in_trade = False
                    if is_win and touch_count < max_touches and not zone_dead:
                        in_probe = False
                        had_bounce = False
                    else:
                        break
            else:
                if zone_dead:
                    break
                touched_entry = (lo <= entry_level) if is_buy else (hi >= entry_level)
                if touch_count == 0:
                    if touched_entry:
                        age_ns = times_ns[j] - created_ns
                        if age_ns < min_fill_age_ns:
                            break
                        in_trade = True
                        touch_count = 1
                        cur_entry_j = j
                else:
                    if (not require_bounce) or had_bounce:
                        if touched_entry:
                            in_trade = True
                            touch_count += 1
                            cur_entry_j = j

            j += 1

    return (out_touch[:out_count], out_entry_idx[:out_count], out_is_win[:out_count],
            out_stop_risk[:out_count])


def build_trades(instrument, entry_frac, stop_mult, target_mult, min_fill_age_h,
                  require_bounce, session_filter, max_touches=4):
    z = rs._prep_zones(instrument, pg.MIN_RETRACEMENT)
    times, lows, highs, df = rs.load_instrument(instrument)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")

    risk_pct = (stop_mult * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
    ok = risk_pct >= pg.MIN_RISK_PCT
    top, bottom = z["top"][ok], z["bottom"][ok]
    is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]

    touch_num, entry_idx, is_win, stop_risk = _simulate_entryfrac(
        top, bottom, is_buy, start_idx, c_ns, times_ns, lows, highs,
        int(min_fill_age_h * 3600 * 1e9), entry_frac, stop_mult, target_mult,
        require_bounce, pg.PROBE_DEPTH, 1.0, max_touches, pg.MAX_WINDOW_BARS)

    if len(entry_idx) == 0:
        return pd.DataFrame(columns=["entry_time", "net_r", "is_win", "instrument"])

    pip = pg.PIP_SIZE[instrument]
    spread_price = pg.SPREAD_PIPS_TYPICAL * pip
    r_gross = np.where(is_win, target_mult / stop_mult, -1.0)
    cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
    net_r = r_gross - cost_in_r
    entry_times_ns = times[entry_idx]

    keep = np.ones(len(entry_idx), dtype=bool)
    if session_filter:
        allowed_hours = SESSION_HOUR_FILTER.get(instrument)
        if allowed_hours is not None:
            entry_hour = pd.DatetimeIndex(entry_times_ns).hour
            keep = np.isin(entry_hour, list(allowed_hours))

    return pd.DataFrame({
        "entry_time": pd.to_datetime(entry_times_ns[keep]),
        "net_r": net_r[keep],
        "is_win": is_win[keep],
        "instrument": instrument,
    })
