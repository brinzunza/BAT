"""General-purpose, faithful multi-touch grid search over PROD's actual filter knobs:
stop distance, target R:R, min fill age, session filter, and whether re-touches
require a confirmed excursion bounce (PROD currently runs WITHOUT that requirement --
unconditional re-arm on any win, up to MAX_ZONE_TOUCHES).

This reuses the exact bar-by-bar state machine validated in cana_faithful_backtest.py
(continuous probe/bounce tracking including while a trade is open, trade resolution
independent of zone invalidation, unit-correct age comparison) generalized to:
  - asymmetric stop/target (stop_mult, target_mult independent)
  - require_bounce as a toggle (False = PROD's current unconditional re-arm behavior)
  - session filter applied as a post-hoc entry-hour mask (same method used throughout
    this project's single-touch SOMEGA work), not baked into the bar walk itself.

Cost model: 1.6-pip spread, charged only on stop-outs (passive-fill target/soft-stop
assumption, matching the live hybrid-stop design).
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
from dtfx_somega_params import SESSION_HOUR_FILTER

try:
    import dtfx_engine
    _HAS_CPP_ENGINE = True
except ImportError:
    _HAS_CPP_ENGINE = False

INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
MIN_RETRACEMENT = 2
MIN_RISK_PCT = 0.000170
PROBE_DEPTH = 0.50
MAX_ZONE_TOUCHES = 4
MAX_WINDOW_BARS = 60 * 24 * 90
SPREAD_PIPS_TYPICAL = 1.6
PIP_SIZE = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}


@njit(cache=True)
def _simulate_generic_numba(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                             zone_created_ns, times_ns, lows, highs, min_fill_age_ns,
                             stop_mult, target_mult, require_bounce, probe_depth, exc_mult,
                             max_touches, max_window):
    n = len(zone_top)
    n_times = len(lows)
    max_out = n * max_touches
    out_touch = np.zeros(max_out, dtype=np.int64)
    out_entry_idx = np.full(max_out, -1, dtype=np.int64)  # true fill bar, NOT the resolution bar (fixed -- see live-readiness audit)
    out_is_win = np.zeros(max_out, dtype=np.bool_)
    out_mid = np.zeros(max_out, dtype=np.float64)
    out_stop_risk = np.zeros(max_out, dtype=np.float64)
    out_count = 0

    for i in range(n):
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        is_buy = zone_is_buy[i]
        zone_width = top - bottom
        half_width = zone_width / 2.0
        stop_risk = stop_mult * half_width
        target_risk = target_mult * half_width
        invalidation = bottom if is_buy else top
        probe_lvl = (top - probe_depth * zone_width) if is_buy else (bottom + probe_depth * zone_width)
        exc_target = (top + exc_mult * zone_width) if is_buy else (bottom - exc_mult * zone_width)
        stop_level = mid - stop_risk if is_buy else mid + stop_risk
        target_level = mid + target_risk if is_buy else mid - target_risk
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
                    out_mid[out_count] = mid
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
                touched_mid = (lo <= mid) if is_buy else (hi >= mid)
                if touch_count == 0:
                    if touched_mid:
                        age_ns = times_ns[j] - created_ns
                        if age_ns < min_fill_age_ns:
                            break
                        in_trade = True
                        touch_count = 1
                        cur_entry_j = j
                else:
                    if (not require_bounce) or had_bounce:
                        if touched_mid:
                            in_trade = True
                            touch_count += 1
                            cur_entry_j = j

            j += 1

    return (out_touch[:out_count], out_entry_idx[:out_count], out_is_win[:out_count],
            out_mid[:out_count], out_stop_risk[:out_count])


def _simulate_generic(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                       zone_created_ns, times_ns, lows, highs, min_fill_age_ns,
                       stop_mult, target_mult, require_bounce, probe_depth, exc_mult,
                       max_touches, max_window):
    """General-purpose multi-touch trade simulator. Uses the C++ dtfx_engine extension
    (cpp_engine/, see cpp_engine/README.md) when built -- validated byte-identical to the
    numba path below (cpp_engine/validate.py) -- falling back to the original numba
    implementation if the extension isn't built. Every strategy knob stays a plain argument
    either way; only the execution backend changes."""
    if _HAS_CPP_ENGINE:
        return dtfx_engine.simulate_generic(
            zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx, zone_created_ns,
            times_ns, lows, highs, min_fill_age_ns, stop_mult, target_mult, require_bounce,
            probe_depth, exc_mult, max_touches, max_window)
    return _simulate_generic_numba(
        zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx, zone_created_ns,
        times_ns, lows, highs, min_fill_age_ns, stop_mult, target_mult, require_bounce,
        probe_depth, exc_mult, max_touches, max_window)


def build_trades(instrument, stop_mult, target_mult, min_fill_age_h, require_bounce,
                  session_filter, exc_mult=1.0):
    z = rs._prep_zones(instrument, MIN_RETRACEMENT)
    times, lows, highs, df = rs.load_instrument(instrument)
    times_ns = times.astype("datetime64[ns]").astype("int64")
    created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")

    risk_pct = (stop_mult * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
    ok = risk_pct >= MIN_RISK_PCT
    top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
    is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]

    touch_num, entry_idx, is_win, entry_mid, stop_risk = _simulate_generic(
        top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
        int(min_fill_age_h * 3600 * 1e9), stop_mult, target_mult, require_bounce,
        PROBE_DEPTH, exc_mult, MAX_ZONE_TOUCHES, MAX_WINDOW_BARS)

    if len(entry_idx) == 0:
        return pd.DataFrame(columns=["entry_time", "net_r", "is_win", "touch_num", "instrument"])

    pip = PIP_SIZE[instrument]
    spread_price = SPREAD_PIPS_TYPICAL * pip
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
        "touch_num": touch_num[keep],
        "instrument": instrument,
    })


def score(trades, years):
    if len(trades) < 30:
        return None
    w = trades.loc[trades.net_r > 0, "net_r"].sum()
    l = -trades.loc[trades.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    return dict(trades=len(trades), trades_yr=round(len(trades) / years, 0),
                win_rate=round(100 * trades.is_win.mean(), 2), pf=round(pf, 3),
                avg_r=round(trades.net_r.mean(), 4))


def main():
    for inst in INSTRUMENTS:
        rs.load_instrument(inst)
        rs._prep_zones(inst, MIN_RETRACEMENT)

    grid = list(itertools.product(
        [2.0, 3.343496],          # stop_mult
        [1.0, 0.5, 0.35],         # rr
        [2.0, 3.366682],          # age
        [False, True],            # require_bounce
        [False, True],            # session filter
    ))
    print(f"Running {len(grid)} configs x {len(INSTRUMENTS)} instruments...\n", flush=True)
    rows = []
    for stop_mult, rr, age, bounce, session in grid:
        target_mult = stop_mult * rr
        frames = [build_trades(inst, stop_mult, target_mult, age, bounce, session) for inst in INSTRUMENTS]
        combined = pd.concat(frames, ignore_index=True)
        if len(combined) < 30:
            continue
        years = (combined.entry_time.max() - combined.entry_time.min()).days / 365.25
        s = score(combined, years)
        if s is None:
            continue
        s.update(stop_mult=stop_mult, rr=rr, age_h=age, bounce=bounce, session=session)
        rows.append(s)

    res = pd.DataFrame(rows)
    cols = ["stop_mult", "rr", "age_h", "bounce", "session", "trades", "trades_yr", "win_rate", "pf", "avg_r"]
    res = res[cols].sort_values("pf", ascending=False)
    out_path = os.path.join(DATA_DIR, "prod_filter_grid_search.csv")
    res.to_csv(out_path, index=False)
    print("Top 20 by PF:")
    print(res.head(20).to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
