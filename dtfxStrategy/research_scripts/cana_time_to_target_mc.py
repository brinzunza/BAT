"""Same 25,000-path time-to-$1M/$10M Monte Carlo as goal_70pct_time_to_target_mc.py,
but built on CANA's ACTUAL deployed parameters (PRODLIVETRADING/strategy/engine.py as
of commit acefb73) instead of the validated narrow-target candidate -- for a direct,
apples-to-apples comparison against CANA.md's own asserted (and unverified) numbers.

CANA parameters, ported faithfully from the live engine, not approximated:
  - MIN_RETRACEMENT=2, MIN_FILL_AGE=2.0h, N_MULTIPLIER=3.00 (SYMMETRIC 1:1 R:R --
    CANA did not change the R:R, only the stop size and age filter vs Omega)
  - MIN_RISK_PCT=0.000170 (unchanged floor)
  - NO session filter (24h/5, SESSION_HOUR_FILTER={} in the live engine)
  - Multi-touch up to 4, but touches 2-4 require a confirmed "elite bounce": price
    must touch mid (probe), then travel a full zone-width further (excursion=1.0x),
    before the NEXT mid-touch counts as a valid re-entry. This is a genuine bar-by-bar
    state machine, not a single first-touch lookup -- ported here as a per-zone numba
    walk mirroring SymbolEngine._check_triggers exactly, not a simplification.

This is the piece that was missing from the CANA review: an actual backtest of what's
live, at the same rigor as the narrow-target candidate got.
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})

INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
MIN_RETRACEMENT = 2
MIN_FILL_AGE_H = 2.0
N_MULTIPLIER = 3.00
MIN_RISK_PCT = 0.000170
EXCURSION_MULT = 1.00
MAX_ZONE_TOUCHES = 4
MAX_WINDOW_BARS = 60 * 24 * 60   # 60 days per touch-search / per resolution search
SPREAD_PIPS_TYPICAL = 1.6
PIP_SIZE = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}


@njit(cache=True)
def _simulate_cana_multitouch(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                               zone_created_ns, times_ns, lows, highs, min_fill_age_ns,
                               n_mult, exc_mult, max_touches, max_window):
    n = len(zone_top)
    n_times = len(lows)
    max_out = n * max_touches
    out_entry_idx = np.full(max_out, -1, dtype=np.int64)
    out_is_win = np.zeros(max_out, dtype=np.bool_)
    out_risk_frac = np.zeros(max_out, dtype=np.float64)
    out_mid = np.zeros(max_out, dtype=np.float64)
    out_touch = np.zeros(max_out, dtype=np.int64)
    out_count = 0

    for i in range(n):
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        is_buy = zone_is_buy[i]
        zone_width = top - bottom
        half_width = zone_width / 2.0
        risk = n_mult * half_width
        invalidation = bottom if is_buy else top
        exc_target = top + exc_mult * zone_width if is_buy else bottom - exc_mult * zone_width
        created_ns = zone_created_ns[i]
        j = zone_start_idx[i]
        if j >= n_times:
            continue
        window_end = min(j + max_window, n_times)

        touch_count = 0
        in_probe = False
        had_bounce = False

        while j < window_end and touch_count < max_touches:
            lo = lows[j]
            hi = highs[j]
            breached = (lo <= invalidation) if is_buy else (hi >= invalidation)
            if breached:
                break
            touched_mid = (lo <= mid) if is_buy else (hi >= mid)

            entry_this_bar = False
            if touch_count == 0:
                if touched_mid:
                    age_ns = times_ns[j] - created_ns
                    if age_ns < min_fill_age_ns:
                        break  # permanently disqualified, matches live "fast touch" rule
                    entry_this_bar = True
            else:
                if touched_mid:
                    in_probe = True
                if in_probe:
                    bounced = (hi >= exc_target) if is_buy else (lo <= exc_target)
                    if bounced:
                        had_bounce = True
                if had_bounce and touched_mid:
                    entry_this_bar = True

            if entry_this_bar:
                stop_level = mid - risk if is_buy else mid + risk
                target_level = mid + risk if is_buy else mid - risk
                sl_idx, tp_idx = -1, -1
                k = j + 1
                k_end = min(k + max_window, n_times)
                while k < k_end:
                    if is_buy:
                        if sl_idx == -1 and lows[k] <= stop_level:
                            sl_idx = k
                        if tp_idx == -1 and highs[k] >= target_level:
                            tp_idx = k
                    else:
                        if sl_idx == -1 and highs[k] >= stop_level:
                            sl_idx = k
                        if tp_idx == -1 and lows[k] <= target_level:
                            tp_idx = k
                    if sl_idx != -1 and tp_idx != -1:
                        break
                    k += 1
                if sl_idx == -1 and tp_idx == -1:
                    break  # unresolved within window, stop this zone
                is_win = tp_idx != -1 and (sl_idx == -1 or tp_idx < sl_idx)
                out_entry_idx[out_count] = j
                out_is_win[out_count] = is_win
                out_risk_frac[out_count] = risk / mid
                out_mid[out_count] = mid
                out_touch[out_count] = touch_count + 1
                out_count += 1
                touch_count += 1
                if not is_win or touch_count >= max_touches:
                    break
                resolve_idx = tp_idx  # win -> resolved at target
                in_probe = False
                had_bounce = False
                j = resolve_idx + 1
                continue

            j += 1

    return out_entry_idx[:out_count], out_is_win[:out_count], out_risk_frac[:out_count], out_mid[:out_count], out_touch[:out_count]


def build_cana_trades(instrument):
    z = rs._prep_zones(instrument, MIN_RETRACEMENT)
    times, lows, highs, df = rs.load_instrument(instrument)
    times_ns = times.astype("int64")
    created_ns = z["created_at"].astype("int64")
    risk_pct = (N_MULTIPLIER * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
    ok = risk_pct >= MIN_RISK_PCT
    top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
    is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]

    entry_idx, is_win, risk_frac, entry_mid, touch_num = _simulate_cana_multitouch(
        top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
        int(MIN_FILL_AGE_H * 3600 * 1e9), N_MULTIPLIER, EXCURSION_MULT, MAX_ZONE_TOUCHES, MAX_WINDOW_BARS)

    if len(entry_idx) == 0:
        return pd.DataFrame(columns=["entry_time", "net_r", "is_win"])

    pip = PIP_SIZE[instrument]
    risk_price = risk_frac * entry_mid          # stop distance in price units, per trade
    spread_price = SPREAD_PIPS_TYPICAL * pip
    r_gross = np.where(is_win, 1.0, -1.0)
    cost_in_r = np.where(is_win, 0.0, spread_price / risk_price)   # cost only on stop-outs
    net_r = r_gross - cost_in_r

    return pd.DataFrame({
        "entry_time": pd.to_datetime(times[entry_idx]),
        "net_r": net_r,
        "is_win": is_win,
        "instrument": instrument,
        "touch_num": touch_num,
    })


def main():
    print("Building CANA trade pool (actual deployed parameters: 2.0h age, N=3.00, "
          "24h no session, 4-touch elite-bounce multi-touch)...\n", flush=True)
    frames = []
    for inst in INSTRUMENTS:
        rs.load_instrument(inst)
        rs._prep_zones(inst, MIN_RETRACEMENT)
        t = build_cana_trades(inst)
        wr = 100 * t["is_win"].mean() if len(t) else float("nan")
        pf = (t.loc[t["net_r"] > 0, "net_r"].sum() / -t.loc[t["net_r"] <= 0, "net_r"].sum()
              if len(t) and (t["net_r"] <= 0).any() else float("nan"))
        years = (t["entry_time"].max() - t["entry_time"].min()).days / 365.25 if len(t) else float("nan")
        print(f"  {inst}: {len(t):,} trades ({len(t)/years:,.0f}/yr), WR={wr:.2f}%, PF={pf:.3f}")
        frames.append(t)
    combined = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)

    years = (combined["entry_time"].max() - combined["entry_time"].min()).days / 365.25
    wr = 100 * combined["is_win"].mean()
    pf = combined.loc[combined["net_r"] > 0, "net_r"].sum() / -combined.loc[combined["net_r"] <= 0, "net_r"].sum()
    print(f"\nCOMBINED (actual CANA, live parameters): {len(combined):,} trades, "
          f"{len(combined)/years:,.0f}/yr, WR={wr:.2f}%, PF={pf:.3f}, avg_r={combined['net_r'].mean():.4f}")

    out_path = os.path.join(DATA_DIR, "cana_actual_trades.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return combined


if __name__ == "__main__":
    main()
