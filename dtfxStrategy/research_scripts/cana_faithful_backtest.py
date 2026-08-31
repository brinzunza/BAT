"""Corrected, faithful CANA backtest -- fixes both bugs found in cana_time_to_target_mc.py:

BUG 1 (units): the old script computed age as times.astype('int64') minus
created_at.astype('int64') without normalizing that `times` is datetime64[ns] while
`created_at` is datetime64[us] -- a 1000x mismatch that made the age filter never
actually fire. Fixed here by explicitly casting BOTH to datetime64[ns] before the
int64 conversion.

BUG 2 (state-machine fidelity): the old script resolved each touch's trade via a
separate forward scan, then jumped straight to the resolve bar +1 to look for the
next touch -- meaning probe/bounce tracking never ran DURING an open trade. The real
engine.py's _check_triggers updates probe/bounce EVERY bar, including while a trade
is open (the in_trade check happens AFTER the probe/bounce block, not before). Fixed
here with a genuine single-pass, bar-by-bar walk per zone that tracks probe/bounce
continuously and resolves the open trade's stop/target independently of the zone's
own "invalidated" bookkeeping state (which only gates FUTURE touches, matching how
bot.py's actual order management is fully independent of engine.py's active_zones
dict -- a zone entry disappearing from that dict doesn't affect an already-open
trade's real stop/target resolution at the broker).
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
PROBE_DEPTH = 0.50
EXCURSION_MULT = 1.00
MAX_ZONE_TOUCHES = 4
MAX_WINDOW_BARS = 60 * 24 * 90   # 90 days per zone lifetime cap
SPREAD_PIPS_TYPICAL = 1.6
PIP_SIZE = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}


@njit(cache=True)
def _simulate_cana_faithful(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                             zone_created_ns, times_ns, lows, highs, min_fill_age_ns,
                             n_mult, probe_depth, exc_mult, max_touches, max_window):
    n = len(zone_top)
    n_times = len(lows)
    max_out = n * max_touches
    out_touch_num = np.zeros(max_out, dtype=np.int64)
    out_entry_idx = np.full(max_out, -1, dtype=np.int64)
    out_is_win = np.zeros(max_out, dtype=np.bool_)
    out_mid = np.zeros(max_out, dtype=np.float64)
    out_risk = np.zeros(max_out, dtype=np.float64)
    out_count = 0

    for i in range(n):
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        is_buy = zone_is_buy[i]
        zone_width = top - bottom
        half_width = zone_width / 2.0
        risk = n_mult * half_width
        invalidation = bottom if is_buy else top
        probe_lvl = (top - probe_depth * zone_width) if is_buy else (bottom + probe_depth * zone_width)
        exc_target = (top + exc_mult * zone_width) if is_buy else (bottom - exc_mult * zone_width)
        stop_level = mid - risk if is_buy else mid + risk
        target_level = mid + risk if is_buy else mid - risk
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

        while j < window_end and touch_count < max_touches:
            lo = lows[j]
            hi = highs[j]

            if not zone_dead:
                breached = (lo <= invalidation) if is_buy else (hi >= invalidation)
                if breached:
                    zone_dead = True
                else:
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
                    is_win = tp_hit and not sl_hit   # tie on the same bar -> conservative: stop wins
                    out_touch_num[out_count] = touch_count
                    out_entry_idx[out_count] = j
                    out_is_win[out_count] = is_win
                    out_mid[out_count] = mid
                    out_risk[out_count] = risk
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
                else:
                    if had_bounce and touched_mid:
                        in_trade = True
                        touch_count += 1

            j += 1

    return (out_touch_num[:out_count], out_entry_idx[:out_count], out_is_win[:out_count],
            out_mid[:out_count], out_risk[:out_count])


def build_cana_trades(instrument):
    z = rs._prep_zones(instrument, MIN_RETRACEMENT)
    times, lows, highs, df = rs.load_instrument(instrument)
    times_ns = times.astype("datetime64[ns]").astype("int64")               # FIX: normalize unit
    created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")   # FIX: normalize unit

    risk_pct = (N_MULTIPLIER * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
    ok = risk_pct >= MIN_RISK_PCT
    top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
    is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]

    touch_num, entry_idx, is_win, entry_mid, entry_risk = _simulate_cana_faithful(
        top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
        int(MIN_FILL_AGE_H * 3600 * 1e9), N_MULTIPLIER, PROBE_DEPTH, EXCURSION_MULT,
        MAX_ZONE_TOUCHES, MAX_WINDOW_BARS)

    if len(entry_idx) == 0:
        return pd.DataFrame(columns=["entry_time", "net_r", "is_win", "touch_num", "instrument"])

    pip = PIP_SIZE[instrument]
    spread_price = SPREAD_PIPS_TYPICAL * pip
    r_gross = np.where(is_win, 1.0, -1.0)
    cost_in_r = np.where(is_win, 0.0, spread_price / entry_risk)
    net_r = r_gross - cost_in_r

    stop_pips = entry_risk / pip

    return pd.DataFrame({
        "entry_time": pd.to_datetime(times[entry_idx]),
        "net_r": net_r,
        "is_win": is_win,
        "touch_num": touch_num,
        "instrument": instrument,
        "stop_pips": stop_pips,
    })


def main():
    print("Corrected CANA backtest: unit bug fixed, continuous bar-by-bar probe/bounce "
          "tracking (including in-trade), independent trade-resolution vs zone-invalidation.\n",
          flush=True)
    frames = []
    for inst in INSTRUMENTS:
        rs.load_instrument(inst)
        rs._prep_zones(inst, MIN_RETRACEMENT)
        t = build_cana_trades(inst)
        years = (t["entry_time"].max() - t["entry_time"].min()).days / 365.25
        wins = t.loc[t.net_r > 0, "net_r"].sum()
        losses = -t.loc[t.net_r <= 0, "net_r"].sum()
        pf = wins / losses if losses > 0 else float("inf")
        print(f"  {inst}: {len(t):,} trades ({len(t)/years:,.0f}/yr), "
              f"WR={100*t['is_win'].mean():.2f}%, PF={pf:.3f}")
        frames.append(t)

    combined = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    years = (combined["entry_time"].max() - combined["entry_time"].min()).days / 365.25
    wins = combined.loc[combined.net_r > 0, "net_r"].sum()
    losses = -combined.loc[combined.net_r <= 0, "net_r"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    print(f"\nCOMBINED: {len(combined):,} trades ({len(combined)/years:,.0f}/yr), "
          f"WR={100*combined['is_win'].mean():.2f}%, PF={pf:.3f}, avg_r={combined['net_r'].mean():.4f}")

    print("\nBy touch number:")
    for t_num, sub in combined.groupby("touch_num"):
        w = sub.loc[sub.net_r > 0, "net_r"].sum()
        l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
        pf_t = w / l if l > 0 else float("inf")
        print(f"  touch {t_num}: trades={len(sub):,} WR={100*sub.is_win.mean():.2f}% "
              f"PF={pf_t:.3f} avg_r={sub.net_r.mean():.4f}")

    out_path = os.path.join(DATA_DIR, "cana_faithful_trades.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return combined


if __name__ == "__main__":
    main()
