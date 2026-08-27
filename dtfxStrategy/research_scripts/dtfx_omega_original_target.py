"""Omega with the ORIGINAL (pre-1:1) stop/target, everything else unchanged.

Omega's zone construction has always recorded a "tp" field per zone -- the trigger price
(the new higher-high / lower-low) that confirmed the zone in the first place (see
run_structure_fsm in dtfx_eurusd_backtest.py). The live bot and the whole robust-search
pipeline (dtfx_robust_search.py) never use that field: commit bc03444 in PRODLIVETRADING
switched stop/target to a symmetric N_MULTIPLIER-scaled level (an exact 1:1 R:R by
construction) because it beat the structural-target version on cost-adjusted Sharpe in
that comparison. This backtests going back to the structural version:

  stop   = the zone's own far edge (bottom for a buy, top for a sell) -- same as before,
           and same as what the numba sim's "invalidation" level already was
  target = zone["tp"] -- the structural trigger price, generally NOT the same distance
           from mid as the stop, so R:R is whatever that specific zone's geometry gives,
           not fixed at 1:1

Everything else stays exactly Omega: MIN_RETRACEMENT=2, MIN_RISK_PCT=0.00017,
MIN_FILL_AGE_HOURS=3.366682, RSI/ATR filter on. N_MULTIPLIER no longer applies (there's
nothing left for it to scale) once stop/target come from the zone's own structure.
"""
import numpy as np
import pandas as pd
from numba import njit

from dtfx_eurusd_backtest_strict import find_pivots_strict
from dtfx_eurusd_backtest import run_structure_fsm
from dtfx_robust_search import load_instrument, score_trades, DATA_DIR, _indicator_cache

OMEGA_PARAMS = dict(min_retracement=2, min_risk_pct=0.000170, min_fill_age_hours=3.366682)
FILL_SEARCH_FAST_BARS = 60 * 24 * 5
MAX_BARS_TO_RESOLVE = 60 * 24 * 30
FLAT_SPREAD_PIPS = 1.6
PIP_SIZE = 0.0001


@njit(cache=True)
def _simulate_original_numba(zone_top, zone_bottom, zone_mid, zone_tp, zone_is_buy,
                              zone_start_idx, lows, highs, fast_bars, max_bars):
    n = len(zone_top)
    n_times = len(lows)
    entry_idx_out = np.full(n, -1, dtype=np.int64)
    status_out = np.zeros(n, dtype=np.int8)  # 0=unresolved, 1=stopped_out, 2=take_profit

    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom, mid, tp = zone_top[i], zone_bottom[i], zone_mid[i], zone_tp[i]
        invalidation = bottom if is_buy else top
        stop_level = invalidation  # structural stop == the zone's own far edge
        target_level = tp          # structural target == the trigger price, NOT symmetric
        start = zone_start_idx[i]
        if start >= n_times:
            continue

        fast_end = start + fast_bars
        if fast_end > n_times:
            fast_end = n_times
        breach_idx, mid_idx = -1, -1
        for j in range(start, fast_end):
            if is_buy:
                if breach_idx == -1 and lows[j] <= invalidation:
                    breach_idx = j
                if mid_idx == -1 and lows[j] <= mid:
                    mid_idx = j
            else:
                if breach_idx == -1 and highs[j] >= invalidation:
                    breach_idx = j
                if mid_idx == -1 and highs[j] >= mid:
                    mid_idx = j
            if breach_idx != -1 and mid_idx != -1:
                break

        if mid_idx == -1 and breach_idx == -1 and fast_end < n_times:
            for j in range(fast_end, n_times):
                if is_buy:
                    if breach_idx == -1 and lows[j] <= invalidation:
                        breach_idx = j
                    if mid_idx == -1 and lows[j] <= mid:
                        mid_idx = j
                else:
                    if breach_idx == -1 and highs[j] >= invalidation:
                        breach_idx = j
                    if mid_idx == -1 and highs[j] >= mid:
                        mid_idx = j
                if breach_idx != -1 and mid_idx != -1:
                    break

        if mid_idx == -1:
            continue
        if breach_idx != -1 and breach_idx <= mid_idx:
            continue

        entry_idx = mid_idx
        after_start = entry_idx + 1
        after_end = after_start + max_bars
        if after_end > n_times:
            after_end = n_times
        if after_start >= n_times:
            continue

        sl_idx, tp_idx = -1, -1
        for j in range(after_start, after_end):
            if is_buy:
                if sl_idx == -1 and lows[j] <= stop_level:
                    sl_idx = j
                if tp_idx == -1 and highs[j] >= target_level:
                    tp_idx = j
            else:
                if sl_idx == -1 and highs[j] >= stop_level:
                    sl_idx = j
                if tp_idx == -1 and lows[j] <= target_level:
                    tp_idx = j
            if sl_idx != -1 and tp_idx != -1:
                break

        if sl_idx == -1 and tp_idx == -1:
            continue
        entry_idx_out[i] = entry_idx
        if sl_idx != -1 and (tp_idx == -1 or sl_idx <= tp_idx):
            status_out[i] = 1  # stopped_out
        else:
            status_out[i] = 2  # take_profit

    return entry_idx_out, status_out


def build_original_target_trades(instrument):
    times, lows, highs, df = load_instrument(instrument)
    zigzag = find_pivots_strict(df, OMEGA_PARAMS["min_retracement"])
    _, zones = run_structure_fsm(zigzag)
    rsi14, atr_ratio = _indicator_cache[instrument]

    n = len(zones)
    top = np.array([z["top"] for z in zones], dtype=np.float64)
    bottom = np.array([z["bottom"] for z in zones], dtype=np.float64)
    mid = np.array([z["mid"] for z in zones], dtype=np.float64)
    tp = np.array([z["tp"] for z in zones], dtype=np.float64)
    is_buy = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
    created_at = np.array([np.datetime64(z["created_at"]) for z in zones])
    start_idx = np.searchsorted(times, created_at, side="right").astype(np.int64)

    entry_idx, status = _simulate_original_numba(
        top, bottom, mid, tp, is_buy, start_idx, lows, highs,
        FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE)

    filled = entry_idx >= 0
    risk = np.where(is_buy, mid - bottom, top - mid)  # structural risk = half zone width, always
    risk_pct_of_price = risk / mid
    entry_times_ns = times[np.where(filled, entry_idx, 0)]
    fill_hours = (entry_times_ns - created_at) / np.timedelta64(1, "h")

    keep = (filled & (risk_pct_of_price >= OMEGA_PARAMS["min_risk_pct"])
            & (fill_hours >= OMEGA_PARAMS["min_fill_age_hours"]))
    r14 = rsi14[np.where(filled, entry_idx, 0)]
    ar = atr_ratio[np.where(filled, entry_idx, 0)]
    rule_ok = ~np.isnan(r14) & ~np.isnan(ar) & ~((r14 >= 45) & (r14 <= 55)) & ~(ar < 0.7)
    keep = keep & rule_ok

    idx = np.where(keep)[0]
    reward = np.where(is_buy, tp - mid, mid - tp)  # asymmetric -- NOT forced equal to risk
    r_gross = np.where(status[idx] == 2, reward[idx] / risk[idx], -1.0)

    out = pd.DataFrame({
        "entry_time": pd.to_datetime(entry_times_ns[idx]),
        "r_gross": r_gross,
        "win": status[idx] == 2,
        "risk_price": risk[idx],
        "reward_to_risk": (reward[idx] / risk[idx]),
    }).sort_values("entry_time").reset_index(drop=True)
    return out


def analyze(instrument):
    print(f"\n{'='*78}\n{instrument} -- ORIGINAL (structural) stop/target\n{'='*78}")
    df = build_original_target_trades(instrument)
    n = len(df)
    print(f"{n:,} trades, {df['entry_time'].min().date()} -> {df['entry_time'].max().date()}")
    print(f"  win_rate={df['win'].mean():.3f}  avg gross R={df['r_gross'].mean():.3f}  "
          f"median R:R on wins={df.loc[df['win'],'reward_to_risk'].median():.2f}")

    gross_frame = df[["entry_time", "r_gross"]].rename(columns={"r_gross": "net_r"})
    gross_score, gross_stats = score_trades(gross_frame)
    print(f"  GROSS: score={gross_score}", gross_stats)

    cost_in_r = (FLAT_SPREAD_PIPS * PIP_SIZE) / df["risk_price"]
    df["net_r"] = df["r_gross"] - np.where(df["r_gross"] < 0, cost_in_r, 0.0)
    net_score, net_stats = score_trades(df[["entry_time", "net_r"]])
    print(f"  NET (asymmetric cost, {FLAT_SPREAD_PIPS} pip): score={net_score}", net_stats)

    out_path = f"{DATA_DIR}/dtfx_omega_original_target_{instrument}.csv"
    df.to_csv(out_path, index=False)
    print(f"  saved -> {out_path}")
    return df, gross_stats, net_stats


if __name__ == "__main__":
    for instrument in ["EURUSD", "GBPUSD", "EURGBP"]:
        analyze(instrument)
