"""Hybrid stop: exit as a LIMIT order sitting exactly at the original stop level (fills at
the stop price with zero spread cost, same as a take-profit does today) with a wider
BACKSTOP hard stop further out that only fires as a real market order if price gaps past
the limit without ever trading at it. This is the middle ground between "market stop" (pays
spread on every loss, always fills) and "pure limit stop" (never pays spread, but can leave
the position open indefinitely on a gap) discussed in the prior analysis.

Mechanics per stop-out trade:
  - If the bar that first reaches the original stop level did NOT gap past it (its open was
    still on the favorable side), the limit fills there cleanly: r=-1.0, zero cost -- this is
    the common case (>97% of stop-outs historically, per the gap-frequency check).
  - If that bar DID gap past the stop level, the limit doesn't fill. From there, whichever
    happens first wins: price reverses back to the original stop level (limit fills late,
    r=-1.0 still, zero cost -- just delayed, no worse P&L), price reaches the backstop level
    (a real market exit, r=-backstop_multiple, DOES pay spread cost), or price reverses all
    the way to the original target (take-profit anyway, r=+1.0, zero cost).
  - If none of those happen before the resolution window closes, the trade is left
    unresolved (excluded), matching how the base simulation already treats zones that never
    resolve.
"""
import numpy as np
import pandas as pd
from numba import njit

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
from dtfx_robust_search import _prep_zones, load_instrument, _indicator_cache, score_trades, DATA_DIR
from dtfx_omega_winloss_factors import BASE_PARAMS as OMEGA_PARAMS
from dtfx_somega_params import SESSION_HOUR_FILTER

FILL_SEARCH_FAST_BARS = 60 * 24 * 5
MAX_BARS_TO_RESOLVE = 60 * 24 * 30
PIP_SIZE = 0.0001

STATUS_UNRESOLVED = 0
STATUS_TARGET = 2
STATUS_STOP_LIMIT = 1     # clean or delayed limit fill at the original stop -- zero cost
STATUS_STOP_BACKSTOP = 3  # backstop market exit -- pays cost


@njit
def sim_hybrid(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
               opens, lows, highs, n_multiplier, backstop_multiplier, fast_bars, max_bars):
    n = len(zone_top)
    n_times = len(lows)
    entry_idx_out = np.full(n, -1, dtype=np.int64)
    status_out = np.zeros(n, dtype=np.int8)
    exit_r_out = np.zeros(n, dtype=np.float64)  # signed R realized (before cost)

    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        risk = n_multiplier * (top - bottom) / 2.0
        invalidation = bottom if is_buy else top
        start = zone_start_idx[i]
        if start >= n_times:
            continue

        fast_end = start + fast_bars
        if fast_end > n_times:
            fast_end = n_times
        breach_idx, mid_idx = -1, -1
        for j in range(start, fast_end):
            if is_buy:
                if breach_idx == -1 and lows[j] <= invalidation: breach_idx = j
                if mid_idx == -1 and lows[j] <= mid: mid_idx = j
            else:
                if breach_idx == -1 and highs[j] >= invalidation: breach_idx = j
                if mid_idx == -1 and highs[j] >= mid: mid_idx = j
            if breach_idx != -1 and mid_idx != -1:
                break
        if mid_idx == -1 and breach_idx == -1 and fast_end < n_times:
            for j in range(fast_end, n_times):
                if is_buy:
                    if breach_idx == -1 and lows[j] <= invalidation: breach_idx = j
                    if mid_idx == -1 and lows[j] <= mid: mid_idx = j
                else:
                    if breach_idx == -1 and highs[j] >= invalidation: breach_idx = j
                    if mid_idx == -1 and highs[j] >= mid: mid_idx = j
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

        stop_level = mid - risk if is_buy else mid + risk
        target_level = mid + risk if is_buy else mid - risk
        backstop_level = mid - risk * backstop_multiplier if is_buy else mid + risk * backstop_multiplier

        sl_idx, tp_idx = -1, -1
        for j in range(after_start, after_end):
            if is_buy:
                if sl_idx == -1 and lows[j] <= stop_level: sl_idx = j
                if tp_idx == -1 and highs[j] >= target_level: tp_idx = j
            else:
                if sl_idx == -1 and highs[j] >= stop_level: sl_idx = j
                if tp_idx == -1 and lows[j] <= target_level: tp_idx = j
            if sl_idx != -1 and tp_idx != -1:
                break

        if sl_idx == -1 and tp_idx == -1:
            continue
        entry_idx_out[i] = entry_idx

        if tp_idx != -1 and (sl_idx == -1 or tp_idx <= sl_idx):
            status_out[i] = STATUS_TARGET
            exit_r_out[i] = 1.0
            continue

        # sl_idx comes first (or ties, same convention as the base sim) -- check for a gap
        gapped = (opens[sl_idx] < stop_level) if is_buy else (opens[sl_idx] > stop_level)
        if not gapped:
            status_out[i] = STATUS_STOP_LIMIT
            exit_r_out[i] = -1.0
            continue

        # gapped: limit didn't fill this bar. Scan forward for backstop / limit-refill / target,
        # worst-case-first tie-break within a single bar (consistent with the base sim's own
        # "breach wins ties" convention).
        resolved = False
        for j in range(sl_idx, after_end):
            if is_buy:
                backstop_hit = lows[j] <= backstop_level
                refill_hit = highs[j] >= stop_level
                target_hit = highs[j] >= target_level
            else:
                backstop_hit = highs[j] >= backstop_level
                refill_hit = lows[j] <= stop_level
                target_hit = lows[j] <= target_level
            if backstop_hit:
                status_out[i] = STATUS_STOP_BACKSTOP
                exit_r_out[i] = -backstop_multiplier
                resolved = True
                break
            if refill_hit:
                status_out[i] = STATUS_STOP_LIMIT
                exit_r_out[i] = -1.0
                resolved = True
                break
            if target_hit:
                status_out[i] = STATUS_TARGET
                exit_r_out[i] = 1.0
                resolved = True
                break
        # if never resolved within the window, leave as unresolved (entry_idx_out stays set,
        # status_out stays 0 -- filtered out downstream same as any other unresolved trade)

    return entry_idx_out, status_out, exit_r_out


def build_hybrid_trades(instrument, backstop_multiplier):
    z = _prep_zones(instrument, OMEGA_PARAMS["min_retracement"])
    times, lows, highs, df = load_instrument(instrument)
    opens = df["open"].values
    rsi14, atr_ratio = _indicator_cache[instrument]

    entry_idx, status, exit_r = sim_hybrid(
        z["top"], z["bottom"], z["mid"], z["is_buy"], z["start_idx"], opens, lows, highs,
        float(OMEGA_PARAMS["n_multiplier"]), float(backstop_multiplier),
        FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE)

    filled = (entry_idx >= 0) & (status != STATUS_UNRESOLVED)
    risk = OMEGA_PARAMS["n_multiplier"] * (z["top"] - z["bottom"]) / 2.0
    risk_pct_of_price = risk / z["mid"]
    entry_times_ns = times[np.where(filled, entry_idx, 0)]
    fill_hours = (entry_times_ns - z["created_at"]) / np.timedelta64(1, "h")

    keep = (filled & (risk_pct_of_price >= OMEGA_PARAMS["min_risk_pct"])
            & (fill_hours >= OMEGA_PARAMS["min_fill_age_hours"]))
    r14 = rsi14[np.where(filled, entry_idx, 0)]
    ar = atr_ratio[np.where(filled, entry_idx, 0)]
    rule_ok = ~np.isnan(r14) & ~np.isnan(ar) & ~((r14 >= 45) & (r14 <= 55)) & ~(ar < 0.7)
    keep = keep & rule_ok

    idx = np.where(keep)[0]
    entry_times = pd.to_datetime(entry_times_ns[idx])
    out = pd.DataFrame({
        "entry_time": entry_times,
        "status": status[idx],
        "r_gross": exit_r[idx],
        "risk_price": risk[idx],
    }).sort_values("entry_time").reset_index(drop=True)
    out["hour"] = out["entry_time"].dt.hour
    return out


def apply_cost(df, spread_price):
    """Only STATUS_STOP_BACKSTOP pays spread -- limit fills (clean, delayed, or target) don't."""
    cost_in_r = spread_price / df["risk_price"]
    pays_cost = df["status"] == STATUS_STOP_BACKSTOP
    return df["r_gross"] - np.where(pays_cost, cost_in_r, 0.0)


def main():
    # risk_price from build_hybrid_trades is already in each instrument's own native price
    # units (n_multiplier * half the zone's raw top-bottom range) -- no pct-of-price
    # conversion needed, spread_price just needs to be in that same native scale.
    configs = [
        ("EURUSD", None, 1.6 * PIP_SIZE),
        ("GBPUSD", "GBPUSD", 1.6 * PIP_SIZE),
        ("EURGBP", "EURGBP", 1.6 * PIP_SIZE),
        ("USDJPY", "USDJPY", 0.015),
    ]
    rows = []
    for instrument, filt_key, spread_price in configs:
        for backstop_mult in [1.5, 2.0]:
            trades = build_hybrid_trades(instrument, backstop_mult)
            if filt_key:
                trades = trades[trades["hour"].isin(SESSION_HOUR_FILTER[filt_key])].copy()
            trades["net_r"] = apply_cost(trades, spread_price)

            score, stats_ = score_trades(trades[["entry_time", "net_r"]])
            n_backstop = (trades["status"] == STATUS_STOP_BACKSTOP).sum()
            n_stop_total = (trades["r_gross"] < 0).sum()
            backstop_rate = n_backstop / n_stop_total if n_stop_total else 0.0
            row = dict(instrument=instrument, backstop_mult=backstop_mult, n_trades=len(trades),
                       backstop_hit_rate=backstop_rate, score=score, **(stats_ or {}))
            rows.append(row)
            print(f"{instrument:8s} backstop={backstop_mult}R  n={len(trades)}  "
                  f"backstop_hit={backstop_rate*100:.2f}% of stops  "
                  f"sharpe={stats_['ann_sharpe']:.2f}  pf={stats_['profit_factor']:.3f}  "
                  f"ann_return={stats_['ann_return']*100:.1f}%  maxdd={stats_['maxdd']*100:.1f}%")
    out = pd.DataFrame(rows)
    out.to_csv(f"{DATA_DIR}/dtfx_omega_hybrid_stop_results.csv", index=False)
    print(f"\nSaved -> {DATA_DIR}/dtfx_omega_hybrid_stop_results.csv")


if __name__ == "__main__":
    main()
