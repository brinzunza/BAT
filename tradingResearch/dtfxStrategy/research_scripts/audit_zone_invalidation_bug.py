"""Audit of the Zone Invalidation & False Swing Discovery:
Investigating:
1. Missing Invalidation Gate during Overshoot Phase:
   - For BUY zone: If price breaches below bottom before overshooting top, the zone is already dead.
   - For SELL zone: If price breaches above top before overshooting bottom, the zone is already dead.
2. Impact of Strict Pre-Overshoot Invalidation on Strategy Win Rate, PF, and Cleanliness.
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal95_every_swing_zones as g95
import goal79_two_account_portfolio as g79
import audit_and_retest_full_spread_and_fixed_mc as audited

@njit(cache=True)
def resolve_trades_strict_invalidation(
    z_created, z_top, z_bottom, z_is_buy, high, low,
    spread_price, min_stop_price, max_window, min_zone_height,
    sl_mult, tp_mult
):
    n_zones = len(z_created)
    n_bars = len(high)

    entry_idx = np.full(n_zones, -1, dtype=np.int64)
    exit_idx = np.full(n_zones, -1, dtype=np.int64)
    outcome = np.zeros(n_zones, dtype=np.int8)  # 0: unres, 1: win, 2: loss
    net_r = np.zeros(n_zones, dtype=np.float64)
    entry_price_out = np.zeros(n_zones, dtype=np.float64)
    stop_risk_out = np.zeros(n_zones, dtype=np.float64)

    half_spread = spread_price / 2.0

    for zi in range(n_zones):
        top = z_top[zi]
        bottom = z_bottom[zi]
        height = top - bottom
        if height < min_zone_height:
            continue

        h = height / 2.0
        mid = (top + bottom) / 2.0
        c_bar = z_created[zi]
        is_buy = z_is_buy[zi]

        # 1. STRICT Overshoot search: INVALIDATE if price breaches opposite edge before overshooting!
        overshot = False
        overshoot_bar = -1
        w_end = min(c_bar + max_window, n_bars)

        for bi in range(c_bar, w_end):
            if is_buy:
                # If price drops below bottom before overshooting top -> ZONE IS INVALIDATED
                if low[bi] <= bottom:
                    break
                if high[bi] > top:
                    overshot = True
                    overshoot_bar = bi
                    break
            else:
                # If price rises above top before overshooting bottom -> ZONE IS INVALIDATED
                if high[bi] >= top:
                    break
                if low[bi] < bottom:
                    overshot = True
                    overshoot_bar = bi
                    break

        if not overshot:
            continue

        # 2. Wait for 50% midpoint touch
        touched = False
        touch_bar = -1
        w_end2 = min(overshoot_bar + max_window, n_bars)

        for bi in range(overshoot_bar, w_end2):
            if is_buy:
                if low[bi] <= bottom:
                    break
                if low[bi] <= mid:
                    touched = True
                    touch_bar = bi
                    break
            else:
                if high[bi] >= top:
                    break
                if high[bi] >= mid:
                    touched = True
                    touch_bar = bi
                    break

        if not touched:
            continue

        # 3. Setup Price Levels with FULL SYMMETRIC SPREAD
        if is_buy:
            entry_p = mid + half_spread
            raw_stop = mid - sl_mult * h
            stop_p = raw_stop - half_spread
            raw_target = mid + tp_mult * h
            target_p = raw_target - half_spread
            risk = entry_p - stop_p
            if risk < min_stop_price:
                continue
            win_r_val = (target_p - entry_p) / risk if risk > 0 else 0.0
        else:
            entry_p = mid - half_spread
            raw_stop = mid + sl_mult * h
            stop_p = raw_stop + half_spread
            raw_target = mid - tp_mult * h
            target_p = raw_target + half_spread
            risk = stop_p - entry_p
            if risk < min_stop_price:
                continue
            win_r_val = (entry_p - target_p) / risk if risk > 0 else 0.0

        # 4. Resolve Exit
        w_end3 = min(touch_bar + max_window, n_bars)
        for bi in range(touch_bar, w_end3):
            if is_buy:
                if low[bi] <= stop_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 2
                    net_r[zi] = -1.0
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
                elif high[bi] >= target_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 1
                    net_r[zi] = win_r_val
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
            else:
                if high[bi] >= stop_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 2
                    net_r[zi] = -1.0
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break
                elif low[bi] <= target_p:
                    entry_idx[zi] = touch_bar
                    exit_idx[zi] = bi
                    outcome[zi] = 1
                    net_r[zi] = win_r_val
                    entry_price_out[zi] = entry_p
                    stop_risk_out[zi] = risk
                    break

    return entry_idx, exit_idx, outcome, net_r, entry_price_out, stop_risk_out


def main():
    cache = audited.load_cache()
    all_times = [d["times"].min() for d in cache.values()] + [d["times"].max() for d in cache.values()]
    total_days = (max(all_times) - min(all_times)).total_seconds() / 86400.0
    trading_days = total_days * (5.0 / 7.0)

    min_pips = {"USDJPY": 10.0, "EURUSD": 10.0, "GBPUSD": 10.0, "EURGBP": 10.0}

    print("==========================================================================")
    print("  AUDITING PRE-OVERSHOOT INVALIDATION GATE (10-PIP ALL PAIRS)")
    print("==========================================================================")

    # 1. Un-gated baseline (the flawed version)
    trades_flawed = audited.build_trades(cache, 1.0, 0.8, min_pips, spread_pips=1.6)
    res_f, _, _, _, _ = g79.simulate_two_account(trades_flawed, risk_pct=0.03, start_per_side=100.0, max_leverage=50.0, cap=10_000_000.0)
    acc_f = res_f[res_f["accepted"]]
    wr_f = 100.0 * (acc_f["net_r"] > 0).mean()
    gw_f = acc_f.loc[acc_f["net_r"] > 0, "net_r"].sum()
    gl_f = -acc_f.loc[acc_f["net_r"] <= 0, "net_r"].sum()
    pf_f = gw_f / gl_f if gl_f > 0 else 0.0

    print(f"\n1. FLAVORED BASELINE (No pre-overshoot invalidation):")
    print(f"   Execs: {len(acc_f):,} | Trades/Day: {len(acc_f)/trading_days:.1f} | WR: {wr_f:.2f}% | PF: {pf_f:.2f}")

    # 2. Strict Invalidation
    frames_strict = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = 1.6 * pip
        min_stop_price = 3.0 * pip
        min_zone_height = min_pips[inst] * pip

        ei, xi, outcome, net_r, entry_p, stop_r = resolve_trades_strict_invalidation(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            spread_price, min_stop_price, 1440, min_zone_height, 1.0, 0.8
        )
        resolved = outcome > 0
        frames_strict.append(pd.DataFrame({
            "instrument": inst, "symbol": inst,
            "entry_time": d["times"][ei[resolved]], "exit_time": d["times"][xi[resolved]],
            "entry_price": entry_p[resolved], "stop_risk": stop_r[resolved],
            "net_r": net_r[resolved], "is_buy": d["z_is_buy"][resolved]
        }))

    trades_strict = pd.concat(frames_strict, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    res_s, _, _, _, _ = g79.simulate_two_account(trades_strict, risk_pct=0.03, start_per_side=100.0, max_leverage=50.0, cap=10_000_000.0)
    acc_s = res_s[res_s["accepted"]]
    wr_s = 100.0 * (acc_s["net_r"] > 0).mean()
    gw_s = acc_s.loc[acc_s["net_r"] > 0, "net_r"].sum()
    gl_s = -acc_s.loc[acc_s["net_r"] <= 0, "net_r"].sum()
    pf_s = gw_s / gl_s if gl_s > 0 else 0.0
    med_days_s, med_dd_s = audited.run_mc_fixed(trades_strict, n_sims=300)

    print(f"\n2. STRICT PRE-OVERSHOOT INVALIDATION (Zones that blow through opposite edge killed):")
    print(f"   Execs: {len(acc_s):,} | Trades/Day: {len(acc_s)/trading_days:.1f} | WR: {wr_s:.2f}% | PF: {pf_s:.2f} | MC $1M: {med_days_s:.1f}d | MC DD: {med_dd_s:.1f}%")

if __name__ == "__main__":
    main()
