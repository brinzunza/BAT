"""Corrected re-run of compute_combined_comparison_metrics.py (originally written by
Gemini, verified 2026-09-08 -- see dtfx.md). Two real bugs fixed:

1. SPREAD COST: original resolve_trades_grid left entry at raw zone-midpoint (zero spread
   cost) and only shifted the two exits by half_spread -- charging HALF a spread per round
   trip. This project's own established convention (bot.py::_place_trade,
   goal99_dtfx_base_backtest.py's symmetric_exit_spread=True) charges a FULL spread per
   round trip: half on entry, half on each exit. Fixed here to match.
2. MC MARGIN ACCOUNTING: run_mc_fast decremented margin_used[side] on BOTH open and close
   (should increment on open), letting margin capacity drift negative and silently stop
   binding as the sim progresses -- inflating apparent risk capacity and understating time
   to hit the $1M target. Fixed to increment on open, decrement on close (matches the
   already-correct goal79_two_account_portfolio.py::simulate_two_account).

The fabricated "Live Success Probability" / "Primary Live Risk" columns from the original
Gemini table are NOT reproduced here -- they were never computed by any code, just narrated.
In their place: PF under a stressed 2.5-pip spread, a genuine, computed spread-robustness
metric (the original grid_search_*_mc_race.py scripts already computed this in their own
main(), just not surfaced in the combined comparison table).
"""
import os
import sys
import time
import heapq
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import grid_search_optimal_sl_tp_mc_race as g5
import grid_search_10pip_sl_tp_mc_race as g10

RISK_PCT = 0.03
START_PER_SIDE = 100.0
MAX_LEVERAGE = 50.0
ACCOUNT_CAP = 10_000_000.0


@njit(cache=True)
def resolve_trades_grid_correct(
    z_created, z_top, z_bottom, z_is_buy, high, low,
    spread_price, min_stop_price, max_window, min_zone_height,
    sl_mult, tp_mult
):
    """Same walk as the original resolve_trades_grid, but with the FULL-spread convention:
    entry = mid +/- half_spread (against you), both exits = level -/+ half_spread (against
    you) -- a full round trip costs exactly one spread, split across both legs."""
    n_zones = len(z_created)
    n_bars = len(high)

    entry_idx = np.full(n_zones, -1, dtype=np.int64)
    exit_idx = np.full(n_zones, -1, dtype=np.int64)
    outcome = np.zeros(n_zones, dtype=np.int8)
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

        overshot = False
        overshoot_bar = -1
        w_end = min(c_bar + max_window, n_bars)
        for bi in range(c_bar, w_end):
            if is_buy:
                if high[bi] > top:
                    overshot = True
                    overshoot_bar = bi
                    break
            else:
                if low[bi] < bottom:
                    overshot = True
                    overshoot_bar = bi
                    break
        if not overshot:
            continue

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

        w_end3 = min(touch_bar + max_window, n_bars)
        for bi in range(touch_bar, w_end3):
            if is_buy:
                if low[bi] <= stop_p:
                    entry_idx[zi] = touch_bar; exit_idx[zi] = bi; outcome[zi] = 2
                    net_r[zi] = -1.0; entry_price_out[zi] = entry_p; stop_risk_out[zi] = risk
                    break
                elif high[bi] >= target_p:
                    entry_idx[zi] = touch_bar; exit_idx[zi] = bi; outcome[zi] = 1
                    net_r[zi] = win_r_val; entry_price_out[zi] = entry_p; stop_risk_out[zi] = risk
                    break
            else:
                if high[bi] >= stop_p:
                    entry_idx[zi] = touch_bar; exit_idx[zi] = bi; outcome[zi] = 2
                    net_r[zi] = -1.0; entry_price_out[zi] = entry_p; stop_risk_out[zi] = risk
                    break
                elif low[bi] <= target_p:
                    entry_idx[zi] = touch_bar; exit_idx[zi] = bi; outcome[zi] = 1
                    net_r[zi] = win_r_val; entry_price_out[zi] = entry_p; stop_risk_out[zi] = risk
                    break

    return entry_idx, exit_idx, outcome, net_r, entry_price_out, stop_risk_out


def build_trades_correct(cache, min_zone_pips, sl_mult, tp_mult, spread_pips):
    frames = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = spread_pips * pip
        min_stop_price = g5.MIN_STOP_PIPS * pip
        min_zone_height = min_zone_pips * pip

        entry_idx, exit_idx, outcome, net_r, entry_price, stop_risk = resolve_trades_grid_correct(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            spread_price, min_stop_price, g5.MAX_WINDOW, min_zone_height, sl_mult, tp_mult
        )
        resolved = outcome > 0
        ei = entry_idx[resolved]
        xi = exit_idx[resolved]
        frames.append(pd.DataFrame({
            "instrument": inst, "symbol": inst,
            "entry_time": d["times"][ei], "exit_time": d["times"][xi],
            "entry_price": entry_price[resolved], "stop_risk": stop_risk[resolved],
            "net_r": net_r[resolved], "is_buy": d["z_is_buy"][resolved],
            "bars_held": xi - ei,
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def run_mc_fixed(trades, n_sims=300, target=1_000_000.0, seed=42):
    """Same block-bootstrap MC as the original run_mc_fast, with the margin_used sign bug
    fixed: increment on open, decrement on close (matches simulate_two_account)."""
    rng = np.random.default_rng(seed)
    n = len(trades)
    entry_ns = trades.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = trades.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = trades.entry_price.values
    stop_risk = trades.stop_risk.values
    net_r = trades.net_r.values
    instrument = trades.instrument.values
    is_buy = trades.is_buy.values

    BLOCK = 50
    RUIN_FLOOR = 20.0

    days_to_target = []
    max_dds = []

    for sim_i in range(n_sims):
        equity = {True: START_PER_SIDE, False: START_PER_SIDE}
        margin_used = {True: 0.0, False: 0.0}
        open_count = {True: 0, False: 0}
        heap = []
        clock = 0
        peak = START_PER_SIDE * 2.0
        worst_dd = 0.0
        ruined = False
        hit_target = False
        t_hit_days = -1.0
        trades_count = 0

        while trades_count < 50_000 and not hit_target and not ruined:
            start_idx = rng.integers(0, n)
            end_idx = min(start_idx + BLOCK, n)
            offset = clock - entry_ns[start_idx]
            block_exit_max = clock

            for idx in range(start_idx, end_idx):
                et = entry_ns[idx] + offset
                xt = exit_ns[idx] + offset
                if xt > block_exit_max:
                    block_exit_max = xt

                while heap and heap[0][0] <= et:
                    _, r_dollars, margin, r, side = heapq.heappop(heap)
                    equity[side] += r_dollars * r
                    margin_used[side] -= margin
                    open_count[side] -= 1
                    comb = equity[True] + equity[False]
                    if comb > peak:
                        peak = comb
                    dd = 100.0 * (comb - peak) / peak
                    if dd < worst_dd:
                        worst_dd = dd
                    if comb <= RUIN_FLOOR or equity[True] <= 1.0 or equity[False] <= 1.0:
                        ruined = True

                if ruined:
                    break

                side = bool(is_buy[idx])
                sizing_base = min(equity[True], equity[False])
                _, risk_dollars, margin = g5.g79.position_size_and_margin(
                    instrument[idx], sizing_base, entry_price[idx], stop_risk[idx], RISK_PCT,
                    MAX_LEVERAGE, ACCOUNT_CAP)
                avail = min(equity[side], ACCOUNT_CAP) - margin_used[side]

                if margin <= avail and open_count[side] < 1:
                    heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx], side))
                    margin_used[side] += margin  # FIXED: was -= margin
                    open_count[side] += 1
                    trades_count += 1

                comb = equity[True] + equity[False]
                if comb >= target:
                    hit_target = True
                    t_hit_days = et / 86400e9
                    break

            clock = block_exit_max

        if hit_target:
            days_to_target.append(t_hit_days)
        max_dds.append(worst_dd)

    med_days = float(np.median(days_to_target)) if len(days_to_target) else 999.0
    med_dd = float(np.median(max_dds))
    return med_days, med_dd


def stats_for(trades):
    res, eqb, eqs, dd, _ = g5.g79.simulate_two_account(
        trades, risk_pct=RISK_PCT, start_per_side=START_PER_SIDE,
        max_leverage=MAX_LEVERAGE, cap=ACCOUNT_CAP, track_detail=False)
    acc = res[res["accepted"]]
    n = len(acc)
    wr = 100.0 * (acc["net_r"] > 0).mean() if n else 0.0
    avg_win = acc.loc[acc["net_r"] > 0, "net_r"].mean() if (acc["net_r"] > 0).any() else 0.0
    avg_loss = acc.loc[acc["net_r"] <= 0, "net_r"].mean() if (acc["net_r"] <= 0).any() else 0.0
    gw = acc.loc[acc["net_r"] > 0, "net_r"].sum()
    gl = -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    return n, wr, avg_win, avg_loss, pf


def main():
    t0 = time.time()
    cache5 = g5.load_cache()
    cache10 = g10.load_cache()

    all_times = []
    for c in (cache5,):
        for inst, d in c.items():
            all_times.append(d["times"].min()); all_times.append(d["times"].max())
    t_min, t_max = min(all_times), max(all_times)
    total_days = (t_max - t_min).total_seconds() / 86400.0
    print(f"Dataset span: {t_min} to {t_max} ({total_days:.1f} days / {total_days/365.25:.1f} years)")

    configs_5p = [
        (1.0, 1.0, "5-Pip: 1.0SL : 1.0TP"),
        (1.0, 1.25, "5-Pip: 1.0SL : 1.25TP"),
        (1.0, 0.8, "5-Pip: 1.0SL : 0.8TP"),
        (1.25, 1.0, "5-Pip: 1.25SL : 1.0TP"),
        (1.25, 1.25, "5-Pip: 1.25SL : 1.25TP"),
        (1.5, 1.0, "5-Pip: 1.5SL : 1.0TP"),
        (2.0, 1.0, "5-Pip: 2.0SL : 1.0TP"),
    ]
    configs_10p = [
        (1.0, 1.0, "10-Pip: 1.0SL : 1.0TP"),
        (1.0, 0.8, "10-Pip: 1.0SL : 0.8TP"),
        (1.25, 1.0, "10-Pip: 1.25SL : 1.0TP"),
        (1.0, 1.25, "10-Pip: 1.0SL : 1.25TP"),
        (1.25, 0.8, "10-Pip: 1.25SL : 0.8TP"),
        (1.5, 1.0, "10-Pip: 1.5SL : 1.0TP"),
        (2.0, 0.8, "10-Pip: 2.0SL : 0.8TP"),
        (3.0, 0.8, "10-Pip: 3.0SL : 0.8TP"),
    ]

    rows = []
    for min_zone, configs, cache in ((5.0, configs_5p, cache5), (10.0, configs_10p, cache10)):
        for sl, tp, name in configs:
            tc0 = time.time()
            trades_std = build_trades_correct(cache, min_zone, sl, tp, spread_pips=1.6)
            trades_stress = build_trades_correct(cache, min_zone, sl, tp, spread_pips=2.5)

            n_exec, wr, avg_win, avg_loss, pf = stats_for(trades_std)
            _, _, _, _, pf_stress = stats_for(trades_stress)
            tpd = n_exec / (total_days * (5 / 7))
            med_days, med_dd = run_mc_fixed(trades_std, n_sims=300)

            rows.append(dict(name=name, n_exec=n_exec, tpd=tpd, wr=wr, avg_win=avg_win,
                              avg_loss=avg_loss, pf=pf, pf_stress=pf_stress,
                              med_days=med_days, med_dd=med_dd))
            print(f"{name:<25} | Execs: {n_exec:<7} | Trades/Day: {tpd:>5.1f} | WR: {wr:>5.1f}% | "
                  f"AvgWin: +{avg_win:.2f}R | AvgLoss: {avg_loss:.2f}R | PF(1.6p): {pf:>4.2f} | "
                  f"PF(2.5p): {pf_stress:>4.2f} | Days to $1M: {med_days:>6.1f}d | MC DD: {med_dd:>5.1f}%  "
                  f"({time.time()-tc0:.1f}s)")

    pd.DataFrame(rows).to_csv(os.path.join(SCRIPT_DIR, "..", "data",
                                            "combined_comparison_metrics_FIXED.csv"), index=False)
    print(f"\nSaved -> data/combined_comparison_metrics_FIXED.csv")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
