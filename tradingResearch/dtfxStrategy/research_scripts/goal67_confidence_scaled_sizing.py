"""/goal (2026-09-04): "keep researching, developing, and inventing around the factors of
dtfx to find a better strategy."

New mechanism (not a new signal): **confidence-scaled position sizing**. Every candidate
this session sizes every accepted trade at the SAME fixed risk_pct (found once, by grid
search, as whatever single number maximizes CAGR over the whole trade stream). But
meta-labeling already produces a per-trade quality score (meta_prob) that goal64 has so far
only used as a binary filter (take it or don't). The natural next step from meta-labeling's
own literature (López de Prado positions this as BOTH a filter AND a bet-sizing signal) is
to size EACH accepted trade in proportion to its own confidence, not uniformly -- risking
more on the trades the model is most confident about, less on the ones that only barely
cleared the filter.

Implementation: a variable-risk variant of goal30::simulate_two_account (same two-sub-
account, margin, leverage, and one-open-position-per-side mechanics, unchanged) that accepts
a per-trade risk_pct array instead of one scalar, linearly mapping each trade's meta_prob
into a [min_risk, max_risk] band.
"""
import os
import sys
import heapq

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal20_speed_king_margin_report as g20
import goal30_apex_king_deployment as g30
import goal64_meta_labeling as g64

position_size_and_margin = g20.position_size_and_margin


def simulate_two_account_variable_risk(trades, risk_pct_array, start_per_side=g30.START_PER_SIDE,
                                        max_leverage=g30.MAX_LEVERAGE, cap=g30.ACCOUNT_CAP):
    """Identical to goal30::simulate_two_account except risk_pct is looked up per-trade
    (risk_pct_array[i], aligned to the SAME entry_time sort order) instead of one constant
    scalar -- every other mechanic (margin, leverage, cap, one-open-position-per-side,
    two-separate-sub-accounts) is untouched."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    assert len(risk_pct_array) == n, "risk_pct_array must align with trades sorted by entry_time"
    entry_time = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_time = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values
    is_buy = t.is_buy.values

    equity = {True: start_per_side, False: start_per_side}
    margin_used = {True: 0.0, False: 0.0}
    open_count = {True: 0, False: 0}
    heap = []
    peak_combined = start_per_side * 2
    max_dd_pct = 0.0

    for i in range(n):
        et = entry_time[i]
        while heap and heap[0][0] <= et:
            xt, r_dollars, margin, r, side = heapq.heappop(heap)
            equity[side] += r_dollars * r
            margin_used[side] -= margin
            open_count[side] -= 1
            combined = equity[True] + equity[False]
            if combined > peak_combined:
                peak_combined = combined
            dd = 100 * (combined - peak_combined) / peak_combined
            if dd < max_dd_pct:
                max_dd_pct = dd
        margin_used[True] = max(margin_used[True], 0.0)
        margin_used[False] = max(margin_used[False], 0.0)

        side = bool(is_buy[i])
        sizing_base = min(equity[True], equity[False])
        _, risk_dollars, margin = position_size_and_margin(
            instrument[i], sizing_base, entry_price[i], stop_risk[i], risk_pct_array[i],
            max_leverage, cap)
        avail = min(equity[side], cap) - margin_used[side]
        if margin <= avail and open_count[side] < 1:
            heapq.heappush(heap, (exit_time[i], risk_dollars, margin, net_r[i], side))
            margin_used[side] += margin
            open_count[side] += 1

    while heap:
        xt, r_dollars, margin, r, side = heapq.heappop(heap)
        equity[side] += r_dollars * r
        combined = equity[True] + equity[False]
        if combined > peak_combined:
            peak_combined = combined
        dd = 100 * (combined - peak_combined) / peak_combined
        if dd < max_dd_pct:
            max_dd_pct = dd

    return equity[True], equity[False], max_dd_pct


def confidence_scaled_risk(meta_prob, prob_floor, min_risk, max_risk):
    prob_ceil = 1.0
    frac = np.clip((meta_prob - prob_floor) / max(prob_ceil - prob_floor, 1e-9), 0.0, 1.0)
    return min_risk + (max_risk - min_risk) * frac


if __name__ == "__main__":
    import time
    t0 = time.time()
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        res = g64.run_meta_labeling("EURGBP", 35.0, 0.05, 0.05)
    test = res["test"]
    filt = test[test.meta_prob >= 0.85].copy()
    acc = g64.true_accepted(filt, 0.05, "EURGBP")
    if "meta_prob" not in acc.columns:
        acc = acc.merge(filt[["entry_time", "is_buy", "meta_prob"]],
                         on=["entry_time", "is_buy"], how="left")
    print(f"accepted={len(acc)}, meta_prob attached (nulls: {acc.meta_prob.isna().sum()})")
    print(f"meta_prob range among accepted: [{acc.meta_prob.min():.3f}, {acc.meta_prob.max():.3f}], "
          f"std={acc.meta_prob.std():.4f}")

    years = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25

    print("\n=== FIXED risk (baseline, grid search) ===")
    best_fixed = None
    for rp in np.arange(0.02, 0.61, 0.02):
        eq_buy, eq_sell, dd = simulate_two_account_variable_risk(acc, np.full(len(acc), rp))
        if best_fixed is None or (eq_buy + eq_sell) > best_fixed[0]:
            best_fixed = (eq_buy + eq_sell, rp, dd)
    fixed_final, fixed_risk, fixed_dd = best_fixed
    fixed_cagr = (fixed_final / 200.0) ** (1 / years) - 1
    fixed_y1m = np.log(1_000_000 / 200.0) / np.log(1 + fixed_cagr) if fixed_cagr > 0 else float("inf")
    print(f"best fixed risk={fixed_risk:.2f}: final=${fixed_final:,.0f} CAGR={100*fixed_cagr:.2f}% "
          f"years_to_1m={fixed_y1m:.2f} maxDD={fixed_dd:.1f}%")

    print("\n=== CONFIDENCE-SCALED risk (meta_prob -> [min_risk, max_risk]) ===")
    best_scaled = None
    for min_risk in [0.02, 0.05, 0.08]:
        for max_risk in [0.15, 0.20, 0.30, 0.40, 0.50, 0.60]:
            if max_risk <= min_risk:
                continue
            rp_array = confidence_scaled_risk(acc.meta_prob.values, 0.85, min_risk, max_risk)
            eq_buy, eq_sell, dd = simulate_two_account_variable_risk(acc, rp_array)
            final = eq_buy + eq_sell
            cagr = (final / 200.0) ** (1 / years) - 1
            y1m = np.log(1_000_000 / 200.0) / np.log(1 + cagr) if cagr > 0 else float("inf")
            print(f"  min={min_risk:.2f} max={max_risk:.2f}: final=${final:,.0f} "
                  f"CAGR={100*cagr:.2f}% years_to_1m={y1m:.2f} maxDD={dd:.1f}%")
            if best_scaled is None or final > best_scaled[0]:
                best_scaled = (final, min_risk, max_risk, y1m, dd)

    print(f"\nBEST fixed:  years_to_1m={fixed_y1m:.2f} @ risk={fixed_risk:.2f}, maxDD={fixed_dd:.1f}%")
    print(f"BEST scaled: years_to_1m={best_scaled[3]:.2f} @ min={best_scaled[1]:.2f}/"
          f"max={best_scaled[2]:.2f}, maxDD={best_scaled[4]:.1f}%")
    improvement = fixed_y1m - best_scaled[3]
    print(f"improvement: {improvement:.2f} years {'FASTER' if improvement > 0 else 'SLOWER'} "
          f"with confidence-scaled sizing")

    print(f"\n(total: {time.time()-t0:.1f}s)")
