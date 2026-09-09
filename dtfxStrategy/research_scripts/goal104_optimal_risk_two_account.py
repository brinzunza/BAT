"""Direct user request (2026-09-06): compare minimum zone height for variant A (fixed 1:1R)
under the project's ACTUAL two-account live architecture -- two separate $100 sub-accounts
(buy-only, sell-only, never pooled), position sizing = risk_pct * min(buy_equity,
sell_equity), max ONE confirmed open position per side across all four pairs, 50x leverage,
1.6-pip spread charged on every entry -- and find the risk_pct that's actually optimal
under that architecture, not just "5% because that's what's live" (`RISK_PCT` elsewhere in
this project).

Reuses, unmodified: goal95 (zones) + goal99 (overshoot-then-tag entry gate, fixed 1:1R
target, the raw min-zone-height filter added for the previous sweep) for the trade list;
goal79's `simulate_two_account` (deterministic historical replay) and
`run_monte_carlo_two_account` (block-bootstrap MC) for the account mechanics -- both are
pure Python (heapq-based), not numba, so trade counts are kept in check by only testing
zone heights >= 5 pips (this project's own prior finding for where the spread stops eating
the edge) and by capping Monte Carlo scale explicitly below.

Methodology for "optimal risk": for each zone-height, sweep risk_pct through the exact
historical trade sequence via the deterministic `simulate_two_account` (fast, but a single
path -- prone to overfitting the specific historical sequence, the classic problem with
"optimal f"/full-Kelly sizing) and record CAGR + max drawdown. The risk_pct that maximizes
CAGR on that single path is reported as "growth-optimal" (the full-Kelly analog here), but
is not the final recommendation -- full-Kelly sizing is well known to produce drawdowns most
traders can't survive operationally or emotionally. A half-Kelly risk (half the
growth-optimal fraction) is reported alongside as the practical recommendation, then
verified with a proper block-bootstrap Monte Carlo (not just the one historical path) at
both risk levels for the best zone-height candidate.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal95_every_swing_zones as g95
import goal99_dtfx_base_backtest as g99
import goal79_two_account_portfolio as g79

INSTRUMENTS = {
    "EURUSD": ("lse_eurusd_1min_full.csv", 1e-4),
    "GBPUSD": ("lse_gbpusd_1min_full.csv", 1e-4),
    "EURGBP": ("lse_eurgbp_1min_full.csv", 1e-4),
    "USDJPY": ("lse_usdjpy_1min_2018on.csv", 1e-2),
}
SPREAD_PIPS = 1.6
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440
MIN_ZONE_PIPS_LIST = [5, 10, 20, 30, 50]
RISK_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]

START_PER_SIDE = g79.START_PER_SIDE   # $100
MAX_LEVERAGE = g79.MAX_LEVERAGE
ACCOUNT_CAP = g79.ACCOUNT_CAP
RUIN_FLOOR = g79.RUIN_FLOOR


def load_and_build_zones():
    """Load + compute swings/zones ONCE per instrument -- reused across every zone-height
    threshold in the sweep, since only the cheap resolve step below actually depends on it."""
    cache = {}
    for inst, (fname, pip) in INSTRUMENTS.items():
        t1 = time.time()
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path, parse_dates=["date"])
        times = df["date"]
        high_ = df["high"].values.astype(np.float64)
        low_ = df["low"].values.astype(np.float64)
        close_ = df["close"].values.astype(np.float64)
        n_bars = len(df)

        confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high_, low_, close_)
        (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
         ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)
        last_high_of_bar, last_low_of_bar = g99.build_last_confirmed(confirm_idx, is_high, price, n_bars)

        cache[inst] = dict(pip=pip, times=times, high=high_, low=low_,
                            z_created=z_created, z_top=z_top, z_bottom=z_bottom, z_is_buy=z_is_buy,
                            last_high_of_bar=last_high_of_bar, last_low_of_bar=last_low_of_bar)
        print(f"  {inst}: loaded + zones built, {len(z_created):,} zones  ({time.time()-t1:.1f}s)")
    return cache


def build_candidate_trades(cache, min_zone_pips, symmetric_exit_spread=True):
    """symmetric_exit_spread (default True as of 2026-09-07): matches
    PRODLIVETRADING/bot.py's actual marketable-limit order convention -- entry AND both
    exits (stop, target) each pay half the spread, rather than the whole spread landing on
    entry alone. See resolve_base_strategy's own docstring for the full derivation. Pass
    False to reproduce every number in dtfx.md computed before this date."""
    frames = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = min_zone_pips * pip

        entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            d["last_high_of_bar"], d["last_low_of_bar"], spread_price, min_stop_price, MAX_WINDOW,
            False, min_zone_height, symmetric_exit_spread)

        resolved = outcome > 0
        ei = entry_idx[resolved]; xi = exit_idx[resolved]
        is_buy = d["z_is_buy"][resolved]
        raw_stop = np.where(is_buy, d["z_bottom"][resolved], d["z_top"][resolved])
        if symmetric_exit_spread:
            half_spread = spread_price / 2.0
            stop_price = np.where(is_buy, raw_stop - half_spread, raw_stop + half_spread)
        else:
            stop_price = raw_stop
        stop_risk = np.abs(entry_price[resolved] - stop_price)
        frames.append(pd.DataFrame({
            "instrument": inst,
            "entry_time": d["times"].values[ei],
            "exit_time": d["times"].values[xi],
            "entry_price": entry_price[resolved],
            "stop_risk": stop_risk,
            "net_r": net_r[resolved],
            "is_buy": is_buy,
        }))
    trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    return trades


def cagr_from_run(trades_span_years, final_equity, start=START_PER_SIDE * 2):
    if final_equity <= RUIN_FLOOR + 1e-9:
        return -1.0  # total loss
    return (final_equity / start) ** (1.0 / trades_span_years) - 1.0


def main():
    t0 = time.time()
    print("Loading all 4 instruments + building zones once...")
    cache = load_and_build_zones()

    grid_rows = []
    per_height_trades = {}

    for min_pips in MIN_ZONE_PIPS_LIST:
        t1 = time.time()
        trades = build_candidate_trades(cache, min_pips)
        per_height_trades[min_pips] = trades
        span_years = (trades["exit_time"].max() - trades["entry_time"].min()) / np.timedelta64(365, "D")
        print(f"\nmin_zone={min_pips}pips: {len(trades):,} candidate trades, {span_years:.1f}yr span  ({time.time()-t1:.1f}s build)")

        for risk_pct in RISK_GRID:
            t2 = time.time()
            result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
                trades, risk_pct=risk_pct, start_per_side=START_PER_SIDE,
                max_leverage=MAX_LEVERAGE, cap=ACCOUNT_CAP, track_detail=False)
            accepted = int(result["accepted"].sum())
            final_equity = eq_buy + eq_sell
            cagr = cagr_from_run(span_years, final_equity)
            grid_rows.append(dict(min_pips=min_pips, risk_pct=risk_pct, accepted=accepted,
                                   final_equity=final_equity, max_dd_pct=max_dd, cagr=cagr))
            print(f"  risk={risk_pct:>5.0%}  accepted={accepted:>7,}  final=${final_equity:>14,.2f}  "
                  f"maxDD={max_dd:>7.2f}%  CAGR={cagr:>8.2%}  ({time.time()-t2:.1f}s)")

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(os.path.join(DATA_DIR, "goal104_risk_zone_grid.csv"), index=False)

    print(f"\n{'='*70}\nGrowth-optimal risk per zone height (max CAGR on the historical path)\n{'='*70}")
    best_rows = []
    for min_pips, g in grid.groupby("min_pips"):
        best = g.loc[g["cagr"].idxmax()]
        half_kelly_risk = best["risk_pct"] / 2.0
        half_row = g.iloc[(g["risk_pct"] - half_kelly_risk).abs().argsort()[:1]]
        print(f"min_zone={min_pips:>3}pips: growth-optimal risk={best['risk_pct']:.0%} "
              f"(CAGR={best['cagr']:.1%}, maxDD={best['max_dd_pct']:.1f}%, n={best['accepted']:,})  |  "
              f"nearest-to-half-Kelly risk={half_row['risk_pct'].values[0]:.0%} "
              f"(CAGR={half_row['cagr'].values[0]:.1%}, maxDD={half_row['max_dd_pct'].values[0]:.1f}%)")
        best_rows.append(dict(min_pips=min_pips, growth_optimal_risk=best["risk_pct"],
                               growth_optimal_cagr=best["cagr"], growth_optimal_dd=best["max_dd_pct"],
                               growth_optimal_n=best["accepted"]))

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(os.path.join(DATA_DIR, "goal104_growth_optimal_by_height.csv"), index=False)
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")

    return grid, per_height_trades


if __name__ == "__main__":
    main()
