"""Two follow-ups to dtfx_omega_session_liquidity_filter.py:

1. ASYMMETRIC COST MODEL. Entries and take-profit exits are placed as LIMIT orders (you're
   the passive/resting side -- fill at your price or better, no spread crossed). Stop-losses
   fire as MARKET orders once triggered (you're the aggressor -- you DO cross the spread,
   and can slip past it in a fast market). The session-liquidity backtest charged spread
   cost uniformly to every trade regardless of outcome, which overstates cost on winners and
   may understate it on losers. This re-checks the same trade population with cost applied
   ONLY to stop-out trades.

2. STOP-WIDTH SWEEP. cost_in_r = spread_price / stop_distance_price -- for a fixed spread,
   the only strategy-side lever that shrinks cost's share of R is a wider stop. Sweeps
   MIN_RISK_PCT (the live engine's existing "skip zones with too-tight a stop" filter) to
   find where cost genuinely becomes small relative to the edge, and reports what that costs
   in trade count and edge retention -- there's no free lunch, a wider stop keeps fewer,
   different zones.

Same caveat as before: no real historical spread data exists. Two flat spread levels are
used here (1.2 pips = OANDA's marketed "as low as" figure, 1.6 pips = this account's actual
live-measured EUR/USD spread, taken during this project) -- deliberately NOT session-varying
this time, to isolate the stop-width and cost-asymmetry effects cleanly from the
already-established session effect.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import (
    _prep_zones, load_instrument, _simulate_zones_numba, _indicator_cache, score_trades, DATA_DIR
)

BASE_PARAMS = dict(min_retracement=2, n_multiplier=3.343496, min_fill_age_hours=3.366682)
LIVE_MIN_RISK_PCT = 0.000170
FILL_SEARCH_FAST_BARS = 60 * 24 * 5
MAX_BARS_TO_RESOLVE = 60 * 24 * 30
PIP_SIZE = 0.0001
SPREAD_LEVELS_PIPS = [1.2, 1.6]
RISK_PCT_SWEEP = [0.000170, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035]
INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP"]


def raw_trades(instrument, min_risk_pct):
    """status kept this time (1=stopped_out, 2=take_profit) -- needed for the asymmetric
    cost model, unlike the session script which only needed win/loss (r_gross sign)."""
    z = _prep_zones(instrument, BASE_PARAMS["min_retracement"])
    times, lows, highs, df = load_instrument(instrument)
    rsi14, atr_ratio = _indicator_cache[instrument]

    entry_idx, status = _simulate_zones_numba(
        z["top"], z["bottom"], z["mid"], z["is_buy"], z["start_idx"],
        lows, highs, float(BASE_PARAMS["n_multiplier"]), FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE)

    filled = entry_idx >= 0
    risk = BASE_PARAMS["n_multiplier"] * (z["top"] - z["bottom"]) / 2.0
    risk_pct_of_price = risk / z["mid"]
    entry_times_ns = times[np.where(filled, entry_idx, 0)]
    fill_hours = (entry_times_ns - z["created_at"]) / np.timedelta64(1, "h")

    keep = (filled & (risk_pct_of_price >= min_risk_pct)
            & (fill_hours >= BASE_PARAMS["min_fill_age_hours"]))
    r14 = rsi14[np.where(filled, entry_idx, 0)]
    ar = atr_ratio[np.where(filled, entry_idx, 0)]
    rule_ok = ~np.isnan(r14) & ~np.isnan(ar) & ~((r14 >= 45) & (r14 <= 55)) & ~(ar < 0.7)
    keep = keep & rule_ok

    idx = np.where(keep)[0]
    return pd.DataFrame({
        "entry_time": pd.to_datetime(entry_times_ns[idx]),
        "r_gross": np.where(status[idx] == 2, 1.0, -1.0),
        "is_stop": status[idx] == 1,
        "risk_price": risk[idx],
    }).sort_values("entry_time").reset_index(drop=True)


def score(trades, r_col):
    scored = trades.rename(columns={r_col: "net_r"})[["entry_time", "net_r"]]
    return score_trades(scored)


def part1_asymmetric_cost():
    print("\n" + "=" * 78)
    print("PART 1 -- cost only on stop-outs (limit entries/targets don't cross the spread)")
    print("=" * 78)
    rows = []
    for instrument in INSTRUMENTS:
        trades = raw_trades(instrument, LIVE_MIN_RISK_PCT)
        for spread_pips in SPREAD_LEVELS_PIPS:
            cost_price = spread_pips * PIP_SIZE
            cost_in_r = cost_price / trades["risk_price"]
            trades["net_r_uniform"] = trades["r_gross"] - cost_in_r
            trades["net_r_asymmetric"] = trades["r_gross"] - np.where(trades["is_stop"], cost_in_r, 0.0)

            s_uni, st_uni = score(trades, "net_r_uniform")
            s_asym, st_asym = score(trades, "net_r_asymmetric")
            rows.append(dict(instrument=instrument, spread_pips=spread_pips, model="uniform (every trade)",
                              score=s_uni, **(st_uni or {})))
            rows.append(dict(instrument=instrument, spread_pips=spread_pips, model="asymmetric (stops only)",
                              score=s_asym, **(st_asym or {})))
    df = pd.DataFrame(rows)
    cols = ["instrument", "spread_pips", "model", "score", "ann_return", "maxdd", "calmar",
            "ann_sharpe", "profit_factor", "win_rate", "avg_r", "trades"]
    print(df[cols].to_string(index=False))
    df.to_csv(os.path.join(DATA_DIR, "dtfx_omega_asymmetric_cost.csv"), index=False)
    return df


def part2_stop_width_sweep():
    print("\n" + "=" * 78)
    print("PART 2 -- MIN_RISK_PCT sweep (wider stops shrink cost's share of R), asymmetric cost model")
    print("=" * 78)
    rows = []
    for instrument in ["EURUSD", "GBPUSD"]:  # EUR/GBP already established as not worth rescuing
        for min_risk_pct in RISK_PCT_SWEEP:
            trades = raw_trades(instrument, min_risk_pct)
            if len(trades) == 0:
                continue
            for spread_pips in SPREAD_LEVELS_PIPS:
                cost_price = spread_pips * PIP_SIZE
                cost_in_r = cost_price / trades["risk_price"]
                trades["net_r"] = trades["r_gross"] - np.where(trades["is_stop"], cost_in_r, 0.0)
                avg_stop_cost_in_r = cost_in_r[trades["is_stop"]].mean() if trades["is_stop"].any() else np.nan
                s, st = score(trades, "net_r")
                rows.append(dict(instrument=instrument, min_risk_pct=min_risk_pct, spread_pips=spread_pips,
                                  n_trades=len(trades), avg_stop_cost_in_r=avg_stop_cost_in_r,
                                  score=s, **(st or {})))
    df = pd.DataFrame(rows)
    cols = ["instrument", "min_risk_pct", "spread_pips", "n_trades", "avg_stop_cost_in_r",
            "score", "ann_return", "maxdd", "calmar", "ann_sharpe", "profit_factor", "win_rate", "avg_r"]
    print(df[cols].to_string(index=False))
    df.to_csv(os.path.join(DATA_DIR, "dtfx_omega_stop_width_sweep.csv"), index=False)
    return df


if __name__ == "__main__":
    part1_asymmetric_cost()
    part2_stop_width_sweep()
