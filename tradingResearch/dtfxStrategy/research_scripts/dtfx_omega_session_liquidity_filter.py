"""Tests the theory: does restricting Omega trades to high-liquidity trading hours improve
net performance, given that real spread cost -- unlike the flat IBKR-proxy cost used in the
main walk-forward search (IBKR_TYPICAL_PCT in dtfx_robust_search.py) -- actually varies a
lot by session in FX markets?

Two things kept deliberately separate:
1. Is the STRATEGY'S EDGE ITSELF session-dependent? (gross R, before any cost at all --
   a pure "does DTFX work better/worse at certain hours" question, independent of broker.)
2. Does accounting for REALISTIC, session-varying spread cost change the conclusion about
   whether hour-filtering is worth doing, versus the flat-cost assumption the main search
   used (which is close to Interactive Brokers' tight, ~constant all-in cost -- not what
   the live bot actually pays on OANDA, which measured 1.4-1.7 pips just now on EUR/USD).

IMPORTANT CAVEAT, stated once here rather than buried in a comment further down: there is
NO real historical bid/ask spread data available for this. The LSE-vendor OHLC data has no
spread or volume column, and OANDA doesn't expose historical spread as a queryable series
(we only have live current spread snapshots taken during this project). The session-based
spread multipliers below are a MODELED ASSUMPTION grounded in well-documented FX
microstructure -- spreads widen in thin Asian-session and post-NY-close/rollover hours,
tighten during the London/NY overlap -- not measured fact specific to this account. Two
multiplier scenarios (mild, aggressive) are run side by side so the conclusion isn't
hostage to one arbitrary guess.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import (
    _prep_zones, load_instrument, _simulate_zones_numba, _indicator_cache, score_trades, DATA_DIR
)

# Omega's validated, currently-live parameters (see PRODLIVETRADING/strategy/engine.py)
OMEGA_PARAMS = dict(min_retracement=2, n_multiplier=3.343496, min_risk_pct=0.000170,
                     min_fill_age_hours=3.366682)
FILL_SEARCH_FAST_BARS = 60 * 24 * 5
MAX_BARS_TO_RESOLVE = 60 * 24 * 30

PIP_SIZE = 0.0001  # EUR/USD, GBP/USD, EUR/GBP all quoted to 4dp
BASELINE_SPREAD_PIPS = 1.2  # tightest-session baseline (close to OANDA's own quoted "as low as ~1.1-1.2")

SESSION_MULTIPLIERS = {
    "mild":       {"overlap": 1.0, "single": 1.3, "asian": 1.6, "thin": 2.0},
    "aggressive": {"overlap": 1.0, "single": 1.6, "asian": 2.5, "thin": 4.0},
}
INSTRUMENTS = ["EURUSD", "GBPUSD", "EURGBP"]


def session_bucket(hour):
    if 12 <= hour < 16:
        return "overlap"   # London/NY overlap -- highest liquidity
    if 7 <= hour < 12 or 16 <= hour < 21:
        return "single"    # London-only or NY-only
    if 0 <= hour < 7:
        return "asian"
    return "thin"           # 21:00-24:00 UTC -- late NY / rollover, thinnest


def build_trades_with_detail(instrument):
    z = _prep_zones(instrument, OMEGA_PARAMS["min_retracement"])
    times, lows, highs, df = load_instrument(instrument)
    rsi14, atr_ratio = _indicator_cache[instrument]

    entry_idx, status = _simulate_zones_numba(
        z["top"], z["bottom"], z["mid"], z["is_buy"], z["start_idx"],
        lows, highs, float(OMEGA_PARAMS["n_multiplier"]), FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE)

    filled = entry_idx >= 0
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
    r_gross = np.where(status[idx] == 2, 1.0, -1.0)
    out = pd.DataFrame({
        "entry_time": entry_times,
        "r_gross": r_gross,
        "risk_price": risk[idx],
    }).sort_values("entry_time").reset_index(drop=True)
    out["hour"] = out["entry_time"].dt.hour
    out["session"] = out["hour"].apply(session_bucket)
    return out


def apply_spread_cost(trades, scenario):
    mult = trades["session"].map(SESSION_MULTIPLIERS[scenario])
    spread_pips = mult * BASELINE_SPREAD_PIPS
    cost_price = spread_pips * PIP_SIZE
    cost_in_r = cost_price / trades["risk_price"]
    return trades["r_gross"] - cost_in_r


def by_session_table(trades, r_col):
    rows = []
    for session in ["overlap", "single", "asian", "thin"]:
        sub = trades[trades["session"] == session]
        if len(sub) == 0:
            continue
        rows.append(dict(session=session, trades=len(sub),
                          win_rate=(sub[r_col] > 0).mean(), avg_r=sub[r_col].mean(),
                          total_r=sub[r_col].sum()))
    return pd.DataFrame(rows)


def headline_ab(trades, r_col, label):
    """All-hours vs. restricted-to-overlap-only vs. restricted-to-overlap+single (i.e.
    'London or NY, not Asian/thin') -- using the SAME score_trades() the main search uses,
    so these numbers are directly comparable to Omega's published stats."""
    variants = {
        "all_hours": trades,
        "overlap_only": trades[trades["session"] == "overlap"],
        "overlap_or_single": trades[trades["session"].isin(["overlap", "single"])],
    }
    rows = []
    for name, sub in variants.items():
        scored = sub.rename(columns={r_col: "net_r"})[["entry_time", "net_r"]]
        score, stats = score_trades(scored)
        row = dict(variant=name, n_trades=len(sub), score=score)
        row.update(stats or {})
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"\n  [{label}]")
    print(df.to_string(index=False))
    return df


def main():
    all_results = {}
    for instrument in INSTRUMENTS:
        print(f"\n{'='*70}\n{instrument}\n{'='*70}")
        trades = build_trades_with_detail(instrument)
        print(f"  {len(trades):,} Omega-qualifying trades, "
              f"{trades['entry_time'].min()} -> {trades['entry_time'].max()}")

        print("\n  --- GROSS (zero cost) edge by session ---")
        print(by_session_table(trades, "r_gross").to_string(index=False))

        for scenario in ["mild", "aggressive"]:
            trades[f"net_r_{scenario}"] = apply_spread_cost(trades, scenario)
            print(f"\n  --- NET R by session, {scenario} session-spread assumption ---")
            print(by_session_table(trades, f"net_r_{scenario}").to_string(index=False))
            ab = headline_ab(trades, f"net_r_{scenario}", f"{instrument} / {scenario}")
            all_results[(instrument, scenario)] = ab

        out_path = os.path.join(DATA_DIR, f"dtfx_omega_session_liquidity_{instrument}.csv")
        trades.to_csv(out_path, index=False)
        print(f"\n  Saved per-trade detail -> {out_path}")

    summary_rows = []
    for (instrument, scenario), df in all_results.items():
        for _, r in df.iterrows():
            summary_rows.append(dict(instrument=instrument, scenario=scenario, **r))
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(DATA_DIR, "dtfx_omega_session_liquidity_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
