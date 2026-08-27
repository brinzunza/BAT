"""Does Omega's edge -- and the late-UTC-hour finding -- generalize beyond EUR/USD,
GBP/USD, EUR/GBP? Runs the identical zone-construction/simulation pipeline (same
MIN_RETRACEMENT, N_MULTIPLIER, MIN_RISK_PCT, MIN_FILL_AGE_HOURS, RSI/ATR filter -- nothing
re-tuned) against gold, USD/JPY, and BTC/USD, then checks hour-of-day win-rate patterns on
each the same way the GBP-pairs finding was discovered.

Deliberately excludes SPY/AAPL/ES futures from this pass: those trade ~6.5h/day (equities)
or roll contracts (ES futures), which breaks the "continuous wall-clock time" assumption
MIN_FILL_AGE_HOURS and the hour-of-day analysis both depend on -- overnight/weekend gaps in
the data are real closures, not missing data, so a zone spanning a close would look "aged"
by wall-clock hours that were never actually trading hours. That needs separate handling,
not a plug-and-play run through the same pipeline.
"""
import numpy as np
import pandas as pd
from scipy import stats

import dtfx_robust_search as rs
from dtfx_robust_search import score_trades, DATA_DIR

# Extend the module's instrument registry so _prep_zones/load_instrument work unchanged
rs.INSTRUMENTS.update({
    "XAUUSD": "lse_xauusd_full_1min.csv",
    "USDJPY": "lse_usdjpy_1min_2018on.csv",
    "BTCUSD": "lse_btcusd_1min_2018on.csv",
})

OMEGA_PARAMS = dict(min_retracement=2, n_multiplier=3.343496, min_risk_pct=0.000170,
                     min_fill_age_hours=3.366682)
FILL_SEARCH_FAST_BARS = 60 * 24 * 5
MAX_BARS_TO_RESOLVE = 60 * 24 * 30

# Rough, illustrative spread assumptions per instrument, in the SAME price units the
# instrument itself is quoted in (not "pips" -- avoids yen/gold/crypto pip-size confusion).
# Not researched per-broker the way EUR/USD's 1.6 pips was -- treat net-of-cost numbers
# here as directional, not precise.
SPREAD_PRICE_UNITS = {
    "XAUUSD": 0.35,   # ~$0.35 typical spot gold spread
    "USDJPY": 0.015,  # ~1.5 pips (JPY pip = 0.01)
    "BTCUSD": 15.0,   # ~$15 typical spread at current BTC price levels
}


def build_trades_with_hour(instrument):
    z = rs._prep_zones(instrument, OMEGA_PARAMS["min_retracement"])
    times, lows, highs, df = rs.load_instrument(instrument)
    rsi14, atr_ratio = rs._indicator_cache[instrument]

    entry_idx, status = rs._simulate_zones_numba(
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
    out = pd.DataFrame({
        "entry_time": entry_times,
        "r_gross": np.where(status[idx] == 2, 1.0, -1.0),
        "win": status[idx] == 2,
        "risk_price": risk[idx],
    }).sort_values("entry_time").reset_index(drop=True)
    out["hour"] = out["entry_time"].dt.hour
    return out


def analyze(instrument):
    print(f"\n{'='*78}\n{instrument}\n{'='*78}")
    df = build_trades_with_hour(instrument)
    n = len(df)
    if n < 30:
        print(f"only {n} qualifying trades -- too few to say anything, skipping")
        return

    spread = SPREAD_PRICE_UNITS[instrument]
    cost_in_r = spread / df["risk_price"]
    df["net_r"] = df["r_gross"] - np.where(df["r_gross"] < 0, cost_in_r, 0.0)

    gross_frame = df[["entry_time", "r_gross"]].rename(columns={"r_gross": "net_r"})
    gross_score, gross_stats = score_trades(gross_frame)
    net_score, net_stats = score_trades(df[["entry_time", "net_r"]])
    print(f"{n:,} trades, {df['entry_time'].min().date()} -> {df['entry_time'].max().date()}")
    print(f"  GROSS (no cost): win_rate={df['win'].mean():.3f} avg_r={df['r_gross'].mean():.3f} "
          f"score={gross_score if gross_score else 'N/A (too few/degenerate)'}"
          + (f" sharpe={gross_stats['ann_sharpe']:.2f} pf={gross_stats['profit_factor']:.2f} "
             f"ann_return={gross_stats['ann_return']:.2f} maxdd={gross_stats['maxdd']:.2f}" if gross_stats else ""))
    print(f"  NET (illustrative spread={spread}): win_rate={(df['net_r']>0).mean():.3f} avg_r={df['net_r'].mean():.3f} "
          f"score={net_score if net_score else 'N/A (too few/degenerate)'}"
          + (f" sharpe={net_stats['ann_sharpe']:.2f} pf={net_stats['profit_factor']:.2f} "
             f"ann_return={net_stats['ann_return']:.2f} maxdd={net_stats['maxdd']:.2f}" if net_stats else ""))

    print("\n  --- win rate by UTC hour ---")
    hour_table = df.groupby("hour").agg(n=("win", "size"), win_rate=("win", "mean"), avg_r=("r_gross", "mean"))
    print(hour_table.to_string())
    ct = pd.crosstab(df["hour"], df["win"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    print(f"  chi2={chi2:.1f} p={p:.2e}")

    out_path = f"{DATA_DIR}/dtfx_omega_generalize_{instrument}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  saved -> {out_path}")
    return df, p


if __name__ == "__main__":
    results = {}
    for instrument in ["XAUUSD", "USDJPY", "BTCUSD"]:
        results[instrument] = analyze(instrument)
