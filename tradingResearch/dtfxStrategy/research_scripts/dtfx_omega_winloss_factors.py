"""What actually predicts a winning vs. losing Omega trade?

Builds a rich per-trade feature table (time-of-day, day-of-week, day-of-month/NFP-proxy,
month-of-year, retracement geometry, fill delay, RSI/ATR at entry, side, streak/prior
outcome) and checks each against win/loss and net_r, ranked by effect size and sample size
-- not just p-value, since with ~9,600 trades almost anything clears significance; what
matters is whether the split is big enough and the subgroup large enough to trade on.

Deliberately does NOT include macro calendar events (FOMC, CPI) requiring an external
economic-calendar dataset that isn't available locally -- NFP is approximated via
"first Friday of the month" (its actual US release schedule), which IS derivable from the
timestamp alone.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from dtfx_robust_search import (
    _prep_zones, load_instrument, _simulate_zones_numba, _indicator_cache, DATA_DIR
)

BASE_PARAMS = dict(min_retracement=2, n_multiplier=3.343496, min_risk_pct=0.000170,
                    min_fill_age_hours=3.366682)
FILL_SEARCH_FAST_BARS = 60 * 24 * 5
MAX_BARS_TO_RESOLVE = 60 * 24 * 30


def build_feature_trades(instrument):
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

    keep = (filled & (risk_pct_of_price >= BASE_PARAMS["min_risk_pct"])
            & (fill_hours >= BASE_PARAMS["min_fill_age_hours"]))
    r14 = rsi14[np.where(filled, entry_idx, 0)]
    ar = atr_ratio[np.where(filled, entry_idx, 0)]
    rule_ok = ~np.isnan(r14) & ~np.isnan(ar) & ~((r14 >= 45) & (r14 <= 55)) & ~(ar < 0.7)
    keep = keep & rule_ok

    idx = np.where(keep)[0]
    entry_times = pd.to_datetime(entry_times_ns[idx])
    zone_range = (z["top"][idx] - z["bottom"][idx])
    out = pd.DataFrame({
        "entry_time": entry_times,
        "r_gross": np.where(status[idx] == 2, 1.0, -1.0),
        "win": status[idx] == 2,
        "side": np.where(z["is_buy"][idx], "buy", "sell"),
        "risk_pct_of_price": risk_pct_of_price[idx],
        "zone_range_pips": zone_range / 0.0001,
        "fill_hours": fill_hours[idx],
        "rsi14": r14[idx],
        "atr_ratio": ar[idx],
    }).sort_values("entry_time").reset_index(drop=True)

    out["hour"] = out["entry_time"].dt.hour
    out["dow"] = out["entry_time"].dt.day_name()
    out["dow_num"] = out["entry_time"].dt.dayofweek
    out["month"] = out["entry_time"].dt.month
    out["day_of_month"] = out["entry_time"].dt.day
    out["year"] = out["entry_time"].dt.year

    # NFP proxy: first Friday of the month (actual US NFP release schedule), entry within
    # +/- 1 day since a zone can form/fill slightly before or after the exact release bar.
    def is_nfp_week(row):
        d = row["entry_time"]
        first_of_month = d.replace(day=1)
        first_friday = first_of_month + pd.Timedelta(days=(4 - first_of_month.dayofweek) % 7)
        return abs((d.normalize() - first_friday).days) <= 1
    out["nfp_proxy"] = out.apply(is_nfp_week, axis=1)

    out["month_end_proxy"] = out["entry_time"].dt.is_month_end | (out["day_of_month"] >= 28)
    out["prior_win"] = out["win"].shift(1)
    return out


def win_rate_by_bucket(df, col, bins=None, labels=None):
    if bins is not None:
        grouped = pd.cut(df[col], bins=bins, labels=labels)
    else:
        grouped = df[col]
    g = df.groupby(grouped, observed=True)
    table = g.agg(n=("win", "size"), win_rate=("win", "mean"), avg_r=("r_gross", "mean"))
    return table


def chi2_test(df, col):
    ct = pd.crosstab(df[col], df["win"])
    if ct.shape[0] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return chi2, p


def point_biserial(df, col):
    sub = df.dropna(subset=[col])
    r, p = stats.pointbiserialr(sub["win"].astype(int), sub[col])
    return r, p


def analyze(instrument):
    print(f"\n{'='*78}\n{instrument}\n{'='*78}")
    df = build_feature_trades(instrument)
    print(f"{len(df):,} trades, overall win_rate={df['win'].mean():.4f}, avg_r={df['r_gross'].mean():.4f}")

    print("\n--- Hour of day ---")
    print(win_rate_by_bucket(df, "hour").to_string())
    chi2, p = chi2_test(df, "hour")
    print(f"chi2={chi2:.1f} p={p:.2e}")

    print("\n--- Day of week ---")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_table = win_rate_by_bucket(df, "dow").reindex(order).dropna(how="all")
    print(dow_table.to_string())
    chi2, p = chi2_test(df, "dow")
    print(f"chi2={chi2:.1f} p={p:.2e}")

    print("\n--- Month of year ---")
    print(win_rate_by_bucket(df, "month").to_string())
    chi2, p = chi2_test(df, "month")
    print(f"chi2={chi2:.1f} p={p:.2e}")

    print("\n--- Side (buy vs sell) ---")
    print(win_rate_by_bucket(df, "side").to_string())
    chi2, p = chi2_test(df, "side")
    print(f"chi2={chi2:.1f} p={p:.2e}")

    print("\n--- NFP-week proxy (first Friday of month +/- 1 day) ---")
    print(win_rate_by_bucket(df, "nfp_proxy").to_string())
    chi2, p = chi2_test(df, "nfp_proxy")
    print(f"chi2={chi2:.1f} p={p:.2e}")

    print("\n--- Month-end proxy (day >= 28) ---")
    print(win_rate_by_bucket(df, "month_end_proxy").to_string())
    chi2, p = chi2_test(df, "month_end_proxy")
    print(f"chi2={chi2:.1f} p={p:.2e}")

    print("\n--- Prior trade outcome (streak/autocorrelation) ---")
    prior_table = win_rate_by_bucket(df.dropna(subset=["prior_win"]), "prior_win")
    print(prior_table.to_string())
    chi2, p = chi2_test(df.dropna(subset=["prior_win"]), "prior_win")
    print(f"chi2={chi2:.1f} p={p:.2e}")

    print("\n--- Zone retracement range (pips) deciles ---")
    df["range_decile"] = pd.qcut(df["zone_range_pips"], 10, labels=False, duplicates="drop")
    print(win_rate_by_bucket(df, "range_decile").to_string())
    r, p = point_biserial(df, "zone_range_pips")
    print(f"point-biserial r={r:.4f} p={p:.2e}")

    print("\n--- Fill delay (hours from zone creation to fill) deciles ---")
    df["fill_hours_decile"] = pd.qcut(df["fill_hours"], 10, labels=False, duplicates="drop")
    print(win_rate_by_bucket(df, "fill_hours_decile").to_string())
    r, p = point_biserial(df, "fill_hours")
    print(f"point-biserial r={r:.4f} p={p:.2e}")

    print("\n--- RSI14 at entry deciles ---")
    df["rsi_decile"] = pd.qcut(df["rsi14"], 10, labels=False, duplicates="drop")
    print(win_rate_by_bucket(df, "rsi_decile").to_string())
    r, p = point_biserial(df, "rsi14")
    print(f"point-biserial r={r:.4f} p={p:.2e}")

    print("\n--- ATR ratio at entry deciles (recent vol vs long-run avg) ---")
    df["atr_decile"] = pd.qcut(df["atr_ratio"], 10, labels=False, duplicates="drop")
    print(win_rate_by_bucket(df, "atr_decile").to_string())
    r, p = point_biserial(df, "atr_ratio")
    print(f"point-biserial r={r:.4f} p={p:.2e}")

    print("\n--- Risk % of price deciles (stop distance) ---")
    df["risk_decile"] = pd.qcut(df["risk_pct_of_price"], 10, labels=False, duplicates="drop")
    print(win_rate_by_bucket(df, "risk_decile").to_string())
    r, p = point_biserial(df, "risk_pct_of_price")
    print(f"point-biserial r={r:.4f} p={p:.2e}")

    out_path = os.path.join(DATA_DIR, f"dtfx_omega_winloss_factors_{instrument}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved per-trade feature table -> {out_path}")
    return df


if __name__ == "__main__":
    for instrument in ["EURUSD", "GBPUSD", "EURGBP"]:
        analyze(instrument)
