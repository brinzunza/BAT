"""SOMEGA pre-2020 backtest: start of available data → 2019-12-31.

Applies the exact Omega parameters + SOMEGA session-hour filter used in
PRODLIVETRADING, but restricted to historical data up to and including
2019-12-31. This tests the strategy's performance BEFORE the post-2021
macro regime that drove the known edge -- the key question being: was
SOMEGA's session filter able to find *any* edge even in the low-rate,
range-bound pre-2020 environment, or does it share the same regime
dependence as baseline Omega?

Instruments and data files:
  EURUSD: lse_eurusd_1min_full.csv  (from 2003/2004)
  GBPUSD: lse_gbpusd_1min_full.csv  (from 2003)
  EURGBP: lse_eurgbp_1min_full.csv  (from 2003)
  USDJPY: lse_usdjpy_1min_2018on.csv (from 2018 only -- limited window pre-2020)

No spread cost model applied here: results are gross R, matching how the
main CHECKPOINT_NOTES.md reports the validated-filter comparison table.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

# --- resolve paths -----------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_robust_search import (
    _prep_zones, load_instrument, _simulate_zones_numba,
    _indicator_cache, _price_cache, _zone_cache,
    ANALYSIS_START, FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE,
)
from dtfx_somega_params import SESSION_HOUR_FILTER

# --- Omega parameters (identical to PRODLIVETRADING/strategy/engine.py) ------
MIN_RETRACEMENT  = 2
N_MULTIPLIER     = 3.343496
MIN_RISK_PCT     = 0.000170
MIN_FILL_AGE_H   = 3.366682   # hours
USE_RSI_ATR      = False      # dropped in the final Omega/SOMEGA version

# --- Backtest window ---------------------------------------------------------
END_DATE = pd.Timestamp("2019-12-31 23:59:59")

INSTRUMENTS = {
    "EURUSD": "lse_eurusd_1min_full.csv",
    "GBPUSD": "lse_gbpusd_1min_full.csv",
    "EURGBP": "lse_eurgbp_1min_full.csv",
    "USDJPY": "lse_usdjpy_1min_2018on.csv",
}

RISK_PCT   = 0.01   # 1% per trade for equity curve (matches CHECKPOINT_NOTES convention)
START_EQ   = 100.0


def load_instrument_custom(name, filename):
    """Load instrument from DATA_DIR using the custom filename mapping,
    bypassing the INSTRUMENTS dict in dtfx_robust_search (which only knows
    the three core FX pairs)."""
    if name in _price_cache:
        return _price_cache[name]
    path = os.path.join(DATA_DIR, filename)
    print(f"Loading {name} from {os.path.basename(path)} ...", flush=True)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[df.index >= ANALYSIS_START]
    df = df[["open", "high", "low", "close"]]

    from dtfx_robust_search import _rsi, _atr_ratio
    _indicator_cache[name] = (_rsi(df["close"]).values, _atr_ratio(df).values)
    times  = df.index.values
    lows   = np.ascontiguousarray(df["low"].values,  dtype=np.float64)
    highs  = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    _price_cache[name] = (times, lows, highs, df)
    print(f"  {len(df):,} bars  {df.index.min().date()} → {df.index.max().date()}", flush=True)
    return _price_cache[name]


def _prep_zones_custom(name, filename):
    """Zone prep using the custom loader."""
    key = (name, MIN_RETRACEMENT)
    if key in _zone_cache:
        return _zone_cache[key]
    load_instrument_custom(name, filename)   # populates _price_cache / _indicator_cache
    times, lows, highs, df = _price_cache[name]

    from dtfx_eurusd_backtest_strict import find_pivots_strict
    from dtfx_eurusd_backtest import run_structure_fsm
    zigzag = find_pivots_strict(df, MIN_RETRACEMENT)
    _, zones = run_structure_fsm(zigzag)

    n       = len(zones)
    top     = np.array([z["top"]    for z in zones], dtype=np.float64)
    bottom  = np.array([z["bottom"] for z in zones], dtype=np.float64)
    mid     = np.array([z["mid"]    for z in zones], dtype=np.float64)
    is_buy  = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
    created = np.array([np.datetime64(z["created_at"]) for z in zones])
    start_idx = np.searchsorted(times, created, side="right").astype(np.int64)

    packed = dict(top=top, bottom=bottom, mid=mid, is_buy=is_buy,
                  start_idx=start_idx, created_at=created)
    _zone_cache[key] = packed
    print(f"  {name}: {n:,} total zones across full history", flush=True)
    return packed


def build_somega_trades(name, filename):
    """Build filtered trade table for one instrument, SOMEGA-filtered,
    restricted to entries on or before END_DATE."""
    z = _prep_zones_custom(name, filename)
    times, lows, highs, df = _price_cache[name]
    rsi14, atr_ratio = _indicator_cache[name]

    entry_idx, status = _simulate_zones_numba(
        z["top"], z["bottom"], z["mid"], z["is_buy"], z["start_idx"],
        lows, highs, float(N_MULTIPLIER), FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE)

    filled = entry_idx >= 0
    if not filled.any():
        return pd.DataFrame(columns=["entry_time", "r_gross", "win", "side"])

    risk             = N_MULTIPLIER * (z["top"] - z["bottom"]) / 2.0
    risk_pct         = risk / z["mid"]
    entry_times_ns   = times[np.where(filled, entry_idx, 0)]
    fill_hours       = (entry_times_ns - z["created_at"]) / np.timedelta64(1, "h")

    keep = (filled
            & (risk_pct >= MIN_RISK_PCT)
            & (fill_hours >= MIN_FILL_AGE_H))

    # SOMEGA session-hour filter
    allowed_hours = SESSION_HOUR_FILTER.get(name)   # None = no filter (EURUSD, BTCUSD)
    if allowed_hours is not None:
        entry_hours = pd.to_datetime(entry_times_ns).hour
        hour_ok = np.isin(entry_hours, list(allowed_hours))
        keep = keep & hour_ok

    # Restrict to pre-2020 window
    end_ns = np.datetime64(END_DATE)
    keep = keep & (entry_times_ns <= end_ns)

    if not keep.any():
        return pd.DataFrame(columns=["entry_time", "r_gross", "win", "side"])

    idx = np.where(keep)[0]
    entry_times = pd.to_datetime(entry_times_ns[idx])
    r_gross = np.where(status[idx] == 2, 1.0, -1.0)
    sides   = np.where(z["is_buy"][idx], "buy", "sell")

    return pd.DataFrame({
        "entry_time": entry_times,
        "r_gross":    r_gross,
        "win":        status[idx] == 2,
        "side":       sides,
    }).sort_values("entry_time").reset_index(drop=True)


def compute_stats(trades, label):
    """Full stats suite matching the CHECKPOINT_NOTES reporting format."""
    if len(trades) == 0:
        print(f"\n{'='*60}\n{label}\n  *** NO TRADES in window ***\n")
        return {}

    n        = len(trades)
    wins     = trades["win"].sum()
    wr       = wins / n
    avg_win  = trades.loc[trades["r_gross"] > 0, "r_gross"].mean()
    avg_loss = trades.loc[trades["r_gross"] < 0, "r_gross"].mean()
    sum_r    = trades["r_gross"].sum()
    avg_r    = trades["r_gross"].mean()

    gross_win  = trades.loc[trades["r_gross"] > 0, "r_gross"].sum()
    gross_loss = -trades.loc[trades["r_gross"] <= 0, "r_gross"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Equity curve at 1% risk/trade
    equity = START_EQ
    curve_v, curve_t = [], []
    for _, row in trades.iterrows():
        equity = max(equity * (1 + RISK_PCT * row["r_gross"]), 0.0)
        curve_v.append(equity)
        curve_t.append(row["entry_time"])

    curve = pd.Series(curve_v, index=pd.to_datetime(curve_t))
    daily = curve.resample("D").last().ffill()
    ret   = daily.pct_change().dropna()
    ret_nz = ret[ret != 0]

    n_years = (daily.index[-1] - daily.index[0]).days / 365.25
    total_ret = curve.iloc[-1] / START_EQ - 1
    ann_ret   = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1 if n_years > 0 else 0.0
    dd        = (daily / daily.cummax() - 1.0)
    maxdd     = abs(dd.min())
    calmar    = ann_ret / maxdd if maxdd > 0 else 0.0
    sharpe    = (ret_nz.mean() / ret_nz.std() * np.sqrt(252)
                 if len(ret_nz) > 1 and ret_nz.std() > 0 else 0.0)
    skewness  = float(scipy_skew(ret_nz)) if len(ret_nz) > 3 else float("nan")
    exkurt    = float(scipy_kurtosis(ret_nz)) if len(ret_nz) > 3 else float("nan")

    date_min = trades["entry_time"].min().date()
    date_max = trades["entry_time"].max().date()
    tpd      = n / max(n_years * 365, 1)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Window : {date_min} → {date_max}  ({n_years:.1f} years)")
    print(f"  Trades : {n:,}  ({tpd:.2f}/day)")
    print(f"  Wins   : {wins:,}  ({wr*100:.1f}%)")
    print(f"  Avg Win R / Avg Loss R : {avg_win:.3f} / {avg_loss:.3f}")
    print(f"  Sum R  : {sum_r:+.1f}  |  Avg R : {avg_r:+.4f}")
    print(f"  Profit Factor  : {pf:.3f}")
    print(f"  Ann Return     : {ann_ret*100:.1f}%")
    print(f"  Ann Sharpe     : {sharpe:.3f}")
    print(f"  Max Drawdown   : {maxdd*100:.1f}%")
    print(f"  Calmar Ratio   : {calmar:.3f}")
    print(f"  Skew / Ex.Kurt : {skewness:.2f} / {exkurt:.2f}")
    print(f"  Final equity (from ${START_EQ:.0f} @ {RISK_PCT*100:.0f}% risk): ${curve.iloc[-1]:,.2f}")
    print(f"  Buy vs Sell    : {(trades['side']=='buy').sum()} buys / {(trades['side']=='sell').sum()} sells")

    return dict(
        label=label, n=n, win_rate=wr, avg_win_r=avg_win, avg_loss_r=avg_loss,
        sum_r=sum_r, pf=pf, ann_ret=ann_ret, sharpe=sharpe, maxdd=maxdd,
        calmar=calmar, skew=skewness, ex_kurt=exkurt, n_years=n_years,
        final_eq=curve.iloc[-1], tpd=tpd,
    )


def main():
    print(f"\n{'#'*60}")
    print(f"  SOMEGA Pre-2020 Backtest")
    print(f"  Window: start-of-data → {END_DATE.date()}")
    print(f"  Omega params: N={N_MULTIPLIER}, MIN_FILL={MIN_FILL_AGE_H}h, "
          f"MIN_RISK={MIN_RISK_PCT}")
    print(f"  Session filter: EURGBP h>=18, GBPUSD {{20,21,22}}, USDJPY {{19-22}}, "
          f"EURUSD=no filter")
    print(f"{'#'*60}\n")

    all_stats = []
    all_trades = []

    for name, filename in INSTRUMENTS.items():
        print(f"\n--- {name} ---")
        trades = build_somega_trades(name, filename)
        stats  = compute_stats(trades, name)
        if stats:
            all_stats.append(stats)
        if len(trades):
            t = trades.copy()
            t["instrument"] = name
            all_trades.append(t)

    # ----- Combined portfolio summary ----------------------------------------
    if all_trades:
        merged = pd.concat(all_trades, ignore_index=True).sort_values("entry_time")
        print(f"\n{'#'*60}")
        print(f"  COMBINED (all instruments, equal-weight 1% risk each)")
        print(f"{'#'*60}")

        # Pooled equity: each trade risks 1% of a single shared account
        equity = START_EQ
        curve_v, curve_t = [], []
        for _, row in merged.iterrows():
            equity = max(equity * (1 + RISK_PCT * row["r_gross"]), 0.0)
            curve_v.append(equity)
            curve_t.append(row["entry_time"])

        curve  = pd.Series(curve_v, index=pd.to_datetime(curve_t))
        daily  = curve.resample("D").last().ffill()
        ret    = daily.pct_change().dropna()
        ret_nz = ret[ret != 0]
        n_years = (daily.index[-1] - daily.index[0]).days / 365.25
        total_ret = curve.iloc[-1] / START_EQ - 1
        ann_ret   = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1
        dd        = daily / daily.cummax() - 1.0
        maxdd     = abs(dd.min())
        calmar    = ann_ret / maxdd if maxdd > 0 else 0.0
        sharpe    = (ret_nz.mean() / ret_nz.std() * np.sqrt(252)
                     if len(ret_nz) > 1 and ret_nz.std() > 0 else 0.0)
        wins = (merged["r_gross"] > 0).sum()
        wr   = wins / len(merged)

        print(f"\n  Instruments : {list(INSTRUMENTS.keys())}")
        print(f"  Window      : {merged['entry_time'].min().date()} → "
              f"{merged['entry_time'].max().date()}  ({n_years:.1f}y)")
        print(f"  Total Trades: {len(merged):,}")
        print(f"  Win Rate    : {wr*100:.1f}%")
        print(f"  Sum R       : {merged['r_gross'].sum():+.1f}")
        print(f"  Ann Return  : {ann_ret*100:.1f}%")
        print(f"  Ann Sharpe  : {sharpe:.3f}")
        print(f"  Max Drawdown: {maxdd*100:.1f}%")
        print(f"  Calmar      : {calmar:.3f}")
        print(f"  Final equity (from ${START_EQ:.0f}): ${curve.iloc[-1]:,.2f}")

    # ----- Summary table -------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SUMMARY TABLE")
    print(f"  {'Instrument':<10} {'Trades':>7} {'WinRate':>8} {'SumR':>8} "
          f"{'Sharpe':>8} {'MaxDD':>7} {'Calmar':>8}")
    print(f"  {'-'*10} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")
    for s in all_stats:
        print(f"  {s['label']:<10} {s['n']:>7,} {s['win_rate']*100:>7.1f}% "
              f"{s['sum_r']:>+8.1f} {s['sharpe']:>8.3f} {s['maxdd']*100:>6.1f}% "
              f"{s['calmar']:>8.3f}")

    # Save CSV
    out_path = os.path.join(DATA_DIR, "somega_pre2020_backtest_results.csv")
    if all_stats:
        pd.DataFrame(all_stats).to_csv(out_path, index=False)
        print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
