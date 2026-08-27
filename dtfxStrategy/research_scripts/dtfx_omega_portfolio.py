"""Real multi-instrument Omega portfolio: one pooled account, not five independently-
compounding sleeves. Every trade from every instrument, across the whole 2018+ overlap
window, sorted into one chronological stream and applied sequentially to a single equity
curve -- each trade still risks 1% of whatever the CURRENT pooled equity is at that moment,
same convention used everywhere else this session, just no longer partitioned per instrument.

This replaces the earlier "naive" estimate (each instrument compounds independently, then
average the daily returns) with the real thing: what one account actually holding all six
positions concurrently, sized off shared capital, would have produced.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import build_trades as _build_cached, DATA_DIR, _price_cache, _indicator_cache, _rsi, _atr_ratio
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

OMEGA = dict(n_multiplier=3.343496, min_risk_pct=0.000170, min_fill_age_hours=3.366682,
             min_retracement=2, use_rsi_atr_filter=False)
RISK_PCT = 0.01
START_EQUITY = 100.0
WINDOW_START = "2018-01-01"

INSTRUMENTS = {
    "EURUSD": "lse_eurusd_1min_full.csv", "GBPUSD": "lse_gbpusd_1min_full.csv",
    "EURGBP": "lse_eurgbp_1min_full.csv", "USDJPY": "lse_usdjpy_1min_2018on.csv",
    "BTCUSD": "lse_btcusd_1min_2018on.csv", "XAUUSD": "lse_xauusd_1min_2018on.csv",
}


def load_instrument_generic(name, path):
    if name in _price_cache:
        return
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[["open", "high", "low", "close"]]
    _indicator_cache[name] = (_rsi(df["close"]).values, _atr_ratio(df).values)
    times = df.index.values
    lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    _price_cache[name] = (times, lows, highs, df)
    print(f"  {name}: {len(df):,} bars, {df.index.min()} -> {df.index.max()}", flush=True)


def main():
    all_trades = []
    for name, path in INSTRUMENTS.items():
        load_instrument_generic(name, os.path.join(DATA_DIR, path))
        t = _build_cached(name, OMEGA["min_retracement"], OMEGA["n_multiplier"],
                           OMEGA["min_risk_pct"], OMEGA["min_fill_age_hours"], OMEGA["use_rsi_atr_filter"])
        t = t[t["entry_time"] >= WINDOW_START].copy()
        t["instrument"] = name
        all_trades.append(t)
        print(f"{name}: {len(t):,} trades since {WINDOW_START}", flush=True)

    merged = pd.concat(all_trades, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    print(f"\nTotal portfolio trades: {len(merged):,}", flush=True)

    equity = START_EQUITY
    curve_t, curve_v, curve_inst = [], [], []
    for _, row in merged.iterrows():
        equity = max(equity * (1 + RISK_PCT * row["net_r"]), 0.0)
        curve_t.append(row["entry_time"]); curve_v.append(equity); curve_inst.append(row["instrument"])
    curve = pd.Series(curve_v, index=pd.to_datetime(curve_t))
    daily = curve.resample("D").last().ffill()
    ret = daily.pct_change().dropna()
    ret_nz = ret[ret != 0]

    n_years = max((daily.index[-1] - daily.index[0]).days / 365.25, 0.1)
    total_return = curve.iloc[-1] / START_EQUITY - 1
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    dd = (daily / daily.cummax() - 1.0)
    maxdd = abs(dd.min())
    calmar = ann_return / maxdd
    exk = scipy_kurtosis(ret_nz)
    ann_sharpe = ret_nz.mean() / ret_nz.std() * np.sqrt(252)
    gross_win = merged.loc[merged["net_r"] > 0, "net_r"].sum()
    gross_loss = -merged.loc[merged["net_r"] <= 0, "net_r"].sum()
    profit_factor = gross_win / gross_loss
    win_rate = (merged["net_r"] > 0).mean()

    print(f"\n=== PORTFOLIO (6 instruments, pooled account, 1% risk/trade, {WINDOW_START}+) ===")
    print(f"  trades: {len(merged):,}")
    print(f"  win_rate: {win_rate*100:.1f}%")
    print(f"  ann_return: {ann_return*100:.1f}%")
    print(f"  ann_sharpe: {ann_sharpe:.2f}")
    print(f"  calmar: {calmar:.2f}")
    print(f"  profit_factor: {profit_factor:.3f}")
    print(f"  excess_kurtosis: {exk:.2f}")
    print(f"  skew: {scipy_skew(ret_nz):.2f}")
    print(f"  max_drawdown: {maxdd*100:.1f}%")
    print(f"  final_equity (from $100): ${curve.iloc[-1]:,.2f}")

    # per-instrument trade contribution breakdown
    print(f"\n  trades by instrument:")
    print(merged["instrument"].value_counts().to_string())

    out = pd.DataFrame({"date": curve_t, "equity": curve_v, "instrument": curve_inst})
    out.to_csv(f"{DATA_DIR}/dtfx_omega_portfolio_equity_curve.csv", index=False)
    summary = pd.DataFrame([dict(trades=len(merged), win_rate=win_rate, ann_return=ann_return,
                                  ann_sharpe=ann_sharpe, calmar=calmar, profit_factor=profit_factor,
                                  excess_kurtosis=exk, skew=scipy_skew(ret_nz), maxdd=maxdd,
                                  final_equity=curve.iloc[-1])])
    summary.to_csv(f"{DATA_DIR}/dtfx_omega_portfolio_summary.csv", index=False)
    print(f"\nSaved -> dtfx_omega_portfolio_equity_curve.csv, dtfx_omega_portfolio_summary.csv")

    return daily, merged


if __name__ == "__main__":
    main()
