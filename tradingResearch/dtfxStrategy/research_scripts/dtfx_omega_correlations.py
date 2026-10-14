"""Correlation of Omega's daily returns across the 5 proven instruments (EUR/USD, GBP/USD,
EUR/GBP, gold, BTC/USD), plus a naive equal-weighted portfolio Sharpe -- the two numbers
needed to check the "diversification gives a free Sharpe boost" hypothesis instead of
guessing at plausible correlation values.

Uses OMEGA's exact parameters for all 5 (the earlier generalization tests used COMB's
params for USD/JPY, BTC/USD, and gold -- re-running here so every instrument is on the
currently-deployed config, not a mix of two).

Restricted to the overlapping 2018-01-01+ window since USD/JPY, BTC/USD, and gold only have
data fetched from 2018 onward -- EUR/USD, GBP/USD, EUR/GBP all have deeper history, but this
comparison needs the same window across all five to be a fair correlation measurement.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import (
    build_trades as _build_cached, DATA_DIR, _price_cache, _indicator_cache,
    _zone_cache, _rsi, _atr_ratio,
)

OMEGA = dict(n_multiplier=3.343496, min_risk_pct=0.000170, min_fill_age_hours=3.366682,
             min_retracement=2, use_rsi_atr_filter=False)
RISK_PCT_FOR_SHARPE = 0.01
START_EQUITY = 100.0
WINDOW_START = "2018-01-01"

FULL_HISTORY = {"EURUSD": "lse_eurusd_1min_full.csv", "GBPUSD": "lse_gbpusd_1min_full.csv",
                 "EURGBP": "lse_eurgbp_1min_full.csv"}
SHORT_HISTORY = {"USDJPY": "lse_usdjpy_1min_2018on.csv", "BTCUSD": "lse_btcusd_1min_2018on.csv",
                  "XAUUSD": "lse_xauusd_1min_2018on.csv"}
ALL_INSTRUMENTS = {**FULL_HISTORY, **SHORT_HISTORY}


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


def build_trades_generic(name, path, cfg):
    load_instrument_generic(name, path)
    return _build_cached(name, cfg["min_retracement"], cfg["n_multiplier"], cfg["min_risk_pct"],
                          cfg["min_fill_age_hours"], cfg["use_rsi_atr_filter"])


def daily_returns(trades, window_start):
    trades = trades[trades["entry_time"] >= window_start].sort_values("entry_time")
    equity = START_EQUITY
    curve_t, curve_v = [], []
    for _, row in trades.iterrows():
        equity = max(equity * (1 + RISK_PCT_FOR_SHARPE * row["net_r"]), 0.0)
        curve_t.append(row["entry_time"]); curve_v.append(equity)
    curve = pd.Series(curve_v, index=pd.to_datetime(curve_t))
    daily = curve.resample("D").last().ffill()
    ret = daily.pct_change().dropna()
    return ret, len(trades)


def main():
    returns = {}
    sharpes = {}
    for name, path in ALL_INSTRUMENTS.items():
        trades = build_trades_generic(name, os.path.join(DATA_DIR, path), OMEGA)
        ret, n_trades = daily_returns(trades, WINDOW_START)
        returns[name] = ret
        ann_sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0.0
        sharpes[name] = ann_sharpe
        print(f"{name}: {n_trades:,} trades since {WINDOW_START}, "
              f"{len(ret):,} daily obs, ann_sharpe={ann_sharpe:.2f}", flush=True)

    df = pd.DataFrame(returns)  # outer join on date, NaN where an instrument had no move that day
    print(f"\nAligned daily return matrix: {df.shape[0]} days x {df.shape[1]} instruments")
    df_filled = df.fillna(0.0)  # a day with no trade = 0 return for that instrument, not missing

    corr = df_filled.corr()
    print("\n=== Correlation matrix (daily returns, Omega config, 2018+) ===")
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(corr.to_string())
    corr.to_csv(f"{DATA_DIR}/dtfx_omega_correlation_matrix.csv")

    avg_corr = (corr.values.sum() - len(corr)) / (len(corr) ** 2 - len(corr))
    print(f"\nAverage pairwise correlation: {avg_corr:.3f}")

    # naive equal-weighted portfolio: average the 5 instruments' daily returns each day
    portfolio_ret = df_filled.mean(axis=1)
    port_sharpe = portfolio_ret.mean() / portfolio_ret.std() * np.sqrt(252) if portfolio_ret.std() > 0 else 0.0
    avg_single_sharpe = np.mean(list(sharpes.values()))
    print(f"\nAverage single-instrument Sharpe: {avg_single_sharpe:.2f}")
    print(f"Naive equal-weighted 5-instrument portfolio Sharpe: {port_sharpe:.2f}")
    print(f"Uplift vs average single instrument: {(port_sharpe/avg_single_sharpe - 1)*100:+.1f}%")

    # portfolio drawdown too -- diversification should also show up here
    port_curve = (1 + portfolio_ret).cumprod()
    port_dd = abs((port_curve / port_curve.cummax() - 1).min())
    single_dds = []
    for name in ALL_INSTRUMENTS:
        c = (1 + df_filled[name]).cumprod()
        single_dds.append(abs((c / c.cummax() - 1).min()))
    print(f"Average single-instrument max drawdown: {np.mean(single_dds)*100:.1f}%")
    print(f"Portfolio max drawdown: {port_dd*100:.1f}%")


if __name__ == "__main__":
    main()
