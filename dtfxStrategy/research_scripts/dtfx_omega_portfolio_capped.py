"""Risk-capped version of the 4-instrument live Omega portfolio (EUR/USD, GBP/USD, EUR/GBP,
USD/JPY -- the ones actually tradeable on the account; gold and BTC/USD are backtestable but
blocked there). Same 4% risk/trade as what's currently live, but once equity crosses a real,
executable account size, risk freezes at a flat dollar amount instead of continuing to scale
with equity -- the fix for the "total gain in the billions of percent" artifact from
uncapped fixed-fractional compounding at this trade count.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

from dtfx_robust_search import build_trades as _build_cached, DATA_DIR, _price_cache, _indicator_cache, _rsi, _atr_ratio

OMEGA = dict(n_multiplier=3.343496, min_risk_pct=0.000170, min_fill_age_hours=3.366682,
             min_retracement=2, use_rsi_atr_filter=False)
START_EQUITY = 100.0
WINDOW_START = "2018-01-01"
RISK_PCT = 0.04
EQUITY_CAP = 100_000.0

INSTRUMENTS = {
    "EURUSD": "lse_eurusd_1min_full.csv", "GBPUSD": "lse_gbpusd_1min_full.csv",
    "EURGBP": "lse_eurgbp_1min_full.csv", "USDJPY": "lse_usdjpy_1min_2018on.csv",
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


def main():
    all_trades = []
    for name, path in INSTRUMENTS.items():
        load_instrument_generic(name, os.path.join(DATA_DIR, path))
        t = _build_cached(name, OMEGA["min_retracement"], OMEGA["n_multiplier"],
                           OMEGA["min_risk_pct"], OMEGA["min_fill_age_hours"], OMEGA["use_rsi_atr_filter"])
        t = t[t["entry_time"] >= WINDOW_START].copy()
        t["instrument"] = name
        all_trades.append(t)

    merged = pd.concat(all_trades, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    print(f"Total trades: {len(merged):,}  {merged['entry_time'].min()} -> {merged['entry_time'].max()}\n")

    equity = START_EQUITY
    frozen_risk = None
    cap_hit_date = None
    curve_t, curve_v = [], []
    for _, row in merged.iterrows():
        if frozen_risk is None and equity >= EQUITY_CAP:
            frozen_risk = EQUITY_CAP * RISK_PCT
            cap_hit_date = row["entry_time"]
        if frozen_risk is not None:
            equity = max(equity + frozen_risk * row["net_r"], 0.0)
        else:
            equity = max(equity * (1 + RISK_PCT * row["net_r"]), 0.0)
        curve_t.append(row["entry_time"]); curve_v.append(equity)
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
    wins = merged[merged["net_r"] > 0]
    losses = merged[merged["net_r"] <= 0]
    win_rate = len(wins) / len(merged)
    profit_factor = wins["net_r"].sum() / -losses["net_r"].sum()

    print(f"Cap hit on: {cap_hit_date.date() if cap_hit_date is not None else 'never'} "
          f"(risk freezes at ${EQUITY_CAP*RISK_PCT:,.0f}/trade from then on)")
    print(f"\n=== TOTAL ===")
    print(f"  Starting capital: ${START_EQUITY:,.2f}")
    print(f"  Final equity: ${curve.iloc[-1]:,.2f}")
    print(f"  Total gain: {total_return*100:,.1f}%")
    print(f"  Annualized return (pre-cap-blended): {ann_return*100:,.1f}%")
    print(f"  Max drawdown: {maxdd*100:.1f}%")
    print(f"  Ann. Sharpe: {ann_sharpe:.2f}")
    print(f"  Calmar: {calmar:.2f}")
    print(f"  Profit factor: {profit_factor:.3f}")
    print(f"  Excess kurtosis: {exk:.2f}")
    print(f"  Win rate: {win_rate*100:.1f}%")
    print(f"  Trades: {len(merged):,}")

    def period_table(freq, label_fmt):
        rows = []
        g = daily.resample(freq)
        starts, ends = g.first(), g.last()
        for idx in starts.index:
            if pd.isna(starts[idx]):
                continue
            pct = (ends[idx] / starts[idx] - 1) * 100
            rows.append(dict(period=idx.strftime(label_fmt), start_equity=starts[idx],
                              end_equity=ends[idx], pct_change=pct))
        return pd.DataFrame(rows)

    yearly = period_table("YS", "%Y")
    monthly = period_table("MS", "%Y-%m")
    weekly = period_table("W", "%Y-%m-%d")

    yearly.to_csv(f"{DATA_DIR}/dtfx_omega_portfolio_capped_yearly.csv", index=False)
    monthly.to_csv(f"{DATA_DIR}/dtfx_omega_portfolio_capped_monthly.csv", index=False)
    weekly.to_csv(f"{DATA_DIR}/dtfx_omega_portfolio_capped_weekly.csv", index=False)
    curve.to_frame("equity").to_csv(f"{DATA_DIR}/dtfx_omega_portfolio_capped_equity_curve.csv")

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 160)
    print("\n=== YEARLY ===")
    print(yearly.to_string(index=False))

    print(f"\nSaved -> dtfx_omega_portfolio_capped_{{yearly,monthly,weekly,equity_curve}}.csv")


if __name__ == "__main__":
    main()
