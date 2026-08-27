"""Period-level Sharpe (week/month/year/total) for COMB -- the name given to the walk-forward
consensus config (MIN_RETRACEMENT=2, N_MULTIPLIER~4.62, wider risk floor and fill-age gate,
RSI/ATR filter on). Sharpe is computed the same way throughout this project: resample the
compounded equity curve (1% risk/trade) to daily, take pct_change, annualize with sqrt(252),
using only non-zero-return (i.e. trading) days.

Caveat printed alongside the table, not just in a docstring: weekly Sharpe is a genuinely
noisy statistic -- a single week has ~5 trading days, so a handful of weeks will show extreme
values that would smooth out over a longer window. Included because it was asked for, not
because a single week's Sharpe should be trusted on its own.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import build_trades, DATA_DIR

COMB = dict(n_multiplier=4.623254141476141, min_risk_pct=8.837618556064067e-05,
            min_fill_age_hours=3.0892056840204307, min_retracement=2,
            use_rsi_atr_filter=True)
RISK_PCT_FOR_SHARPE = 0.01
START_EQUITY = 100.0


def build_daily_returns(trades):
    trades = trades.sort_values("entry_time")
    equity = START_EQUITY
    curve_t, curve_v = [], []
    for _, row in trades.iterrows():
        equity = max(equity * (1 + RISK_PCT_FOR_SHARPE * row["net_r"]), 0.0)
        curve_t.append(row["entry_time"]); curve_v.append(equity)
    curve = pd.Series(curve_v, index=pd.to_datetime(curve_t))
    daily = curve.resample("D").last().ffill()
    ret = daily.pct_change().dropna()
    return ret[ret != 0]


def sharpe_over(ret_slice, min_obs=3):
    if len(ret_slice) < min_obs or ret_slice.std() == 0:
        return np.nan
    return ret_slice.mean() / ret_slice.std() * np.sqrt(252)


def period_sharpe_table(ret, freq, label_fmt):
    rows = []
    for period_start, g in ret.groupby(pd.Grouper(freq=freq)):
        if len(g) == 0:
            continue
        rows.append(dict(period=period_start.strftime(label_fmt), trading_days=len(g),
                          sharpe=sharpe_over(g)))
    return pd.DataFrame(rows)


def main():
    trades = build_trades("EURUSD", COMB["min_retracement"], COMB["n_multiplier"],
                           COMB["min_risk_pct"], COMB["min_fill_age_hours"],
                           COMB["use_rsi_atr_filter"])
    ret = build_daily_returns(trades)
    print(f"EURUSD COMB: {len(trades):,} trades, {len(ret):,} non-zero trading days, "
          f"{ret.index.min().date()} -> {ret.index.max().date()}\n")

    weekly = period_sharpe_table(ret, "W", "%Y-%m-%d")
    monthly = period_sharpe_table(ret, "MS", "%Y-%m")
    yearly = period_sharpe_table(ret, "YS", "%Y")
    total_sharpe = sharpe_over(ret)

    weekly.to_csv(f"{DATA_DIR}/dtfx_comb_eurusd_sharpe_weekly.csv", index=False)
    monthly.to_csv(f"{DATA_DIR}/dtfx_comb_eurusd_sharpe_monthly.csv", index=False)
    yearly.to_csv(f"{DATA_DIR}/dtfx_comb_eurusd_sharpe_yearly.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    pd.set_option("display.width", 160)
    print("=== YEARLY SHARPE ===")
    print(yearly.to_string(index=False))
    print(f"\n=== MONTHLY SHARPE summary ===  n={len(monthly)}  "
          f"mean={monthly['sharpe'].mean():.2f}  median={monthly['sharpe'].median():.2f}  "
          f"min={monthly['sharpe'].min():.2f}  max={monthly['sharpe'].max():.2f}  "
          f"%months positive={100*(monthly['sharpe']>0).mean():.1f}%")
    print(f"=== WEEKLY SHARPE summary ===  n={len(weekly)}  "
          f"mean={weekly['sharpe'].mean():.2f}  median={weekly['sharpe'].median():.2f}  "
          f"min={weekly['sharpe'].min():.2f}  max={weekly['sharpe'].max():.2f}  "
          f"%weeks positive={100*(weekly['sharpe']>0).mean():.1f}%")
    print(f"\n=== TOTAL (full history) SHARPE === {total_sharpe:.3f}")


if __name__ == "__main__":
    main()
