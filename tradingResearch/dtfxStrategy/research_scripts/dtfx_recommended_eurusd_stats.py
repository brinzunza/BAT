"""Full-history EUR/USD backtest of the recommended (consensus) config from the walk-forward
robustness search, broken into weekly/monthly/yearly/total tables -- same format used
throughout this project for the live-bot baseline, so the two are directly comparable.

Recommended config (fold-median/mode across the 7 walk-forward folds, see
dtfx_robust_search_report.json): N_MULTIPLIER~4.62, MIN_RISK_PCT~0.0088%,
MIN_FILL_AGE_HOURS~3.09h, MIN_RETRACEMENT=2, RSI/ATR filter ON.

Note: this is a FULL-HISTORY (2009-2026) backtest, including the years used to pick these
parameters -- it is NOT the out-of-sample number from the search report. Section 04 of that
report (dtfx_robust_search_report.json / the published artifact) is the trustworthy
out-of-sample figure; this script's total-period Sharpe is the "what would a live account
have shown" number the same way the original live-bot baseline stats were always reported.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import build_trades, DATA_DIR

CONSENSUS = dict(n_multiplier=4.623254141476141, min_risk_pct=8.837618556064067e-05,
                  min_fill_age_hours=3.0892056840204307, min_retracement=2,
                  use_rsi_atr_filter=True)
RISK_PCT_FOR_SHARPE = 0.01
START_EQUITY = 100.0


def period_stats(trades, freq, label_fmt):
    rows = []
    trades = trades.sort_values("entry_time")
    grouper = trades.set_index("entry_time").groupby(pd.Grouper(freq=freq))
    for period_start, g in grouper:
        if len(g) == 0:
            continue
        wins = g[g["net_r"] > 0]
        losses = g[g["net_r"] <= 0]
        rows.append(dict(
            period=period_start.strftime(label_fmt),
            trades=len(g),
            win_rate=len(wins) / len(g),
            avg_win_r=wins["net_r"].mean() if len(wins) else np.nan,
            avg_loss_r=losses["net_r"].mean() if len(losses) else np.nan,
            sum_r=g["net_r"].sum(),
            avg_r=g["net_r"].mean(),
        ))
    return pd.DataFrame(rows)


def compute_equity_and_sharpe(trades):
    trades = trades.sort_values("entry_time")
    equity = START_EQUITY
    curve_t, curve_v = [], []
    for _, row in trades.iterrows():
        equity = max(equity * (1 + RISK_PCT_FOR_SHARPE * row["net_r"]), 0.0)
        curve_t.append(row["entry_time"]); curve_v.append(equity)
    curve = pd.Series(curve_v, index=pd.to_datetime(curve_t))
    daily = curve.resample("D").last().ffill()
    ret = daily.pct_change().dropna()
    ret_nz = ret[ret != 0]
    ann_sharpe = ret_nz.mean() / ret_nz.std() * np.sqrt(252) if ret_nz.std() > 0 else 0.0
    n_years = max((daily.index[-1] - daily.index[0]).days / 365.25, 0.1)
    total_return = curve.iloc[-1] / START_EQUITY - 1
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    dd = (daily / daily.cummax() - 1.0)
    maxdd = abs(dd.min())
    calmar = ann_return / maxdd if maxdd > 0 else np.nan
    from scipy.stats import kurtosis, skew
    return dict(ann_sharpe=ann_sharpe, ann_return=ann_return, total_return=total_return,
                maxdd=maxdd, calmar=calmar, excess_kurtosis=kurtosis(ret_nz), skew=skew(ret_nz),
                final_equity=curve.iloc[-1], n_years=n_years)


def main():
    trades = build_trades("EURUSD", CONSENSUS["min_retracement"], CONSENSUS["n_multiplier"],
                           CONSENSUS["min_risk_pct"], CONSENSUS["min_fill_age_hours"],
                           CONSENSUS["use_rsi_atr_filter"])
    print(f"Total trades: {len(trades):,}  {trades['entry_time'].min()} -> {trades['entry_time'].max()}\n")

    weekly = period_stats(trades, "W", "%Y-%m-%d")
    monthly = period_stats(trades, "MS", "%Y-%m")
    yearly = period_stats(trades, "YS", "%Y")

    weekly.to_csv(f"{DATA_DIR}/dtfx_recommended_eurusd_weekly.csv", index=False)
    monthly.to_csv(f"{DATA_DIR}/dtfx_recommended_eurusd_monthly.csv", index=False)
    yearly.to_csv(f"{DATA_DIR}/dtfx_recommended_eurusd_yearly.csv", index=False)

    total = compute_equity_and_sharpe(trades)
    wins = trades[trades["net_r"] > 0]
    losses = trades[trades["net_r"] <= 0]
    total_row = dict(
        trades=len(trades), win_rate=len(wins) / len(trades),
        avg_win_r=wins["net_r"].mean(), avg_loss_r=losses["net_r"].mean(),
        sum_r=trades["net_r"].sum(), avg_r=trades["net_r"].mean(),
        **total,
    )
    pd.DataFrame([total_row]).to_csv(f"{DATA_DIR}/dtfx_recommended_eurusd_total.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    pd.set_option("display.width", 200)
    print("=== YEARLY ===")
    print(yearly.to_string(index=False))
    print(f"\n=== TOTAL (2009-09-25 -> 2026-08-21, {total['n_years']:.1f} years) ===")
    for k, v in total_row.items():
        print(f"  {k}: {v}")
    print(f"\nWeekly rows: {len(weekly)}  Monthly rows: {len(monthly)}  Yearly rows: {len(yearly)}")
    print("Saved -> dtfx_recommended_eurusd_{weekly,monthly,yearly,total}.csv")


if __name__ == "__main__":
    main()
