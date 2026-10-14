"""$100 start, 5% risk/trade (of CURRENT equity, so this compounds), +$10 added every
calendar day (not just trading days) -- COMB config, EUR/USD full history.

Order of operations per calendar day: deposit first, then that day's trades in chronological
order, each applying equity *= (1 + RISK_PCT * net_r). Deposits happen every day including
weekends/holidays (a standing daily contribution habit), trades only happen when the market
produces one.

5% risk/trade is aggressive -- this session's earlier Kelly-style sweep found returns start
reversing and drawdowns turn catastrophic somewhere around 6-7% risk/trade on this strategy's
R-distribution. 5% is close to that edge, so expect a rougher ride than the 1%-risk numbers
reported elsewhere in this project.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import build_trades, DATA_DIR

COMB = dict(n_multiplier=4.623254141476141, min_risk_pct=8.837618556064067e-05,
            min_fill_age_hours=3.0892056840204307, min_retracement=2,
            use_rsi_atr_filter=True)
START_EQUITY = 100.0
RISK_PCT = 0.05
DAILY_DEPOSIT = 10.0


def main():
    trades = build_trades("EURUSD", COMB["min_retracement"], COMB["n_multiplier"],
                           COMB["min_risk_pct"], COMB["min_fill_age_hours"],
                           COMB["use_rsi_atr_filter"])
    trades = trades.sort_values("entry_time")
    trades["date"] = trades["entry_time"].dt.floor("D")

    start_date = trades["entry_time"].min().floor("D")
    end_date = trades["entry_time"].max().floor("D")
    all_days = pd.date_range(start_date, end_date, freq="D")

    trades_by_day = {d: g["net_r"].tolist() for d, g in trades.groupby("date")}

    equity = START_EQUITY
    total_deposited = 0.0
    daily_rows = []
    hit_zero = False
    for day in all_days:
        equity += DAILY_DEPOSIT
        total_deposited += DAILY_DEPOSIT
        for net_r in trades_by_day.get(day, []):
            equity = equity * (1 + RISK_PCT * net_r)
            if equity <= 0:
                equity = 0.0
                hit_zero = True
        daily_rows.append(dict(date=day, equity=equity, total_deposited=total_deposited + START_EQUITY))

    daily = pd.DataFrame(daily_rows).set_index("date")
    print(f"Trading days simulated: {len(all_days):,} calendar days "
          f"({start_date.date()} -> {end_date.date()})")
    print(f"Hit zero equity at any point: {hit_zero}\n")

    def period_table(freq, label_fmt):
        rows = []
        g = daily["equity"].resample(freq)
        dep = daily["total_deposited"].resample(freq)
        starts = g.first()
        ends = g.last()
        dep_end = dep.last()
        for idx in starts.index:
            rows.append(dict(
                period=idx.strftime(label_fmt),
                start_equity=starts[idx],
                end_equity=ends[idx],
                cumulative_deposited=dep_end[idx],
                trading_pnl_this_period=(ends[idx] - starts[idx]) if idx != starts.index[0] else np.nan,
            ))
        return pd.DataFrame(rows)

    weekly = period_table("W", "%Y-%m-%d")
    monthly = period_table("MS", "%Y-%m")
    yearly = period_table("YS", "%Y")

    weekly.to_csv(f"{DATA_DIR}/dtfx_comb_deposit_sim_weekly.csv", index=False)
    monthly.to_csv(f"{DATA_DIR}/dtfx_comb_deposit_sim_monthly.csv", index=False)
    yearly.to_csv(f"{DATA_DIR}/dtfx_comb_deposit_sim_yearly.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 160)
    print("=== YEARLY (equity at year-end) ===")
    print(yearly.to_string(index=False))

    final_equity = daily["equity"].iloc[-1]
    total_dep_final = daily["total_deposited"].iloc[-1]
    trading_contribution = final_equity - total_dep_final
    print(f"\n=== TOTAL ({start_date.date()} -> {end_date.date()}) ===")
    print(f"  Starting capital: ${START_EQUITY:,.2f}")
    print(f"  Total deposited (incl. starting capital): ${total_dep_final:,.2f}")
    print(f"  Final equity: ${final_equity:,.2f}")
    print(f"  Pure trading contribution (final equity - total deposited): ${trading_contribution:,.2f}")
    print(f"  Multiple on capital contributed: {final_equity/total_dep_final:.2f}x")

    # max drawdown of the equity curve itself (includes deposit effect, which partially masks true trading DD)
    running_max = daily["equity"].cummax()
    dd = (daily["equity"] / running_max - 1.0)
    print(f"  Max drawdown of account equity (deposit-masked): {abs(dd.min())*100:.1f}%")


if __name__ == "__main__":
    main()
