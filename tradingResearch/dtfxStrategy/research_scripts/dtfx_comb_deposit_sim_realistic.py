"""Realistic version of the $100-start, $10/day-deposit simulation: 1% and 2% risk/trade
instead of 5%, AND a $100k equity cap where risk switches from percentage-of-equity to a
frozen flat-dollar amount -- once you're risking $1,000-$2,000/trade on EUR/USD (a real,
executable size), continuing to scale risk with equity is what produced last run's absurd
numbers. Freezing dollar risk at the cap is a one-way switch here (never unfreezes if equity
later dips below the cap) -- the simplest, most defensible convention, called out explicitly
rather than silently assumed.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import build_trades, DATA_DIR

COMB = dict(n_multiplier=4.623254141476141, min_risk_pct=8.837618556064067e-05,
            min_fill_age_hours=3.0892056840204307, min_retracement=2,
            use_rsi_atr_filter=True)
START_EQUITY = 100.0
DAILY_DEPOSIT = 10.0
EQUITY_CAP = 100_000.0


def simulate(trades_by_day, all_days, risk_pct, equity_cap):
    equity = START_EQUITY
    dollar_risk_frozen = None
    total_deposited = START_EQUITY
    rows = []
    for day in all_days:
        equity += DAILY_DEPOSIT
        total_deposited += DAILY_DEPOSIT
        for net_r in trades_by_day.get(day, []):
            if dollar_risk_frozen is None and equity >= equity_cap:
                dollar_risk_frozen = equity_cap * risk_pct
            if dollar_risk_frozen is not None:
                equity = max(equity + dollar_risk_frozen * net_r, 0.0)
            else:
                equity = max(equity * (1 + risk_pct * net_r), 0.0)
        rows.append(dict(date=day, equity=equity, total_deposited=total_deposited,
                          capped=dollar_risk_frozen is not None))
    return pd.DataFrame(rows).set_index("date")


def period_table(daily, freq, label_fmt):
    rows = []
    eq = daily["equity"].resample(freq)
    dep = daily["total_deposited"].resample(freq)
    starts, ends, dep_end = eq.first(), eq.last(), dep.last()
    for idx in starts.index:
        rows.append(dict(period=idx.strftime(label_fmt), start_equity=starts[idx],
                          end_equity=ends[idx], cumulative_deposited=dep_end[idx]))
    return pd.DataFrame(rows)


def run_for_risk(risk_pct, trades_by_day, all_days, tag):
    daily = simulate(trades_by_day, all_days, risk_pct, EQUITY_CAP)
    weekly = period_table(daily, "W", "%Y-%m-%d")
    monthly = period_table(daily, "MS", "%Y-%m")
    yearly = period_table(daily, "YS", "%Y")

    weekly.to_csv(f"{DATA_DIR}/dtfx_comb_deposit_sim_{tag}_weekly.csv", index=False)
    monthly.to_csv(f"{DATA_DIR}/dtfx_comb_deposit_sim_{tag}_monthly.csv", index=False)
    yearly.to_csv(f"{DATA_DIR}/dtfx_comb_deposit_sim_{tag}_yearly.csv", index=False)

    cap_hit_date = daily[daily["capped"]].index.min() if daily["capped"].any() else None
    final_equity = daily["equity"].iloc[-1]
    total_dep_final = daily["total_deposited"].iloc[-1]
    running_max = daily["equity"].cummax()
    dd = (daily["equity"] / running_max - 1.0)

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 160)
    print(f"\n{'='*70}\nRISK={risk_pct*100:.0f}%  (cap ${EQUITY_CAP:,.0f}, frozen risk beyond cap = ${EQUITY_CAP*risk_pct:,.0f}/trade)")
    print(f"{'='*70}")
    print("=== YEARLY ===")
    print(yearly.to_string(index=False))
    print(f"\nCap hit on: {cap_hit_date.date() if cap_hit_date is not None else 'never'}")
    print(f"Total deposited (incl. starting $100): ${total_dep_final:,.2f}")
    print(f"Final equity: ${final_equity:,.2f}")
    print(f"Multiple on capital contributed: {final_equity/total_dep_final:.2f}x")
    print(f"Max drawdown of account equity: {abs(dd.min())*100:.1f}%")
    return dict(risk_pct=risk_pct, final_equity=final_equity, total_deposited=total_dep_final,
                cap_hit_date=cap_hit_date, max_dd=abs(dd.min()))


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

    summary = []
    for risk_pct, tag in [(0.01, "1pct"), (0.02, "2pct")]:
        summary.append(run_for_risk(risk_pct, trades_by_day, all_days, tag))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for s in summary:
        print(f"  {s['risk_pct']*100:.0f}% risk: final=${s['final_equity']:,.2f}  "
              f"deposited=${s['total_deposited']:,.2f}  multiple={s['final_equity']/s['total_deposited']:.2f}x  "
              f"cap_hit={s['cap_hit_date'].date() if s['cap_hit_date'] is not None else 'never'}  "
              f"max_dd={s['max_dd']*100:.1f}%")


if __name__ == "__main__":
    main()
