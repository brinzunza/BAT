"""Direct user request (2026-09-07): backtest the $200 two-account setup (variant A, fixed
1:1R) as if it had started fresh on 2026-01-01 -- not continuing 23 years of compounding --
at 3% risk, comparing minimum zone heights, with the full period-level breakdown (balance,
win rate, trades/day, % profitable days/weeks/months) plus a robustness check.

Reuses goal104's cached zone-building (full history needed for correct trend/structure
context going INTO 2026) and goal99's overshoot-then-tag entry gate + fixed 1:1R target;
only the trade list is filtered to entry_time >= 2026-01-01 before feeding it to goal79's
real two-account simulator, so 2026 starts both sub-accounts fresh at $100 each rather than
inheriting decades of prior compounding.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal79_two_account_portfolio as g79
import goal104_optimal_risk_two_account as g104

RISK_PCT = 0.03
YTD_START = pd.Timestamp("2026-01-01")
MIN_ZONE_PIPS_LIST = [5, 10, 20, 30]


def period_stats(equity, freq):
    e = equity.resample(freq).last().ffill().dropna()
    ret = e.pct_change().dropna()
    n = len(ret)
    n_pos = int((ret > 0).sum())
    pct = 100.0 * n_pos / n if n else float("nan")
    return dict(n_periods=n, n_profitable=n_pos, pct_profitable=pct)


def main():
    t0 = time.time()
    print("Loading + building zones once (full history, needed for structure context)...")
    cache = g104.load_and_build_zones()

    rows = []
    detail = {}
    for min_pips in MIN_ZONE_PIPS_LIST:
        trades_full = g104.build_candidate_trades(cache, min_pips)
        trades = trades_full[trades_full["entry_time"] >= YTD_START].reset_index(drop=True)
        span_days = (trades["exit_time"].max() - YTD_START) / np.timedelta64(1, "D")
        print(f"\nmin_zone={min_pips}pips: {len(trades):,} candidate trades from {YTD_START.date()} "
              f"through {trades['exit_time'].max().date()} ({span_days:.0f} days)")

        g79.RISK_PCT = RISK_PCT
        result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
            trades, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
            max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=True)
        accepted = result[result["accepted"]].reset_index(drop=True)
        n = len(accepted)
        wr = 100.0 * (accepted["net_r"] > 0).mean() if n else float("nan")
        final_bal = eq_buy + eq_sell

        equity = trace.set_index("close_time")["equity_combined"].sort_index()
        equity = equity[~equity.index.duplicated(keep="last")]
        anchor = pd.Series([200.0], index=[YTD_START - pd.Timedelta(seconds=1)])
        equity_full = pd.concat([anchor, equity]).sort_index()
        equity_full = equity_full[~equity_full.index.duplicated(keep="last")]

        daily = period_stats(equity_full, "D")
        weekly = period_stats(equity_full, "W-FRI")
        monthly = period_stats(equity_full, "M")

        tpd = n / span_days if span_days > 0 else float("nan")
        rows.append(dict(min_pips=min_pips, n_trades=n, wr=wr, final_balance=final_bal,
                          buy_final=eq_buy, sell_final=eq_sell, max_dd_pct=max_dd,
                          trades_per_day=tpd, pct_days_profitable=daily["pct_profitable"],
                          pct_weeks_profitable=weekly["pct_profitable"],
                          pct_months_profitable=monthly["pct_profitable"],
                          days_n=daily["n_periods"], weeks_n=weekly["n_periods"], months_n=monthly["n_periods"]))
        detail[min_pips] = dict(accepted=accepted, equity=equity_full, trades_full=trades_full)

        print(f"  accepted={n:,}  WR={wr:.2f}%  final=${final_bal:,.2f} (buy ${eq_buy:,.2f} / sell ${eq_sell:,.2f})  "
              f"maxDD={max_dd:.2f}%  trades/day={tpd:.2f}")
        print(f"  profitable: days {daily['n_profitable']}/{daily['n_periods']} ({daily['pct_profitable']:.1f}%)  "
              f"weeks {weekly['n_profitable']}/{weekly['n_periods']} ({weekly['pct_profitable']:.1f}%)  "
              f"months {monthly['n_profitable']}/{monthly['n_periods']} ({monthly['pct_profitable']:.1f}%)")

    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(DATA_DIR, "goal106_2026_ytd_grid.csv"), index=False)

    # ---- robustness: split the YTD period into thirds, check consistency ----
    print(f"\n{'='*70}\nRobustness: thirds of the 2026 YTD period\n{'='*70}")
    robust_rows = []
    for min_pips, d in detail.items():
        trades = d["trades_full"]
        trades = trades[trades["entry_time"] >= YTD_START].reset_index(drop=True)
        t_min, t_max = YTD_START, trades["exit_time"].max()
        edges = pd.date_range(t_min, t_max, periods=4)
        print(f"\nmin_zone={min_pips}pips:")
        for i in range(3):
            lo, hi = edges[i], edges[i + 1]
            chunk = trades[(trades["entry_time"] >= lo) & (trades["entry_time"] < hi)].reset_index(drop=True)
            if len(chunk) == 0:
                continue
            g79.RISK_PCT = RISK_PCT
            res_c, eb_c, es_c, dd_c, _ = g79.simulate_two_account(
                chunk, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
                max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=False)
            acc_c = res_c[res_c["accepted"]]
            n_c = len(acc_c)
            wr_c = 100.0 * (acc_c["net_r"] > 0).mean() if n_c else float("nan")
            print(f"  third {i+1} ({lo.date()} to {hi.date()}): n={n_c:,} WR={wr_c:.1f}% "
                  f"final=${eb_c+es_c:,.2f} maxDD={dd_c:.1f}%")
            robust_rows.append(dict(min_pips=min_pips, third=i + 1, start=lo.date(), end=hi.date(),
                                     n=n_c, wr=wr_c, final_balance=eb_c + es_c, max_dd_pct=dd_c))

    robust_df = pd.DataFrame(robust_rows)
    robust_df.to_csv(os.path.join(DATA_DIR, "goal106_2026_robustness_thirds.csv"), index=False)

    # ---- per-instrument breakdown for the recommended zone height ----
    REC = 10
    print(f"\n{'='*70}\nPer-instrument breakdown, min_zone={REC}pips\n{'='*70}")
    acc = detail[REC]["accepted"]
    for inst, g in acc.groupby("instrument"):
        wr_i = 100.0 * (g["net_r"] > 0).mean()
        print(f"  {inst}: n={len(g):,} WR={wr_i:.1f}% sum_R={g['net_r'].sum():+.1f}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
