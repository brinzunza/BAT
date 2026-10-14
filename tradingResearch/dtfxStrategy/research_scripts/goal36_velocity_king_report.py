"""Full stats sheet + historical/monthly-reset backtest for Velocity-King, at this
deployment's exact settings: two $100 sub-accounts (buy-only, sell-only, never pooled),
6% risk (goal34's MC-robust optimum) of min(buy_equity, sell_equity), 50x leverage,
1.6-pip spread, max 1 open position per side, EURUSD/GBPUSD/EURGBP/USDJPY.

Reuses goal30's two-account simulator/MC and goal31's stats methodology (streaks,
Sharpe/kurtosis convention, period-reset positive-% methodology) directly -- only the
trade-stream builder is Velocity-King-specific.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal30_apex_king_deployment as g30
import goal31_apex_stats as g31

RISK_PCT = 0.06  # Velocity-King's own MC-robust optimum (goal34)
VELOCITY_CFG = dict(n=5.0, rr=0.20, age_h=0.20, bounce=False, touches=16,
                     session=None, holdout=True)


def build_velocity_trades():
    for inst in g29.TRADEABLE:
        g29.rs.load_instrument(inst)
        g29.rs._prep_zones(inst, g29.pg.MIN_RETRACEMENT)
    return g29.get_true_trades("Velocity-King", VELOCITY_CFG)


def section_historical_backtest(trades, risk_pct):
    print("\n" + "=" * 90 + f"\nHISTORICAL BACKTEST at {100*risk_pct:.1f}% risk\n" + "=" * 90)
    t, eq_buy, eq_sell, max_dd, trace = g30.simulate_two_account(trades, risk_pct)
    acc = t[t.accepted]
    wr = 100 * acc.is_win.mean()
    w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    print(f"Window: {t.entry_time.min()} -> {t.entry_time.max()}  ({years:.1f} years)")
    print(f"Signals: {len(t):,}  Accepted: {len(acc):,} ({100*len(acc)/len(t):.1f}%)")
    print(f"Accepted WR={wr:.2f}%  PF={pf:.3f}")
    print(f"Start: $100 buy + $100 sell = $200 combined")
    print(f"End:   ${eq_buy:,.2f} buy + ${eq_sell:,.2f} sell = ${eq_buy+eq_sell:,.2f} combined "
          f"({100*((eq_buy+eq_sell)/200-1):+.1f}%)")
    print(f"Max drawdown (combined equity): {max_dd:.1f}%")
    return t, trace


def section_monthly_reset(trades, risk_pct):
    print("\n" + "=" * 90 + f"\nMONTHLY GROWTH, account reset to $100/$100 each month\n" + "=" * 90)
    t = trades.copy()
    t["month"] = t.entry_time.dt.to_period("M")
    rows = []
    for month, sub in t.groupby("month"):
        if len(sub) < 5:
            continue
        _, eq_buy, eq_sell, max_dd, _ = g30.simulate_two_account(sub, risk_pct, track_detail=False)
        combined = eq_buy + eq_sell
        rows.append(dict(month=str(month), signals=len(sub), end_buy=eq_buy, end_sell=eq_sell,
                          end_combined=combined, return_pct=100 * (combined / 200 - 1), max_dd_pct=max_dd))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal36_velocity_monthly_reset.csv"), index=False)
    print(df.to_string(index=False))
    print(f"\nMedian monthly return: {df.return_pct.median():+.1f}%   "
          f"Mean: {df.return_pct.mean():+.1f}%   "
          f"Win months: {100*(df.return_pct > 0).mean():.0f}%   "
          f"Worst month: {df.return_pct.min():+.1f}%   Best month: {df.return_pct.max():+.1f}%")
    return df


def section_full_stats(trades, risk_pct):
    print("\nSimulating at deployment settings for stats sheet...")
    t, eq_buy, eq_sell, max_dd, trace = g30.simulate_two_account(trades, risk_pct)
    acc = t[t.accepted].sort_values("entry_time").reset_index(drop=True)
    n = len(acc)
    years = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25
    days = (acc.entry_time.max() - acc.entry_time.min()).days

    wins = acc[acc.net_r > 0]
    losses = acc[acc.net_r <= 0]
    wr = 100 * len(wins) / n
    avg_win_r = wins.net_r.mean()
    avg_loss_r = losses.net_r.mean()
    gross_win = wins.net_r.sum()
    gross_loss = -losses.net_r.sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    trades_per_day = n / days

    win_streaks, loss_streaks = g31.compute_streaks([bool(x) for x in (acc.net_r.values > 0)])

    print("\n" + "=" * 70 + "\nCORE TRADE STATS (accepted trades, R-multiples)\n" + "=" * 70)
    print(f"Accepted trades:        {n:,}  ({wr:.2f}% WR)  over {years:.1f} years")
    print(f"Profit factor:          {pf:.3f}")
    print(f"Avg win:                {avg_win_r:+.4f} R  (constant by construction -- TARGET_RR=0.20)")
    print(f"Avg loss:               {avg_loss_r:+.4f} R  (varies with spread/stop_risk per zone)")
    print(f"Trades/day (accepted):  {trades_per_day:.2f}")
    print(f"Longest win streak:     {max(win_streaks)} trades" if win_streaks else "no wins")
    print(f"Longest loss streak:    {max(loss_streaks)} trades" if loss_streaks else "no losses")
    print(f"Avg win streak length:  {np.mean(win_streaks):.2f} trades  ({len(win_streaks)} streaks)")
    print(f"Avg loss streak length: {np.mean(loss_streaks):.2f} trades  ({len(loss_streaks)} streaks)")

    print("\n" + "=" * 70 + "\nPOSITIVE-PERIOD % (each period independently reset to $100/$100)\n" + "=" * 70)
    t0 = time.perf_counter()
    for label, freq in [("Day", "D"), ("Week", "W"), ("Month", "M"), ("Year", "Y")]:
        pct, n_periods, _ = g31.period_positive_pct(trades, freq, risk_pct=risk_pct)
        print(f"  {label:6s}: {pct:5.1f}% positive  (n={n_periods} {label.lower()}s)  "
              f"[{time.perf_counter()-t0:.0f}s elapsed]")

    print("\n" + "=" * 70 + "\nSHARPE / KURTOSIS (project-standard 1%-fixed-risk daily curve convention)\n" + "=" * 70)
    ann_sharpe, exk, calmar, maxdd, n_obs = g31.sharpe_and_kurtosis(acc)
    print(f"Annualized Sharpe:       {ann_sharpe:.2f}  (n={n_obs} nonzero daily obs)")
    print(f"Excess kurtosis:         {exk:+.2f}  (0 = normal; positive = fat tails)")
    print(f"Calmar ratio:            {calmar:.2f}")
    print(f"Max drawdown (1% ref):   {-100*maxdd:.1f}%")

    print("\n" + "=" * 70 + "\nMONTE CARLO: probability of reaching $1M by various horizons\n" + "=" * 70)
    t1s, t10s, dds, ruins, accs = g30.run_mc(trades, risk_pct, n_paths=500, seed=6161, max_trades=400_000)
    reached = t1s >= 0
    print(f"Paths simulated: 500   Ruined: {100*ruins.mean():.2f}%")
    for horizon in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        hit_by = reached & (t1s <= horizon)
        print(f"  P(reach $1M within {horizon:.2f} yr / {horizon*365.25:.0f} days): {100*hit_by.mean():.1f}%")
    print(f"  P(reach $1M ever within budget): {100*reached.mean():.1f}%")


def main():
    print("Building Velocity-King TRUE holdout trade stream...")
    trades = build_velocity_trades()
    print(f"  {len(trades):,} signal candidates, {trades.entry_time.min()} -> {trades.entry_time.max()}")

    section_historical_backtest(trades, RISK_PCT)
    section_monthly_reset(trades, RISK_PCT)
    section_full_stats(trades, RISK_PCT)


if __name__ == "__main__":
    main()
