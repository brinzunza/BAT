"""Full stats sheet for Apex-King under this deployment's exact setup: two $100
sub-accounts, 1.6-pip spread, 50x leverage, max 1 open position per side, 6.0% risk
(goal30's MC-robust optimum). All win/loss/streak/frequency stats are computed on the
REAL ACCEPTED trade population (post margin + per-side-cap filtering) -- the trades that
would actually execute live, not the raw unconstrained signal population.

Sharpe and excess kurtosis use this project's own established convention
(dtfx_robust_search.py::score_trades): a FIXED 1%-risk-per-trade compounding curve,
resampled to daily closes, nonzero-return days only, sqrt(252) annualization -- decoupled
from the real 6% deployment risk so the number stays on a comparable, sane scale and is
directly comparable to every other strategy's Sharpe elsewhere in this project. Applied to
the REAL accepted-at-6%-risk trade stream (which trades get taken at all still depends on
6% risk's margin/concurrency filtering), not the raw stream score_trades() itself uses.

Positive-day/week/month/year % use a period-RESET methodology (each period simulated
independently from $100/$100, like goal30's monthly-reset table) -- avoids one period's
number being an artifact of how much prior periods had already compounded equity up.
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

import goal30_apex_king_deployment as g30

RISK_PCT = 0.06                # this deployment's real, MC-robust-optimal risk
SHARPE_RISK_PCT = 0.01         # project-standard fixed reference risk for Sharpe/kurtosis only
SHARPE_START = 100.0           # project-standard reference start (dtfx_robust_search.START_EQUITY)


def compute_streaks(is_win_seq):
    """Longest/avg win and loss streaks from a chronological win/loss boolean sequence."""
    win_streaks, loss_streaks = [], []
    cur_val, cur_len = None, 0
    for w in is_win_seq:
        if w == cur_val:
            cur_len += 1
        else:
            if cur_val is True:
                win_streaks.append(cur_len)
            elif cur_val is False:
                loss_streaks.append(cur_len)
            cur_val, cur_len = w, 1
    if cur_val is True:
        win_streaks.append(cur_len)
    elif cur_val is False:
        loss_streaks.append(cur_len)
    return win_streaks, loss_streaks


def period_positive_pct(trades, freq, start_per_side=100.0, risk_pct=RISK_PCT):
    """Reset to $100/$100 at the start of EVERY period (day/week/month/year), simulate
    independently, report % of periods that ended combined-equity-positive."""
    t = trades.copy()
    t["period"] = t.entry_time.dt.to_period(freq)
    results = []
    for period, sub in t.groupby("period"):
        if len(sub) == 0:
            continue
        _, eq_buy, eq_sell, _, _ = g30.simulate_two_account(
            sub, risk_pct, start_per_side=start_per_side, track_detail=False)
        results.append((eq_buy + eq_sell) - 2 * start_per_side)
    results = np.array(results)
    return 100 * (results > 0).mean(), len(results), results


def sharpe_and_kurtosis(trades):
    """Project-standard Sharpe/excess-kurtosis convention: fixed 1%-risk compounding
    curve, daily-resampled, nonzero returns only, sqrt(252) annualized."""
    t = trades.sort_values("entry_time")
    equity = SHARPE_START
    curve_v, curve_t = [], []
    for r in t.net_r.values:
        equity = max(equity * (1 + SHARPE_RISK_PCT * r), 0.0)
        curve_v.append(equity)
    curve_t = t.entry_time.values
    curve = pd.Series(curve_v, index=pd.to_datetime(curve_t))
    daily = curve.resample("D").last().ffill()
    ret = daily.pct_change().dropna()
    ret_nz = ret[ret != 0]
    ann_sharpe = ret_nz.mean() / ret_nz.std() * np.sqrt(252) if ret_nz.std() > 0 else 0.0
    exk = scipy_kurtosis(ret_nz)
    dd = (daily / daily.cummax() - 1.0)
    maxdd = abs(dd.min())
    n_years = max((daily.index[-1] - daily.index[0]).days / 365.25, 0.1)
    total_return = curve.iloc[-1] / SHARPE_START - 1
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    calmar = ann_return / maxdd if maxdd > 1e-9 else float("inf")
    return ann_sharpe, exk, calmar, maxdd, len(ret_nz)


def main():
    print("Building Apex-King TRUE holdout trade stream...")
    trades = g30.build_apex_trades()
    print(f"  {len(trades):,} signal candidates, {trades.entry_time.min()} -> {trades.entry_time.max()}\n")

    print(f"Simulating at deployment settings: $100/$100, {100*RISK_PCT:.0f}% risk, 1/side cap...")
    t, eq_buy, eq_sell, max_dd, trace = g30.simulate_two_account(trades, RISK_PCT)
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

    win_streaks, loss_streaks = compute_streaks([bool(x) for x in (acc.net_r.values > 0)])

    print("\n" + "=" * 70 + "\nCORE TRADE STATS (accepted trades, R-multiples)\n" + "=" * 70)
    print(f"Accepted trades:        {n:,}  ({wr:.2f}% WR)  over {years:.1f} years")
    print(f"Profit factor:          {pf:.3f}")
    print(f"Avg win:                {avg_win_r:+.4f} R  (constant by construction -- every win")
    print(f"                          nets exactly TARGET_RR=0.18, spread cost is zero on")
    print(f"                          limit-filled wins; real variance is only in losses)")
    print(f"Avg loss:                {avg_loss_r:+.4f} R  (varies -- spread cost/stop_risk ratio")
    print(f"                          differs per zone)")
    print(f"Trades/day (accepted):  {trades_per_day:.2f}")
    print(f"Longest win streak:     {max(win_streaks)} trades" if win_streaks else "no wins")
    print(f"Longest loss streak:    {max(loss_streaks)} trades" if loss_streaks else "no losses")
    print(f"Avg win streak length:  {np.mean(win_streaks):.2f} trades  ({len(win_streaks)} streaks)")
    print(f"Avg loss streak length: {np.mean(loss_streaks):.2f} trades  ({len(loss_streaks)} streaks)")

    print("\n" + "=" * 70 + "\nPOSITIVE-PERIOD % (each period independently reset to $100/$100)\n" + "=" * 70)
    t0 = time.perf_counter()
    for label, freq in [("Day", "D"), ("Week", "W"), ("Month", "M"), ("Year", "Y")]:
        pct, n_periods, _ = period_positive_pct(trades, freq)
        print(f"  {label:6s}: {pct:5.1f}% positive  (n={n_periods} {label.lower()}s)  "
              f"[{time.perf_counter()-t0:.0f}s elapsed]")

    print("\n" + "=" * 70 + "\nSHARPE / KURTOSIS (project-standard 1%-fixed-risk daily curve convention)\n" + "=" * 70)
    ann_sharpe, exk, calmar, maxdd, n_obs = sharpe_and_kurtosis(acc)
    print(f"Annualized Sharpe:       {ann_sharpe:.2f}  (n={n_obs} nonzero daily obs)")
    print(f"Excess kurtosis:         {exk:+.2f}  (0 = normal; positive = fat tails)")
    print(f"Calmar ratio:            {calmar:.2f}")
    print(f"Max drawdown (1% ref):   {-100*maxdd:.1f}%")

    print("\n" + "=" * 70 + "\nMONTE CARLO: probability of reaching $1M by various horizons\n" + "=" * 70)
    t1s, t10s, dds, ruins, accs = g30.run_mc(trades, RISK_PCT, n_paths=500, seed=5150, max_trades=400_000)
    reached = t1s >= 0
    print(f"Paths simulated: 500   Ruined: {100*ruins.mean():.2f}%")
    for horizon in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        hit_by = reached & (t1s <= horizon)
        print(f"  P(reach $1M within {horizon:.2f} yr / {horizon*365.25:.0f} days): {100*hit_by.mean():.1f}%")
    print(f"  P(reach $1M ever within budget): {100*reached.mean():.1f}%")


if __name__ == "__main__":
    main()
