"""Direct user request (2026-09-06): full portfolio report across all four forex pairs for
the dtfx.md base strategy (goal95 zones, goal99 overshoot-then-tag entry gate, 50% entry),
comparing the two target conventions from dtfx.md #7 back to back:

  (A) fixed 1:1R
  (C) fixed last-confirmed higher-high / lower-low, read once at entry

Requested: win rate, avg pips per TP and per SL, % profitable days/weeks/months/years,
Sharpe, and a Monte Carlo race-to-$1M -- reuses goal78_portfolio_report.py's exact
equity-curve/period-stats/MC methodology (unmodified: sequential risk ordered by exit time,
5%-of-equity risk capped at $2M, goal10's block-bootstrap MC engine), only swapping in this
session's newer zone-detection + entry-gate + target-convention mechanism in place of
goal77's older one.

Caveat carried over explicitly: USDJPY's data file only goes back to 2018 (lse_usdjpy_1min_2018on.csv)
vs. 2009 for the other three pairs -- USDJPY contributes zero trades to the combined
portfolio before 2018, unlike the other three symbols. Not a bug, just this project's data
coverage.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal95_every_swing_zones as g95
import goal99_dtfx_base_backtest as g99
from goal10_cana_nt_mc_capped import run_mc_capped_numba, START, RISK_FRAC, CAP, T1, T10, RUIN_FLOOR

INSTRUMENTS = {
    "EURUSD": ("lse_eurusd_1min_full.csv", 1e-4),
    "GBPUSD": ("lse_gbpusd_1min_full.csv", 1e-4),
    "EURGBP": ("lse_eurgbp_1min_full.csv", 1e-4),
    "USDJPY": ("lse_usdjpy_1min_2018on.csv", 1e-2),
}
SPREAD_PIPS = 1.6
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440
N_SIMS = 25_000
BLOCK = 20
MAX_TRADES = 60_000
SEED = 42


def build_trades(inst, pip, times, z_top, z_bottom, z_is_buy, entry_idx, exit_idx, outcome, net_r, entry_price):
    resolved = outcome > 0
    ei = entry_idx[resolved]; xi = exit_idx[resolved]
    is_buy = z_is_buy[resolved]
    stop_price = np.where(is_buy, z_bottom[resolved], z_top[resolved])
    eff_risk = np.abs(entry_price[resolved] - stop_price)
    is_win = outcome[resolved] == 1
    r = net_r[resolved]
    pips_risked = eff_risk / pip
    # a win's realized price move = net_r * eff_risk (since net_r = reward/eff_risk); a
    # loss's realized price move is exactly the risked stop distance (a full -1R)
    pips_result = np.where(is_win, r * eff_risk / pip, pips_risked)
    return pd.DataFrame({
        "instrument": inst,
        "entry_time": times.values[ei],
        "exit_time": times.values[xi],
        "net_r": r,
        "is_win": is_win,
        "pips_risked": pips_risked,
        "pips_result": pips_result,
    })


def run_symbol(inst, fname, pip):
    path = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(path, parse_dates=["date"])
    times = df["date"]
    high_ = df["high"].values.astype(np.float64)
    low_ = df["low"].values.astype(np.float64)
    close_ = df["close"].values.astype(np.float64)
    n_bars = len(df)

    confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high_, low_, close_)
    (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
     ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)

    last_high_of_bar, last_low_of_bar = g99.build_last_confirmed(confirm_idx, is_high, price, n_bars)
    last_hh_of_bar, last_ll_of_bar = g99.build_confirmed_hh_ll(confirm_idx, is_high, price, n_bars)

    spread_price = SPREAD_PIPS * pip
    min_stop_price = MIN_STOP_PIPS * pip

    entry_idx_a, exit_idx_a, outcome_a, net_r_a, entry_price_a = g99.resolve_base_strategy(
        z_created, z_top, z_bottom, z_is_buy, high_, low_,
        last_high_of_bar, last_low_of_bar, spread_price, min_stop_price, MAX_WINDOW, False)

    entry_idx_c, exit_idx_c, outcome_c, net_r_c, rr_c, entry_price_c, eff_risk_c = g99.resolve_fixed_structural_target(
        z_created, z_top, z_bottom, z_is_buy, high_, low_,
        last_hh_of_bar, last_ll_of_bar, spread_price, min_stop_price, MAX_WINDOW)

    trades_a = build_trades(inst, pip, times, z_top, z_bottom, z_is_buy,
                             entry_idx_a, exit_idx_a, outcome_a, net_r_a, entry_price_a)
    trades_c = build_trades(inst, pip, times, z_top, z_bottom, z_is_buy,
                             entry_idx_c, exit_idx_c, outcome_c, net_r_c, entry_price_c)
    return trades_a, trades_c, len(z_created)


def build_equity_curve(trades, start=START, risk_frac=RISK_FRAC, cap=CAP, ruin_floor=RUIN_FLOOR):
    eq = start
    equity = np.empty(len(trades))
    for i, r in enumerate(trades["net_r"].values):
        risk_base = min(eq, cap)
        eq = max(eq + risk_base * risk_frac * r, ruin_floor)
        equity[i] = eq
    s = pd.Series(equity, index=pd.DatetimeIndex(trades["exit_time"].values), name="equity")
    return s[~s.index.duplicated(keep="last")].sort_index()


def period_stats(equity, freq):
    e = equity.resample(freq).last().ffill().dropna()
    ret = e.pct_change().dropna()
    n = len(ret)
    n_pos = int((ret > 0).sum())
    pct = 100.0 * n_pos / n if n else float("nan")
    return dict(n_periods=n, n_profitable=n_pos, pct_profitable=pct)


def report(label, trades):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    span_start = trades["entry_time"].min()
    span_end = trades["exit_time"].max()
    years = (span_end - span_start) / np.timedelta64(365, "D")
    days = (span_end - span_start) / np.timedelta64(1, "D")

    net_r = trades["net_r"].values
    wins = trades["is_win"].values
    wr = 100.0 * wins.sum() / len(trades)
    pf = net_r[net_r > 0].sum() / abs(net_r[net_r <= 0].sum())
    avg_pips_tp = trades.loc[wins, "pips_result"].mean()
    avg_pips_sl = trades.loc[~wins, "pips_result"].mean()

    print(f"Trades: {len(trades):,}  span: {span_start.date()} .. {span_end.date()} ({years:.1f}yr)")
    print(f"Win rate: {wr:.2f}%   Profit factor: {pf:.3f}")
    print(f"Avg pips per TP (winners): {avg_pips_tp:.2f}   Avg pips per SL (losers): {avg_pips_sl:.2f}")
    print(f"Trades/year: {len(trades)/years:,.0f}   Trades/day: {len(trades)/days:.1f}")
    print("Per-instrument:")
    for inst, g in trades.groupby("instrument"):
        gw = g["is_win"].values
        print(f"  {inst}: n={len(g):,}  WR={100*gw.mean():.1f}%  "
              f"avg_pips_tp={g.loc[gw,'pips_result'].mean():.2f}  "
              f"avg_pips_sl={g.loc[~gw,'pips_result'].mean():.2f}  sum_R={g['net_r'].sum():+.1f}")

    equity = build_equity_curve(trades)
    print(f"\nCapped-compounding equity: start=${START:.0f} -> final=${equity.iloc[-1]:,.0f} "
          f"(risk={RISK_FRAC*100:.0f}%, cap=${CAP:,.0f})")
    period_results = {}
    for freq, flabel in [("D", "Daily"), ("W-FRI", "Weekly"), ("M", "Monthly"), ("A", "Yearly")]:
        ps = period_stats(equity, freq)
        period_results[flabel] = ps
        print(f"  {flabel:8s}: {ps['n_profitable']:>5d} / {ps['n_periods']:<5d} profitable ({ps['pct_profitable']:.1f}%)")

    daily = equity.resample("D").last().ffill().dropna()
    daily_ret = daily.pct_change().dropna().values
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1)) * np.sqrt(252) if daily_ret.std(ddof=1) > 0 else float("nan")
    exkurt = sps.kurtosis(daily_ret, fisher=True, bias=False)
    print(f"Sharpe (annualized, daily returns): {sharpe:.3f}   Excess kurtosis: {exkurt:.3f}")

    tpy = len(trades) / years
    t1_raw, t10_raw, max_dd, ruined = run_mc_capped_numba(net_r, N_SIMS, BLOCK, MAX_TRADES, SEED)
    t1_hit = t1_raw[t1_raw > 0]
    t1_years = t1_hit / tpy
    pct_reached = 100 * len(t1_hit) / N_SIMS
    print(f"\nMonte Carlo ({N_SIMS:,} sims, block={BLOCK}): race to $1M")
    print(f"  Reached $1M: {pct_reached:.2f}% of sims")
    med_years = float(np.median(t1_years)) if len(t1_years) else float("nan")
    if len(t1_years):
        print(f"  Years to $1M -- median: {med_years:.2f}  mean: {t1_years.mean():.2f}")
    print(f"  %% ruined: {100*ruined.mean():.3f}%   Max DD median: {np.median(max_dd)*100:.2f}%")

    return dict(label=label, n=len(trades), wr=wr, pf=pf, avg_pips_tp=avg_pips_tp,
                avg_pips_sl=avg_pips_sl, tpy=tpy, tpd=len(trades)/days, final_equity=equity.iloc[-1],
                sharpe=sharpe, exkurt=exkurt, pct_reached_1m=pct_reached, med_years_1m=med_years,
                pct_ruined=100*ruined.mean(), period_results=period_results,
                daily_ret_mean=daily_ret.mean(), daily_ret_std=daily_ret.std(ddof=1))


def main():
    t0 = time.time()
    frames_a, frames_c = [], []
    for inst, (fname, pip) in INSTRUMENTS.items():
        t1 = time.time()
        ta, tc, n_zones = run_symbol(inst, fname, pip)
        frames_a.append(ta); frames_c.append(tc)
        print(f"{inst}: {n_zones:,} zones -> A:{len(ta):,} trades  C:{len(tc):,} trades  ({time.time()-t1:.1f}s)")

    trades_a = pd.concat(frames_a, ignore_index=True).sort_values("exit_time").reset_index(drop=True)
    trades_c = pd.concat(frames_c, ignore_index=True).sort_values("exit_time").reset_index(drop=True)

    stats_a = report("Variant A -- fixed 1:1R (all 4 pairs)", trades_a)
    stats_c = report("Variant C -- fixed last-confirmed HH/LL at entry (all 4 pairs)", trades_c)

    trades_a.to_csv(os.path.join(DATA_DIR, "goal101_variant_a_trades.csv"), index=False)
    trades_c.to_csv(os.path.join(DATA_DIR, "goal101_variant_c_trades.csv"), index=False)

    import json
    def clean(s):
        return {k: (v if not isinstance(v, dict) else v) for k, v in s.items()}
    with open(os.path.join(DATA_DIR, "goal101_summary.json"), "w") as f:
        json.dump({"A": clean(stats_a), "C": clean(stats_c)}, f, default=str, indent=2)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
