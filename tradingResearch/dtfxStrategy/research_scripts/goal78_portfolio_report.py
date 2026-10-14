"""Direct user request (2026-09-06): a portfolio-level report of the corrected dynamic-zigzag
zone rule, INVERTED (fade the pullback -- validated in goal77), across all four symbols
(EURUSD, GBPUSD, USDJPY, EURGBP) combined into one trade stream. Reuses goal77's exact
zone-detection/resolution code (unmodified) and goal10's exact capped-compounding /
block-bootstrap Monte Carlo engine (unmodified) -- this script only merges the four
per-symbol trade streams and computes the requested report statistics.

Requested: win rate, R:R, trades/year, trades/day, % profitable days/weeks/months/years,
excess kurtosis, Sharpe ratio, and a Monte Carlo race-to-$1M.

Methodology notes (please read before trusting the numbers):
  - Every trade risks a fixed 5% of current equity, capped at 5% of $2,000,000 once equity
    exceeds that (same realistic-cap convention as goal10_cana_nt_mc_capped.py) -- otherwise
    naive uncapped compounding at this trade frequency overflows to a meaningless number
    within a few years of history.
  - The four symbols' trades are merged into ONE chronological stream ordered by EXIT time
    (when a trade's R is realized, not when it was opened) and treated as sequential risk --
    i.e. this assumes trades are risked one after another, not that four independent
    positions can be open and sized simultaneously. Real concurrent open risk across pairs
    is NOT modeled; treat this as a single combined track record, not a margin-aware
    portfolio simulation.
  - Sharpe and excess kurtosis are computed on DAILY returns of that equity curve
    (business-day calendar), annualized with sqrt(252). "Profitable days/weeks/months/years"
    use the same equity curve resampled to each frequency; a period with zero trades is a
    flat (non-profitable) period, not excluded.
  - This is a backtest on modeled spread cost only -- no slippage, latency, broker fills, or
    live verification. Research-grade, not a live track record.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import goal77_dynamic_zigzag_backtest as g77
from goal10_cana_nt_mc_capped import run_mc_capped_numba, START, RISK_FRAC, CAP, T1, T10, RUIN_FLOOR

rs, pg = g77.rs, g77.pg
INSTRUMENTS = ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"]
MAX_WINDOW = 1440
MIN_STOP_PIPS = 3.0
N_SIMS = 25_000
BLOCK = 20
MAX_TRADES = 60_000
SEED = 42


def run_symbol(inst):
    times, lows, highs, df = rs.load_instrument(inst)
    high_ = df["high"].values.astype(np.float64)
    low_ = df["low"].values.astype(np.float64)
    pip = pg.PIP_SIZE[inst]
    spread_price = pg.SPREAD_PIPS_TYPICAL * pip

    confirm_idx, pivot_idx, is_high, price = g77._zigzag_pivots(high_, low_)
    z_created, z_top, z_bottom, z_is_buy, z_is_flip = g77._build_zones(confirm_idx, is_high, price)
    touch, exit_, outcome, net_r = g77._resolve_zones(
        z_created, z_top, z_bottom, z_is_buy, high_, low_, MAX_WINDOW,
        MIN_STOP_PIPS * pip, spread_price, invert=True)

    resolved = outcome > 0
    entry_idx = touch[resolved]
    exit_idx = exit_[resolved]
    out = pd.DataFrame({
        "instrument": inst,
        "entry_time": times[entry_idx],
        "exit_time": times[exit_idx],
        "net_r": net_r[resolved],
        "is_flip": z_is_flip[resolved],
        "is_win": net_r[resolved] > 0,
    })
    return out


def build_equity_curve(trades, start=START, risk_frac=RISK_FRAC, cap=CAP, ruin_floor=RUIN_FLOOR):
    """Deterministic (non-bootstrapped) capped-compounding equity curve over the ACTUAL
    historical trade sequence, ordered by exit time -- same cap convention as the MC engine,
    so the daily/weekly/monthly/yearly stats and the MC race use one consistent account model."""
    eq = start
    equity = np.empty(len(trades))
    for i, r in enumerate(trades["net_r"].values):
        risk_base = min(eq, cap)
        eq = max(eq + risk_base * risk_frac * r, ruin_floor)
        equity[i] = eq
    s = pd.Series(equity, index=pd.DatetimeIndex(trades["exit_time"].values), name="equity")
    return s[~s.index.duplicated(keep="last")].sort_index()


def period_stats(equity, freq, label):
    e = equity.resample(freq).last().ffill().dropna()
    ret = e.pct_change().dropna()
    n = len(ret)
    n_pos = int((ret > 0).sum())
    pct = 100.0 * n_pos / n if n else float("nan")
    return dict(label=label, n_periods=n, n_profitable=n_pos, pct_profitable=pct)


def main():
    t0 = time.time()
    print("=== Per-symbol backtests (full history, inverted / fade-the-pullback) ===")
    frames = []
    for inst in INSTRUMENTS:
        df = run_symbol(inst)
        frames.append(df)
        wr = 100.0 * df["is_win"].mean()
        years = (df["exit_time"].max() - df["entry_time"].min()) / np.timedelta64(365, "D")
        print(f"  {inst}: {len(df):,} trades, WR={wr:.1f}%, {len(df)/years:.0f}/yr  ({time.time()-t0:.0f}s)")

    trades = pd.concat(frames, ignore_index=True).sort_values("exit_time").reset_index(drop=True)
    print(f"\nPortfolio: {len(trades):,} combined trades ({time.time()-t0:.0f}s)")

    span_start = trades["entry_time"].min()
    span_end = trades["exit_time"].max()
    years = (span_end - span_start) / np.timedelta64(365, "D")
    days = (span_end - span_start) / np.timedelta64(1, "D")

    net_r = trades["net_r"].values
    wins = net_r[net_r > 0]
    losses = net_r[net_r <= 0]
    wr = 100.0 * len(wins) / len(net_r)
    avg_win = wins.mean()
    avg_loss = losses.mean()  # negative
    rr = avg_win / abs(avg_loss)
    pf = wins.sum() / abs(losses.sum())

    print("\n=== Portfolio-level trade stats ===")
    print(f"  Trades: {len(trades):,}  |  span: {span_start.date()} to {span_end.date()} ({years:.1f} yrs)")
    print(f"  Win rate: {wr:.2f}%")
    print(f"  Avg win: {avg_win:.3f}R   Avg loss: {avg_loss:.3f}R   R:R = {rr:.3f}")
    print(f"  Profit factor: {pf:.3f}")
    print(f"  Trades/year: {len(trades)/years:.1f}")
    print(f"  Trades/day:  {len(trades)/days:.2f}")

    print("\n=== Per-symbol contribution ===")
    for inst, g in trades.groupby("instrument"):
        print(f"  {inst}: n={len(g):,} WR={100*g['is_win'].mean():.1f}% "
              f"avg_win={g.loc[g.net_r>0,'net_r'].mean():.3f}R sum_R={g['net_r'].sum():.1f}")

    # ---- equity curve, daily/weekly/monthly/yearly ----
    equity = build_equity_curve(trades)
    print(f"\n=== Capped-compounding equity curve (start=${START:.0f}, risk={RISK_FRAC*100:.0f}%, cap=${CAP:,.0f}) ===")
    print(f"  Final equity: ${equity.iloc[-1]:,.0f}")

    for freq, label in [("D", "Daily"), ("W-FRI", "Weekly"), ("M", "Monthly"), ("A", "Yearly")]:
        ps = period_stats(equity, freq, label)
        print(f"  {label:8s}: {ps['n_profitable']:>5d} / {ps['n_periods']:<5d} profitable "
              f"({ps['pct_profitable']:.1f}%)")

    daily = equity.resample("D").last().ffill().dropna()
    daily_ret = daily.pct_change().dropna().values
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1)) * np.sqrt(252) if daily_ret.std(ddof=1) > 0 else float("nan")
    exkurt = sps.kurtosis(daily_ret, fisher=True, bias=False)
    skew = sps.skew(daily_ret, bias=False)
    print(f"\n=== Return distribution (daily) ===")
    print(f"  Sharpe (annualized): {sharpe:.3f}")
    print(f"  Excess kurtosis:     {exkurt:.3f}")
    print(f"  Skew:                {skew:.3f}")

    # ---- Monte Carlo: race to $1M ----
    print(f"\n=== Monte Carlo: {N_SIMS:,} block-bootstrap sims, block={BLOCK}, race to ${T1:,.0f} ===")
    tpy = len(trades) / years
    t1_raw, t10_raw, max_dd, ruined = run_mc_capped_numba(net_r, N_SIMS, BLOCK, MAX_TRADES, SEED)
    t1_hit = t1_raw[t1_raw > 0]
    t1_years = t1_hit / tpy
    pct_reached = 100 * len(t1_hit) / N_SIMS
    print(f"  Reached $1M: {pct_reached:.2f}% of sims")
    if len(t1_years):
        print(f"  Years to $1M -- median: {np.median(t1_years):.2f}  mean: {t1_years.mean():.2f}")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f"    P{p:2d}: {np.percentile(t1_years, p):.2f} yrs")
    print(f"  % ruined (equity hit ${RUIN_FLOOR:.0f} floor): {100*ruined.mean():.3f}%")
    dd_pct = max_dd * 100
    print(f"  Max drawdown -- median: {np.median(dd_pct):.2f}%  P95 worst: {np.percentile(dd_pct,5):.2f}%")

    trades.to_csv(os.path.join(SCRIPT_DIR, "scratch", "goal78_portfolio_trades.csv"), index=False)
    equity.to_csv(os.path.join(SCRIPT_DIR, "scratch", "goal78_portfolio_equity.csv"))
    print(f"\nSaved trades + equity curve to scratch/. ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
