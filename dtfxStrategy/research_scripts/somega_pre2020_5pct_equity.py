"""Equity simulation: $100 starting capital, 5% risk per trade, SOMEGA pre-2020.

Runs the exact same trade generation as somega_pre2020_backtest.py (data is
re-loaded fresh; Numba JIT cache means zone simulation is fast after the first
run), then simulates all four instruments combined as a single account at 5%
risk/trade. Reports equity at every weekly, monthly, and yearly interval.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_robust_search import (
    _simulate_zones_numba, _indicator_cache, _price_cache, _zone_cache,
    ANALYSIS_START, FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE,
)
from dtfx_somega_params import SESSION_HOUR_FILTER
from dtfx_eurusd_backtest_strict import find_pivots_strict
from dtfx_eurusd_backtest import run_structure_fsm
from dtfx_robust_search import _rsi, _atr_ratio

# ── Parameters ────────────────────────────────────────────────────────────────
MIN_RETRACEMENT = 2
N_MULTIPLIER    = 3.343496
MIN_RISK_PCT    = 0.000170
MIN_FILL_AGE_H  = 3.366682

RISK_PCT   = 0.05          # 5% per trade
START_EQ   = 100.0
END_DATE   = pd.Timestamp("2019-12-31 23:59:59")

INSTRUMENTS = {
    "EURUSD": "lse_eurusd_1min_full.csv",
    "GBPUSD": "lse_gbpusd_1min_full.csv",
    "EURGBP": "lse_eurgbp_1min_full.csv",
    "USDJPY": "lse_usdjpy_1min_2018on.csv",
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_instrument_custom(name, filename):
    if name in _price_cache:
        return _price_cache[name]
    path = os.path.join(DATA_DIR, filename)
    print(f"  Loading {name} ...", flush=True)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[df.index >= ANALYSIS_START]
    df = df[["open", "high", "low", "close"]]
    _indicator_cache[name] = (_rsi(df["close"]).values, _atr_ratio(df).values)
    times = df.index.values
    lows  = np.ascontiguousarray(df["low"].values,  dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    _price_cache[name] = (times, lows, highs, df)
    print(f"    {len(df):,} bars  {df.index.min().date()} → {df.index.max().date()}")
    return _price_cache[name]


def prep_zones(name, filename):
    key = (name, MIN_RETRACEMENT)
    if key in _zone_cache:
        return _zone_cache[key]
    load_instrument_custom(name, filename)
    times, lows, highs, df = _price_cache[name]
    zigzag = find_pivots_strict(df, MIN_RETRACEMENT)
    _, zones = run_structure_fsm(zigzag)
    top     = np.array([z["top"]    for z in zones], dtype=np.float64)
    bottom  = np.array([z["bottom"] for z in zones], dtype=np.float64)
    mid     = np.array([z["mid"]    for z in zones], dtype=np.float64)
    is_buy  = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
    created = np.array([np.datetime64(z["created_at"]) for z in zones])
    start_idx = np.searchsorted(times, created, side="right").astype(np.int64)
    packed = dict(top=top, bottom=bottom, mid=mid, is_buy=is_buy,
                  start_idx=start_idx, created_at=created)
    _zone_cache[key] = packed
    print(f"    {name}: {len(zones):,} zones found")
    return packed


def get_trades(name, filename):
    z = prep_zones(name, filename)
    times, lows, highs, df = _price_cache[name]
    entry_idx, status = _simulate_zones_numba(
        z["top"], z["bottom"], z["mid"], z["is_buy"], z["start_idx"],
        lows, highs, float(N_MULTIPLIER), FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE)

    filled  = entry_idx >= 0
    risk    = N_MULTIPLIER * (z["top"] - z["bottom"]) / 2.0
    risk_pct = risk / z["mid"]
    entry_times_ns = times[np.where(filled, entry_idx, 0)]
    fill_hours = (entry_times_ns - z["created_at"]) / np.timedelta64(1, "h")

    keep = filled & (risk_pct >= MIN_RISK_PCT) & (fill_hours >= MIN_FILL_AGE_H)

    allowed = SESSION_HOUR_FILTER.get(name)
    if allowed is not None:
        hrs = pd.to_datetime(entry_times_ns).hour
        keep = keep & np.isin(hrs, list(allowed))

    keep = keep & (entry_times_ns <= np.datetime64(END_DATE))
    if not keep.any():
        return pd.DataFrame(columns=["entry_time", "r_gross", "instrument"])

    idx = np.where(keep)[0]
    return pd.DataFrame({
        "entry_time":  pd.to_datetime(entry_times_ns[idx]),
        "r_gross":     np.where(status[idx] == 2, 1.0, -1.0),
        "instrument":  name,
    }).sort_values("entry_time").reset_index(drop=True)


# ── Equity simulation ─────────────────────────────────────────────────────────
def simulate(trades, risk_pct, start_eq):
    eq = start_eq
    rows = []
    cumulative_wins  = 0
    cumulative_gross_win  = 0.0
    cumulative_gross_loss = 0.0

    for _, t in trades.iterrows():
        eq = max(eq * (1 + risk_pct * t["r_gross"]), 0.0)
        if t["r_gross"] > 0:
            cumulative_wins += 1
            cumulative_gross_win  += t["r_gross"]
        else:
            cumulative_gross_loss += abs(t["r_gross"])
        pf = (cumulative_gross_win / cumulative_gross_loss
              if cumulative_gross_loss > 0 else float("inf"))
        rows.append({
            "entry_time": t["entry_time"],
            "instrument": t["instrument"],
            "r_gross":    t["r_gross"],
            "equity":     eq,
            "cum_trades": len(rows) + 1,
            "cum_wins":   cumulative_wins,
            "profit_factor": pf,
        })
    return pd.DataFrame(rows)


def period_table(sim, freq, label):
    """Build period-end equity table with trades-in-period and running PF."""
    sim = sim.copy()
    sim["entry_time"] = pd.to_datetime(sim["entry_time"])
    sim = sim.set_index("entry_time").sort_index()

    # Period-end equity: last equity value in each period
    eq_by_period = sim["equity"].resample(freq).last().ffill()

    # Trades per period
    trades_per = sim["r_gross"].resample(freq).count()
    wins_per   = sim["r_gross"].resample(freq).apply(lambda x: (x > 0).sum())

    # Running PF at period end
    pf_by_period = sim["profit_factor"].resample(freq).last().ffill()

    # Period P&L
    prev_eq = eq_by_period.shift(1).fillna(START_EQ)
    pnl = eq_by_period - prev_eq
    pct = (eq_by_period / prev_eq - 1) * 100

    tbl = pd.DataFrame({
        "equity":         eq_by_period.round(2),
        "period_pnl":     pnl.round(2),
        "period_pct":     pct.round(2),
        "trades":         trades_per.astype(int),
        "wins":           wins_per.astype(int),
        "profit_factor":  pf_by_period.round(3),
    }).dropna(subset=["equity"])
    tbl = tbl[tbl["trades"] > 0]
    return tbl


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading instruments and running zone simulation...")
    all_trades = []
    for name, fname in INSTRUMENTS.items():
        t = get_trades(name, fname)
        if len(t):
            all_trades.append(t)
            print(f"  {name}: {len(t):,} trades pre-2020")

    merged = pd.concat(all_trades).sort_values("entry_time").reset_index(drop=True)
    print(f"\nTotal combined trades: {len(merged):,}")

    print(f"\nSimulating ${START_EQ:.0f} at {RISK_PCT*100:.0f}% risk/trade ...\n")
    sim = simulate(merged, RISK_PCT, START_EQ)

    # ── Grand summary ────────────────────────────────────────────────────────
    gross_win  = sim.loc[sim["r_gross"] > 0, "r_gross"].sum()
    gross_loss = sim.loc[sim["r_gross"] < 0, "r_gross"].abs().sum()
    overall_pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    win_rate   = (sim["r_gross"] > 0).mean()
    final_eq   = sim["equity"].iloc[-1]

    # Drawdown
    curve  = sim.set_index("entry_time")["equity"]
    daily  = curve.resample("D").last().ffill()
    ret    = daily.pct_change().dropna()
    ret_nz = ret[ret != 0]
    dd     = daily / daily.cummax() - 1
    maxdd  = abs(dd.min())
    n_years = (daily.index[-1] - daily.index[0]).days / 365.25
    sharpe = (ret_nz.mean() / ret_nz.std() * np.sqrt(252)
              if ret_nz.std() > 0 else 0.0)
    total_ret = final_eq / START_EQ - 1
    ann_ret   = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    print("=" * 65)
    print(f"  GRAND SUMMARY  (${START_EQ:.0f} → 5% risk/trade, all instruments)")
    print("=" * 65)
    print(f"  Start:          {sim['entry_time'].min().date()}")
    print(f"  End:            {sim['entry_time'].max().date()}  ({n_years:.1f} years)")
    print(f"  Total trades:   {len(sim):,}")
    print(f"  Win rate:       {win_rate*100:.1f}%")
    print(f"  Profit Factor:  {overall_pf:.3f}")
    print(f"  Ann. Sharpe:    {sharpe:.2f}")
    print(f"  Ann. Return:    {ann_ret*100:.1f}%")
    print(f"  Max Drawdown:   {maxdd*100:.1f}%")
    print(f"  Final equity:   ${final_eq:,.2f}")
    print(f"  Total return:   {total_ret*100:,.1f}%")

    # ── Weekly table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  WEEKLY EQUITY  (period end, trades during week, profit factor)")
    print("=" * 100)
    weekly = period_table(sim, "W", "Week")
    weekly.index = weekly.index.strftime("%Y-%m-%d")
    with pd.option_context("display.max_rows", 1000, "display.width", 200,
                            "display.float_format", lambda x: f"{x:,.2f}"):
        print(weekly.to_string())

    # ── Monthly table ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  MONTHLY EQUITY  (period end, trades during month, profit factor)")
    print("=" * 100)
    monthly = period_table(sim, "M", "Month")
    monthly.index = monthly.index.strftime("%Y-%m")
    with pd.option_context("display.max_rows", 500, "display.width", 200,
                            "display.float_format", lambda x: f"{x:,.2f}"):
        print(monthly.to_string())

    # ── Yearly table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("  YEARLY EQUITY  (period end, trades during year, profit factor)")
    print("=" * 100)
    yearly = period_table(sim, "Y", "Year")
    yearly.index = yearly.index.strftime("%Y")
    with pd.option_context("display.max_rows", 50, "display.width", 200,
                            "display.float_format", lambda x: f"{x:,.2f}"):
        print(yearly.to_string())

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    weekly.to_csv(os.path.join(DATA_DIR,  "somega_pre2020_5pct_weekly.csv"))
    monthly.to_csv(os.path.join(DATA_DIR, "somega_pre2020_5pct_monthly.csv"))
    yearly.to_csv(os.path.join(DATA_DIR,  "somega_pre2020_5pct_yearly.csv"))
    print(f"\n  CSVs saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
