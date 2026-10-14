"""Equity simulation with HYBRID STOP: $100 starting capital, 5% risk per trade, SOMEGA pre-2020.

Exact reproduction of PRODLIVETRADING's live execution:
- Limit entry (0 spread cost)
- Limit take-profit (0 spread cost)
- Hybrid soft-stop: REDUCE_ONLY limit order at tight 1.0R stop (0 spread cost in >97% clean/delayed limit fills)
- 1.5x Backstop market order: catches gaps past the soft stop (exits at -1.5R, pays 1.6 pips spread cost)

Runs across all four instruments combined (EURUSD, GBPUSD, EURGBP, USDJPY) pre-2020.
Produces yearly, monthly, and weekly breakdown tables with profit factor at each interval.
"""
import os, sys
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import kurtosis as scipy_kurtosis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_somega_params import SESSION_HOUR_FILTER
from dtfx_eurusd_backtest_strict import find_pivots_strict
from dtfx_eurusd_backtest import run_structure_fsm

# ── Parameters ────────────────────────────────────────────────────────────────
MIN_RETRACEMENT     = 2
N_MULTIPLIER        = 3.343496
BACKSTOP_MULTIPLIER = 1.5
MIN_RISK_PCT        = 0.000170
MIN_FILL_AGE_H      = 3.366682

RISK_PCT   = 0.05          # 5% per trade
START_EQ   = 100.0
END_DATE   = pd.Timestamp("2019-12-31 23:59:59")
ANALYSIS_START = pd.Timestamp("2009-09-25")

FILL_SEARCH_FAST_BARS = 60 * 24 * 5
MAX_BARS_TO_RESOLVE = 60 * 24 * 30

STATUS_UNRESOLVED   = 0
STATUS_STOP_LIMIT   = 1     # clean or delayed limit fill at the original stop -- zero cost
STATUS_TARGET       = 2     # target limit fill -- zero cost
STATUS_STOP_BACKSTOP = 3    # backstop market exit -- pays spread cost

SPREAD_PIPS = {
    "EURUSD": 1.6,
    "GBPUSD": 1.6,
    "EURGBP": 1.6,
    "USDJPY": 1.6,
}
PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "EURGBP": 0.0001,
    "USDJPY": 0.01,
}

INSTRUMENTS = {
    "EURUSD": "lse_eurusd_1min_full.csv",
    "GBPUSD": "lse_gbpusd_1min_full.csv",
    "EURGBP": "lse_eurgbp_1min_full.csv",
    "USDJPY": "lse_usdjpy_1min_2018on.csv",
}

_price_cache = {}
_zone_cache  = {}


@njit(cache=True)
def sim_hybrid_numba(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                     opens, lows, highs, n_multiplier, backstop_multiplier, fast_bars, max_bars):
    n = len(zone_top)
    n_times = len(lows)
    entry_idx_out = np.full(n, -1, dtype=np.int64)
    status_out = np.zeros(n, dtype=np.int8)
    exit_r_out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        risk = n_multiplier * (top - bottom) / 2.0
        invalidation = bottom if is_buy else top
        start = zone_start_idx[i]
        if start >= n_times:
            continue

        fast_end = start + fast_bars
        if fast_end > n_times:
            fast_end = n_times
        breach_idx, mid_idx = -1, -1
        for j in range(start, fast_end):
            if is_buy:
                if breach_idx == -1 and lows[j] <= invalidation: breach_idx = j
                if mid_idx == -1 and lows[j] <= mid: mid_idx = j
            else:
                if breach_idx == -1 and highs[j] >= invalidation: breach_idx = j
                if mid_idx == -1 and highs[j] >= mid: mid_idx = j
            if breach_idx != -1 and mid_idx != -1:
                break
        if mid_idx == -1 and breach_idx == -1 and fast_end < n_times:
            for j in range(fast_end, n_times):
                if is_buy:
                    if breach_idx == -1 and lows[j] <= invalidation: breach_idx = j
                    if mid_idx == -1 and lows[j] <= mid: mid_idx = j
                else:
                    if breach_idx == -1 and highs[j] >= invalidation: breach_idx = j
                    if mid_idx == -1 and highs[j] >= mid: mid_idx = j
                if breach_idx != -1 and mid_idx != -1:
                    break
        if mid_idx == -1:
            continue
        if breach_idx != -1 and breach_idx <= mid_idx:
            continue

        entry_idx = mid_idx
        after_start = entry_idx + 1
        after_end = after_start + max_bars
        if after_end > n_times:
            after_end = n_times
        if after_start >= n_times:
            continue

        stop_level = mid - risk if is_buy else mid + risk
        target_level = mid + risk if is_buy else mid - risk
        backstop_level = mid - risk * backstop_multiplier if is_buy else mid + risk * backstop_multiplier

        sl_idx, tp_idx = -1, -1
        for j in range(after_start, after_end):
            if is_buy:
                if sl_idx == -1 and lows[j] <= stop_level: sl_idx = j
                if tp_idx == -1 and highs[j] >= target_level: tp_idx = j
            else:
                if sl_idx == -1 and highs[j] >= stop_level: sl_idx = j
                if tp_idx == -1 and lows[j] <= target_level: tp_idx = j
            if sl_idx != -1 and tp_idx != -1:
                break

        if sl_idx == -1 and tp_idx == -1:
            continue
        entry_idx_out[i] = entry_idx

        if tp_idx != -1 and (sl_idx == -1 or tp_idx <= sl_idx):
            status_out[i] = STATUS_TARGET
            exit_r_out[i] = 1.0
            continue

        # sl_idx comes first -- check for gap
        gapped = (opens[sl_idx] < stop_level) if is_buy else (opens[sl_idx] > stop_level)
        if not gapped:
            status_out[i] = STATUS_STOP_LIMIT
            exit_r_out[i] = -1.0
            continue

        # gapped: scan forward for backstop, refill limit, or target
        for j in range(sl_idx, after_end):
            if is_buy:
                backstop_hit = lows[j] <= backstop_level
                refill_hit = highs[j] >= stop_level
                target_hit = highs[j] >= target_level
            else:
                backstop_hit = highs[j] >= backstop_level
                refill_hit = lows[j] <= stop_level
                target_hit = lows[j] <= target_level
            if backstop_hit:
                status_out[i] = STATUS_STOP_BACKSTOP
                exit_r_out[i] = -backstop_multiplier
                break
            if refill_hit:
                status_out[i] = STATUS_STOP_LIMIT
                exit_r_out[i] = -1.0
                break
            if target_hit:
                status_out[i] = STATUS_TARGET
                exit_r_out[i] = 1.0
                break

    return entry_idx_out, status_out, exit_r_out


def load_instrument(name, filename):
    if name in _price_cache:
        return _price_cache[name]
    path = os.path.join(DATA_DIR, filename)
    print(f"  Loading {name} ...", flush=True)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[df.index >= ANALYSIS_START]
    df = df[["open", "high", "low", "close"]]
    times = df.index.values
    opens = np.ascontiguousarray(df["open"].values, dtype=np.float64)
    lows  = np.ascontiguousarray(df["low"].values,  dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    _price_cache[name] = (times, opens, lows, highs, df)
    print(f"    {len(df):,} bars  {df.index.min().date()} → {df.index.max().date()}")
    return _price_cache[name]


def prep_zones(name, filename):
    key = (name, MIN_RETRACEMENT)
    if key in _zone_cache:
        return _zone_cache[key]
    load_instrument(name, filename)
    times, opens, lows, highs, df = _price_cache[name]
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


def get_hybrid_trades(name, filename):
    z = prep_zones(name, filename)
    times, opens, lows, highs, df = _price_cache[name]
    entry_idx, status, exit_r = sim_hybrid_numba(
        z["top"], z["bottom"], z["mid"], z["is_buy"], z["start_idx"],
        opens, lows, highs, float(N_MULTIPLIER), float(BACKSTOP_MULTIPLIER),
        FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE)

    filled   = (entry_idx >= 0) & (status != STATUS_UNRESOLVED)
    risk     = N_MULTIPLIER * (z["top"] - z["bottom"]) / 2.0
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
        return pd.DataFrame(columns=["entry_time", "net_r", "status", "instrument"])

    idx = np.where(keep)[0]
    realized_r = exit_r[idx]
    
    # Spread cost applied ONLY to STATUS_STOP_BACKSTOP (market exit on gap)
    spread_val = SPREAD_PIPS[name] * PIP_SIZE[name]
    cost_in_r  = spread_val / risk[idx]
    pays_cost  = status[idx] == STATUS_STOP_BACKSTOP
    net_r      = realized_r - np.where(pays_cost, cost_in_r, 0.0)

    return pd.DataFrame({
        "entry_time":  pd.to_datetime(entry_times_ns[idx]),
        "net_r":       net_r,
        "status":      status[idx],
        "instrument":  name,
    }).sort_values("entry_time").reset_index(drop=True)


def simulate(trades, risk_pct, start_eq):
    eq = start_eq
    rows = []
    cumulative_gross_win  = 0.0
    cumulative_gross_loss = 0.0

    for _, t in trades.iterrows():
        eq = max(eq * (1 + risk_pct * t["net_r"]), 0.0)
        if t["net_r"] > 0:
            cumulative_gross_win  += t["net_r"]
        else:
            cumulative_gross_loss += abs(t["net_r"])
        pf = (cumulative_gross_win / cumulative_gross_loss
              if cumulative_gross_loss > 0 else float("inf"))
        rows.append({
            "entry_time": t["entry_time"],
            "instrument": t["instrument"],
            "net_r":      t["net_r"],
            "status":     t["status"],
            "equity":     eq,
            "cum_trades": len(rows) + 1,
            "profit_factor": pf,
        })
    return pd.DataFrame(rows)


def period_table(sim, freq):
    sim = sim.copy()
    sim["entry_time"] = pd.to_datetime(sim["entry_time"])
    sim = sim.set_index("entry_time").sort_index()

    eq_by_period = sim["equity"].resample(freq).last().ffill()
    trades_per   = sim["net_r"].resample(freq).count()
    wins_per     = sim["net_r"].resample(freq).apply(lambda x: (x > 0).sum())
    pf_by_period = sim["profit_factor"].resample(freq).last().ffill()

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


def main():
    print("Loading instruments and simulating Hybrid Stop...")
    all_trades = []
    for name, fname in INSTRUMENTS.items():
        t = get_hybrid_trades(name, fname)
        if len(t):
            all_trades.append(t)
            target_n = (t["status"] == STATUS_TARGET).sum()
            limit_n  = (t["status"] == STATUS_STOP_LIMIT).sum()
            back_n   = (t["status"] == STATUS_STOP_BACKSTOP).sum()
            print(f"  {name}: {len(t):,} trades | Targets: {target_n:,}, Limit Stops: {limit_n:,}, Backstops (Gaps): {back_n:,} ({back_n/(limit_n+back_n)*100:.2f}% of stops)")

    merged = pd.concat(all_trades).sort_values("entry_time").reset_index(drop=True)
    print(f"\nTotal combined trades: {len(merged):,}")

    sim = simulate(merged, RISK_PCT, START_EQ)

    gross_win  = sim.loc[sim["net_r"] > 0, "net_r"].sum()
    gross_loss = sim.loc[sim["net_r"] < 0, "net_r"].abs().sum()
    overall_pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    win_rate   = (sim["net_r"] > 0).mean()
    final_eq   = sim["equity"].iloc[-1]

    curve  = sim.set_index("entry_time")["equity"]
    daily  = curve.resample("D").last().ffill()
    ret    = daily.pct_change().dropna()
    ret_nz = ret[ret != 0]
    dd     = daily / daily.cummax() - 1
    maxdd  = abs(dd.min())
    n_years = (daily.index[-1] - daily.index[0]).days / 365.25
    sharpe = (ret_nz.mean() / ret_nz.std() * np.sqrt(252) if ret_nz.std() > 0 else 0.0)
    total_ret = final_eq / START_EQ - 1
    ann_ret   = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    print("=" * 70)
    print(f"  HYBRID STOP GRAND SUMMARY  ($100 @ 5% risk, pre-2020)")
    print("=" * 70)
    print(f"  Start:          {sim['entry_time'].min().date()}")
    print(f"  End:            {sim['entry_time'].max().date()}  ({n_years:.1f} years)")
    print(f"  Total trades:   {len(sim):,}")
    print(f"  Win rate:       {win_rate*100:.2f}%")
    print(f"  Profit Factor:  {overall_pf:.3f}")
    print(f"  Ann. Sharpe:    {sharpe:.2f}")
    print(f"  Max Drawdown:   {maxdd*100:.1f}%")
    print(f"  Final equity:   ${final_eq:,.2f}")

    # Tables
    weekly = period_table(sim, "W")
    weekly.index = weekly.index.strftime("%Y-%m-%d")

    monthly = period_table(sim, "M")
    monthly.index = monthly.index.strftime("%Y-%m")

    yearly = period_table(sim, "Y")
    yearly.index = yearly.index.strftime("%Y")

    print("\n\n" + "=" * 100)
    print("  YEARLY EQUITY TABLE (HYBRID STOP)")
    print("=" * 100)
    with pd.option_context("display.max_rows", 50, "display.width", 200,
                            "display.float_format", lambda x: f"{x:,.2f}"):
        print(yearly.to_string())

    print("\n\n" + "=" * 100)
    print("  MONTHLY EQUITY TABLE (HYBRID STOP)")
    print("=" * 100)
    with pd.option_context("display.max_rows", 500, "display.width", 200,
                            "display.float_format", lambda x: f"{x:,.2f}"):
        print(monthly.to_string())

    weekly.to_csv(os.path.join(DATA_DIR,  "somega_pre2020_hybrid_5pct_weekly.csv"))
    monthly.to_csv(os.path.join(DATA_DIR, "somega_pre2020_hybrid_5pct_monthly.csv"))
    yearly.to_csv(os.path.join(DATA_DIR,  "somega_pre2020_hybrid_5pct_yearly.csv"))
    print(f"\nSaved CSVs to {DATA_DIR}")


if __name__ == "__main__":
    main()
