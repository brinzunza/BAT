"""Cross-Asset Generalization Backtest for DTFX / Omega / SOMEGA.

Evaluates the strategy across 9 distinct instruments spanning 6 asset classes:
1. Forex Majors: EUR/USD, GBP/USD
2. Forex Crosses: EUR/GBP, USD/JPY
3. Cryptocurrencies: BTC/USD (Bitcoin)
4. Precious Metals / Commodities: XAU/USD (Gold)
5. Equity Index ETFs: SPY (S&P 500 ETF)
6. Single Stock Equities: AAPL (Apple Inc)
7. Index Futures: ES_F (E-mini S&P 500 Continuous Futures)

Tests:
- Baseline Omega performance (1:1 RR, 3.37h fill age, 0.017% min risk)
- Win Rate, Profit Factor, Annualized Sharpe, Max Drawdown, CAGR (at 1% risk)
- Hour-of-day edge analysis for each market
"""
import os, sys
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_eurusd_backtest_strict import find_pivots_strict
from dtfx_eurusd_backtest import run_structure_fsm
from dtfx_somega_params import SESSION_HOUR_FILTER

MIN_RETRACEMENT     = 2
N_MULTIPLIER        = 3.343496
MIN_RISK_PCT        = 0.000170
MIN_FILL_AGE_HOURS  = 3.366682
FILL_SEARCH_FAST_BARS = 60 * 24 * 5
MAX_BARS_TO_RESOLVE = 60 * 24 * 30

MARKETS = {
    # Forex
    "EURUSD": {"file": "lse_eurusd_1min_full.csv", "class": "Forex Major", "spread_pips": 1.6, "pip_size": 0.0001},
    "GBPUSD": {"file": "lse_gbpusd_1min_full.csv", "class": "Forex Major", "spread_pips": 1.6, "pip_size": 0.0001},
    "EURGBP": {"file": "lse_eurgbp_1min_full.csv", "class": "Forex Cross", "spread_pips": 1.6, "pip_size": 0.0001},
    "USDJPY": {"file": "lse_usdjpy_1min_2018on.csv", "class": "Forex Cross", "spread_pips": 1.5, "pip_size": 0.01},
    # Crypto
    "BTCUSD": {"file": "lse_btcusd_1min_2018on.csv", "class": "Crypto", "spread_pips": 15.0, "pip_size": 1.0},
    # Commodities / Metals
    "XAUUSD": {"file": "lse_xauusd_full_1min.csv", "class": "Commodity / Gold", "spread_pips": 3.5, "pip_size": 0.1},
    # Equities & Indices
    "SPY":    {"file": "lse_spy_1min.csv", "class": "Index ETF (S&P 500)", "spread_pips": 1.0, "pip_size": 0.01},
    "AAPL":   {"file": "lse_aapl_1min.csv", "class": "Single Stock (Apple)", "spread_pips": 1.0, "pip_size": 0.01},
    "ES_F":   {"file": "lse_esf_1min.csv", "class": "Index Futures (E-mini)", "spread_pips": 1.0, "pip_size": 0.25},
}

_cache = {}


@njit(cache=True)
def simulate_zones_fast(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                        lows, highs, n_multiplier, fast_bars, max_bars):
    n = len(zone_top)
    n_times = len(lows)
    entry_idx_out = np.full(n, -1, dtype=np.int64)
    status_out = np.zeros(n, dtype=np.int8)

    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        risk = n_multiplier * (top - bottom) / 2.0
        invalidation = bottom if is_buy else top
        start = zone_start_idx[i]
        if start >= n_times:
            continue

        fast_end = min(start + fast_bars, n_times)
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
        after_end = min(after_start + max_bars, n_times)
        if after_start >= n_times:
            continue

        stop_level = mid - risk if is_buy else mid + risk
        target_level = mid + risk if is_buy else mid - risk

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
            status_out[i] = 2  # target
        else:
            status_out[i] = 1  # stop

    return entry_idx_out, status_out


def run_asset(name, meta):
    print(f"\nProcessing {name} ({meta['class']})...", flush=True)
    path = os.path.join(DATA_DIR, meta["file"])
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None
        
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[["open", "high", "low", "close"]].dropna()
    
    n_bars = len(df)
    times = df.index.values
    lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    years = (df.index.max() - df.index.min()).days / 365.25
    print(f"  {n_bars:,} bars | {start_date} to {end_date} ({years:.1f} years)", flush=True)
    
    zigzag = find_pivots_strict(df, MIN_RETRACEMENT)
    _, zones = run_structure_fsm(zigzag)
    n_zones = len(zones)
    print(f"  {n_zones:,} zones detected", flush=True)
    
    top = np.array([z["top"] for z in zones], dtype=np.float64)
    bottom = np.array([z["bottom"] for z in zones], dtype=np.float64)
    mid = np.array([z["mid"] for z in zones], dtype=np.float64)
    is_buy = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
    created_at = np.array([np.datetime64(z["created_at"]) for z in zones])
    start_idx = np.searchsorted(times, created_at, side="right").astype(np.int64)
    
    entry_idx, status = simulate_zones_fast(
        top, bottom, mid, is_buy, start_idx,
        lows, highs, float(N_MULTIPLIER),
        FILL_SEARCH_FAST_BARS, MAX_BARS_TO_RESOLVE
    )
    
    filled = entry_idx >= 0
    risk = N_MULTIPLIER * (top - bottom) / 2.0
    risk_pct = risk / mid
    entry_times_ns = times[np.where(filled, entry_idx, 0)]
    fill_hours = (entry_times_ns - created_at) / np.timedelta64(1, "h")
    
    # ── BASELINE OMEGA FILTER ──
    keep_base = (filled & (risk_pct >= MIN_RISK_PCT) & (fill_hours >= MIN_FILL_AGE_HOURS))
    idx_base = np.where(keep_base)[0]
    
    t_df = pd.DataFrame({
        "entry_time": pd.to_datetime(entry_times_ns[idx_base]),
        "status": status[idx_base],
        "win": status[idx_base] == 2,
        "r_gross": np.where(status[idx_base] == 2, 1.0, -1.0),
        "hour": pd.to_datetime(entry_times_ns[idx_base]).hour,
    }).sort_values("entry_time").reset_index(drop=True)
    
    # Calculate Metrics at 1% Risk Compounding
    def score_equity(trades_df):
        if len(trades_df) < 10:
            return 0.0, 0.0, 0.0, 0.0
        eq = 100.0
        curve = []
        for _, r in trades_df.iterrows():
            eq = max(eq * (1 + 0.01 * r["r_gross"]), 0.0)
            curve.append((r["entry_time"], eq))
        s = pd.Series([c[1] for c in curve], index=[c[0] for c in curve])
        daily = s.resample("D").last().ffill()
        d_ret = daily.pct_change().dropna()
        d_ret_nz = d_ret[d_ret != 0]
        sharpe = (d_ret_nz.mean() / d_ret_nz.std() * np.sqrt(252)) if d_ret_nz.std() > 0 else 0.0
        dd = daily / daily.cummax() - 1.0
        maxdd = abs(dd.min())
        tot_ret = s.iloc[-1] / 100.0 - 1.0
        cagr = (1 + tot_ret) ** (1 / max(years, 0.1)) - 1.0
        return sharpe, maxdd, cagr, s.iloc[-1]

    # Baseline Stats
    n_base = len(t_df)
    wr_base = t_df["win"].mean() if n_base else 0.0
    wins_base = t_df["win"].sum()
    losses_base = n_base - wins_base
    pf_base = wins_base / losses_base if losses_base > 0 else float("inf")
    sharpe_base, maxdd_base, cagr_base, _ = score_equity(t_df)
    
    # ── SOMEGA SESSION FILTER STATS ──
    allowed_hrs = SESSION_HOUR_FILTER.get(name)
    if allowed_hrs is not None:
        t_somega = t_df[t_df["hour"].isin(allowed_hrs)].copy()
    else:
        t_somega = t_df.copy()
        
    n_som = len(t_somega)
    wr_som = t_somega["win"].mean() if n_som else 0.0
    wins_som = t_somega["win"].sum()
    losses_som = n_som - wins_som
    pf_som = wins_som / losses_som if losses_som > 0 else float("inf")
    sharpe_som, maxdd_som, cagr_som, _ = score_equity(t_somega)
    
    # Check best UTC hour for this instrument
    hour_stats = t_df.groupby("hour").agg(n=("win", "size"), wr=("win", "mean")).sort_values("wr", ascending=False)
    best_hours = list(hour_stats[hour_stats["n"] >= 30].head(3).index)
    
    return {
        "instrument": name,
        "asset_class": meta["class"],
        "years": f"{years:.1f}y",
        "date_range": f"{start_date} → {end_date}",
        "raw_zones": n_zones,
        # Baseline Omega
        "base_trades": n_base,
        "base_win_rate": wr_base,
        "base_pf": pf_base,
        "base_sharpe": sharpe_base,
        "base_maxdd": maxdd_base,
        "base_cagr": cagr_base,
        # SOMEGA / Best Hours
        "somega_trades": n_som,
        "somega_win_rate": wr_som,
        "somega_pf": pf_som,
        "somega_sharpe": sharpe_som,
        "somega_maxdd": maxdd_som,
        "best_utc_hours": str(best_hours),
    }


def main():
    print("="*90)
    print("  CROSS-ASSET GENERALIZATION BACKTEST: ALL MARKETS & TICKERS")
    print("="*90)
    
    results = []
    for name, meta in MARKETS.items():
        res = run_asset(name, meta)
        if res:
            results.append(res)
            
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*115)
    print("  CROSS-MARKET BACKTEST SUMMARY (BASELINE OMEGA vs SOMEGA)")
    print("="*115)
    
    disp = df_res.copy()
    disp["Base WR"] = disp["base_win_rate"].apply(lambda x: f"{x*100:.1f}%")
    disp["Base PF"] = disp["base_pf"].apply(lambda x: f"{x:.2f}")
    disp["Base Sharpe"] = disp["base_sharpe"].apply(lambda x: f"{x:.2f}")
    disp["Base MaxDD"] = disp["base_maxdd"].apply(lambda x: f"{x*100:.1f}%")
    
    disp["SOMEGA Trades"] = disp["somega_trades"]
    disp["SOMEGA WR"] = disp["somega_win_rate"].apply(lambda x: f"{x*100:.1f}%")
    disp["SOMEGA PF"] = disp["somega_pf"].apply(lambda x: f"{x:.2f}")
    disp["SOMEGA Sharpe"] = disp["somega_sharpe"].apply(lambda x: f"{x:.2f}")
    disp["SOMEGA MaxDD"] = disp["somega_maxdd"].apply(lambda x: f"{x*100:.1f}%")
    
    cols = ["instrument", "asset_class", "years", "base_trades", "Base WR", "Base PF", "Base Sharpe", "Base MaxDD", 
            "SOMEGA Trades", "SOMEGA WR", "SOMEGA PF", "SOMEGA Sharpe", "SOMEGA MaxDD", "best_utc_hours"]
    print(disp[cols].to_string(index=False))
    
    # Save CSV
    out_path = os.path.join(DATA_DIR, "cross_asset_generalization_results.csv")
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved cross-asset results to {out_path}")


if __name__ == "__main__":
    main()
