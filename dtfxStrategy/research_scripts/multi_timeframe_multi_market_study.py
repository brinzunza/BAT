"""Massive Multi-Timeframe & Cross-Market Backtesting Study.

Evaluates the DTFX structural framework across:
1. 7 Timeframes: 1-Min, 5-Min, 15-Min, 1-Hour, 4-Hour, 1-Day (Daily), 1-Week (Weekly)
2. 7 Diverse Markets:
   - Forex Majors: EUR/USD, GBP/USD, USD/JPY
   - Forex Cross: EUR/GBP
   - Commodities: Gold (XAU/USD)
   - Crypto: Bitcoin (BTC/USD)
   - Equity Indices: S&P 500 (SPY)
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_eurusd_backtest_strict import find_pivots_strict, run_structure_fsm

MARKETS = {
    "EUR/USD (Forex)": "lse_eurusd_1min_full.csv",
    "GBP/USD (Forex)": "lse_gbpusd_1min_full.csv",
    "EUR/GBP (Forex)": "lse_eurgbp_1min_full.csv",
    "USD/JPY (Forex)": "lse_usdjpy_1min_2018on.csv",
    "XAU/USD (Gold)": "lse_xauusd_full_1min.csv",
    "BTC/USD (Crypto)": "lse_btcusd_1min_2018on.csv",
    "SPY (S&P 500)": "lse_spy_1min.csv",
}

TIMEFRAMES = {
    "1-Min (1M)": None,
    "5-Min (5M)": "5min",
    "15-Min (15M)": "15min",
    "1-Hour (1H)": "1h",
    "4-Hour (4H)": "4h",
    "1-Day (1D)": "1D",
    "1-Week (1W)": "1W",
}

N_MULTIPLIER = 3.343496
MIN_RETRACEMENT = 2
MAX_TOUCHES = 4
MAX_BARS_TO_RESOLVE = 2880

_raw_cache = {}

def load_raw_1min(filename):
    if filename in _raw_cache:
        return _raw_cache[filename]
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    print(f"Loading {filename} ...", flush=True)
    df = pd.read_csv(path)
    for col in df.columns:
        if col.lower() in ("time", "date", "datetime", "timestamp"):
            df["date"] = pd.to_datetime(df[col])
            break
    df = df.set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[["open", "high", "low", "close"]].dropna()
    _raw_cache[filename] = df
    return df

def resample_df(df, rule):
    if rule is None:
        return df
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()
    return resampled

@njit
def sim_mtf_numba(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                  lows, highs, n_multiplier, min_fill_bars, max_bars, max_touches=4):
    n = len(zone_top)
    n_times = len(lows)

    touch_entry_idx = np.full((n, max_touches), -1, dtype=np.int64)
    touch_exit_idx = np.full((n, max_touches), -1, dtype=np.int64)
    touch_status = np.zeros((n, max_touches), dtype=np.int8)

    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        zone_width = top - bottom
        risk = n_multiplier * zone_width / 2.0
        invalidation = bottom if is_buy else top
        stop_level = mid - risk if is_buy else mid + risk
        target_level = mid + risk if is_buy else mid - risk

        start = zone_start_idx[i]
        if start >= n_times:
            continue

        earliest_valid_entry = start + min_fill_bars
        if earliest_valid_entry >= n_times:
            continue

        current_bar = start
        first_touch_bar = -1
        while current_bar < n_times:
            if is_buy and lows[current_bar] <= invalidation:
                break
            elif (not is_buy) and highs[current_bar] >= invalidation:
                break

            mid_hit = (lows[current_bar] <= mid) if is_buy else (highs[current_bar] >= mid)
            if mid_hit:
                if current_bar >= earliest_valid_entry:
                    first_touch_bar = current_bar
                    break
                else:
                    break
            current_bar += 1

        if first_touch_bar == -1:
            continue

        sim_bar = first_touch_bar
        for t in range(max_touches):
            touch_entry_idx[i, t] = sim_bar
            after_start = sim_bar + 1
            after_end = min(after_start + max_bars, n_times)

            sl_bar, tp_bar = -1, -1
            for j in range(after_start, after_end):
                if is_buy:
                    if sl_bar == -1 and lows[j] <= stop_level:
                        sl_bar = j
                    if tp_bar == -1 and highs[j] >= target_level:
                        tp_bar = j
                else:
                    if sl_bar == -1 and highs[j] >= stop_level:
                        sl_bar = j
                    if tp_bar == -1 and lows[j] <= target_level:
                        tp_bar = j
                if sl_bar != -1 and tp_bar != -1:
                    break

            if sl_bar == -1 and tp_bar == -1:
                touch_status[i, t] = 0
                touch_exit_idx[i, t] = n_times - 1
                break

            if sl_bar != -1 and (tp_bar == -1 or sl_bar <= tp_bar):
                touch_status[i, t] = 1  # loss
                touch_exit_idx[i, t] = sl_bar
                break
            else:
                touch_status[i, t] = 2  # win
                touch_exit_idx[i, t] = tp_bar

            re_search_start = tp_bar + 1
            re_search_end = min(re_search_start + max_bars, n_times)
            next_entry_bar = -1
            search_bar = re_search_start
            while search_bar < re_search_end:
                if is_buy and lows[search_bar] <= invalidation:
                    break
                elif (not is_buy) and highs[search_bar] >= invalidation:
                    break
                re_mid_hit = (lows[search_bar] <= mid) if is_buy else (highs[search_bar] >= mid)
                if re_mid_hit:
                    next_entry_bar = search_bar
                    break
                search_bar += 1

            if next_entry_bar == -1:
                break
            sim_bar = next_entry_bar

    return touch_entry_idx, touch_exit_idx, touch_status

def evaluate_market_timeframe(df_bars, tf_name):
    if len(df_bars) < 100:
        return None

    zigzag = find_pivots_strict(df_bars, min_retracement=MIN_RETRACEMENT)
    _, zones = run_structure_fsm(zigzag)

    if not zones:
        return None

    times = df_bars.index.values
    lows = np.ascontiguousarray(df_bars["low"].values, dtype=np.float64)
    highs = np.ascontiguousarray(df_bars["high"].values, dtype=np.float64)

    top = np.array([z["top"] for z in zones], dtype=np.float64)
    bottom = np.array([z["bottom"] for z in zones], dtype=np.float64)
    mid = np.array([z["mid"] for z in zones], dtype=np.float64)
    is_buy = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
    created_at = np.array([np.datetime64(z["created_at"]) for z in zones])
    start_idx = np.searchsorted(times, created_at, side="right").astype(np.int64)

    # Minimum fill bars scaling by timeframe:
    # 1M: 200 bars (3.33h)
    # 5M: 40 bars (3.33h)
    # 15M: 14 bars (3.5h)
    # 1H: 4 bars (4h)
    # 4H: 2 bars (8h)
    # 1D: 2 bars (2 days)
    # 1W: 2 bars (2 weeks)
    tf_fill_map = {
        "1-Min (1M)": 200,
        "5-Min (5M)": 40,
        "15-Min (15M)": 14,
        "1-Hour (1H)": 4,
        "4-Hour (4H)": 2,
        "1-Day (1D)": 2,
        "1-Week (1W)": 2
    }
    min_fill_bars = tf_fill_map.get(tf_name, 2)
    max_bars = 2880 if "Min" in tf_name else 1000

    e_idx, x_idx, status = sim_mtf_numba(
        top, bottom, mid, is_buy, start_idx,
        lows, highs, N_MULTIPLIER, min_fill_bars, max_bars, max_touches=MAX_TOUCHES
    )

    trades = []
    for i in range(len(top)):
        for t in range(MAX_TOUCHES):
            if e_idx[i, t] != -1:
                stat = status[i, t]
                if stat in (1, 2):
                    trades.append({
                        "entry_time": pd.Timestamp(times[e_idx[i, t]]),
                        "exit_time": pd.Timestamp(times[x_idx[i, t]]),
                        "is_win": 1 if stat == 2 else 0,
                        "pnl_r": 1.0 if stat == 2 else -1.0
                    })

    df_t = pd.DataFrame(trades)
    if len(df_t) == 0:
        return None
    df_t = df_t.sort_values("entry_time").reset_index(drop=True)

    n = len(df_t)
    w = (df_t["is_win"] == 1).sum()
    l = (df_t["is_win"] == 0).sum()
    wr = w / n * 100.0
    pf = w / l if l > 0 else float("inf")
    tot_r = df_t["pnl_r"].sum()
    
    start_d = df_t["entry_time"].min()
    end_d = df_t["entry_time"].max()
    years = max((end_d - start_d).days / 365.25, 0.5)
    ann_r = tot_r / years
    trades_per_yr = n / years

    # Average trade duration
    dur_days = (df_t["exit_time"] - df_t["entry_time"]).dt.total_seconds() / 86400.0
    avg_dur = dur_days.mean()
    if avg_dur < 1.0:
        dur_str = f"{avg_dur*24:.1f} hrs"
    else:
        dur_str = f"{avg_dur:.1f} days"

    return {
        "Trades": n,
        "Win Rate %": wr,
        "Profit Factor": pf,
        "Total Net R": tot_r,
        "Annual R/yr": ann_r,
        "Trades/Yr": trades_per_yr,
        "Avg Duration": dur_str,
        "Years": years
    }

def main():
    print("=" * 125)
    print("  MASSIVE MULTI-TIMEFRAME & MULTI-MARKET SYSTEMATIC STUDY")
    print("=" * 125)

    all_results = []

    for m_name, filename in MARKETS.items():
        raw_df = load_raw_1min(filename)
        if raw_df is None:
            print(f"Skipping {m_name} (file not found).")
            continue

        print(f"\n🔹 Evaluating Market: {m_name} ({len(raw_df):,} 1-min raw candles)")
        print("-" * 125)

        for tf_name, rule in TIMEFRAMES.items():
            df_resampled = resample_df(raw_df, rule)
            res = evaluate_market_timeframe(df_resampled, tf_name)
            if res is not None:
                all_results.append({
                    "Market": m_name,
                    "Timeframe": tf_name,
                    "Trades": f"{res['Trades']:,}",
                    "Win Rate %": f"{res['Win Rate %']:.1f}%",
                    "Profit Factor": f"{res['Profit Factor']:.2f}",
                    "Total Net R": f"{res['Total Net R']:+,.1f} R",
                    "Annual R/yr": f"{res['Annual R/yr']:+,.1f} R/yr",
                    "Trades/Yr": f"{res['Trades/Yr']:,.1f}/yr",
                    "Avg Duration": res['Avg Duration'],
                    "Span": f"{res['Years']:.1f} yrs"
                })
                print(f"   [{tf_name:12s}] {res['Trades']:5,d} trades | WR: {res['Win Rate %']:5.1f}% | PF: {res['Profit Factor']:4.2f} | Net: {res['Total Net R']:+8.1f} R | Ann: {res['Annual R/yr']:+6.1f} R/yr | Dur: {res['Avg Duration']}")

    df_master = pd.DataFrame(all_results)

    print("\n" + "=" * 135)
    print("  MASTER SCORECARD: MULTI-TIMEFRAME & CROSS-MARKET GENERALIZATION")
    print("=" * 135)
    print(df_master.to_string(index=False))

    # Aggregate by Timeframe across all markets
    print("\n" + "=" * 135)
    print("  TIMEFRAME AGGREGATE SUMMARY (AVERAGED ACROSS ALL 7 MARKETS):")
    print("=" * 135)
    tf_summary = []
    for tf_name in TIMEFRAMES:
        sub = [r for r in all_results if r["Timeframe"] == tf_name]
        if sub:
            tot_tr = sum(int(r["Trades"].replace(",", "")) for r in sub)
            avg_wr = np.mean([float(r["Win Rate %"].replace("%", "")) for r in sub])
            avg_pf = np.mean([float(r["Profit Factor"]) for r in sub])
            tot_net = sum(float(r["Total Net R"].replace(" R", "").replace(",", "")) for r in sub)
            avg_ann = np.mean([float(r["Annual R/yr"].replace(" R/yr", "").replace(",", "")) for r in sub])
            tf_summary.append({
                "Timeframe": tf_name,
                "Total Trades (7 mkts)": f"{tot_tr:,}",
                "Avg Win Rate %": f"{avg_wr:.1f}%",
                "Avg Profit Factor": f"{avg_pf:.2f}",
                "Combined Net R": f"{tot_net:+,.1f} R",
                "Avg Annual Return": f"{avg_ann:+,.1f} R/yr"
            })
    print(pd.DataFrame(tf_summary).to_string(index=False))
    print("=" * 135 + "\n")

if __name__ == "__main__":
    main()
