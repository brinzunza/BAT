"""Comprehensive Penetration Depth Audit across All 4 Tickers and Multi-Touches (2005 - 2026).

Tests whether 50% midpoint (p = 0.50) is optimal for:
1. Global Portfolio (Uniform Penetration p in [0.10, 0.20, 0.25, 0.30, 0.382, 0.50, 0.618, 0.70, 0.75, 0.80, 0.90])
2. Per-Touch Independent Optimization (Touch 1, Touch 2, Touch 3, Touch 4)
3. Per-Symbol Analysis (EURUSD, EURGBP, GBPUSD, USDJPY)
4. Comprehensive Risk-Adjusted Evaluation (Win Rate %, PF, Net R, Sharpe Ratio, Max DD, Annual R/yr)
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from dtfx_somega_params import SESSION_HOUR_FILTER
from dtfx_eurusd_backtest_strict import find_pivots_strict, run_structure_fsm

INSTRUMENTS = {
    "EURUSD": "lse_eurusd_1min_full.csv",
    "GBPUSD": "lse_gbpusd_1min_full.csv",
    "EURGBP": "lse_eurgbp_1min_full.csv",
    "USDJPY": "lse_usdjpy_1min_2018on.csv",
}

BASE_PARAMS = dict(
    min_retracement=2,
    n_multiplier=3.343496,
    min_risk_pct=0.000170,
    min_fill_age_hours=3.366682
)

MAX_TOUCHES = 4
MAX_BARS_TO_RESOLVE = 2880

_cache = {}

def load_instrument(symbol):
    if symbol in _cache:
        return _cache[symbol]
    path = os.path.join(DATA_DIR, INSTRUMENTS[symbol])
    print(f"Loading {symbol} from {path} ...", flush=True)
    df = pd.read_csv(path)
    for col in df.columns:
        if col.lower() in ("time", "date", "datetime", "timestamp"):
            df["date"] = pd.to_datetime(df[col])
            break
    df = df.set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[["open", "high", "low", "close"]]

    times = df.index.values
    lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    _cache[symbol] = (times, lows, highs, df)
    return _cache[symbol]

_zone_cache = {}

def prep_symbol_zones(symbol):
    if symbol in _zone_cache:
        return _zone_cache[symbol]
    times, lows, highs, df = load_instrument(symbol)
    zigzag = find_pivots_strict(df, BASE_PARAMS["min_retracement"])
    _, zones = run_structure_fsm(zigzag)

    top = np.array([z["top"] for z in zones], dtype=np.float64)
    bottom = np.array([z["bottom"] for z in zones], dtype=np.float64)
    mid = np.array([z["mid"] for z in zones], dtype=np.float64)
    is_buy = np.array([z["side"] == "buy" for z in zones], dtype=np.bool_)
    created_at = np.array([np.datetime64(z["created_at"]) for z in zones])
    start_idx = np.searchsorted(times, created_at, side="right").astype(np.int64)

    _zone_cache[symbol] = dict(top=top, bottom=bottom, mid=mid, is_buy=is_buy, start_idx=start_idx)
    return _zone_cache[symbol]

@njit
def sim_multitouch_penetration(zone_top, zone_bottom, zone_is_buy, zone_start_idx,
                               lows, highs, n_multiplier, min_fill_bars, max_bars,
                               penetration_array, max_touches=4):
    """
    penetration_array: array of length max_touches specifying penetration depth fraction for each touch.
    For Buy zone: entry_price = top - p * (top - bottom)
    For Sell zone: entry_price = bottom + p * (top - bottom)
    """
    n = len(zone_top)
    n_times = len(lows)

    touch_entry_idx = np.full((n, max_touches), -1, dtype=np.int64)
    touch_exit_idx = np.full((n, max_touches), -1, dtype=np.int64)
    touch_status = np.zeros((n, max_touches), dtype=np.int8)  # 0=none, 1=loss, 2=win

    for i in range(n):
        is_buy = zone_is_buy[i]
        top, bottom = zone_top[i], zone_bottom[i]
        zone_width = top - bottom
        risk = n_multiplier * zone_width / 2.0
        invalidation = bottom if is_buy else top

        start = zone_start_idx[i]
        if start >= n_times:
            continue

        earliest_valid_entry = start + min_fill_bars
        if earliest_valid_entry >= n_times:
            continue

        # Touch 0 (Touch #1)
        p0 = penetration_array[0]
        entry_lvl_0 = (top - p0 * zone_width) if is_buy else (bottom + p0 * zone_width)

        current_bar = start
        first_touch_bar = -1
        while current_bar < n_times:
            if is_buy and lows[current_bar] <= invalidation:
                break
            elif (not is_buy) and highs[current_bar] >= invalidation:
                break

            entry_hit = (lows[current_bar] <= entry_lvl_0) if is_buy else (highs[current_bar] >= entry_lvl_0)
            if entry_hit:
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
            p_t = penetration_array[t]
            entry_lvl_t = (top - p_t * zone_width) if is_buy else (bottom + p_t * zone_width)
            stop_level = entry_lvl_t - risk if is_buy else entry_lvl_t + risk
            target_level = entry_lvl_t + risk if is_buy else entry_lvl_t - risk

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

            if t == max_touches - 1:
                break

            # Search next touch
            p_next = penetration_array[t + 1]
            entry_lvl_next = (top - p_next * zone_width) if is_buy else (bottom + p_next * zone_width)
            re_search_start = tp_bar + 1
            re_search_end = min(re_search_start + max_bars, n_times)
            next_entry_bar = -1
            search_bar = re_search_start
            while search_bar < re_search_end:
                if is_buy and lows[search_bar] <= invalidation:
                    break
                elif (not is_buy) and highs[search_bar] >= invalidation:
                    break
                re_hit = (lows[search_bar] <= entry_lvl_next) if is_buy else (highs[search_bar] >= entry_lvl_next)
                if re_hit:
                    next_entry_bar = search_bar
                    break
                search_bar += 1

            if next_entry_bar == -1:
                break
            sim_bar = next_entry_bar

    return touch_entry_idx, touch_exit_idx, touch_status

def evaluate_portfolio(pen_array):
    all_trades = []
    min_fill_bars = int(round(BASE_PARAMS["min_fill_age_hours"] * 60))

    for sym in INSTRUMENTS:
        times, lows, highs, _ = load_instrument(sym)
        z = prep_symbol_zones(sym)

        zone_widths = z["top"] - z["bottom"]
        risks = BASE_PARAMS["n_multiplier"] * zone_widths / 2.0
        risk_pcts = risks / z["mid"]
        valid_mask = risk_pcts >= BASE_PARAMS["min_risk_pct"]

        z_top = np.ascontiguousarray(z["top"][valid_mask], dtype=np.float64)
        z_bottom = np.ascontiguousarray(z["bottom"][valid_mask], dtype=np.float64)
        z_buy = np.ascontiguousarray(z["is_buy"][valid_mask], dtype=np.bool_)
        z_start = np.ascontiguousarray(z["start_idx"][valid_mask], dtype=np.int64)

        e_idx, x_idx, status = sim_multitouch_penetration(
            z_top, z_bottom, z_buy, z_start,
            lows, highs, BASE_PARAMS["n_multiplier"], min_fill_bars, MAX_BARS_TO_RESOLVE,
            pen_array, max_touches=MAX_TOUCHES
        )

        somega_hours = SESSION_HOUR_FILTER.get(sym)

        for i in range(len(z_top)):
            for t in range(MAX_TOUCHES):
                if e_idx[i, t] != -1:
                    stat = status[i, t]
                    if stat in (1, 2):
                        e_time = pd.Timestamp(times[e_idx[i, t]])
                        x_time = pd.Timestamp(times[x_idx[i, t]])
                        if somega_hours is None or e_time.hour in somega_hours:
                            all_trades.append({
                                "symbol": sym,
                                "touch_count": t + 1,
                                "entry_time": e_time,
                                "exit_time": x_time,
                                "is_win": 1 if stat == 2 else 0,
                                "pnl_r": 1.0 if stat == 2 else -1.0
                            })

    df = pd.DataFrame(all_trades)
    if len(df) == 0:
        return None
    df = df[df["exit_time"] >= pd.Timestamp("2005-09-25")].sort_values("entry_time").reset_index(drop=True)
    return df

def calc_metrics(df):
    if df is None or len(df) == 0:
        return {}
    n = len(df)
    w = (df["is_win"] == 1).sum()
    l = (df["is_win"] == 0).sum()
    wr = w / n * 100.0
    pf = w / l if l > 0 else 0
    tot_r = df["pnl_r"].sum()
    years = (df["entry_time"].max() - df["entry_time"].min()).days / 365.25
    ann_r = tot_r / years if years > 0 else 0

    daily = df.groupby(df["entry_time"].dt.date)["pnl_r"].sum()
    daily_mean = daily.mean()
    daily_std = daily.std()
    sharpe = (daily_mean / daily_std * np.sqrt(260)) if daily_std > 0 else 0

    cum_r = df["pnl_r"].cumsum()
    peak = cum_r.cummax()
    dd = peak - cum_r
    max_dd = dd.max()

    return {
        "Trades": n,
        "Win Rate %": wr,
        "Profit Factor": pf,
        "Total Net R": tot_r,
        "Annual R / yr": ann_r,
        "Sharpe Ratio": sharpe,
        "Max DD (R)": max_dd
    }

def main():
    print("=" * 115)
    print("  PENETRATION DEPTH OPTIMIZATION AUDIT: IS 50% MIDPOINT OPTIMAL?")
    print("=" * 115)

    # 1. Uniform Penetration Sweep (All Touches use same p)
    p_levels = [0.10, 0.20, 0.25, 0.30, 0.382, 0.50, 0.618, 0.70, 0.75, 0.80, 0.90]
    uniform_results = []

    print("\n🔹 PART 1: UNIFORM PENETRATION SWEEP (p = 10% to 90% across all touches):")
    print("-" * 115)

    for p in p_levels:
        pen_arr = np.full(MAX_TOUCHES, p, dtype=np.float64)
        df_trades = evaluate_portfolio(pen_arr)
        m = calc_metrics(df_trades)
        uniform_results.append({
            "Penetration": f"{p*100:4.1f}%",
            "Trades": f"{m['Trades']:,}",
            "Win Rate %": f"{m['Win Rate %']:.1f}%",
            "Profit Factor": f"{m['Profit Factor']:.2f}",
            "Total Net R": f"{m['Total Net R']:+,.1f} R",
            "Annual R/yr": f"{m['Annual R / yr']:+,.1f} R/yr",
            "Sharpe": f"{m['Sharpe Ratio']:.2f}",
            "Max DD": f"{m['Max DD (R)']:.1f} R"
        })
        print(f"  • Tested p = {p*100:4.1f}% -> {m['Trades']:,} trades | WR: {m['Win Rate %']:.1f}% | PF: {m['Profit Factor']:.2f} | Net: {m['Total Net R']:+,.1f} R | Sharpe: {m['Sharpe Ratio']:.2f}")

    print("\n" + "=" * 115)
    print("  1. GLOBAL UNIFORM PENETRATION SCORECARD:")
    print("=" * 115)
    print(pd.DataFrame(uniform_results).to_string(index=False))

    # 2. Touch-by-Touch Performance at Different Depths
    print("\n" + "=" * 115)
    print("  2. PER-TOUCH BREAKDOWN ACROSS KEY PENETRATION LEVELS (25%, 38.2%, 50%, 61.8%, 75%):")
    print("=" * 115)

    key_p = [0.25, 0.382, 0.50, 0.618, 0.75]
    touch_matrix = []

    for p in key_p:
        pen_arr = np.full(MAX_TOUCHES, p, dtype=np.float64)
        df_trades = evaluate_portfolio(pen_arr)
        for t in range(1, MAX_TOUCHES + 1):
            sub = df_trades[df_trades["touch_count"] == t]
            n_t = len(sub)
            w_t = (sub["is_win"] == 1).sum()
            l_t = (sub["is_win"] == 0).sum()
            wr_t = w_t / n_t * 100.0 if n_t > 0 else 0
            pf_t = w_t / l_t if l_t > 0 else 0
            r_t = sub["pnl_r"].sum()
            touch_matrix.append({
                "Penetration": f"{p*100:4.1f}%",
                "Touch": f"Touch #{t}",
                "Trades": f"{n_t:,}",
                "Win Rate %": f"{wr_t:.1f}%",
                "Profit Factor": f"{pf_t:.2f}",
                "Total Net R": f"{r_t:+,.1f} R",
                "Avg R / Trade": f"{sub['pnl_r'].mean():+.3f} R"
            })

    print(pd.DataFrame(touch_matrix).to_string(index=False))

    # 3. Adaptive Penetration Profiles vs Flat 50%
    print("\n" + "=" * 115)
    print("  3. ADAPTIVE PENETRATION PROFILES VS UNIFORM 50%:")
    print("=" * 115)

    profiles = {
        "Flat 50% (Baseline)": np.array([0.50, 0.50, 0.50, 0.50]),
        "Progressive Deepening (40%-50%-60%-70%)": np.array([0.40, 0.50, 0.60, 0.70]),
        "Progressive Shallowing (60%-50%-40%-30%)": np.array([0.60, 0.50, 0.40, 0.30]),
        "Golden Ratio (61.8% All)": np.array([0.618, 0.618, 0.618, 0.618]),
        "Shallow Outer Edge (25% All)": np.array([0.25, 0.25, 0.25, 0.25]),
        "Deep Discount (75% All)": np.array([0.75, 0.75, 0.75, 0.75]),
    }

    adaptive_results = []
    for name, p_arr in profiles.items():
        df_trades = evaluate_portfolio(p_arr)
        m = calc_metrics(df_trades)
        adaptive_results.append({
            "Profile Description": name,
            "Trades": f"{m['Trades']:,}",
            "Win Rate %": f"{m['Win Rate %']:.1f}%",
            "Profit Factor": f"{m['Profit Factor']:.2f}",
            "Total Net R": f"{m['Total Net R']:+,.1f} R",
            "Annual R/yr": f"{m['Annual R / yr']:+,.1f} R/yr",
            "Sharpe Ratio": f"{m['Sharpe Ratio']:.2f}",
            "Max DD (R)": f"{m['Max DD (R)']:.1f} R"
        })

    print(pd.DataFrame(adaptive_results).to_string(index=False))
    print("=" * 115 + "\n")

if __name__ == "__main__":
    main()
