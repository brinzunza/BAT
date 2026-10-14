import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
from dtfx_2025_backtest import find_pivots, build_zigzag, run_structure_fsm, _first_true
from dtfx_timeframe_backtest import resample_ohlc, simulate_equity, RISK_PER_TRADE_PCT, ROUND_TRIP_FEE_PCT, STARTING_EQUITY

CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"
TICKERS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XAUT-USDT"]
TIMEFRAMES = {"15min": "15min", "1h": "1h", "4h": "4h", "1D": "1D"}
ENTRY_FRACS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
PIVOT_WINDOW = 3


def entry_level(zone, entry_frac):
    """entry_frac=0 -> at the zone's outer edge (top for a buy zone, bottom for a sell zone,
    i.e. right where price entered the zone, minimal retracement). entry_frac=1 -> at the far
    edge (maximal retracement, right before invalidation). 0.5 = the original midpoint rule."""
    if zone["side"] == "buy":
        return zone["top"] - entry_frac * (zone["top"] - zone["bottom"])
    else:
        return zone["bottom"] + entry_frac * (zone["top"] - zone["bottom"])


def simulate_zone_param(zone, times, lows, highs, entry_frac):
    is_buy_zone = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy_zone else zone["top"]
    entry_lvl = entry_level(zone, entry_frac)
    stop_level = invalidation_level
    target_level = zone["tp"]
    going_long = is_buy_zone

    start_idx = np.searchsorted(times, np.datetime64(zone["created_at"]), side="right")
    if start_idx >= len(times):
        return dict(**zone, status="pending", entry_time=None, exit_time=None, exit_price=None, entry_price=None)

    sub_lows, sub_highs, sub_times = lows[start_idx:], highs[start_idx:], times[start_idx:]
    breach_mask = sub_lows <= invalidation_level if is_buy_zone else sub_highs >= invalidation_level
    entry_mask = sub_lows <= entry_lvl if is_buy_zone else sub_highs >= entry_lvl
    breach_idx = _first_true(breach_mask)
    entry_idx = _first_true(entry_mask)

    if breach_idx <= entry_idx:
        if breach_idx == len(sub_lows):
            return dict(**zone, status="pending", entry_time=None, exit_time=None, exit_price=None, entry_price=None)
        return dict(**zone, status="invalidated", entry_time=None,
                    exit_time=pd.Timestamp(sub_times[breach_idx]), exit_price=invalidation_level, entry_price=None)

    entry_idx_abs = start_idx + entry_idx
    entry_time = pd.Timestamp(times[entry_idx_abs])
    after_start = entry_idx_abs + 1
    if after_start >= len(times):
        return dict(**zone, status="open", entry_time=entry_time, exit_time=None, exit_price=None, entry_price=entry_lvl)

    a_lows, a_highs, a_times = lows[after_start:], highs[after_start:], times[after_start:]
    sl_mask = a_lows <= stop_level if going_long else a_highs >= stop_level
    tp_mask = a_highs >= target_level if going_long else a_lows <= target_level
    sl_idx = _first_true(sl_mask)
    tp_idx = _first_true(tp_mask)

    if sl_idx == len(a_lows) and tp_idx == len(a_lows):
        return dict(**zone, status="open", entry_time=entry_time, exit_time=None, exit_price=None, entry_price=entry_lvl)
    if sl_idx <= tp_idx:
        return dict(**zone, status="stopped_out", entry_time=entry_time,
                    exit_time=pd.Timestamp(a_times[sl_idx]), exit_price=stop_level, entry_price=entry_lvl)
    return dict(**zone, status="take_profit", entry_time=entry_time,
                exit_time=pd.Timestamp(a_times[tp_idx]), exit_price=target_level, entry_price=entry_lvl)


def r_multiple(tr):
    if tr["entry_time"] is None or tr["entry_price"] is None:
        return None
    is_buy_zone = tr["side"] == "buy"
    risk = (tr["entry_price"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["entry_price"])
    if risk <= 0:
        return None
    if tr["status"] == "take_profit":
        reward = (tr["tp"] - tr["entry_price"]) if is_buy_zone else (tr["entry_price"] - tr["tp"])
        return reward / risk
    if tr["status"] == "stopped_out":
        return -1.0
    return None


def get_trades(df, entry_frac, pivot_window):
    swing_highs, swing_lows = find_pivots(df["high"], df["low"], window=pivot_window)
    zigzag = build_zigzag(swing_highs, swing_lows)
    _, zones = run_structure_fsm(zigzag)
    times, lows, highs = df.index.values, df["low"].values, df["high"].values
    trades = [simulate_zone_param(z, times, lows, highs, entry_frac) for z in zones]
    completed = [tr for tr in trades if tr["status"] in ("take_profit", "stopped_out")]
    out = []
    for tr in completed:
        r = r_multiple(tr)
        if r is not None:
            tr["r"] = r
            is_buy_zone = tr["side"] == "buy"
            risk_price = (tr["entry_price"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["entry_price"])
            tr["risk_pct_of_price"] = abs(risk_price) / tr["entry_price"]
            out.append(tr)
    out.sort(key=lambda tr: tr["entry_time"])
    return out, len(zones)


results = []
for ticker in TICKERS:
    df1m = pd.read_csv(f"{CACHE_DIR}/polygon_{ticker.replace('-','_')}_2025.csv", parse_dates=["date"]).set_index("date").sort_index()
    df1m = df1m[["open", "high", "low", "close"]]
    n_days = (df1m.index.max() - df1m.index.min()).days + 1

    for tf_label, rule in TIMEFRAMES.items():
        tf_df = resample_ohlc(df1m, rule)
        if len(tf_df) < 30:
            continue
        for entry_frac in ENTRY_FRACS:
            trades, n_zones = get_trades(tf_df, entry_frac, PIVOT_WINDOW)
            wins = [tr for tr in trades if tr["status"] == "take_profit"]
            losses = [tr for tr in trades if tr["status"] == "stopped_out"]
            win_rate = len(wins) / len(trades) if trades else np.nan
            avg_win_r = np.mean([tr["r"] for tr in wins]) if wins else np.nan
            avg_loss_r = np.mean([tr["r"] for tr in losses]) if losses else np.nan
            sum_r = sum(tr["r"] for tr in trades)
            final_gross = simulate_equity(trades, with_fees=False)
            final_net = simulate_equity(trades, with_fees=True)
            results.append(dict(
                ticker=ticker, timeframe=tf_label, entry_frac=entry_frac, pivot_window=PIVOT_WINDOW,
                zones=n_zones, trades=len(trades), win_rate=win_rate,
                avg_win_r=avg_win_r, avg_loss_r=avg_loss_r, sum_r=sum_r,
                trades_per_day=len(trades) / n_days, final_gross=final_gross, final_net=final_net,
            ))
    print(f"{ticker} done", flush=True)

summary = pd.DataFrame(results)
summary.to_csv(f"{CACHE_DIR}/dtfx_param_sweep_deep_summary.csv", index=False)

print("\n=== TOP 15 BY final_net (across all tickers/timeframes/entry_fracs) ===")
top = summary.sort_values("final_net", ascending=False).head(15)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
pd.set_option("display.width", 220)
print(top.to_string(index=False))

print("\n=== BEST entry_frac per ticker+timeframe (by final_net) ===")
best_per_group = summary.loc[summary.groupby(["ticker", "timeframe"])["final_net"].idxmax()]
print(best_per_group.sort_values("final_net", ascending=False).to_string(index=False))
