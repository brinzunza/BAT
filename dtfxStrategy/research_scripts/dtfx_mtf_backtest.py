import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
from dtfx_2025_backtest import find_pivots, build_zigzag, run_structure_fsm
from dtfx_timeframe_backtest import (
    resample_ohlc, simulate_zone_original, r_multiple as r_multiple_base,
    simulate_equity, RISK_PER_TRADE_PCT, ROUND_TRIP_FEE_PCT, STARTING_EQUITY,
)

CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"
TICKERS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XAUT-USDT"]
HTF_RULE = "4h"
PIVOT_WINDOW = 3


def find_pivots_confirmed(high, low, window=3):
    """Same as find_pivots, but also returns the timestamp at which each pivot becomes
    knowable without lookahead: a centered rolling window needs `window` MORE bars after the
    pivot bar before it's confirmed as a genuine local extreme."""
    idx = high.index
    is_high = high == high.rolling(window * 2 + 1, center=True).max()
    is_low = low == low.rolling(window * 2 + 1, center=True).min()

    def with_confirmation(mask, series):
        positions = np.flatnonzero(mask.values)
        rows = []
        for pos in positions:
            confirm_pos = pos + window
            if confirm_pos >= len(idx):
                continue  # not yet confirmed within available data
            rows.append((idx[pos], series.iloc[pos], idx[confirm_pos]))
        return rows

    highs = with_confirmation(is_high, high)
    lows = with_confirmation(is_low, low)
    return highs, lows


def build_htf_trend_timeline(htf_df, window=PIVOT_WINDOW):
    """Returns a sorted list of (confirmed_at, trend) marking every point the HTF trend
    became knowable as up/down, starting from the bootstrap trend ('up', by the same
    convention as the main FSM). Query with a 1-minute timestamp to get the HTF trend that
    was ACTUALLY available at that moment (no lookahead)."""
    highs, lows = find_pivots_confirmed(htf_df["high"], htf_df["low"], window=window)
    tagged = [(t, v, "H", conf) for t, v, conf in highs] + [(t, v, "L", conf) for t, v, conf in lows]
    tagged.sort(key=lambda x: x[0])

    zigzag = []  # (time, value, type, confirmed_at), alternating
    for t, v, typ, conf in tagged:
        if zigzag and zigzag[-1][2] == typ:
            _, prev_v, _, _ = zigzag[-1]
            more_extreme = (typ == "H" and v > prev_v) or (typ == "L" and v < prev_v)
            if more_extreme:
                zigzag[-1] = (t, v, typ, conf)
            continue
        zigzag.append((t, v, typ, conf))

    trend = None
    confirmed_high = confirmed_low = None
    pending_low_idx = pending_high_idx = None
    events = []  # [time, value, type, role, confirmed_at]
    timeline = []  # (confirmed_at, trend)

    i = 0
    seen_low, seen_high = None, None
    while i < len(zigzag) and (seen_low is None or seen_high is None):
        t, v, typ, conf = zigzag[i]
        if typ == "L" and seen_low is None:
            seen_low = (t, v, conf)
        if typ == "H" and seen_high is None:
            seen_high = (t, v, conf)
        i += 1
    if not (seen_low and seen_high):
        return [(pd.Timestamp.min, "up")]
    confirmed_low, confirmed_high = seen_low[1], seen_high[1]
    trend = "up"
    bootstrap_conf = max(seen_low[2], seen_high[2])
    timeline.append((bootstrap_conf, "up"))

    for t, v, typ, conf in zigzag[i:]:
        if typ == "L":
            if trend == "up":
                if v < confirmed_low:
                    confirmed_high, confirmed_low = confirmed_high, v
                    trend = "down"
                    pending_high_idx = None
                    timeline.append((conf, "down"))
                    events.append([t, v, "L", "flip_down", conf])
                else:
                    pending_low_idx = len(events)
                    events.append([t, v, "L", "pending", conf])
            else:
                if confirmed_low is None or v < confirmed_low:
                    confirmed_low = v
                    if pending_high_idx is not None:
                        confirmed_high = events[pending_high_idx][1]
                        pending_high_idx = None
                    events.append([t, v, "L", "confirmed", conf])
                else:
                    events.append([t, v, "L", "pending", conf])
        else:
            if trend == "down":
                if v > confirmed_high:
                    confirmed_low, confirmed_high = confirmed_low, v
                    trend = "up"
                    pending_low_idx = None
                    timeline.append((conf, "up"))
                    events.append([t, v, "H", "flip_up", conf])
                else:
                    pending_high_idx = len(events)
                    events.append([t, v, "H", "pending", conf])
            else:
                if confirmed_high is None or v > confirmed_high:
                    confirmed_high = v
                    if pending_low_idx is not None:
                        confirmed_low = events[pending_low_idx][1]
                        pending_low_idx = None
                    events.append([t, v, "H", "confirmed", conf])
                else:
                    events.append([t, v, "H", "pending", conf])

    timeline.sort(key=lambda x: x[0])
    return timeline


def trend_at(timeline, ts):
    """Latest trend whose confirmation time is <= ts (no lookahead)."""
    lo, hi = 0, len(timeline)
    while lo < hi:
        mid = (lo + hi) // 2
        if timeline[mid][0] <= ts:
            lo = mid + 1
        else:
            hi = mid
    return timeline[lo - 1][1] if lo > 0 else None


def run_mtf_backtest(ticker):
    df1m = pd.read_csv(f"{CACHE_DIR}/polygon_{ticker.replace('-','_')}_2025.csv", parse_dates=["date"]).set_index("date").sort_index()
    df1m = df1m[["open", "high", "low", "close"]]
    df4h = resample_ohlc(df1m, HTF_RULE)

    htf_timeline = build_htf_trend_timeline(df4h)

    swing_highs, swing_lows = find_pivots(df1m["high"], df1m["low"], window=PIVOT_WINDOW)
    zigzag = build_zigzag(swing_highs, swing_lows)
    _, zones = run_structure_fsm(zigzag)

    times, lows, highs = df1m.index.values, df1m["low"].values, df1m["high"].values
    all_trades = [simulate_zone_original(z, times, lows, highs) for z in zones]

    aligned_trades = []
    for z, tr in zip(zones, all_trades):
        htf_trend = trend_at(htf_timeline, z["created_at"])
        wanted_trend = "up" if z["side"] == "buy" else "down"
        if htf_trend == wanted_trend:
            aligned_trades.append(tr)

    return all_trades, aligned_trades, len(zones)


def summarize(trades, ticker, label, n_days):
    completed = [tr for tr in trades if tr["status"] in ("take_profit", "stopped_out")]
    for tr in completed:
        tr["r"] = r_multiple_base(tr, "original")
        is_buy_zone = tr["side"] == "buy"
        risk_price = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
        tr["risk_pct_of_price"] = abs(risk_price) / tr["mid"]
    completed = [tr for tr in completed if tr["r"] is not None]
    completed.sort(key=lambda tr: tr["entry_time"])

    wins = [tr for tr in completed if tr["status"] == "take_profit"]
    losses = [tr for tr in completed if tr["status"] == "stopped_out"]
    win_rate = len(wins) / len(completed) if completed else np.nan
    avg_win_r = np.mean([tr["r"] for tr in wins]) if wins else np.nan
    avg_loss_r = np.mean([tr["r"] for tr in losses]) if losses else np.nan
    sum_r = sum(tr["r"] for tr in completed)
    final_gross = simulate_equity(completed, with_fees=False)
    final_net = simulate_equity(completed, with_fees=True)
    return dict(
        ticker=ticker, label=label, trades=len(completed), win_rate=win_rate,
        avg_win_r=avg_win_r, avg_loss_r=avg_loss_r, sum_r=sum_r,
        trades_per_day=len(completed) / n_days, final_gross=final_gross, final_net=final_net,
    )


results = []
for ticker in TICKERS:
    print(f"=== {ticker} ===", flush=True)
    all_trades, aligned_trades, n_zones = run_mtf_backtest(ticker)
    n_days = 365
    results.append(summarize(all_trades, ticker, "baseline_1min", n_days))
    results.append(summarize(aligned_trades, ticker, "mtf_4h_aligned", n_days))
    print(f"  zones={n_zones}  baseline_trades={len([t for t in all_trades if t['status'] in ('take_profit','stopped_out')])}  "
          f"aligned_trades={len([t for t in aligned_trades if t['status'] in ('take_profit','stopped_out')])}", flush=True)

summary = pd.DataFrame(results)
summary.to_csv(f"{CACHE_DIR}/dtfx_mtf_summary.csv", index=False)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
pd.set_option("display.width", 200)
print("\n" + summary.to_string(index=False))
