import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
from dtfx_2025_backtest import find_pivots, build_zigzag, run_structure_fsm, _first_true

CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"
TICKERS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XAUT-USDT"]
TIMEFRAMES = {"1min": "1min", "5min": "5min", "15min": "15min", "1h": "1h", "4h": "4h", "1D": "1D"}
RISK_PER_TRADE_PCT = 0.01
ROUND_TRIP_FEE_PCT = 0.001
STARTING_EQUITY = 10000.0


def resample_ohlc(df, rule):
    out = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna()


def simulate_zone_original(zone, times, lows, highs):
    """Original DTFX: long a buy-zone / short a sell-zone at 50%, stop at the zone's own far
    edge, target the trigger price that created the zone (outside the zone)."""
    is_buy_zone = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy_zone else zone["top"]
    stop_level = invalidation_level
    target_level = zone["tp"]
    going_long = is_buy_zone
    return _run_sim(zone, times, lows, highs, is_buy_zone, invalidation_level, stop_level, target_level, going_long)


def simulate_zone_reverse_symmetric(zone, times, lows, highs):
    """New reverse definition: enter at 50% same as always, but take the OPPOSITE position,
    with stop-loss and take-profit at the zone's top and bottom exactly (not the external
    trigger price). Since entry = exact midpoint, this makes every trade exactly 1:1 R:R by
    construction: a short buy-zone trade targets the zone bottom and stops at the zone top;
    a long sell-zone trade targets the zone top and stops at the zone bottom."""
    is_buy_zone = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy_zone else zone["top"]
    going_long = not is_buy_zone
    if is_buy_zone:
        stop_level, target_level = zone["top"], zone["bottom"]
    else:
        stop_level, target_level = zone["bottom"], zone["top"]
    return _run_sim(zone, times, lows, highs, is_buy_zone, invalidation_level, stop_level, target_level, going_long)


def _run_sim(zone, times, lows, highs, is_buy_zone, invalidation_level, stop_level, target_level, going_long):
    start_idx = np.searchsorted(times, np.datetime64(zone["created_at"]), side="right")
    if start_idx >= len(times):
        return dict(**zone, status="pending", entry_time=None, exit_time=None, exit_price=None)

    sub_lows, sub_highs, sub_times = lows[start_idx:], highs[start_idx:], times[start_idx:]
    breach_mask = sub_lows <= invalidation_level if is_buy_zone else sub_highs >= invalidation_level
    mid_mask = sub_lows <= zone["mid"] if is_buy_zone else sub_highs >= zone["mid"]
    breach_idx = _first_true(breach_mask)
    mid_idx = _first_true(mid_mask)

    if breach_idx <= mid_idx:
        if breach_idx == len(sub_lows):
            return dict(**zone, status="pending", entry_time=None, exit_time=None, exit_price=None)
        return dict(**zone, status="invalidated", entry_time=None,
                    exit_time=pd.Timestamp(sub_times[breach_idx]), exit_price=invalidation_level)

    entry_idx_abs = start_idx + mid_idx
    entry_time = pd.Timestamp(times[entry_idx_abs])
    after_start = entry_idx_abs + 1
    if after_start >= len(times):
        return dict(**zone, status="open", entry_time=entry_time, exit_time=None, exit_price=None)

    a_lows, a_highs, a_times = lows[after_start:], highs[after_start:], times[after_start:]
    sl_mask = a_lows <= stop_level if going_long else a_highs >= stop_level
    tp_mask = a_highs >= target_level if going_long else a_lows <= target_level
    sl_idx = _first_true(sl_mask)
    tp_idx = _first_true(tp_mask)

    if sl_idx == len(a_lows) and tp_idx == len(a_lows):
        return dict(**zone, status="open", entry_time=entry_time, exit_time=None, exit_price=None)
    if sl_idx <= tp_idx:
        return dict(**zone, status="stopped_out", entry_time=entry_time,
                    exit_time=pd.Timestamp(a_times[sl_idx]), exit_price=stop_level)
    return dict(**zone, status="take_profit", entry_time=entry_time,
                exit_time=pd.Timestamp(a_times[tp_idx]), exit_price=target_level)


def r_multiple(tr, variant):
    if tr["entry_time"] is None:
        return None
    is_buy_zone = tr["side"] == "buy"
    if variant == "original":
        risk = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
        reward_level, going_long = tr["tp"], is_buy_zone
    else:  # reverse_symmetric: exactly 1:1 by construction
        risk = (tr["top"] - tr["mid"]) if is_buy_zone else (tr["mid"] - tr["bottom"])
        reward_level = tr["bottom"] if is_buy_zone else tr["top"]
        going_long = not is_buy_zone
    if risk <= 0:
        return None
    if tr["status"] == "take_profit":
        reward = (reward_level - tr["mid"]) if going_long else (tr["mid"] - reward_level)
        return reward / risk
    if tr["status"] == "stopped_out":
        return -1.0
    return None


def get_trades(df, variant):
    swing_highs, swing_lows = find_pivots(df["high"], df["low"], window=3)
    zigzag = build_zigzag(swing_highs, swing_lows)
    _, zones = run_structure_fsm(zigzag)
    times, lows, highs = df.index.values, df["low"].values, df["high"].values
    sim_fn = simulate_zone_original if variant == "original" else simulate_zone_reverse_symmetric
    trades = [sim_fn(z, times, lows, highs) for z in zones]
    completed = [tr for tr in trades if tr["status"] in ("take_profit", "stopped_out")]
    completed.sort(key=lambda tr: tr["entry_time"])
    out = []
    for tr in completed:
        r = r_multiple(tr, variant)
        if r is not None:
            tr["r"] = r
            is_buy_zone = tr["side"] == "buy"
            if variant == "original":
                risk_price = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
            else:
                risk_price = (tr["top"] - tr["mid"]) if is_buy_zone else (tr["mid"] - tr["bottom"])
            tr["risk_pct_of_price"] = abs(risk_price) / tr["mid"]
            out.append(tr)
    return out, len(zones)


def simulate_equity(trades, with_fees):
    equity = STARTING_EQUITY
    for tr in trades:
        risk_dollars = equity * RISK_PER_TRADE_PCT
        pnl = tr["r"] * risk_dollars
        if with_fees:
            notional = risk_dollars / tr["risk_pct_of_price"] if tr["risk_pct_of_price"] > 0 else 0
            pnl -= notional * ROUND_TRIP_FEE_PCT
        equity = max(equity + pnl, 0.0)
        if equity <= 0:
            break
    return equity


results = []
for ticker in TICKERS:
    df1m = pd.read_csv(f"{CACHE_DIR}/polygon_{ticker.replace('-','_')}_2025.csv", parse_dates=["date"]).set_index("date").sort_index()
    df1m = df1m[["open", "high", "low", "close"]]
    n_days = (df1m.index.max() - df1m.index.min()).days + 1

    for tf_label, rule in TIMEFRAMES.items():
        tf_df = df1m if tf_label == "1min" else resample_ohlc(df1m, rule)
        if len(tf_df) < 30:
            continue
        for variant in ["original", "reverse_symmetric"]:
            trades, n_zones = get_trades(tf_df, variant)
            wins = [tr for tr in trades if tr["status"] == "take_profit"]
            losses = [tr for tr in trades if tr["status"] == "stopped_out"]
            win_rate = len(wins) / len(trades) if trades else np.nan
            avg_win_r = np.mean([tr["r"] for tr in wins]) if wins else np.nan
            avg_loss_r = np.mean([tr["r"] for tr in losses]) if losses else np.nan
            sum_r = sum(tr["r"] for tr in trades)
            final_gross = simulate_equity(trades, with_fees=False)
            final_net = simulate_equity(trades, with_fees=True)
            results.append(dict(
                ticker=ticker, timeframe=tf_label, variant=variant, bars=len(tf_df),
                zones=n_zones, trades=len(trades), win_rate=win_rate,
                avg_win_r=avg_win_r, avg_loss_r=avg_loss_r, sum_r=sum_r,
                trades_per_day=len(trades) / n_days,
                final_gross=final_gross, final_net=final_net,
            ))
        print(f"{ticker} {tf_label} done", flush=True)

summary = pd.DataFrame(results)
summary.to_csv(f"{CACHE_DIR}/dtfx_timeframe_summary.csv", index=False)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print("\n" + summary.to_string(index=False))
