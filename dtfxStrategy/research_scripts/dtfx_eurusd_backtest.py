import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"
RISK_PER_TRADE_PCT = 0.01
ROUND_TRIP_COST_PCT = 0.0002  # ~2 pips round trip on EURUSD, a realistic retail spread cost
STARTING_EQUITY = 10000.0


def resample_ohlc(df, rule):
    out = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna()


def find_pivots(high, low, window=3):
    is_high = high == high.rolling(window * 2 + 1, center=True).max()
    is_low = low == low.rolling(window * 2 + 1, center=True).min()
    return high[is_high].dropna(), low[is_low].dropna()


def build_zigzag(swing_highs, swing_lows):
    raw = [(t, v, "H") for t, v in swing_highs.items()] + [(t, v, "L") for t, v in swing_lows.items()]
    raw.sort(key=lambda x: x[0])
    zigzag = []
    for t, v, typ in raw:
        if zigzag and zigzag[-1][2] == typ:
            _, prev_v, _ = zigzag[-1]
            more_extreme = (typ == "H" and v > prev_v) or (typ == "L" and v < prev_v)
            if more_extreme:
                zigzag[-1] = (t, v, typ)
            continue
        zigzag.append((t, v, typ))
    return zigzag


def run_structure_fsm(zigzag):
    events, zones = [], []
    trend = confirmed_high = confirmed_low = None
    pending_low_idx = pending_high_idx = None
    i = 0
    seen_low, seen_high = None, None
    while i < len(zigzag) and (seen_low is None or seen_high is None):
        t, v, typ = zigzag[i]
        if typ == "L" and seen_low is None:
            seen_low = (t, v)
        if typ == "H" and seen_high is None:
            seen_high = (t, v)
        i += 1
    if not (seen_low and seen_high):
        return events, zones
    confirmed_low, confirmed_high = seen_low[1], seen_high[1]
    trend = "up"
    events.append([seen_low[0], seen_low[1], "L", "bootstrap"])
    events.append([seen_high[0], seen_high[1], "H", "bootstrap"])

    for t, v, typ in zigzag[i:]:
        if typ == "L":
            if trend == "up":
                if v < confirmed_low:
                    confirmed_high, confirmed_low = confirmed_high, v
                    trend = "down"
                    pending_high_idx = None
                    events.append([t, v, "L", "flip_down"])
                else:
                    pending_low_idx = len(events)
                    events.append([t, v, "L", "pending"])
            else:
                if confirmed_low is None or v < confirmed_low:
                    old_low = confirmed_low
                    zone_top = events[pending_high_idx][1] if pending_high_idx is not None else confirmed_high
                    confirmed_low = v
                    if pending_high_idx is not None:
                        confirmed_high = events[pending_high_idx][1]
                        events[pending_high_idx][3] = "confirmed"
                        pending_high_idx = None
                    events.append([t, v, "L", "confirmed"])
                    if old_low is not None and zone_top is not None and zone_top > old_low:
                        zones.append(dict(top=zone_top, bottom=old_low, mid=(zone_top + old_low) / 2,
                                           tp=v, created_at=t, side="sell"))
                else:
                    events.append([t, v, "L", "pending"])
        else:
            if trend == "down":
                if v > confirmed_high:
                    confirmed_low, confirmed_high = confirmed_low, v
                    trend = "up"
                    pending_low_idx = None
                    events.append([t, v, "H", "flip_up"])
                else:
                    pending_high_idx = len(events)
                    events.append([t, v, "H", "pending"])
            else:
                if confirmed_high is None or v > confirmed_high:
                    old_high = confirmed_high
                    zone_bottom = events[pending_low_idx][1] if pending_low_idx is not None else confirmed_low
                    confirmed_high = v
                    if pending_low_idx is not None:
                        confirmed_low = events[pending_low_idx][1]
                        events[pending_low_idx][3] = "confirmed"
                        pending_low_idx = None
                    events.append([t, v, "H", "confirmed"])
                    if old_high is not None and zone_bottom is not None and old_high > zone_bottom:
                        zones.append(dict(top=old_high, bottom=zone_bottom, mid=(old_high + zone_bottom) / 2,
                                           tp=v, created_at=t, side="buy"))
                else:
                    events.append([t, v, "H", "pending"])
    return events, zones


def _first_true(mask):
    idx = np.flatnonzero(mask)
    return idx[0] if len(idx) else len(mask)


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


def simulate_zone_original(zone, times, lows, highs):
    """Buy/sell at 50% of the zone. SL at the zone's own far edge. TP at the trigger
    price that created the zone (the new higher-high / lower-low, beyond the zone)."""
    is_buy_zone = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy_zone else zone["top"]
    return _run_sim(zone, times, lows, highs, is_buy_zone, invalidation_level,
                     invalidation_level, zone["tp"], going_long=is_buy_zone)


def simulate_zone_1to1(zone, times, lows, highs):
    """Same direction as original (buy a buy-zone, sell a sell-zone) at 50%. SL at the
    zone's own far edge, but TP at the zone's OWN near edge (not the external trigger),
    which is exactly 1:1 R:R by construction since entry = the exact midpoint."""
    is_buy_zone = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy_zone else zone["top"]
    target_level = zone["top"] if is_buy_zone else zone["bottom"]
    return _run_sim(zone, times, lows, highs, is_buy_zone, invalidation_level,
                     invalidation_level, target_level, going_long=is_buy_zone)


def r_multiple(tr, variant):
    if tr["entry_time"] is None:
        return None
    is_buy_zone = tr["side"] == "buy"
    risk = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
    if risk <= 0:
        return None
    if tr["status"] == "take_profit":
        if variant == "original":
            reward = (tr["tp"] - tr["mid"]) if is_buy_zone else (tr["mid"] - tr["tp"])
        else:  # 1to1: exactly 1:1 by construction
            reward = risk
        return reward / risk
    if tr["status"] == "stopped_out":
        return -1.0
    return None


def get_trades(df, variant):
    swing_highs, swing_lows = find_pivots(df["high"], df["low"], window=3)
    zigzag = build_zigzag(swing_highs, swing_lows)
    _, zones = run_structure_fsm(zigzag)
    times, lows, highs = df.index.values, df["low"].values, df["high"].values
    sim_fn = simulate_zone_original if variant == "original" else simulate_zone_1to1
    trades = [sim_fn(z, times, lows, highs) for z in zones]
    completed = [tr for tr in trades if tr["status"] in ("take_profit", "stopped_out")]
    completed.sort(key=lambda tr: tr["entry_time"])
    out = []
    for tr in completed:
        r = r_multiple(tr, variant)
        if r is not None:
            tr["r"] = r
            is_buy_zone = tr["side"] == "buy"
            risk_price = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
            tr["risk_pct_of_price"] = abs(risk_price) / tr["mid"]
            out.append(tr)
    return out, len(zones)


def equity_curve(trades, with_fees):
    equity = STARTING_EQUITY
    curve = [(trades[0]["entry_time"], equity)] if trades else []
    for tr in trades:
        risk_dollars = equity * RISK_PER_TRADE_PCT
        pnl = tr["r"] * risk_dollars
        if with_fees:
            notional = risk_dollars / tr["risk_pct_of_price"] if tr["risk_pct_of_price"] > 0 else 0
            pnl -= notional * ROUND_TRIP_COST_PCT
        equity = max(equity + pnl, 0.0)
        curve.append((tr["exit_time"], equity))
        if equity <= 0:
            break
    return curve


def stats_for(trades, n_days):
    wins = [tr for tr in trades if tr["status"] == "take_profit"]
    losses = [tr for tr in trades if tr["status"] == "stopped_out"]
    win_rate = len(wins) / len(trades) if trades else np.nan
    avg_win_r = np.mean([tr["r"] for tr in wins]) if wins else np.nan
    avg_loss_r = np.mean([tr["r"] for tr in losses]) if losses else np.nan
    sum_r = sum(tr["r"] for tr in trades) if trades else 0.0
    gross_curve = equity_curve(trades, with_fees=False)
    net_curve = equity_curve(trades, with_fees=True)
    return dict(
        trades=len(trades), win_rate=win_rate, avg_win_r=avg_win_r, avg_loss_r=avg_loss_r,
        sum_r=sum_r, trades_per_day=len(trades) / n_days if n_days else np.nan,
        final_gross=gross_curve[-1][1] if gross_curve else STARTING_EQUITY,
        final_net=net_curve[-1][1] if net_curve else STARTING_EQUITY,
    ), gross_curve, net_curve


if __name__ == "__main__":
    parts = []
    for yr in ["2024", "2025", "2026"]:
        d = pd.read_csv(f"{CACHE_DIR}/polygon_EURUSD_{yr}.csv", parse_dates=["date"]).set_index("date").sort_index()
        parts.append(d[["open", "high", "low", "close"]])
    df1m = pd.concat(parts).sort_index()
    df1m = df1m[~df1m.index.duplicated(keep="first")]
    df1h = resample_ohlc(df1m, "1h")
    print(f"Loaded EURUSD: {len(df1m)} 1min bars / {len(df1h)} 1h bars, {df1m.index.min()} to {df1m.index.max()}", flush=True)

    all_rows = []
    curves = {}
    for tf_label, df in [("1min", df1m), ("1h", df1h)]:
        for variant in ["original", "1to1"]:
            trades, n_zones = get_trades(df, variant)
            risks = np.array([tr["risk_pct_of_price"] for tr in trades]) if trades else np.array([0.0])
            print(f"[{tf_label}] {variant}: {n_zones} zones, {len(trades)} completed trades, "
                  f"median risk_pct_of_price={np.median(risks)*100:.4f}% (implied leverage {1/max(np.median(risks),1e-9):.0f}x)",
                  flush=True)

            all_days = (df.index.max() - df.index.min()).days + 1
            all_stats, gross_curve, net_curve = stats_for(trades, all_days)
            all_rows.append(dict(timeframe=tf_label, period="2024-2026 (all)", variant=variant, **all_stats))
            if tf_label == "1h":
                curves[variant] = dict(gross=gross_curve, net=net_curve)

            for yr in [2024, 2025, 2026]:
                yr_trades = [tr for tr in trades if tr["entry_time"].year == yr]
                yr_start = df.index[df.index.year == yr].min()
                yr_end = df.index[df.index.year == yr].max()
                if pd.isna(yr_start):
                    continue
                n_days = (yr_end - yr_start).days + 1
                yr_stats, _, _ = stats_for(yr_trades, n_days)
                all_rows.append(dict(timeframe=tf_label, period=str(yr), variant=variant, **yr_stats))

    summary = pd.DataFrame(all_rows)
    summary = summary[["timeframe", "period", "variant", "trades", "win_rate", "avg_win_r", "avg_loss_r",
                        "sum_r", "trades_per_day", "final_gross", "final_net"]]
    summary.to_csv(f"{CACHE_DIR}/dtfx_eurusd_summary.csv", index=False)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    pd.set_option("display.width", 200)
    print("\n" + summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = {"original": "#1f77b4", "1to1": "#d62728"}
    for variant, c in colors.items():
        gc = curves[variant]["gross"]
        nc = curves[variant]["net"]
        if gc:
            xs, ys = zip(*gc)
            ax.plot(xs, ys, color=c, linewidth=1.6, label=f"{variant} (gross)")
        if nc:
            xs, ys = zip(*nc)
            ax.plot(xs, ys, color=c, linewidth=1.2, linestyle="--", alpha=0.75, label=f"{variant} (net, ~2 pip cost)")
    ax.axhline(STARTING_EQUITY, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("DTFX on EUR-USD 1h entries, 2024-2026: Original (TP=trigger high) vs 1:1 (TP=zone edge)\n"
                  "$10,000 start, 1% risk/trade, compounded")
    ax.set_ylabel("Equity ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{CACHE_DIR}/dtfx_eurusd_pnl.png", dpi=130)
    print(f"Saved chart -> {CACHE_DIR}/dtfx_eurusd_pnl.png", flush=True)
