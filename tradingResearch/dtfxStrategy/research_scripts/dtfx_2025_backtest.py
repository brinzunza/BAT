import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"
TICKERS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XAUT-USDT"]


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
    """Index of the first True in a boolean numpy array, or len(mask) if none."""
    idx = np.flatnonzero(mask)
    return idx[0] if len(idx) else len(mask)


def simulate_zone_vectorized(zone, times, lows, highs, reversed_signal=False):
    """Same logic as the row-by-row version, but using numpy boolean masks + first-nonzero
    instead of a Python-level per-bar loop -- orders of magnitude faster at 500k+ rows."""
    is_buy_zone = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy_zone else zone["top"]
    going_long = is_buy_zone if not reversed_signal else not is_buy_zone
    if not reversed_signal:
        stop_level = zone["bottom"] if is_buy_zone else zone["top"]
        target_level = zone["tp"]
    else:
        stop_level = zone["tp"]
        target_level = zone["bottom"] if is_buy_zone else zone["top"]

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


def r_multiple(tr, reversed_signal=False):
    if tr["entry_time"] is None:
        return None
    is_buy_zone = tr["side"] == "buy"
    if not reversed_signal:
        risk = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
        reward_level = tr["tp"]
        going_long = is_buy_zone
    else:
        risk = (tr["tp"] - tr["mid"]) if is_buy_zone else (tr["mid"] - tr["tp"])
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


def stats_for(df_slice, reversed_signal, zones=None):
    if zones is None:
        swing_highs, swing_lows = find_pivots(df_slice["high"], df_slice["low"], window=3)
        zigzag = build_zigzag(swing_highs, swing_lows)
        _, zones = run_structure_fsm(zigzag)
    times = df_slice.index.values
    lows = df_slice["low"].values
    highs = df_slice["high"].values
    trades = [simulate_zone_vectorized(z, times, lows, highs, reversed_signal=reversed_signal) for z in zones]
    completed = [tr for tr in trades if tr["status"] in ("take_profit", "stopped_out")]
    r_values = [r for r in (r_multiple(tr, reversed_signal=reversed_signal) for tr in completed) if r is not None]
    wins = sum(1 for tr in completed if tr["status"] == "take_profit")
    return dict(
        zones_created=len(zones), trades_taken=len(completed), wins=wins, losses=len(completed) - wins,
        win_rate=wins / len(completed) if completed else np.nan,
        sum_R=sum(r_values) if r_values else 0.0,
        avg_R=np.mean(r_values) if r_values else np.nan,
    )


results = []
for ticker in TICKERS:
    cache_path = f"{CACHE_DIR}/polygon_{ticker.replace('-','_')}_2025.csv"
    df = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date").sort_index()
    df = df[["open", "high", "low", "close"]]
    print(f"{ticker}: {len(df)} bars loaded")

    df["quarter"] = df.index.quarter
    for q in [1, 2, 3, 4]:
        q_df = df[df["quarter"] == q][["open", "high", "low", "close"]]
        if len(q_df) < 50:
            continue
        swing_highs, swing_lows = find_pivots(q_df["high"], q_df["low"], window=3)
        zigzag = build_zigzag(swing_highs, swing_lows)
        _, q_zones = run_structure_fsm(zigzag)
        for variant, rev in [("original", False), ("reversed", True)]:
            s = stats_for(q_df, rev, zones=q_zones)
            results.append(dict(ticker=ticker, period=f"Q{q}_2025", variant=variant, bars=len(q_df), **s))

    full_df = df[["open", "high", "low", "close"]]
    swing_highs, swing_lows = find_pivots(full_df["high"], full_df["low"], window=3)
    zigzag = build_zigzag(swing_highs, swing_lows)
    _, full_zones = run_structure_fsm(zigzag)
    for variant, rev in [("original", False), ("reversed", True)]:
        s = stats_for(full_df, rev, zones=full_zones)
        results.append(dict(ticker=ticker, period="FULL_2025", variant=variant, bars=len(df), **s))
    print(f"  {ticker} done", flush=True)

summary = pd.DataFrame(results)
print("\n" + "=" * 120)
print(summary.to_string(index=False))
summary.to_csv(f"{CACHE_DIR}/dtfx_2025_summary.csv", index=False)

# ---- full-year totals chart ----
full_year = summary[summary["period"] == "FULL_2025"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

pivot_r = full_year.pivot(index="ticker", columns="variant", values="sum_R")
pivot_r.plot(kind="bar", ax=axes[0], color=["#1565c0", "#b71c1c"])
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("Full-Year 2025: Sum R by Ticker")
axes[0].set_ylabel("Sum of R multiples")

pivot_wr = full_year.pivot(index="ticker", columns="variant", values="win_rate") * 100
pivot_wr.plot(kind="bar", ax=axes[1], color=["#1565c0", "#b71c1c"])
axes[1].axhline(50, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Full-Year 2025: Win Rate by Ticker")
axes[1].set_ylabel("Win rate (%)")

plt.tight_layout()
plt.savefig(f"{CACHE_DIR}/dtfx_2025_fullyear.png", dpi=140)

# ---- quarterly consistency chart (reversed variant, since that's the promising one) ----
quarterly = summary[(summary["period"] != "FULL_2025") & (summary["variant"] == "reversed")]
fig2, ax2 = plt.subplots(figsize=(12, 6))
qp = quarterly.pivot(index="period", columns="ticker", values="sum_R")
qp = qp.reindex(["Q1_2025", "Q2_2025", "Q3_2025", "Q4_2025"])
qp.plot(kind="bar", ax=ax2)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("Reversed DTFX: Sum R by Quarter, 2025")
ax2.set_ylabel("Sum of R multiples")
plt.tight_layout()
plt.savefig(f"{CACHE_DIR}/dtfx_2025_quarterly.png", dpi=140)

print("\nWrote charts and summary CSV")
