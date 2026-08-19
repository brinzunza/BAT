import time
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

INST_ID = "ETH-USDT"
OKX_BASE = "https://www.okx.com/api/v5/market"


def fetch_okx_1m(inst_id, target_bars=700):
    all_rows = []
    r = requests.get(f"{OKX_BASE}/candles", params={"instId": inst_id, "bar": "1m", "limit": 300}, timeout=20)
    r.raise_for_status()
    data = r.json()["data"]
    all_rows.extend(data)
    after = min(int(row[0]) for row in data)
    while len(all_rows) < target_bars:
        r = requests.get(
            f"{OKX_BASE}/history-candles",
            params={"instId": inst_id, "bar": "1m", "limit": 100, "after": after},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()["data"]
        if not data:
            break
        all_rows.extend(data)
        after = min(int(row[0]) for row in data)
        time.sleep(0.25)
    cols = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
    df = pd.DataFrame(all_rows, columns=cols)
    df = df[df["confirm"] == "1"]
    df["date"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
    df = df.drop_duplicates(subset="date").set_index("date").sort_index()
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "vol"]]


df = fetch_okx_1m(INST_ID, target_bars=2000)
print(f"Fetched {len(df)} real 1-minute bars: {df.index.min()} to {df.index.max()}")


def find_pivots(high, low, window=3):
    is_high = high == high.rolling(window * 2 + 1, center=True).max()
    is_low = low == low.rolling(window * 2 + 1, center=True).min()
    return high[is_high].dropna(), low[is_low].dropna()


swing_highs, swing_lows = find_pivots(df["high"], df["low"], window=3)

raw = [(t, v, "H") for t, v in swing_highs.items()] + [(t, v, "L") for t, v in swing_lows.items()]
raw.sort(key=lambda x: x[0])
zigzag = []
for t, v, typ in raw:
    if zigzag and zigzag[-1][2] == typ:
        prev_t, prev_v, prev_typ = zigzag[-1]
        more_extreme = (typ == "H" and v > prev_v) or (typ == "L" and v < prev_v)
        if more_extreme:
            zigzag[-1] = (t, v, typ)
        continue
    zigzag.append((t, v, typ))
print(f"Zigzag pivots: {len(zigzag)}")


# ---- structure FSM (same rules as before), now also emitting a zone every time
#      a new higher high confirms while in an uptrend ----
events = []
zones = []  # each: dict(top, bottom, mid, tp, created_at, trigger_high_time)
trend = None
confirmed_high = None
confirmed_low = None
pending_low_idx = None
pending_high_idx = None

i = 0
seen_low, seen_high = None, None
while i < len(zigzag) and (seen_low is None or seen_high is None):
    t, v, typ = zigzag[i]
    if typ == "L" and seen_low is None:
        seen_low = (t, v)
    if typ == "H" and seen_high is None:
        seen_high = (t, v)
    i += 1
if seen_low and seen_high:
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
        else:  # trend == 'down'
            if confirmed_low is None or v < confirmed_low:
                old_low = confirmed_low  # "the previous lower low" -> zone bottom
                # zone top = the lower high actually being confirmed by THIS breakdown
                # (the recent bounce high), not the already-stale confirmed_high from one cycle back
                zone_top = events[pending_high_idx][1] if pending_high_idx is not None else confirmed_high
                confirmed_low = v
                if pending_high_idx is not None:
                    confirmed_high = events[pending_high_idx][1]
                    events[pending_high_idx][3] = "confirmed"
                    pending_high_idx = None
                events.append([t, v, "L", "confirmed"])
                if old_low is not None and zone_top is not None and zone_top > old_low:
                    zones.append(dict(
                        top=zone_top, bottom=old_low, mid=(zone_top + old_low) / 2,
                        tp=v, created_at=t, side="sell",
                    ))
            else:
                events.append([t, v, "L", "pending"])
    else:  # H
        if trend == "down":
            if v > confirmed_high:
                confirmed_low, confirmed_high = confirmed_low, v
                trend = "up"
                pending_low_idx = None
                events.append([t, v, "H", "flip_up"])
            else:
                pending_high_idx = len(events)
                events.append([t, v, "H", "pending"])
        else:  # trend == 'up'
            if confirmed_high is None or v > confirmed_high:
                old_high = confirmed_high  # "the previous higher high" -> zone top
                # zone bottom = the higher low that's actually being confirmed by THIS breakout
                # (the recent pullback low), not the already-stale confirmed_low from one cycle back
                zone_bottom = events[pending_low_idx][1] if pending_low_idx is not None else confirmed_low
                confirmed_high = v
                if pending_low_idx is not None:
                    confirmed_low = events[pending_low_idx][1]
                    events[pending_low_idx][3] = "confirmed"
                    pending_low_idx = None
                events.append([t, v, "H", "confirmed"])
                if old_high is not None and zone_bottom is not None and old_high > zone_bottom:
                    zones.append(dict(
                        top=old_high, bottom=zone_bottom, mid=(old_high + zone_bottom) / 2,
                        tp=v, created_at=t, side="buy",
                    ))
            else:
                events.append([t, v, "H", "pending"])

print(f"\nZones created: {len(zones)}")
for z in zones:
    print(f"  created {z['created_at']}  top={z['top']:.2f}  bottom={z['bottom']:.2f}  "
          f"mid={z['mid']:.2f}  tp={z['tp']:.2f}")


# ---- simulate each zone forward bar-by-bar: entry at 50%, SL at bottom, TP at trigger high ----
def simulate_zone(zone, bars):
    """buy: price must fall INTO the zone from above (entry at mid, invalidated below bottom,
             stopped out below bottom, take-profit above tp).
    sell: mirror image — price must rise INTO the zone from below (entry at mid, invalidated
             above top, stopped out above top, take-profit below tp)."""
    is_buy = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy else zone["top"]
    stop_level = zone["bottom"] if is_buy else zone["top"]

    future = bars.loc[bars.index > zone["created_at"]]
    entry_time = None
    for t, row in future.iterrows():
        breached = row["low"] <= invalidation_level if is_buy else row["high"] >= invalidation_level
        if breached:
            return dict(**zone, status="invalidated", entry_time=None, exit_time=t, exit_price=invalidation_level)
        reached_mid = row["low"] <= zone["mid"] if is_buy else row["high"] >= zone["mid"]
        if reached_mid:
            entry_time = t
            break
    if entry_time is None:
        return dict(**zone, status="pending", entry_time=None, exit_time=None, exit_price=None)

    after_entry = future.loc[future.index > entry_time]
    for t, row in after_entry.iterrows():
        hit_sl = row["low"] <= stop_level if is_buy else row["high"] >= stop_level
        hit_tp = row["high"] >= zone["tp"] if is_buy else row["low"] <= zone["tp"]
        if hit_sl:
            return dict(**zone, status="stopped_out", entry_time=entry_time, exit_time=t, exit_price=stop_level)
        if hit_tp:
            return dict(**zone, status="take_profit", entry_time=entry_time, exit_time=t, exit_price=zone["tp"])
    return dict(**zone, status="open", entry_time=entry_time, exit_time=None, exit_price=None)


trades = [simulate_zone(z, df) for z in zones]
print("\nTrade outcomes:")
for tr in trades:
    r_multiple = None
    if tr["entry_time"] is not None:
        if tr["side"] == "buy":
            risk = tr["mid"] - tr["bottom"]
            reward_potential = tr["tp"] - tr["mid"]
        else:
            risk = tr["top"] - tr["mid"]
            reward_potential = tr["mid"] - tr["tp"]
        r_multiple = reward_potential / risk if risk > 0 else float("nan")
    line = f"  [{tr['side']:4s}] zone@{tr['created_at']}  status={tr['status']:12s}  entry={tr['entry_time']}  exit={tr['exit_time']}"
    if r_multiple is not None:
        line += f"  planned R:R={r_multiple:.2f}"
    print(line)


# ---- plot ----
fig, ax = plt.subplots(figsize=(18, 9))
bar_width_days = (df.index[1] - df.index[0]).total_seconds() / 86400 * 0.7
for t, row in df.iterrows():
    color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
    t_num = mdates.date2num(t)
    ax.plot([t_num, t_num], [row["low"], row["high"]], color=color, linewidth=0.6, zorder=2, alpha=0.8)
    body_low = min(row["open"], row["close"])
    body_height = abs(row["close"] - row["open"]) or 0.01
    ax.add_patch(Rectangle((t_num - bar_width_days / 2, body_low), bar_width_days, body_height,
                            facecolor=color, edgecolor=color, zorder=3, alpha=0.8))

status_color = {"take_profit": "#1b5e20", "stopped_out": "#b71c1c", "invalidated": "#9e9e9e", "pending": "#f9a825", "open": "#1565c0"}

for tr in trades:
    t0 = mdates.date2num(tr["created_at"])
    t_end = mdates.date2num(tr["exit_time"]) if tr["exit_time"] is not None else mdates.date2num(df.index[-1])
    color = status_color[tr["status"]]

    ax.add_patch(Rectangle((t0, tr["bottom"]), t_end - t0, tr["top"] - tr["bottom"],
                            facecolor=color, alpha=0.12, edgecolor=color, linewidth=1, zorder=1))
    ax.plot([t0, t_end], [tr["mid"], tr["mid"]], color=color, linestyle="--", linewidth=1, alpha=0.7, zorder=4)

    if tr["entry_time"] is not None:
        t_entry = mdates.date2num(tr["entry_time"])
        is_buy = tr["side"] == "buy"
        entry_marker = "^" if is_buy else "v"
        entry_color = "#1565c0" if is_buy else "#6a1b9a"
        stop_level = tr["bottom"] if is_buy else tr["top"]
        ax.scatter([t_entry], [tr["mid"]], marker=entry_marker, s=90, color=entry_color,
                    edgecolor="black", zorder=6)
        ax.plot([t_entry, t_end], [stop_level, stop_level], color="#b71c1c",
                 linewidth=1.3, alpha=0.8, zorder=4)
        ax.plot([t_entry, t_end], [tr["tp"], tr["tp"]], color="#1b5e20",
                 linewidth=1.3, alpha=0.8, zorder=4)
        if tr["status"] == "take_profit":
            ax.scatter([mdates.date2num(tr["exit_time"])], [tr["exit_price"]], marker="*", s=220,
                        color="#1b5e20", edgecolor="black", zorder=7)
        elif tr["status"] == "stopped_out":
            ax.scatter([mdates.date2num(tr["exit_time"])], [tr["exit_price"]], marker="X", s=120,
                        color="#b71c1c", edgecolor="black", zorder=7)

legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#1565c0", markeredgecolor="black",
           markersize=9, label="buy signal (uptrend zone, price hit 50% from above)"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#6a1b9a", markeredgecolor="black",
           markersize=9, label="sell signal (downtrend zone, price hit 50% from below)"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#1b5e20", markeredgecolor="black",
           markersize=14, label="take profit hit (back to trigger high/low)"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor="#b71c1c", markeredgecolor="black",
           markersize=10, label="stopped out (far side of zone broken)"),
    Rectangle((0, 0), 1, 1, facecolor="#f9a825", alpha=0.3, label="zone never touched (still pending)"),
    Rectangle((0, 0), 1, 1, facecolor="#9e9e9e", alpha=0.3, label="zone invalidated before 50% reached"),
]
ax.legend(handles=legend_elements, fontsize=8.5, loc="upper left")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title(f"{INST_ID} 1-Minute — Retracement Zones, Both Directions (uptrend HH-to-HL buys, downtrend LL-to-LH sells), 50% Entry")
ax.set_ylabel("Price (USDT)")
plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/zone_retracement_demo.png"
plt.savefig(out_path, dpi=140)
print("\nWrote", out_path)
