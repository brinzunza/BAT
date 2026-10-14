import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

rng = np.random.default_rng(42)

# ---- 1. build an idealized key-point price path: clean uptrend (each pullback finds
#      support exactly at the zone's 50% level and continues to a new high), then a
#      clean downtrend (mirror image) ----

keypoints = [100.0]  # start
N_UP_LEGS = 4
N_DOWN_LEGS = 4

price = 100.0
H_prev, L_prev = None, None

# bootstrap: first rally + first pullback
H0 = price + 15
keypoints.append(H0)
L0 = H0 - 7
keypoints.append(L0)
price = L0
H_prev, L_prev = H0, L0

for i in range(N_UP_LEGS):
    leg_size = 12 + i * 3
    H_new = price + leg_size
    keypoints.append(H_new)  # breakout -> creates zone (top=H_prev, bottom=L_prev)

    zone_top, zone_bottom = H_prev, L_prev
    mid = (zone_top + zone_bottom) / 2

    if i < N_UP_LEGS - 1:
        # pull back down through mid, to a new higher low that stays above zone_bottom
        L_new = zone_bottom + (mid - zone_bottom) * 0.35
        keypoints.append(L_new)
        price = L_new
        H_prev, L_prev = H_new, L_new
    else:
        # final leg: break DOWN through the zone bottom instead -> flips the trend
        breakdown = zone_bottom - 9
        keypoints.append(breakdown)
        price = breakdown
        flip_high, flip_low = H_new, breakdown  # carried into the downtrend

H_prev_dt, L_prev_dt = flip_high, flip_low
# first bounce after the flip
bounce0 = price + 7
keypoints.append(bounce0)
price = bounce0
H_prev_dt = bounce0

for i in range(N_DOWN_LEGS):
    leg_size = 12 + i * 3
    L_new = price - leg_size
    keypoints.append(L_new)  # breakdown -> creates SELL zone (top=H_prev_dt, bottom=L_prev_dt)

    zone_top, zone_bottom = H_prev_dt, L_prev_dt
    mid = (zone_top + zone_bottom) / 2

    if i < N_DOWN_LEGS - 1:
        # bounce back up through mid, to a new lower high that stays below zone_top
        H_new = zone_top - (zone_top - mid) * 0.35
        keypoints.append(H_new)
        price = H_new
        L_prev_dt, H_prev_dt = L_new, H_new
    else:
        # final leg: a small closing bounce, nothing further
        keypoints.append(L_new + 5)

print("Key points:", [round(k, 2) for k in keypoints])

# ---- 2. turn the key points into minute-bar OHLC candles (interpolate + light noise) ----
BARS_PER_LEG = 25
closes = []
for a, b in zip(keypoints[:-1], keypoints[1:]):
    seg = np.linspace(a, b, BARS_PER_LEG)
    noise = rng.normal(0, abs(b - a) * 0.01, BARS_PER_LEG)
    noise[0] = noise[-1] = 0  # keep endpoints exact so key levels are hit precisely
    closes.append(seg + noise)
closes = np.concatenate(closes)

idx = pd.date_range("2026-01-01", periods=len(closes), freq="min")
close = pd.Series(closes, index=idx)
open_ = close.shift(1).fillna(close.iloc[0])
high = pd.concat([open_, close], axis=1).max(axis=1) + rng.uniform(0, 0.15, len(close))
low = pd.concat([open_, close], axis=1).min(axis=1) - rng.uniform(0, 0.15, len(close))
df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
print(f"Synthetic bars: {len(df)}")


# ---- 3. run the EXACT SAME pivot/FSM/zone logic as the real-data notebook ----
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

events = []
zones = []
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

print(f"\nZones created: {len(zones)}")
for z in zones:
    print(f"  [{z['side']}] created {z['created_at']}  top={z['top']:.2f}  bottom={z['bottom']:.2f}  "
          f"mid={z['mid']:.2f}  tp={z['tp']:.2f}")


def simulate_zone(zone, bars):
    is_buy = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy else zone["top"]
    stop_level = invalidation_level
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
    print(f"  [{tr['side']:4s}] zone@{tr['created_at']}  status={tr['status']:12s}  "
          f"entry={tr['entry_time']}  exit={tr['exit_time']}")


# ---- 4. plot ----
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

status_color = {"take_profit": "#1b5e20", "stopped_out": "#b71c1c", "invalidated": "#9e9e9e",
                 "pending": "#f9a825", "open": "#1565c0"}

for tr in trades:
    t0 = mdates.date2num(tr["created_at"])
    t_end = mdates.date2num(tr["exit_time"]) if tr["exit_time"] is not None else mdates.date2num(df.index[-1])
    color = status_color[tr["status"]]
    ax.add_patch(Rectangle((t0, tr["bottom"]), t_end - t0, tr["top"] - tr["bottom"],
                            facecolor=color, alpha=0.15, edgecolor=color, linewidth=1, zorder=1))
    ax.plot([t0, t_end], [tr["mid"], tr["mid"]], color=color, linestyle="--", linewidth=1, alpha=0.8, zorder=4)
    if tr["entry_time"] is not None:
        t_entry = mdates.date2num(tr["entry_time"])
        is_buy = tr["side"] == "buy"
        entry_marker = "^" if is_buy else "v"
        entry_color = "#1565c0" if is_buy else "#6a1b9a"
        stop_level = tr["bottom"] if is_buy else tr["top"]
        ax.scatter([t_entry], [tr["mid"]], marker=entry_marker, s=110, color=entry_color,
                    edgecolor="black", zorder=6)
        ax.plot([t_entry, t_end], [stop_level, stop_level], color="#b71c1c", linewidth=1.3, alpha=0.8, zorder=4)
        ax.plot([t_entry, t_end], [tr["tp"], tr["tp"]], color="#1b5e20", linewidth=1.3, alpha=0.8, zorder=4)
        if tr["status"] == "take_profit":
            ax.scatter([mdates.date2num(tr["exit_time"])], [tr["exit_price"]], marker="*", s=260,
                        color="#1b5e20", edgecolor="black", zorder=7)
        elif tr["status"] == "stopped_out":
            ax.scatter([mdates.date2num(tr["exit_time"])], [tr["exit_price"]], marker="X", s=140,
                        color="#b71c1c", edgecolor="black", zorder=7)

legend_elements = [
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#1565c0", markeredgecolor="black",
           markersize=10, label="buy signal"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#6a1b9a", markeredgecolor="black",
           markersize=10, label="sell signal"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#1b5e20", markeredgecolor="black",
           markersize=16, label="take profit hit"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor="#b71c1c", markeredgecolor="black",
           markersize=11, label="stopped out"),
]
ax.legend(handles=legend_elements, fontsize=10, loc="upper left")
ax.set_title("Synthetic Idealized Uptrend -> Downtrend: Retracement Zone Strategy, Both Directions")
ax.set_ylabel("Price")
ax.set_xticks([])
plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/synthetic_ideal_trend.png"
plt.savefig(out_path, dpi=140)
print("\nWrote", out_path)
