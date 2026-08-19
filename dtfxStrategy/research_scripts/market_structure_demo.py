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


def fetch_okx_1m(inst_id, target_bars=500):
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


df = fetch_okx_1m(INST_ID, target_bars=500)
print(f"Fetched {len(df)} real 1-minute bars: {df.index.min()} to {df.index.max()}")


# ---- 1. raw swing pivots (same construct as before) ----
def find_pivots(high, low, window=3):
    is_high = high == high.rolling(window * 2 + 1, center=True).max()
    is_low = low == low.rolling(window * 2 + 1, center=True).min()
    return high[is_high].dropna(), low[is_low].dropna()


swing_highs, swing_lows = find_pivots(df["high"], df["low"], window=3)

# ---- 2. collapse into a strictly alternating zigzag (keep the more extreme point
#         whenever two same-type pivots occur back to back with no opposite pivot between) ----
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

print(f"Zigzag pivots after alternation cleanup: {len(zigzag)}")


# ---- 3. the market-structure state machine, exactly as specified ----
#   Uptrend: confirmed_high updates immediately on any new higher high.
#            confirmed_low only updates (gets "confirmed") once a high beats confirmed_high.
#            Trend flips to DOWN the moment a low prints below confirmed_low.
#   Downtrend: symmetric (lows confirm immediately, highs need a lower-low to confirm).
#   On a flip, the pivot that just broke structure is auto-validated as the first
#   confirmed point of the new trend, and the old confirmed extreme on the other side
#   is reinterpreted (unchanged value) as the new trend's opposite reference.

events = []  # [time, value, type, role]; role in {'confirmed','pending','flip_up','flip_down','bootstrap'}
trend = None
confirmed_high = None
confirmed_low = None
pending_low_idx = None   # index into `events` of the most recent not-yet-confirmed low
pending_high_idx = None  # index into `events` of the most recent not-yet-confirmed high
flips = []  # (time, value, 'to_up'/'to_down')

# bootstrap using the first low and first high seen, in whichever order they occur
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
    trend = "up"  # simplifying assumption for the bootstrap phase only
    events.append([seen_low[0], seen_low[1], "L", "bootstrap"])
    events.append([seen_high[0], seen_high[1], "H", "bootstrap"])

for t, v, typ in zigzag[i:]:
    if typ == "L":
        if trend == "up":
            if v < confirmed_low:
                confirmed_high, confirmed_low = confirmed_high, v  # old high -> new ceiling; this low -> new floor
                trend = "down"
                pending_high_idx = None
                flips.append((t, v, "to_down"))
                events.append([t, v, "L", "flip_down"])
            else:
                pending_low_idx = len(events)
                events.append([t, v, "L", "pending"])
        else:  # trend == 'down'
            if confirmed_low is None or v < confirmed_low:
                confirmed_low = v
                if pending_high_idx is not None:
                    confirmed_high = events[pending_high_idx][1]
                    events[pending_high_idx][3] = "confirmed"  # retroactively promote the pending high
                    pending_high_idx = None
                events.append([t, v, "L", "confirmed"])
            else:
                events.append([t, v, "L", "pending"])
    else:  # typ == 'H'
        if trend == "down":
            if v > confirmed_high:
                confirmed_low, confirmed_high = confirmed_low, v  # old low -> new floor; this high -> new ceiling
                trend = "up"
                pending_low_idx = None
                flips.append((t, v, "to_up"))
                events.append([t, v, "H", "flip_up"])
            else:
                pending_high_idx = len(events)
                events.append([t, v, "H", "pending"])
        else:  # trend == 'up'
            if confirmed_high is None or v > confirmed_high:
                confirmed_high = v
                if pending_low_idx is not None:
                    confirmed_low = events[pending_low_idx][1]
                    events[pending_low_idx][3] = "confirmed"  # retroactively promote the pending low
                    pending_low_idx = None
                events.append([t, v, "H", "confirmed"])
            else:
                events.append([t, v, "H", "pending"])

print(f"\nTrend flips detected: {len(flips)}")
for t, v, direction in flips:
    print(f"  {t}  price={v:.2f}  {direction}")

print("\nFull event log:")
for t, v, typ, role in events:
    print(f"  {t}  {typ}  {v:.2f}  -> {role}")


# ---- 4. plot ----
fig, ax = plt.subplots(figsize=(17, 8.5))

bar_width_days = (df.index[1] - df.index[0]).total_seconds() / 86400 * 0.7
for t, row in df.iterrows():
    color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
    t_num = mdates.date2num(t)
    ax.plot([t_num, t_num], [row["low"], row["high"]], color=color, linewidth=0.6, zorder=2, alpha=0.85)
    body_low = min(row["open"], row["close"])
    body_height = abs(row["close"] - row["open"]) or 0.01
    ax.add_patch(Rectangle((t_num - bar_width_days / 2, body_low), bar_width_days, body_height,
                            facecolor=color, edgecolor=color, zorder=3, alpha=0.85))

# only confirmed/bootstrap/flip points are shown — pending (unconfirmed) swings are dropped entirely
shown = [e for e in events if e[3] != "pending"]

struct_times = [e[0] for e in shown]
struct_vals = [e[1] for e in shown]
seg_trend = []
running_trend = "up"
for e in shown:
    if e[3] == "flip_down":
        running_trend = "down"
    elif e[3] == "flip_up":
        running_trend = "up"
    seg_trend.append(running_trend)

for j in range(len(shown) - 1):
    color = "#1b5e20" if seg_trend[j] == "up" else "#b71c1c"
    ax.plot([struct_times[j], struct_times[j + 1]], [struct_vals[j], struct_vals[j + 1]],
            color=color, linewidth=2, alpha=0.85, zorder=5)

role_style = {
    "confirmed": dict(marker="o", s=70, edgecolor="black", zorder=7),
    "bootstrap": dict(marker="o", s=70, edgecolor="black", zorder=7),
    "flip_down": dict(marker="X", s=140, edgecolor="black", zorder=8),
    "flip_up": dict(marker="X", s=140, edgecolor="black", zorder=8),
}
for t, v, typ, role in shown:
    color = "#1b5e20" if (typ == "L") else "#b71c1c"
    if role in ("flip_down", "flip_up"):
        color = "#b71c1c" if role == "flip_down" else "#1b5e20"
    style = role_style[role]
    ax.scatter([t], [v], color=color, **style)

for t, v, direction in flips:
    ax.axvline(t, color="black", linestyle=":", linewidth=1, alpha=0.6, zorder=1)
    label = "trend flip: up -> down" if direction == "to_down" else "trend flip: down -> up"
    ax.annotate(label, xy=(t, v), xytext=(8, 12 if direction == "to_up" else -18),
                textcoords="offset points", fontsize=8, fontweight="bold",
                color="#1b5e20" if direction == "to_up" else "#b71c1c")

legend_elements = [
    Line2D([0], [0], color="#1b5e20", lw=2, label="structure line (uptrend regime)"),
    Line2D([0], [0], color="#b71c1c", lw=2, label="structure line (downtrend regime)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black",
           markersize=8, label="confirmed swing (validated HH/HL or LH/LL)"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor="gray", markeredgecolor="black",
           markersize=11, label="structure break (trend flip)"),
]
ax.legend(handles=legend_elements, fontsize=8.5, loc="upper left")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title(f"{INST_ID} 1-Minute — Market Structure (HH/HL confirmed uptrend, LH/LL confirmed downtrend)")
ax.set_ylabel("Price (USDT)")
plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/market_structure_demo.png"
plt.savefig(out_path, dpi=140)
print("\nWrote", out_path)

# ---- zoomed-in verification view around the first flip ----
zoom_start = pd.Timestamp("2026-08-18 10:55:00")
zoom_end = pd.Timestamp("2026-08-18 13:00:00")
zoom_df = df.loc[zoom_start:zoom_end]

fig2, ax2 = plt.subplots(figsize=(15, 7))
bw = (zoom_df.index[1] - zoom_df.index[0]).total_seconds() / 86400 * 0.7
for t, row in zoom_df.iterrows():
    color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
    t_num = mdates.date2num(t)
    ax2.plot([t_num, t_num], [row["low"], row["high"]], color=color, linewidth=1, zorder=2)
    body_low = min(row["open"], row["close"])
    body_height = abs(row["close"] - row["open"]) or 0.01
    ax2.add_patch(Rectangle((t_num - bw / 2, body_low), bw, body_height,
                             facecolor=color, edgecolor=color, zorder=3))

ax2.axhline(1899.23, color="#1b5e20", linestyle="--", linewidth=1.5,
            label="confirmed_low = 1899.23 (set 11:06, validated by the 11:27 higher high)")
ax2.scatter([pd.Timestamp("2026-08-18 11:06:00")], [1899.23], color="#1b5e20", s=90, zorder=6, edgecolor="black")
ax2.scatter([pd.Timestamp("2026-08-18 12:45:00")], [1899.04], color="#b71c1c", s=130, marker="X",
            zorder=6, edgecolor="black", label="flip low = 1899.04 (12:45)")
ax2.annotate("1899.23", (pd.Timestamp("2026-08-18 11:06:00"), 1899.23),
             xytext=(-45, 5), textcoords="offset points", fontsize=9, fontweight="bold", color="#1b5e20")
ax2.annotate("1899.04", (pd.Timestamp("2026-08-18 12:45:00"), 1899.04),
             xytext=(5, -15), textcoords="offset points", fontsize=9, fontweight="bold", color="#b71c1c")

ax2.set_ylim(1897, 1906)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax2.set_title("Zoomed: confirmed_low (1899.23) vs. the breaking low (1899.04) — real, ~$0.19 break")
ax2.legend(fontsize=9, loc="upper right")
plt.tight_layout()
out_path2 = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/market_structure_zoom.png"
plt.savefig(out_path2, dpi=140)
print("Wrote", out_path2)
