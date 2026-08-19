import time
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

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


def find_pivots(high, low, window=3):
    is_high = high == high.rolling(window * 2 + 1, center=True).max()
    is_low = low == low.rolling(window * 2 + 1, center=True).min()
    swing_highs = high[is_high]
    swing_lows = low[is_low]
    return swing_highs.dropna(), swing_lows.dropna()


swing_highs, swing_lows = find_pivots(df["high"], df["low"], window=3)
print(f"swing highs found: {len(swing_highs)}, swing lows found: {len(swing_lows)}")


def find_monotonic_runs(pivots, direction, min_len=3):
    """direction='up' -> strictly increasing values (higher lows).
    direction='down' -> strictly decreasing values (lower highs).
    Returns list of (start_idx_label, end_idx_label, start_val, end_val, run_length)."""
    idx = pivots.index
    vals = pivots.values
    runs = []
    run_start = 0
    for i in range(1, len(vals) + 1):
        broke = True
        if i < len(vals):
            if direction == "up" and vals[i] > vals[i - 1]:
                broke = False
            elif direction == "down" and vals[i] < vals[i - 1]:
                broke = False
        if broke:
            run_len = i - run_start
            if run_len >= min_len:
                runs.append((idx[run_start], idx[i - 1], vals[run_start], vals[i - 1], run_len))
            run_start = i
    return runs


bullish_runs = find_monotonic_runs(swing_lows, "up", min_len=3)
bearish_runs = find_monotonic_runs(swing_highs, "down", min_len=3)
print(f"bullish (higher-low) runs of 3+: {len(bullish_runs)}")
for r in bullish_runs:
    print("  ", r)
print(f"bearish (lower-high) runs of 3+: {len(bearish_runs)}")
for r in bearish_runs:
    print("  ", r)


fig, ax = plt.subplots(figsize=(16, 8))

bar_width_days = (df.index[1] - df.index[0]).total_seconds() / 86400 * 0.7
for t, row in df.iterrows():
    color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
    t_num = mdates.date2num(t)
    ax.plot([t_num, t_num], [row["low"], row["high"]], color=color, linewidth=0.7, zorder=2)
    body_low = min(row["open"], row["close"])
    body_height = abs(row["close"] - row["open"]) or 0.01
    ax.add_patch(Rectangle(
        (t_num - bar_width_days / 2, body_low), bar_width_days, body_height,
        facecolor=color, edgecolor=color, zorder=3,
    ))

ax.scatter(swing_highs.index, swing_highs.values, color="#555555", marker="v", s=18, zorder=4, alpha=0.6)
ax.scatter(swing_lows.index, swing_lows.values, color="#555555", marker="^", s=18, zorder=4, alpha=0.6)

extend_minutes = 25
for (t0, t1, v0, v1, n) in bullish_runs:
    t0n, t1n = mdates.date2num(t0), mdates.date2num(t1)
    slope = (v1 - v0) / (t1n - t0n)
    t_end = mdates.date2num(t1 + pd.Timedelta(minutes=extend_minutes))
    v_end = v1 + slope * (t_end - t1n)
    ax.plot([t0n, t1n], [v0, v1], color="#1b5e20", linewidth=2.2, zorder=5)
    ax.plot([t1n, t_end], [v1, v_end], color="#1b5e20", linewidth=2.2, linestyle="--", alpha=0.8, zorder=5)
    ax.scatter([t0n, t1n], [v0, v1], color="#1b5e20", s=55, zorder=6, marker="o")

for (t0, t1, v0, v1, n) in bearish_runs:
    t0n, t1n = mdates.date2num(t0), mdates.date2num(t1)
    slope = (v1 - v0) / (t1n - t0n)
    t_end = mdates.date2num(t1 + pd.Timedelta(minutes=extend_minutes))
    v_end = v1 + slope * (t_end - t1n)
    ax.plot([t0n, t1n], [v0, v1], color="#b71c1c", linewidth=2.2, zorder=5)
    ax.plot([t1n, t_end], [v1, v_end], color="#b71c1c", linewidth=2.2, linestyle="--", alpha=0.8, zorder=5)
    ax.scatter([t0n, t1n], [v0, v1], color="#b71c1c", s=55, zorder=6, marker="o")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="#1b5e20", lw=2.2, label="bullish trendline (3+ higher lows, 'U' swings)"),
    Line2D([0], [0], color="#b71c1c", lw=2.2, label="bearish trendline (3+ lower highs, 'n' swings)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#555555", markersize=6, alpha=0.7, label="swing low"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#555555", markersize=6, alpha=0.7, label="swing high"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title(f"{INST_ID} 1-Minute — TradingView-Style 2-Point Trendlines "
             f"(drawn only when 3+ consecutive swings confirm the direction)")
ax.set_ylabel("Price (USDT)")
plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/tv_trendline_demo.png"
plt.savefig(out_path, dpi=140)
print("Wrote", out_path)
