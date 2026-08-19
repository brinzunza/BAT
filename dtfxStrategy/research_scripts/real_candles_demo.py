import time
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

INST_ID = "ETH-USDT"
OKX_BASE = "https://www.okx.com/api/v5/market"


def fetch_okx_1m(inst_id, target_bars=300):
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


df = fetch_okx_1m(INST_ID, target_bars=240)
df = df.iloc[-240:]
print(f"Fetched {len(df)} real 1-minute bars: {df.index.min()} to {df.index.max()}")


def find_pivots(price, window=4):
    is_high = price == price.rolling(window * 2 + 1, center=True).max()
    is_low = price == price.rolling(window * 2 + 1, center=True).min()
    return price[is_high], price[is_low]


CHANNEL_WINDOW = 20
channel_high = df["close"].rolling(CHANNEL_WINDOW).max().shift(1)
channel_low = df["close"].rolling(CHANNEL_WINDOW).min().shift(1)
breakout_up = df["close"] > channel_high
breakout_down = df["close"] < channel_low
swing_highs, swing_lows = find_pivots(df["close"], window=4)

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]})
ax = axes[0]

bar_width_days = (df.index[1] - df.index[0]).total_seconds() / 86400 * 0.7
for t, row in df.iterrows():
    color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
    t_num = mdates.date2num(t)
    ax.plot([t_num, t_num], [row["low"], row["high"]], color=color, linewidth=0.8, zorder=2)
    body_low = min(row["open"], row["close"])
    body_height = abs(row["close"] - row["open"]) or 0.01
    ax.add_patch(Rectangle(
        (t_num - bar_width_days / 2, body_low), bar_width_days, body_height,
        facecolor=color, edgecolor=color, zorder=3,
    ))

ax.plot(df.index, channel_high, color="darkorange", linewidth=1, linestyle=":", alpha=0.8,
        label=f"{CHANNEL_WINDOW}min rolling high (channel)")
ax.plot(df.index, channel_low, color="purple", linewidth=1, linestyle=":", alpha=0.8,
        label=f"{CHANNEL_WINDOW}min rolling low (channel)")
ax.scatter(swing_highs.index, swing_highs.values, color="black", marker="v", s=30, zorder=4,
           label="swing high")
ax.scatter(swing_lows.index, swing_lows.values, color="black", marker="^", s=30, zorder=4,
           label="swing low")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title(f"{INST_ID} — Real 1-Minute OHLC Candles (OKX, {df.index.min():%Y-%m-%d %H:%M} UTC "
             f"to {df.index.max():%H:%M} UTC)")
ax.set_ylabel("Price (USDT)")
ax.legend(fontsize=8, loc="upper left")

ax2 = axes[1]
ax2.bar(df.index, df["vol"], width=bar_width_days,
        color=["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["open"], df["close"])])
ax2.set_title("Volume")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/real_candles_demo.png"
plt.savefig(out_path, dpi=140)
print("Wrote", out_path)
