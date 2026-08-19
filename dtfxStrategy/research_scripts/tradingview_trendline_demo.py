import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

df = pd.read_csv(
    "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/eth_1m_snapshot.csv",
    index_col=0, parse_dates=True,
)


def find_pivots(price, window=4):
    is_high = price == price.rolling(window * 2 + 1, center=True).max()
    is_low = price == price.rolling(window * 2 + 1, center=True).min()
    return price[is_high], price[is_low]


swing_highs, swing_lows = find_pivots(df["close"], window=4)


def two_point_line(p1_time, p1_price, p2_time, p2_price, end_time):
    p1_time, p2_time, end_time = pd.Timestamp(p1_time), pd.Timestamp(p2_time), pd.Timestamp(end_time)
    slope = (p2_price - p1_price) / (p2_time - p1_time).total_seconds()
    x = pd.date_range(p1_time, end_time, freq="1min")
    y = p1_price + slope * (x - p1_time).total_seconds()
    return x, y


def fit_trendline_regression(pivot_points, n_recent=4):
    recent = pivot_points.iloc[-n_recent:]
    x = (recent.index - recent.index[0]).total_seconds().values / 60.0
    y = recent.values
    slope, intercept = np.polyfit(x, y, 1)
    return recent.index[0], slope, intercept


fig, ax = plt.subplots(figsize=(15, 8))

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

end_time = df.index[-1] + pd.Timedelta(minutes=15)

# --- TradingView-style: literal 2-point lines, manually anchored, extended right ---
r_x, r_y = two_point_line("2026-08-18 12:34:00", 1901.88, "2026-08-18 13:31:00", 1898.21, end_time)
ax.plot(r_x, r_y, color="#1f6feb", linewidth=2.2, zorder=5,
        label="TradingView-style: 2-point resistance (12:34 -> 13:31, extended)")
ax.scatter([pd.Timestamp("2026-08-18 12:34:00"), pd.Timestamp("2026-08-18 13:31:00")],
           [1901.88, 1898.21], color="#1f6feb", s=90, zorder=6, edgecolor="white", linewidth=1.2)

s_x, s_y = two_point_line("2026-08-18 15:10:00", 1908.62, "2026-08-18 15:44:00", 1917.02, end_time)
ax.plot(s_x, s_y, color="#1f6feb", linewidth=2.2, zorder=5,
        label="TradingView-style: 2-point support (15:10 -> 15:44, extended)")
ax.scatter([pd.Timestamp("2026-08-18 15:10:00"), pd.Timestamp("2026-08-18 15:44:00")],
           [1908.62, 1917.02], color="#1f6feb", s=90, zorder=6, edgecolor="white", linewidth=1.2)

# --- For direct contrast: my notebook's regression-fit line through the same early highs ---
early_highs = swing_highs[swing_highs.index <= pd.Timestamp("2026-08-18 14:04:00")]
x0, slope, intercept = fit_trendline_regression(early_highs, n_recent=4)
x_line = pd.date_range(x0, end_time, freq="1min")
minutes_elapsed = (x_line - x0).total_seconds() / 60.0
y_line = intercept + slope * minutes_elapsed
ax.plot(x_line, y_line, color="#d62728", linewidth=1.6, linestyle="--", zorder=5,
        label="My notebook's fit_trendline(): regression through last 4 swing highs")
used = early_highs.iloc[-4:]
ax.scatter(used.index, used.values, color="#d62728", s=45, marker="D", zorder=6)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_title("ETH-USDT, Real 1-Minute Candles: TradingView-Style 2-Point Trendline vs. Regression Fit")
ax.set_ylabel("Price (USDT)")
ax.legend(fontsize=8.5, loc="upper left")

plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/tradingview_trendline_demo.png"
plt.savefig(out_path, dpi=140)
print("Wrote", out_path)
