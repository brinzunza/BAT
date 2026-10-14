import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

n = 160
idx = pd.date_range("2026-01-01", periods=n, freq="D")


def wobble(length, cycles, amp, seed_noise):
    t = np.linspace(0, cycles * 2 * np.pi, length)
    return amp * np.sin(t) + seed_noise


up1 = np.linspace(100, 126, 60) + wobble(60, 4.5, 2.3, rng.normal(0, 0.5, 60).cumsum() * 0.1)
down1 = np.linspace(up1[-1], up1[-1] - 9, 50) + wobble(50, 3.5, 2.0, rng.normal(0, 0.5, 50).cumsum() * 0.1)
up2 = np.linspace(down1[-1], down1[-1] + 20, 50) + wobble(50, 4, 2.0, rng.normal(0, 0.5, 50).cumsum() * 0.1)

price = np.concatenate([up1, down1, up2])
price = pd.Series(price, index=idx, name="close")


# ---- exact functions from the notebooks ----

def find_pivots(price, window=5):
    is_high = price == price.rolling(window * 2 + 1, center=True).max()
    is_low = price == price.rolling(window * 2 + 1, center=True).min()
    return price[is_high], price[is_low]


def fit_trendline(pivot_points, n_recent=4):
    if len(pivot_points) < 2:
        return None
    recent = pivot_points.iloc[-n_recent:]
    x = (recent.index - recent.index[0]).days.values.astype(float)
    y = recent.values
    slope, intercept = np.polyfit(x, y, 1)
    return recent.index[0], slope, intercept


def rolling_trend_pct(price, window=20):
    def slope(y):
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    raw_slope = price.rolling(window).apply(slope, raw=True)
    return raw_slope * window / price


CHANNEL_WINDOW = 20
channel_high = price.rolling(CHANNEL_WINDOW).max().shift(1)
channel_low = price.rolling(CHANNEL_WINDOW).min().shift(1)
breakout_up = price > channel_high
breakout_down = price < channel_low

swing_highs, swing_lows = find_pivots(price, window=5)
trend_pct = rolling_trend_pct(price, window=20)

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1.4, 1.2]})

# --- Panel 1: price + swing pivots + fitted trendlines + Donchian channel ---
ax = axes[0]
ax.plot(price.index, price, color="tab:blue", linewidth=1.3, label="close (\"bars\")", zorder=2)
ax.scatter(swing_highs.index, swing_highs.values, color="red", s=45, zorder=4, label="swing high (pivot)")
ax.scatter(swing_lows.index, swing_lows.values, color="green", s=45, zorder=4, label="swing low (pivot)")

for pivots, color, label in [(swing_highs, "red", "trendline thru recent highs"),
                              (swing_lows, "green", "trendline thru recent lows")]:
    fit = fit_trendline(pivots, n_recent=4)
    if fit:
        x0, slope, intercept = fit
        x_line = pd.date_range(x0, price.index[-1])
        y_line = intercept + slope * (x_line - x0).days
        ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=1.8, alpha=0.85, label=label, zorder=3)

ax.plot(price.index, channel_high, color="darkorange", linewidth=1, linestyle=":", alpha=0.9,
        label=f"{CHANNEL_WINDOW}d rolling high (channel)")
ax.plot(price.index, channel_low, color="purple", linewidth=1, linestyle=":", alpha=0.9,
        label=f"{CHANNEL_WINDOW}d rolling low (channel)")

up_pts = price.index[breakout_up.fillna(False)]
down_pts = price.index[breakout_down.fillna(False)]
ax.scatter(up_pts, price[up_pts], color="black", marker="^", s=55, zorder=5, label="breakout UP bar")
ax.scatter(down_pts, price[down_pts], color="black", marker="v", s=55, zorder=5, label="breakout DOWN bar")

ax.set_title("Sample Price Series: Pivots, Fitted Trendlines, and Donchian Channel Breakouts")
ax.legend(fontsize=7.5, ncol=2, loc="upper left")

# --- Panel 2: rolling regression-slope trend_pct signal ---
ax2 = axes[1]
ax2.plot(trend_pct.index, trend_pct * 100, color="purple")
ax2.axhline(0, color="gray", linewidth=0.8)
ax2.fill_between(trend_pct.index, trend_pct * 100, 0, where=(trend_pct > 0), color="green", alpha=0.15)
ax2.fill_between(trend_pct.index, trend_pct * 100, 0, where=(trend_pct < 0), color="red", alpha=0.15)
ax2.set_title("Tradeable Signal #1: Rolling 20-Day Regression Slope (\"trend_pct\", causal)")
ax2.set_ylabel("% move / window")

# --- Panel 3: breakout state (the other tradeable signal) ---
ax3 = axes[2]
state = pd.Series(0, index=price.index)
state[breakout_up.fillna(False)] = 1
state[breakout_down.fillna(False)] = -1
ax3.fill_between(state.index, state, 0, step="pre", color="black", alpha=0.6)
ax3.set_yticks([-1, 0, 1])
ax3.set_yticklabels(["down breakout", "inside channel", "up breakout"])
ax3.set_title("Tradeable Signal #2: Channel-Breakout State (causal, this is what gates entries)")

plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/trend_demo.png"
plt.savefig(out_path, dpi=140)
print("Wrote", out_path)
