import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

OUT = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/strict_pivots_demo.png"

# ---- synthetic OHLC, hand-built to contain specific scenarios ----
# each row: (open, close, high, low)
bars = [
    (100.0, 100.4, 100.5, 99.9), (100.4, 100.9, 101.0, 100.3), (100.9, 101.3, 101.4, 100.8),
    (101.3, 101.0, 101.4, 100.9),   # single red pullback -- noise, should NOT validate a low
    (101.0, 101.6, 101.7, 100.95), (101.6, 102.1, 102.2, 101.5), (102.1, 102.6, 102.7, 102.0),
    (102.6, 103.0, 103.1, 102.5),
    (103.0, 102.7, 104.0, 102.6),   # upside wick spike to 104.0, closes red -- should NOT set a new high
    (102.7, 103.2, 103.3, 102.6), (103.2, 103.5, 103.6, 103.1), (103.5, 103.8, 103.9, 103.4),
    (103.8, 104.2, 104.3, 103.7), (104.2, 104.6, 104.7, 104.1),
    (104.6, 104.4, 104.8, 104.3),   # wick pokes to 104.8 but closes red -- real top stays at bar13 (104.7)
    (104.4, 104.0, 104.5, 103.9), (104.0, 103.6, 104.1, 103.5),  # 3rd consecutive red -> validates bar13 high
    (103.6, 103.1, 103.7, 103.0), (103.1, 102.6, 103.2, 102.5),
    (102.6, 102.9, 103.0, 102.5),   # single green bounce -- noise, should NOT validate a low
    (102.9, 102.3, 103.0, 102.2),
    (102.3, 102.5, 102.6, 101.0),   # downside wick spike to 101.0, closes green -- should NOT set a new low
    (102.5, 102.0, 102.6, 101.9), (102.0, 101.5, 102.1, 101.4), (101.5, 101.1, 101.6, 101.05),
    (101.1, 100.6, 101.2, 100.5), (100.6, 100.2, 100.7, 100.1),
    (100.2, 100.6, 100.7, 100.15), (100.6, 101.0, 101.1, 100.55), (101.0, 101.4, 101.5, 100.95),
    # 3rd consecutive green -> validates bar26 low (100.1)
    (101.4, 101.9, 102.0, 101.35), (101.9, 102.4, 102.5, 101.85), (102.4, 102.9, 103.0, 102.35),
    (102.9, 102.6, 103.1, 102.5), (102.6, 103.1, 103.2, 102.5), (103.1, 103.5, 103.6, 103.0),
]
idx = pd.date_range("2026-01-05 09:30:00", periods=len(bars), freq="1min")
df = pd.DataFrame(bars, columns=["open", "close", "high", "low"], index=idx)
df = df[["open", "high", "low", "close"]]


# ---- old method: centered rolling-window wick pivots (no candle-color / no close confirmation) ----
def find_pivots_loose(high, low, window=3):
    is_high = high == high.rolling(window * 2 + 1, center=True).max()
    is_low = low == low.rolling(window * 2 + 1, center=True).min()
    return high[is_high].dropna(), low[is_low].dropna()


swing_highs, swing_lows = find_pivots_loose(df["high"], df["low"], window=3)
old_pivots = sorted(
    [(t, v, "H") for t, v in swing_highs.items()] + [(t, v, "L") for t, v in swing_lows.items()],
    key=lambda x: x[0],
)


# ---- new method: >=3 consecutive opposite-color candles to validate, and only a full CLOSE
#      beyond the candidate's own high/low (never a wick) extends it ----
def find_pivots_strict(df, min_retracement=3):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    t = df.index
    pivots = []
    confirmations = []  # (extreme_time, confirm_time) for the connector line

    looking_for = "high"
    candidate_price, candidate_i = h[0], 0
    streak = 0

    for i in range(1, len(df)):
        is_red = c[i] < o[i]
        is_green = c[i] > o[i]

        if looking_for == "high":
            if c[i] > candidate_price:
                candidate_price, candidate_i, streak = h[i], i, 0
            elif is_red:
                streak += 1
                if streak >= min_retracement:
                    pivots.append((t[candidate_i], candidate_price, "H"))
                    confirmations.append((t[candidate_i], t[i]))
                    looking_for, candidate_price, candidate_i, streak = "low", l[i], i, 0
            else:
                streak = 0
        else:
            if c[i] < candidate_price:
                candidate_price, candidate_i, streak = l[i], i, 0
            elif is_green:
                streak += 1
                if streak >= min_retracement:
                    pivots.append((t[candidate_i], candidate_price, "L"))
                    confirmations.append((t[candidate_i], t[i]))
                    looking_for, candidate_price, candidate_i, streak = "high", h[i], i, 0
            else:
                streak = 0

    return pivots, confirmations


new_pivots, confirmations = find_pivots_strict(df, min_retracement=3)

print("OLD method (wick-based, no confirmation) pivots:")
for t, v, typ in old_pivots:
    print(f"  {t.strftime('%H:%M')}  {typ}  {v:.2f}")
print(f"\nNEW method (3-candle-close-confirmed) validated pivots:")
for t, v, typ in new_pivots:
    print(f"  {t.strftime('%H:%M')}  {typ}  {v:.2f}")


# ---- plot: two panels, same candles, old vs new pivot markers ----
def draw_candles(ax, df):
    bw = (df.index[1] - df.index[0]).total_seconds() / 86400 * 0.6
    for t, row in df.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        tn = mdates.date2num(t)
        ax.plot([tn, tn], [row["low"], row["high"]], color=color, linewidth=1, zorder=2)
        body_lo = min(row["open"], row["close"])
        body_h = abs(row["close"] - row["open"]) or 0.01
        ax.add_patch(Rectangle((tn - bw / 2, body_lo), bw, body_h, facecolor=color, edgecolor=color, zorder=3))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

# --- top: old / loose method ---
draw_candles(ax1, df)
for t, v, typ in old_pivots:
    color = "#b71c1c" if typ == "H" else "#1b5e20"
    marker = "v" if typ == "H" else "^"
    ax1.scatter([t], [v], color=color, marker=marker, s=110, edgecolor="black", zorder=6)
ax1.set_title("BEFORE — raw wick pivot (any local high/low counts, no confirmation)", fontsize=12, fontweight="bold", pad=34)
ax1.set_ylim(99.3, 106.1)
ax1.annotate("wick spike flagged\nas a swing high", xy=(df.index[8], 104.0), xytext=(-95, 8),
             textcoords="offset points", fontsize=8.5, color="#b71c1c", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#b71c1c"))
ax1.annotate("rejection wick flagged\ninstead of the real top", xy=(df.index[14], 104.8), xytext=(20, 24),
             textcoords="offset points", fontsize=8.5, color="#b71c1c", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#b71c1c"))
ax1.annotate("downside wick spike\nflagged as a swing low", xy=(df.index[21], 101.0), xytext=(5, -35),
             textcoords="offset points", fontsize=8.5, color="#1b5e20", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#1b5e20"))

legend1 = [
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#b71c1c", markeredgecolor="black", markersize=10, label="swing high (wick)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#1b5e20", markeredgecolor="black", markersize=10, label="swing low (wick)"),
]
ax1.legend(handles=legend1, fontsize=9, loc="lower right")
ax1.set_ylabel("Price")

# --- bottom: new / strict method ---
draw_candles(ax2, df)
for extreme_t, confirm_t in confirmations:
    ax2.axvline(confirm_t, color="black", linestyle=":", linewidth=0.8, alpha=0.5, zorder=1)
    ax2.plot([extreme_t, confirm_t], [df.loc[extreme_t:confirm_t, "high"].max() + 0.05] * 2,
             color="gray", linewidth=0, zorder=1)
for t, v, typ in new_pivots:
    color = "#b71c1c" if typ == "H" else "#1b5e20"
    marker = "v" if typ == "H" else "^"
    ax2.scatter([t], [v], color=color, marker=marker, s=140, edgecolor="black", linewidth=1.4, zorder=7)
    label = f"{v:.2f}  validated" if typ == "H" else f"{v:.2f}  validated"
    ax2.annotate(label, (t, v), xytext=(0, 12 if typ == "H" else -18), textcoords="offset points",
                 fontsize=8.5, fontweight="bold", ha="center", color=color)

ax2.annotate("single red candle\n-> ignored (needs 3)", xy=(df.index[3], 101.0), xytext=(-70, -35),
             textcoords="offset points", fontsize=8.5, color="#555", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#555"))
ax2.annotate("wick to 104.0, closed red\n-> never confirms a new high", xy=(df.index[8], 102.7), xytext=(-25, -42),
             textcoords="offset points", fontsize=8.5, color="#555", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#555"))
ax2.annotate("single green candle\n-> ignored (needs 3)", xy=(df.index[19], 102.9), xytext=(-115, 38),
             textcoords="offset points", fontsize=8.5, color="#555", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#555"))
ax2.annotate("wick to 101.0, closed green\n-> never confirms a new low", xy=(df.index[21], 102.5), xytext=(35, -50),
             textcoords="offset points", fontsize=8.5, color="#555", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#555"))

ax2.set_title("AFTER — requires a full candle CLOSE beyond the level + 3 consecutive opposite-color candles to validate",
              fontsize=12, fontweight="bold", pad=34)
ax2.set_ylim(99.3, 106.1)
legend2 = [
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#b71c1c", markeredgecolor="black", markersize=10, label="validated swing high"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#1b5e20", markeredgecolor="black", markersize=10, label="validated swing low"),
    Line2D([0], [0], color="black", linestyle=":", linewidth=1, label="confirmation point (3rd opposite candle)"),
]
ax2.legend(handles=legend2, fontsize=9, loc="lower right")
ax2.set_ylabel("Price")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

fig.suptitle("Stricter swing-validation rule: 3-candle retracement + close-only confirmation (synthetic data)",
             fontsize=13, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUT, dpi=140)
print("\nWrote", OUT)
