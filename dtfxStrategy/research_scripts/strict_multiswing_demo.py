import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

OUT = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/strict_multiswing_demo.png"

rng = np.random.default_rng(7)

# ---- synthetic multi-regime path: alternating up/down legs with biased candle color,
#      so the strict rule gets clean, realistic 3-candle retracements to confirm on ----
regimes = [("up", 20), ("down", 16), ("up", 18), ("down", 15), ("up", 17), ("down", 14), ("up", 16)]

o, h, l, c = [], [], [], []
price = 100.0
for direction, n in regimes:
    p_trend = 0.72
    for _ in range(n):
        is_trend_candle = rng.random() < p_trend
        if direction == "up":
            body = rng.uniform(0.15, 0.55) if is_trend_candle else -rng.uniform(0.1, 0.4)
        else:
            body = -rng.uniform(0.15, 0.55) if is_trend_candle else rng.uniform(0.1, 0.4)
        op = price
        cl = op + body
        wick_up = rng.uniform(0.03, 0.18)
        wick_dn = rng.uniform(0.03, 0.18)
        hi = max(op, cl) + wick_up
        lo = min(op, cl) - wick_dn
        o.append(op); h.append(hi); l.append(lo); c.append(cl)
        price = cl

idx = pd.date_range("2026-02-02 09:30:00", periods=len(o), freq="1min")
df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)
print(f"Generated {len(df)} synthetic bars across {len(regimes)} regimes")


def find_pivots_strict(df, min_retracement=3):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    t = df.index
    pivots, confirmations = [], []
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


pivots, confirmations = find_pivots_strict(df, min_retracement=3)
print(f"Validated {len(pivots)} swings\n")

# classify HH/LH/HL/LL against the previous swing of the same type
last_h, last_l = None, None
labeled = []
for t, v, typ in pivots:
    if typ == "H":
        if last_h is None:
            tag = "H"
        else:
            tag = "HH" if v > last_h else "LH"
        last_h = v
    else:
        if last_l is None:
            tag = "L"
        else:
            tag = "HL" if v > last_l else "LL"
        last_l = v
    labeled.append((t, v, typ, tag))
    print(f"  {t.strftime('%H:%M')}  {typ}  {v:7.2f}   {tag}")


fig, ax = plt.subplots(figsize=(18, 8.5))
bw = (df.index[1] - df.index[0]).total_seconds() / 86400 * 0.6
for t, row in df.iterrows():
    color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
    tn = mdates.date2num(t)
    ax.plot([tn, tn], [row["low"], row["high"]], color=color, linewidth=0.9, zorder=2)
    body_lo = min(row["open"], row["close"])
    body_h = abs(row["close"] - row["open"]) or 0.01
    ax.add_patch(Rectangle((tn - bw / 2, body_lo), bw, body_h, facecolor=color, edgecolor=color, zorder=3))

# zigzag line, colored by leg direction (green = swing up into a high, red = swing down into a low)
for j in range(len(labeled) - 1):
    t0, v0, typ0, _ = labeled[j]
    t1, v1, typ1, _ = labeled[j + 1]
    seg_color = "#1b5e20" if typ1 == "H" else "#b71c1c"
    ax.plot([t0, t1], [v0, v1], color=seg_color, linewidth=2, alpha=0.85, zorder=5)

tag_color = {"HH": "#1b5e20", "H": "#1b5e20", "LH": "#e65100", "HL": "#2e7d32", "LL": "#b71c1c", "L": "#b71c1c"}
for t, v, typ, tag in labeled:
    marker = "v" if typ == "H" else "^"
    color = tag_color[tag]
    ax.scatter([t], [v], color=color, marker=marker, s=150, edgecolor="black", linewidth=1.2, zorder=7)
    ax.annotate(tag, (t, v), xytext=(0, 13 if typ == "H" else -19), textcoords="offset points",
                fontsize=10, fontweight="bold", ha="center", color=color)

for extreme_t, confirm_t in confirmations:
    ax.axvline(confirm_t, color="gray", linestyle=":", linewidth=0.6, alpha=0.4, zorder=1)

legend_elements = [
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#1b5e20", markeredgecolor="black", markersize=10, label="HH — higher high"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="#e65100", markeredgecolor="black", markersize=10, label="LH — lower high"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#2e7d32", markeredgecolor="black", markersize=10, label="HL — higher low"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#b71c1c", markeredgecolor="black", markersize=10, label="LL — lower low"),
    Line2D([0], [0], color="#1b5e20", lw=2, label="swing up (bullish leg)"),
    Line2D([0], [0], color="#b71c1c", lw=2, label="swing down (bearish leg)"),
    Line2D([0], [0], color="gray", linestyle=":", linewidth=1, label="validation point (3rd opposite candle)"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper left", ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_ylabel("Price")
ax.set_title("Multiple bullish & bearish swings — strict rule: 3-candle retracement + close-only confirmation (synthetic data)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT, dpi=140)
print("\nWrote", OUT)
