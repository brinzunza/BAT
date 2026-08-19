import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dtfx_eurusd_backtest import CACHE_DIR, resample_ohlc, run_structure_fsm

DATA_PATH = f"{CACHE_DIR}/lse_eurusd_1min_full.csv"


def find_pivots_strict_confirmed(df, min_retracement=3):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    t = df.index.values
    pivots, confirmed_ats = [], []
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
                    pivots.append((pd.Timestamp(t[candidate_i]), candidate_price, "H"))
                    confirmed_ats.append(pd.Timestamp(t[i]))
                    looking_for, candidate_price, candidate_i, streak = "low", l[i], i, 0
            else:
                streak = 0
        else:
            if c[i] < candidate_price:
                candidate_price, candidate_i, streak = l[i], i, 0
            elif is_green:
                streak += 1
                if streak >= min_retracement:
                    pivots.append((pd.Timestamp(t[candidate_i]), candidate_price, "L"))
                    confirmed_ats.append(pd.Timestamp(t[i]))
                    looking_for, candidate_price, candidate_i, streak = "high", h[i], i, 0
            else:
                streak = 0
    return pivots, confirmed_ats


df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
df.index = df.index.tz_localize(None)
df = df[["open", "high", "low", "close"]]
df4h = resample_ohlc(df, "4h")
print(f"1min: {len(df)} bars. 4H: {len(df4h)} bars, {df4h.index.min()} -> {df4h.index.max()}", flush=True)

zigzag, confirmed_ats = find_pivots_strict_confirmed(df4h, 3)
events, zones4h = run_structure_fsm(zigzag)
print(f"4H: {len(zigzag)} validated swings, {len(events)} events, {len(zones4h)} 4H zones", flush=True)

confirm_map = {(t, v, typ): conf for (t, v, typ), conf in zip(zigzag, confirmed_ats)}

trend_timeline = []  # (confirmed_at, trend_direction)
for e in events:
    t_, v_, typ_, role = e
    conf = confirm_map[(t_, v_, typ_)]
    if role == "bootstrap":
        trend_timeline.append((conf, "up"))  # matches run_structure_fsm's simplifying bootstrap assumption
    elif role == "flip_up":
        trend_timeline.append((conf, "up"))
    elif role == "flip_down":
        trend_timeline.append((conf, "down"))
trend_timeline.sort(key=lambda x: x[0])
tt_times = np.array([x[0] for x in trend_timeline], dtype="datetime64[ns]")
tt_dirs = np.array([x[1] for x in trend_timeline])
print(f"4H trend timeline: {len(trend_timeline)} confirmed regime points", flush=True)


def trend_at(query_time):
    idx = np.searchsorted(tt_times, np.datetime64(query_time), side="right") - 1
    if idx < 0:
        return None
    return tt_dirs[idx]


t = pd.read_csv(f"{CACHE_DIR}/dtfx_diag_trades.csv", parse_dates=["entry_time"])
t["htf_trend"] = [trend_at(e) for e in t["entry_time"]]
t = t.dropna(subset=["htf_trend"])
t["aligned"] = np.where(
    ((t["side"] == "buy") & (t["htf_trend"] == "up")) | ((t["side"] == "sell") & (t["htf_trend"] == "down")),
    "aligned", "against"
)
t["year"] = t["entry_time"].dt.year
t["regime"] = np.where(t["year"] >= 2021, "2021-2026", "2009-2020")


def agg(g):
    wins = (g["r"] > 0).sum()
    n = len(g)
    return pd.Series({"trades": n, "win_rate": wins / n, "avg_r": g["r"].mean(), "sum_r": g["r"].sum()})


print("\n=== 3. 4H TREND ALIGNMENT (no-lookahead) ===")
by_align = t.groupby("aligned", observed=True).apply(agg)
print(by_align.to_string())

by_align_regime = t.groupby(["regime", "aligned"], observed=True).apply(agg)
print("\n--- by regime ---")
print(by_align_regime.to_string())

by_align_side = t.groupby(["side", "aligned"], observed=True).apply(agg)
print("\n--- by side ---")
print(by_align_side.to_string())

aligned_only = t[t["aligned"] == "aligned"]
print(f"\nALIGNED-ONLY FILTER: trades={len(aligned_only)} sum_r={aligned_only['r'].sum():.1f} "
      f"win_rate={(aligned_only['r']>0).mean():.3f}  (baseline: trades={len(t)} sum_r={t['r'].sum():.1f} "
      f"win_rate={(t['r']>0).mean():.3f})")

t.to_csv(f"{CACHE_DIR}/dtfx_diag4_trades.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
colors = ["#2e7d32" if x == "aligned" else "#b71c1c" for x in by_align.index]
ax.bar(by_align.index, by_align["sum_r"], color=colors)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_title("Sum R: 4H-trend-aligned vs against")
ax.set_ylabel("Sum R")

ax = axes[1]
x = np.arange(2)
width = 0.35
for i, (regime, c) in enumerate([("2009-2020", "#b71c1c"), ("2021-2026", "#2e7d32")]):
    sub = t[t["regime"] == regime].groupby("aligned", observed=True).apply(agg).reindex(["aligned", "against"])
    ax.bar(x + (i - 0.5) * width, sub["sum_r"], width, label=regime, color=c)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["aligned", "against"])
ax.set_ylabel("Sum R")
ax.set_title("Sum R: alignment x regime")
ax.legend()

fig.suptitle("DTFX EUR/USD: does 4H trend alignment separate winners from losers? (original variant)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{CACHE_DIR}/dtfx_diagnostics4_mtf.png", dpi=130)
print(f"\nSaved chart -> {CACHE_DIR}/dtfx_diagnostics4_mtf.png", flush=True)
