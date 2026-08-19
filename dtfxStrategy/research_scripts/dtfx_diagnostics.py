import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dtfx_eurusd_backtest import CACHE_DIR
from dtfx_eurusd_backtest_strict import get_trades_strict  # UNCHANGED rule, no retuning

DATA_PATH = f"{CACHE_DIR}/lse_eurusd_1min_full.csv"
OUT_DIR = CACHE_DIR

df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
df.index = df.index.tz_localize(None)
df = df[["open", "high", "low", "close"]]
print(f"Loaded {len(df)} bars, {df.index.min()} -> {df.index.max()}", flush=True)

trades, n_zones = get_trades_strict(df, "original")
print(f"original: {n_zones} zones, {len(trades)} completed trades", flush=True)

t = pd.DataFrame([{
    "entry_time": tr["entry_time"],
    "side": tr["side"],
    "r": tr["r"],
    "status": tr["status"],
} for tr in trades])
t["year"] = t["entry_time"].dt.year
t["hour"] = t["entry_time"].dt.hour
t["month"] = t["entry_time"].dt.month
t["dow"] = t["entry_time"].dt.day_name()
t["regime"] = np.where(t["year"] >= 2021, "2021-2026", "2009-2020")

t.to_csv(f"{OUT_DIR}/dtfx_diag_trades.csv", index=False)


def agg(g):
    wins = (g["r"] > 0).sum()
    n = len(g)
    return pd.Series({
        "trades": n,
        "win_rate": wins / n if n else np.nan,
        "avg_r": g["r"].mean(),
        "sum_r": g["r"].sum(),
    })


print("\n=== BY SIDE (buy vs sell) ===")
by_side = t.groupby(["regime", "side"]).apply(agg).reset_index()
print(by_side.to_string(index=False))

by_side_all = t.groupby("side").apply(agg).reset_index()
print("\n--- combined (all years) ---")
print(by_side_all.to_string(index=False))

print("\n=== BY HOUR OF DAY (UTC) ===")
by_hour = t.groupby(["regime", "hour"]).apply(agg).reset_index()
by_hour_all = t.groupby("hour").apply(agg).reset_index()
print(by_hour_all.to_string(index=False))

print("\n=== BY MONTH ===")
by_month = t.groupby(["regime", "month"]).apply(agg).reset_index()
by_month_all = t.groupby("month").apply(agg).reset_index()
print(by_month_all.to_string(index=False))

print("\n=== BY DAY OF WEEK ===")
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
by_dow_all = t.groupby("dow").apply(agg).reindex(dow_order).reset_index()
print(by_dow_all.to_string(index=False))

# ---- worst buckets across dimensions (combined, all years) ----
print("\n=== WORST HOUR BUCKETS (sum_r, all years) ===")
print(by_hour_all.sort_values("sum_r").head(5).to_string(index=False))
print("\n=== WORST MONTH BUCKETS (sum_r, all years) ===")
print(by_month_all.sort_values("sum_r").head(5).to_string(index=False))

# ---- what if we filtered out the worst hours/months? ----
worst_hours = set(by_hour_all.sort_values("sum_r").head(6)["hour"])
worst_months = set(by_month_all.sort_values("sum_r").head(3)["month"])
filtered = t[~t["hour"].isin(worst_hours) & ~t["month"].isin(worst_months)]
print(f"\n=== FILTER TEST: drop worst 6 hours {sorted(worst_hours)} + worst 3 months {sorted(worst_months)} ===")
print(f"  before: trades={len(t)} sum_r={t['r'].sum():.1f} win_rate={(t['r']>0).mean():.3f}")
print(f"  after:  trades={len(filtered)} sum_r={filtered['r'].sum():.1f} win_rate={(filtered['r']>0).mean():.3f}")

# ---- charts ----
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

ax = axes[0, 0]
for regime, c in [("2009-2020", "#b71c1c"), ("2021-2026", "#2e7d32")]:
    sub = by_hour[by_hour["regime"] == regime].sort_values("hour")
    ax.plot(sub["hour"], sub["sum_r"], marker="o", label=regime, color=c)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("Entry hour (UTC)")
ax.set_ylabel("Sum R")
ax.set_title("Sum R by hour of day, split by regime")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
for regime, c in [("2009-2020", "#b71c1c"), ("2021-2026", "#2e7d32")]:
    sub = by_month[by_month["regime"] == regime].sort_values("month")
    ax.plot(sub["month"], sub["sum_r"], marker="o", label=regime, color=c)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("Month")
ax.set_ylabel("Sum R")
ax.set_title("Sum R by calendar month, split by regime")
ax.set_xticks(range(1, 13))
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
x = np.arange(len(dow_order))
width = 0.35
for i, (regime, c) in enumerate([("2009-2020", "#b71c1c"), ("2021-2026", "#2e7d32")]):
    sub = t[t["regime"] == regime].groupby("dow").apply(agg).reindex(dow_order)
    ax.bar(x + (i - 0.5) * width, sub["sum_r"], width, label=regime, color=c)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([d[:3] for d in dow_order])
ax.set_ylabel("Sum R")
ax.set_title("Sum R by day of week, split by regime")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
x = np.arange(2)
for i, (regime, c) in enumerate([("2009-2020", "#b71c1c"), ("2021-2026", "#2e7d32")]):
    sub = t[t["regime"] == regime].groupby("side").apply(agg).reindex(["buy", "sell"])
    ax.bar(x + (i - 0.5) * width, sub["sum_r"], width, label=regime, color=c)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["buy", "sell"])
ax.set_ylabel("Sum R")
ax.set_title("Sum R: buy vs sell, split by regime")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("DTFX EUR/USD diagnostics — where does R come from / get lost? (original variant, 2009-2026)",
             fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/dtfx_diagnostics.png", dpi=130)
print(f"\nSaved chart -> {OUT_DIR}/dtfx_diagnostics.png", flush=True)
