import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dtfx_eurusd_backtest import CACHE_DIR
from dtfx_eurusd_backtest_strict import find_pivots_strict, simulate_zone_original
from dtfx_eurusd_backtest import run_structure_fsm

DATA_PATH = f"{CACHE_DIR}/lse_eurusd_1min_full.csv"
OUT_DIR = CACHE_DIR

df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
df.index = df.index.tz_localize(None)
df = df[["open", "high", "low", "close"]]
print(f"Loaded {len(df)} bars, {df.index.min()} -> {df.index.max()}", flush=True)

zigzag = find_pivots_strict(df, 3)
_, zones = run_structure_fsm(zigzag)
print(f"{len(zones)} zones total", flush=True)

# swing-position-in-trend: index within each maximal run of consecutive same-side zones
sides = [z["side"] for z in zones]
run_id = 0
pos_in_trend = []
prev_side = None
counter = 0
for s in sides:
    if s != prev_side:
        counter = 0
        run_id += 1
    counter += 1
    pos_in_trend.append(counter)
    prev_side = s

times, lows, highs = df.index.values, df["low"].values, df["high"].values
records = []
for z, pos in zip(zones, pos_in_trend):
    tr = simulate_zone_original(z, times, lows, highs)
    if tr["status"] not in ("take_profit", "stopped_out"):
        continue
    is_buy_zone = tr["side"] == "buy"
    risk = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
    if risk <= 0:
        continue
    if tr["status"] == "take_profit":
        reward = (tr["tp"] - tr["mid"]) if is_buy_zone else (tr["mid"] - tr["tp"])
        r = reward / risk
    else:
        r = -1.0
    records.append(dict(
        entry_time=tr["entry_time"], side=tr["side"], r=r, status=tr["status"],
        risk_pct_of_price=abs(risk) / tr["mid"], pos_in_trend=pos,
    ))

t = pd.DataFrame(records).sort_values("entry_time").reset_index(drop=True)
t["year"] = t["entry_time"].dt.year
t["regime"] = np.where(t["year"] >= 2021, "2021-2026", "2009-2020")
t.to_csv(f"{OUT_DIR}/dtfx_diag2_trades.csv", index=False)
print(f"{len(t)} completed trades", flush=True)


def agg(g):
    wins = (g["r"] > 0).sum()
    n = len(g)
    return pd.Series({"trades": n, "win_rate": wins / n, "avg_r": g["r"].mean(), "sum_r": g["r"].sum()})


# ---- 1. volatility / zone-width regime ----
print("\n=== 1. ZONE WIDTH (risk_pct_of_price) QUINTILES ===")
t["vol_bucket"] = pd.qcut(t["risk_pct_of_price"], 5, labels=["Q1 tightest", "Q2", "Q3", "Q4", "Q5 widest"])
by_vol = t.groupby("vol_bucket", observed=True).apply(agg)
print(by_vol.to_string())
by_vol_regime = t.groupby(["regime", "vol_bucket"], observed=True).apply(agg)
print("\n--- by regime ---")
print(by_vol_regime.to_string())

# ---- 2. position in trend ----
print("\n=== 2. POSITION IN TREND (swing count since last flip) ===")
t["pos_bucket"] = np.where(t["pos_in_trend"] >= 6, "6+", t["pos_in_trend"].astype(str))
by_pos = t.groupby("pos_bucket", observed=True).apply(agg)
order = ["1", "2", "3", "4", "5", "6+"]
by_pos = by_pos.reindex(order)
print(by_pos.to_string())
by_pos_regime = t.groupby(["regime", "pos_bucket"], observed=True).apply(agg)
print("\n--- by regime ---")
print(by_pos_regime.to_string())

# ---- 5. losing-streak autocorrelation ----
print("\n=== 5. STREAK AUTOCORRELATION (does a loss predict the next trade's outcome?) ===")
t["is_win"] = t["r"] > 0
t["prev_outcome"] = t["is_win"].shift(1)
by_prev = t.dropna(subset=["prev_outcome"]).groupby("prev_outcome").apply(agg)
print(by_prev.to_string())
lag1_corr = t["r"].autocorr(lag=1)
print(f"\nlag-1 autocorrelation of r: {lag1_corr:.4f}")

# consecutive-loss-count -> next trade performance
t["loss_streak"] = 0
streak = 0
streaks = []
for w in t["is_win"]:
    streaks.append(streak)
    streak = 0 if w else streak + 1
t["loss_streak_before"] = streaks
t["streak_bucket"] = np.where(t["loss_streak_before"] >= 4, "4+", t["loss_streak_before"].astype(str))
by_streak = t.groupby("streak_bucket", observed=True).apply(agg).reindex(["0", "1", "2", "3", "4+"])
print("\n--- performance conditioned on # consecutive losses immediately before this trade ---")
print(by_streak.to_string())

# ---- charts ----
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

ax = axes[0, 0]
ax.bar(by_vol.index.astype(str), by_vol["sum_r"], color="#1f77b4")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_title("Sum R by zone-width quintile (Q1=tightest/lowest-vol, Q5=widest)")
ax.set_ylabel("Sum R")

ax2 = axes[0, 0].twinx()
ax2.plot(by_vol.index.astype(str), by_vol["win_rate"] * 100, color="black", marker="o")
ax2.set_ylabel("Win rate (%)")

ax = axes[0, 1]
ax.bar(by_pos.index.astype(str), by_pos["sum_r"], color="#2e7d32")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_title("Sum R by position-in-trend (1st, 2nd, ... swing since flip)")
ax.set_ylabel("Sum R")
ax2 = axes[0, 1].twinx()
ax2.plot(by_pos.index.astype(str), by_pos["win_rate"] * 100, color="black", marker="o")
ax2.set_ylabel("Win rate (%)")

ax = axes[1, 0]
ax.bar(["after loss", "after win"], [by_prev.loc[False, "avg_r"], by_prev.loc[True, "avg_r"]],
       color=["#b71c1c", "#2e7d32"])
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_title(f"Avg R conditioned on previous trade outcome (lag-1 autocorr={lag1_corr:.3f})")
ax.set_ylabel("Avg R next trade")

ax = axes[1, 1]
ax.bar(by_streak.index.astype(str), by_streak["avg_r"], color="#d62728")
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_title("Avg R by # consecutive losses immediately before")
ax.set_xlabel("Consecutive prior losses")
ax.set_ylabel("Avg R")

fig.suptitle("DTFX EUR/USD diagnostics round 2: zone width, trend age, streaks (original variant, 2009-2026)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/dtfx_diagnostics2.png", dpi=130)
print(f"\nSaved chart -> {OUT_DIR}/dtfx_diagnostics2.png", flush=True)
