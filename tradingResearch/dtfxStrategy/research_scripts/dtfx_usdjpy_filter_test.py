import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dtfx_eurusd_backtest import CACHE_DIR
from dtfx_eurusd_backtest_strict import get_trades_strict
from dtfx_indicator_diagnostics import build_indicators

DATA_PATH = f"{CACHE_DIR}/lse_usdjpy_1min_full.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
df.index = df.index.tz_localize(None)
df = df[["open", "high", "low", "close"]]
print(f"Loaded {len(df)} bars, {df.index.min()} -> {df.index.max()} "
      f"({(df.index.max()-df.index.min()).days/365.25:.1f}y)", flush=True)

trades, n_zones = get_trades_strict(df, "original")
print(f"{n_zones} zones, {len(trades)} completed trades", flush=True)

t = pd.DataFrame([{"entry_time": tr["entry_time"], "side": tr["side"], "r": tr["r"]} for tr in trades])
ind = build_indicators(df)
t = t.set_index("entry_time").join(ind, how="left")
t.index.name = "entry_time"
t = t.reset_index()
t = t.dropna(subset=["rsi14", "atr_ratio"])
print(f"{len(t)} trades with indicator data", flush=True)

baseline = t.sort_values("entry_time")
combined_filter = baseline[~((baseline["rsi14"] >= 45) & (baseline["rsi14"] <= 55)) &
                            (baseline["atr_ratio"] >= 0.7)]


def stats(g, label):
    n = len(g)
    return dict(filter=label, trades=n, win_rate=(g["r"] > 0).mean(), avg_r=g["r"].mean(), sum_r=g["r"].sum())


rows = [stats(baseline, "BASELINE (all trades)"), stats(combined_filter, "COMBINED FILTER (exclude RSI 45-55 + ATR<0.7x)")]
summary = pd.DataFrame(rows)
summary["sum_r_vs_baseline_pct"] = (summary["sum_r"] / summary["sum_r"].iloc[0] - 1) * 100
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
print("\n" + summary.to_string(index=False))


def build_curve(g, fixed_risk=2.0, start=100.0):
    g = g.sort_values("entry_time")
    equity = start
    times = [g["entry_time"].iloc[0]]
    equities = [equity]
    for _, row in g.iterrows():
        equity += row["r"] * fixed_risk
        times.append(row["entry_time"])
        equities.append(equity)
    return times, equities

bx, by = build_curve(baseline)
fx, fy = build_curve(combined_filter)

print(f"\nBASELINE final equity: ${by[-1]:,.2f}")
print(f"FILTERED final equity: ${fy[-1]:,.2f}")

fig, ax = plt.subplots(figsize=(14, 6.5))
ax.plot(bx, by, color="#1f77b4", linewidth=1.0, label=f"Baseline, all trades (final: ${by[-1]:,.0f})")
ax.plot(fx, fy, color="#d62728", linewidth=1.2, label=f"Combined filter: no RSI 45-55, no ATR<0.7x (final: ${fy[-1]:,.0f})")
ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
ax.set_title("USD/JPY 1-minute, 2009-2026 -- THIRD PAIR TEST of the EUR/USD-derived combined filter\n"
              "$100 start, fixed $2/trade risk, no compounding, gross (fees ignored)")
ax.set_ylabel("Account equity ($)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{CACHE_DIR}/dtfx_usdjpy_filter_test.png", dpi=130)
print(f"\nSaved chart -> {CACHE_DIR}/dtfx_usdjpy_filter_test.png", flush=True)

t.to_csv(f"{CACHE_DIR}/dtfx_ind_usdjpy.csv", index=False)
