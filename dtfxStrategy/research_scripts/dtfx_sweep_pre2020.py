import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dtfx_eurusd_backtest import stats_for, CACHE_DIR
from dtfx_eurusd_backtest_strict import get_trades_strict  # UNCHANGED rule, no retuning

SWEEP_DIR = f"{CACHE_DIR}/lse_sweep"

INSTRUMENTS = [
    ("EUR/USD (fx)",      f"{SWEEP_DIR}/EURUSD_pre2020.csv"),
    ("XAU/USD (gold)",    f"{SWEEP_DIR}/XAUUSD_pre2020.csv"),
    ("SPX500 (index)",    f"{SWEEP_DIR}/SPX500_pre2020.csv"),
    ("ES.F (SPX futures)",f"{SWEEP_DIR}/ESF_pre2020.csv"),
    ("AAPL (stock)",      f"{SWEEP_DIR}/AAPL_pre2020.csv"),
    ("USD/JPY (fx)",      f"{SWEEP_DIR}/USDJPY_2009_2016.csv"),
    ("AUD/USD (fx)",      f"{SWEEP_DIR}/AUDUSD_2009_2016.csv"),
]

rows = []
for label, path in INSTRUMENTS:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[["open", "high", "low", "close"]]
    span_years = (df.index.max() - df.index.min()).days / 365.25
    n_days = (df.index.max() - df.index.min()).days + 1
    print(f"=== {label}: {len(df)} bars, {df.index.min()} -> {df.index.max()} ({span_years:.1f}y) ===", flush=True)

    for variant in ["original", "1to1"]:
        trades, n_zones = get_trades_strict(df, variant)
        stats, _, _ = stats_for(trades, n_days)
        rows.append(dict(instrument=label, span_years=round(span_years, 1), variant=variant,
                          zones=n_zones, **stats))
        print(f"  {variant}: trades={stats['trades']} win_rate={stats['win_rate']:.3f} "
              f"avg_win_r={stats['avg_win_r']:.3f} sum_r={stats['sum_r']:.2f} "
              f"trades_per_day={stats['trades_per_day']:.3f}", flush=True)

summary = pd.DataFrame(rows)
summary = summary[["instrument", "span_years", "variant", "trades", "win_rate", "avg_win_r",
                    "avg_loss_r", "sum_r", "trades_per_day"]]
summary.to_csv(f"{SWEEP_DIR}/dtfx_sweep_pre2020_summary.csv", index=False)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
pd.set_option("display.width", 220)
print("\n" + summary.to_string(index=False))

# comparison bar chart: win rate and R-per-trade by instrument, for the "original" variant
orig = summary[summary["variant"] == "original"].copy()
orig["r_per_trade"] = orig["sum_r"] / orig["trades"]
breakeven = 1 / (1 + orig["avg_win_r"].mean())

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
colors = ["#2e7d32" if x > breakeven else "#b71c1c" for x in orig["win_rate"]]
axes[0].barh(orig["instrument"], orig["win_rate"] * 100, color=colors)
axes[0].axvline(breakeven * 100, color="black", linestyle="--", linewidth=1, label=f"breakeven ~{breakeven*100:.1f}%")
axes[0].set_xlabel("Win rate (%)")
axes[0].set_title("Original variant — win rate by instrument (pre-2020)")
axes[0].legend(fontsize=9)
axes[0].invert_yaxis()

colors2 = ["#2e7d32" if x > 0 else "#b71c1c" for x in orig["r_per_trade"]]
axes[1].barh(orig["instrument"], orig["r_per_trade"], color=colors2)
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_xlabel("R per trade (sum_r / trades)")
axes[1].set_title("Original variant — average edge per trade (pre-2020)")
axes[1].invert_yaxis()

fig.suptitle("DTFX strict-pivot rule (UNCHANGED), pre-2020 data across instruments", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{SWEEP_DIR}/dtfx_sweep_pre2020_chart.png", dpi=130)
print(f"\nSaved chart -> {SWEEP_DIR}/dtfx_sweep_pre2020_chart.png", flush=True)
