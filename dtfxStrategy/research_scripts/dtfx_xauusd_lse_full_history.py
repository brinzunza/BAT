import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dtfx_eurusd_backtest import stats_for, CACHE_DIR
from dtfx_eurusd_backtest_strict import get_trades_strict  # UNCHANGED rule, no retuning

DATA_PATH = f"{CACHE_DIR}/lse_xauusd_1min_full.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date").sort_index()
df.index = df.index.tz_localize(None)
df = df[["open", "high", "low", "close"]]
print(f"Loaded {len(df)} 1-minute bars, {df.index.min()} to {df.index.max()} "
      f"({(df.index.max()-df.index.min()).days/365.25:.1f} years)", flush=True)

all_rows = []
curves = {}
for variant in ["original", "1to1"]:
    trades, n_zones = get_trades_strict(df, variant)
    risks = np.array([tr["risk_pct_of_price"] for tr in trades]) if trades else np.array([0.0])
    print(f"{variant}: {n_zones} zones, {len(trades)} completed trades, "
          f"median risk_pct_of_price={np.median(risks)*100:.4f}% "
          f"(implied leverage {0.01/max(np.median(risks),1e-9):.1f}x)", flush=True)
    curves[variant] = trades

    all_days = (df.index.max() - df.index.min()).days + 1
    all_stats, _, _ = stats_for(trades, all_days)
    all_rows.append(dict(period="2006-2026 (full history)", variant=variant, **all_stats))

    for yr in range(2006, 2027):
        yr_trades = [tr for tr in trades if tr["entry_time"].year == yr]
        yr_start = df.index[df.index.year == yr].min()
        yr_end = df.index[df.index.year == yr].max()
        if pd.isna(yr_start):
            continue
        n_days = (yr_end - yr_start).days + 1
        yr_stats, _, _ = stats_for(yr_trades, n_days)
        all_rows.append(dict(period=str(yr), variant=variant, **yr_stats))

summary = pd.DataFrame(all_rows)
summary = summary[["period", "variant", "trades", "win_rate", "avg_win_r", "avg_loss_r",
                    "sum_r", "trades_per_day", "final_gross", "final_net"]]
summary.to_csv(f"{CACHE_DIR}/dtfx_xauusd_lse_full_history_summary.csv", index=False)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
pd.set_option("display.width", 200)
print("\n" + summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(15, 6.5))
colors = {"original": "#1f77b4", "1to1": "#d62728"}
FIXED_RISK, START = 2.0, 100.0
for variant, c in colors.items():
    trades = curves[variant]
    equity = START
    curve = [(trades[0]["entry_time"], equity)] if trades else []
    for tr in trades:
        equity += tr["r"] * FIXED_RISK
        curve.append((tr["exit_time"], equity))
    xs, ys = zip(*curve)
    min_eq = min(ys)
    ax.plot(xs, ys, color=c, linewidth=1.0, label=f"{variant}  (final: ${ys[-1]:,.2f}, min: ${min_eq:,.2f})")
    print(variant, "full-history final equity ($2/trade, $100 start):", f"${ys[-1]:,.2f}", "min:", f"${min_eq:,.2f}")

ax.axhline(START, color="gray", linewidth=0.8, linestyle=":")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("DTFX (strict pivots, UNCHANGED rule) on XAU-USD 1-minute entries, 2006-2026 -- full available history\n"
              "$100 start, fixed $2/trade risk, no compounding, gross (fees ignored)")
ax.set_ylabel("Account equity ($)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{CACHE_DIR}/dtfx_xauusd_lse_full_history_fixed2.png", dpi=130)
print(f"Saved chart -> {CACHE_DIR}/dtfx_xauusd_lse_full_history_fixed2.png", flush=True)
