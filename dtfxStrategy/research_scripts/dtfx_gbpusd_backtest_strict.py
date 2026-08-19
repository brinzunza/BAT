import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dtfx_eurusd_backtest import resample_ohlc, stats_for, equity_curve, CACHE_DIR, STARTING_EQUITY
from dtfx_eurusd_backtest_strict import get_trades_strict  # UNCHANGED rule, no retuning

if __name__ == "__main__":
    parts = []
    for yr in ["2024", "2025", "2026"]:
        d = pd.read_csv(f"{CACHE_DIR}/polygon_GBPUSD_{yr}.csv", parse_dates=["date"]).set_index("date").sort_index()
        parts.append(d[["open", "high", "low", "close"]])
    df1m = pd.concat(parts).sort_index()
    df1m = df1m[~df1m.index.duplicated(keep="first")]
    df1h = resample_ohlc(df1m, "1h")
    print(f"Loaded GBPUSD: {len(df1m)} 1min bars / {len(df1h)} 1h bars, {df1m.index.min()} to {df1m.index.max()}", flush=True)

    all_rows = []
    curves = {}
    for tf_label, df in [("1min", df1m), ("1h", df1h)]:
        for variant in ["original", "1to1"]:
            trades, n_zones = get_trades_strict(df, variant)
            risks = np.array([tr["risk_pct_of_price"] for tr in trades]) if trades else np.array([0.0])
            print(f"[{tf_label}] {variant}: {n_zones} zones, {len(trades)} completed trades, "
                  f"median risk_pct_of_price={np.median(risks)*100:.4f}% "
                  f"(implied leverage {0.01/max(np.median(risks),1e-9):.1f}x)", flush=True)

            all_days = (df.index.max() - df.index.min()).days + 1
            all_stats, gross_curve, net_curve = stats_for(trades, all_days)
            all_rows.append(dict(timeframe=tf_label, period="2024-2026 (all)", variant=variant, **all_stats))
            if tf_label == "1min":
                curves[variant] = trades

            for yr in [2024, 2025, 2026]:
                yr_trades = [tr for tr in trades if tr["entry_time"].year == yr]
                yr_start = df.index[df.index.year == yr].min()
                yr_end = df.index[df.index.year == yr].max()
                if pd.isna(yr_start):
                    continue
                n_days = (yr_end - yr_start).days + 1
                yr_stats, _, _ = stats_for(yr_trades, n_days)
                all_rows.append(dict(timeframe=tf_label, period=str(yr), variant=variant, **yr_stats))

    summary = pd.DataFrame(all_rows)
    summary = summary[["timeframe", "period", "variant", "trades", "win_rate", "avg_win_r", "avg_loss_r",
                        "sum_r", "trades_per_day", "final_gross", "final_net"]]
    summary.to_csv(f"{CACHE_DIR}/dtfx_gbpusd_strict_summary.csv", index=False)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    pd.set_option("display.width", 200)
    print("\n" + summary.to_string(index=False))

    # fixed $2/trade, $100 start, gross, 1-minute -- matching the exact framing already shown for EURUSD
    fig, ax = plt.subplots(figsize=(13, 6.5))
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
        ax.plot(xs, ys, color=c, linewidth=1.2, label=f"{variant}  (final: ${ys[-1]:,.2f}, min: ${min_eq:,.2f})")
        print(variant, "1min final equity ($2/trade, $100 start):", f"${ys[-1]:,.2f}", "min:", f"${min_eq:,.2f}")

    ax.axhline(START, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("DTFX (strict pivots, UNCHANGED rule) on GBP-USD 1-minute entries, 2024-2026 -- OUT OF SAMPLE\n"
                  "$100 start, fixed $2/trade risk, no compounding, gross (fees ignored)")
    ax.set_ylabel("Account equity ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{CACHE_DIR}/dtfx_gbpusd_strict_fixed2.png", dpi=130)
    print(f"Saved chart -> {CACHE_DIR}/dtfx_gbpusd_strict_fixed2.png", flush=True)
