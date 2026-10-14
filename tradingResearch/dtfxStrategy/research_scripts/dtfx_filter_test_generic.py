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


def build_curve(g, fixed_risk=2.0, start=100.0):
    g = g.sort_values("entry_time")
    equity = start
    equities = [equity]
    for r in g["r"]:
        equity += r * fixed_risk
        equities.append(equity)
    times = [g["entry_time"].iloc[0]] + list(g["entry_time"])
    return times, equities


def test_instrument(label, csv_path, out_prefix, tz_localize=True):
    print(f"\n{'='*20} {label} {'='*20}", flush=True)
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
    if tz_localize and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["open", "high", "low", "close"]]
    print(f"{len(df)} bars, {df.index.min()} -> {df.index.max()} "
          f"({(df.index.max()-df.index.min()).days/365.25:.1f}y)", flush=True)

    trades, n_zones = get_trades_strict(df, "original")
    print(f"{n_zones} zones, {len(trades)} completed trades", flush=True)
    if len(trades) < 50:
        print("Too few trades, skipping.", flush=True)
        return None

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

    def stats(g):
        n = len(g)
        return dict(trades=n, win_rate=(g["r"] > 0).mean() if n else np.nan,
                    avg_r=g["r"].mean() if n else np.nan, sum_r=g["r"].sum())

    b_stats, f_stats = stats(baseline), stats(combined_filter)
    pct_change = (f_stats["sum_r"] / b_stats["sum_r"] - 1) * 100 if b_stats["sum_r"] != 0 else float("nan")
    print(f"BASELINE: {b_stats}")
    print(f"FILTERED: {f_stats}")
    print(f"sum_r change: {pct_change:+.1f}%")

    bx, by = build_curve(baseline)
    fx, fy = build_curve(combined_filter)
    print(f"BASELINE final equity ($100 start, $2/trade): ${by[-1]:,.2f}")
    print(f"FILTERED final equity: ${fy[-1]:,.2f}")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(bx, by, color="#1f77b4", linewidth=1.0, label=f"Baseline (final: ${by[-1]:,.0f})")
    ax.plot(fx, fy, color="#d62728", linewidth=1.2, label=f"Combined filter (final: ${fy[-1]:,.0f})")
    ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(f"{label} -- combined filter test\n$100 start, fixed $2/trade, no compounding, gross")
    ax.set_ylabel("Account equity ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = f"{CACHE_DIR}/{out_prefix}_filter_test.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved -> {out_png}", flush=True)

    return dict(label=label, span_years=round((df.index.max()-df.index.min()).days/365.25, 1),
                baseline_trades=b_stats["trades"], baseline_sum_r=b_stats["sum_r"],
                baseline_win_rate=b_stats["win_rate"], baseline_final=by[-1],
                filtered_trades=f_stats["trades"], filtered_sum_r=f_stats["sum_r"],
                filtered_win_rate=f_stats["win_rate"], filtered_final=fy[-1],
                pct_change=pct_change, chart=out_png)
