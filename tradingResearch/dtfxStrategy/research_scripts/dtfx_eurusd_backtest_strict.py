import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dtfx_eurusd_backtest import (
    resample_ohlc, run_structure_fsm, simulate_zone_original, simulate_zone_1to1,
    equity_curve, stats_for, CACHE_DIR, STARTING_EQUITY,
)

MIN_RETRACEMENT = 3


def find_pivots_strict(df, min_retracement=MIN_RETRACEMENT):
    """Stricter swing rule: a candidate high/low only ever extends on a bar whose CLOSE
    (never a wick) goes beyond the candidate's own high/low, and only gets validated as an
    actual swing point after >= min_retracement CONSECUTIVE opposite-color candles."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    t = df.index.values
    pivots = []
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
                    looking_for, candidate_price, candidate_i, streak = "high", h[i], i, 0
            else:
                streak = 0
    return pivots


def get_trades_strict(df, variant):
    zigzag = find_pivots_strict(df, MIN_RETRACEMENT)
    _, zones = run_structure_fsm(zigzag)
    times, lows, highs = df.index.values, df["low"].values, df["high"].values
    sim_fn = simulate_zone_original if variant == "original" else simulate_zone_1to1
    trades = [sim_fn(z, times, lows, highs) for z in zones]
    completed = [tr for tr in trades if tr["status"] in ("take_profit", "stopped_out")]
    completed.sort(key=lambda tr: tr["entry_time"])
    out = []
    for tr in completed:
        is_buy_zone = tr["side"] == "buy"
        risk = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
        if risk <= 0:
            continue
        if tr["status"] == "take_profit":
            if variant == "original":
                reward = (tr["tp"] - tr["mid"]) if is_buy_zone else (tr["mid"] - tr["tp"])
            else:
                reward = risk
            r = reward / risk
        else:
            r = -1.0
        tr["r"] = r
        tr["risk_pct_of_price"] = abs(risk) / tr["mid"]
        out.append(tr)
    return out, len(zones)


if __name__ == "__main__":
    parts = []
    for yr in ["2024", "2025", "2026"]:
        d = pd.read_csv(f"{CACHE_DIR}/polygon_EURUSD_{yr}.csv", parse_dates=["date"]).set_index("date").sort_index()
        parts.append(d[["open", "high", "low", "close"]])
    df1m = pd.concat(parts).sort_index()
    df1m = df1m[~df1m.index.duplicated(keep="first")]
    df1h = resample_ohlc(df1m, "1h")
    print(f"Loaded EURUSD: {len(df1m)} 1min bars / {len(df1h)} 1h bars, {df1m.index.min()} to {df1m.index.max()}", flush=True)

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
            if tf_label == "1h":
                curves[variant] = dict(gross=gross_curve, net=net_curve)

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
    summary.to_csv(f"{CACHE_DIR}/dtfx_eurusd_strict_summary.csv", index=False)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    pd.set_option("display.width", 200)
    print("\n" + summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(13, 6))
    colors = {"original": "#1f77b4", "1to1": "#d62728"}
    for variant, c in colors.items():
        gc = curves[variant]["gross"]
        nc = curves[variant]["net"]
        if gc:
            xs, ys = zip(*gc)
            ax.plot(xs, ys, color=c, linewidth=1.6, label=f"{variant} (gross)")
        if nc:
            xs, ys = zip(*nc)
            ax.plot(xs, ys, color=c, linewidth=1.2, linestyle="--", alpha=0.75, label=f"{variant} (net, ~2 pip cost)")
    ax.axhline(STARTING_EQUITY, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title("DTFX (STRICT pivots) on EUR-USD 1h entries, 2024-2026: Original vs 1:1\n"
                  "$10,000 start, 1% risk/trade, compounded")
    ax.set_ylabel("Equity ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{CACHE_DIR}/dtfx_eurusd_strict_pnl.png", dpi=130)
    print(f"Saved chart -> {CACHE_DIR}/dtfx_eurusd_strict_pnl.png", flush=True)
