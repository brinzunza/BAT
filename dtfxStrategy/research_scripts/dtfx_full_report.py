import sys
import json
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dtfx_eurusd_backtest import CACHE_DIR
from dtfx_eurusd_backtest_strict import get_trades_strict
from dtfx_indicator_diagnostics import build_indicators

LABEL = sys.argv[1]
CSV_PATH = sys.argv[2]
PREFIX = sys.argv[3]
DAY = pd.Timestamp(sys.argv[4]) if len(sys.argv) > 4 else pd.Timestamp("2026-08-18")

OUT = CACHE_DIR


def load(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["open", "high", "low", "close"]]


print(f"[{LABEL}] loading {CSV_PATH}", flush=True)
df = load(CSV_PATH)
print(f"[{LABEL}] {len(df)} bars, {df.index.min()} -> {df.index.max()}", flush=True)

trades, n_zones = get_trades_strict(df, "original")
print(f"[{LABEL}] {n_zones} zones, {len(trades)} completed trades", flush=True)

ind = build_indicators(df)

rows = []
for tr in trades:
    et = tr["entry_time"]
    if et not in ind.index:
        continue
    row = dict(entry_time=et, exit_time=tr["exit_time"], side=tr["side"], status=tr["status"], r=tr["r"],
               top=tr["top"], bottom=tr["bottom"], mid=tr["mid"], tp=tr["tp"], created_at=tr["created_at"],
               exit_price=tr["exit_price"],
               rsi14=ind.loc[et, "rsi14"], atr_ratio=ind.loc[et, "atr_ratio"])
    rows.append(row)
t = pd.DataFrame(rows).dropna(subset=["rsi14", "atr_ratio"]).sort_values("entry_time").reset_index(drop=True)
t["passes_filter"] = ~((t["rsi14"] >= 45) & (t["rsi14"] <= 55)) & (t["atr_ratio"] >= 0.7)
filt = t[t["passes_filter"]].reset_index(drop=True)
print(f"[{LABEL}] {len(t)} trades w/ indicators, {len(filt)} pass combined filter", flush=True)

t.to_csv(f"{OUT}/{PREFIX}_full_trades.csv", index=False)


def rich_stats(g, span_days):
    wins = g[g["r"] > 0]
    losses = g[g["r"] <= 0]
    n = len(g)
    win_rate = len(wins) / n if n else np.nan
    avg_win = wins["r"].mean() if len(wins) else np.nan
    avg_loss = losses["r"].mean() if len(losses) else np.nan
    trades_per_day = n / span_days if span_days else np.nan

    equity = 100.0
    times, equities = [g["entry_time"].iloc[0]], [equity]
    for _, row in g.iterrows():
        equity *= (1 + row["r"] * 0.01)
        equity = max(equity, 0.0)
        times.append(row["entry_time"])
        equities.append(equity)
    ec = pd.Series(equities, index=pd.to_datetime(times))
    ec = ec[~ec.index.duplicated(keep="last")]
    daily = ec.resample("1D").last().ffill().dropna()
    daily_ret = daily.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else np.nan
    running_max = daily.cummax()
    drawdown = (daily - running_max) / running_max
    max_dd = drawdown.min()

    return dict(
        trades=n, wins=len(wins), losses=len(losses), win_rate=win_rate,
        avg_win_r=avg_win, avg_loss_r=avg_loss, trades_per_day=trades_per_day,
        final_balance=equities[-1] if equities else 100.0, total_return_pct=(equities[-1] / 100 - 1) * 100 if equities else 0.0,
        sharpe=sharpe, max_drawdown_pct=max_dd * 100 if pd.notna(max_dd) else np.nan,
    ), times, equities


span_days = (df.index.max() - df.index.min()).days
stats, eq_times, eq_vals = rich_stats(filt, span_days)
stats["label"] = LABEL
stats["span_years"] = round(span_days / 365.25, 1)
print(f"[{LABEL}] STATS: {json.dumps(stats, default=str)}", flush=True)
with open(f"{OUT}/{PREFIX}_stats.json", "w") as f:
    json.dump(stats, f, default=str, indent=2)

# ---- equity curve plot (1% risk, compounding, filtered strategy) ----
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(eq_times, eq_vals, color="#1f77b4", linewidth=1.0)
ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
ax.set_yscale("log")
ax.set_title(f"{LABEL}\n$100 start, 1% risk/trade, compounding, filtered strategy, gross\n"
             f"Final: ${eq_vals[-1]:,.0f}  |  Sharpe: {stats['sharpe']:.2f}  |  Max DD: {stats['max_drawdown_pct']:.1f}%")
ax.set_ylabel("Equity ($, log scale)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/{PREFIX}_equity.png", dpi=130)
plt.close(fig)
print(f"[{LABEL}] saved equity chart", flush=True)

# ---- interactive day chart for DAY (or nearest prior trading day with data) ----
day_start = DAY
for _ in range(5):
    day_end = day_start + pd.Timedelta(days=1)
    day_df = df[(df.index >= day_start) & (df.index < day_end)]
    if len(day_df) > 10:
        break
    day_start -= pd.Timedelta(days=1)
else:
    day_df = pd.DataFrame()

if len(day_df) > 0:
    pad_before = pd.Timedelta(hours=6)
    pad_after = pd.Timedelta(hours=1)
    win_start, win_end = day_start - pad_before, day_end + pad_after
    ind_win = ind[(ind.index >= win_start) & (ind.index <= win_end)]
    plot_df = df[(df.index >= win_start) & (df.index <= win_end)]

    day_trades = t[(t["created_at"] >= win_start) & (t["created_at"] <= win_end)]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
                         subplot_titles=(f"{LABEL} -- {day_start.date()}", "RSI(14)", "ATR ratio (vs 2000-bar avg)"))

    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["open"], high=plot_df["high"], low=plot_df["low"], close=plot_df["close"],
        increasing_line_color="#4f9d6e", decreasing_line_color="#c1533f",
        increasing_fillcolor="#4f9d6e", decreasing_fillcolor="#c1533f",
        name=LABEL, showlegend=False,
    ), row=1, col=1)

    shapes = []
    for _, tr in day_trades.iterrows():
        is_buy = tr["side"] == "buy"
        color = "#c9963c" if is_buy else "#2f7d6b"
        exit_x = min(tr["exit_time"], win_end) if pd.notna(tr["exit_time"]) else win_end
        shapes.append(dict(type="rect", xref="x", yref="y", x0=tr["created_at"], x1=exit_x,
                            y0=tr["bottom"], y1=tr["top"], fillcolor=color, opacity=0.10,
                            line=dict(color=color, width=1, dash="dot")))
        stop_level = tr["bottom"] if is_buy else tr["top"]
        shapes.append(dict(type="line", xref="x", yref="y", x0=tr["entry_time"], x1=exit_x,
                            y0=stop_level, y1=stop_level, line=dict(color="#c1533f", width=1.2, dash="dash")))
        shapes.append(dict(type="line", xref="x", yref="y", x0=tr["entry_time"], x1=exit_x,
                            y0=tr["tp"], y1=tr["tp"], line=dict(color="#4f9d6e", width=1.2, dash="dash")))
        passed = bool(tr["passes_filter"])
        fig.add_trace(go.Scatter(
            x=[tr["entry_time"]], y=[tr["mid"]], mode="markers",
            marker=dict(symbol="triangle-up" if is_buy else "triangle-down", size=15 if passed else 11,
                        color=color, opacity=1.0 if passed else 0.35,
                        line=dict(width=2 if passed else 1, color="#10151c")),
            hovertext=[f"{'BUY' if is_buy else 'SELL'} @ 50%<br>{tr['entry_time']}<br>"
                       f"RSI={tr['rsi14']:.1f}  ATR ratio={tr['atr_ratio']:.2f}<br>"
                       f"{'PASSES FILTER' if passed else 'FILTERED OUT'}"],
            hoverinfo="text", showlegend=False,
        ), row=1, col=1)
        if pd.notna(tr["exit_time"]) and tr["exit_time"] <= win_end:
            won = tr["status"] == "take_profit"
            fig.add_trace(go.Scatter(
                x=[tr["exit_time"]], y=[tr["exit_price"]], mode="markers",
                marker=dict(symbol="star" if won else "x", size=12,
                            color="#4f9d6e" if won else "#c1533f", line=dict(width=1, color="#10151c")),
                hovertext=[f"{'TAKE PROFIT' if won else 'STOPPED OUT'}<br>{tr['exit_time']}<br>{tr['r']:+.2f}R"],
                hoverinfo="text", showlegend=False,
            ), row=1, col=1)

    fig.add_trace(go.Scatter(x=ind_win.index, y=ind_win["rsi14"], mode="lines",
                              line=dict(color="#5b6b7a", width=1.3), name="RSI(14)", showlegend=False),
                  row=2, col=1)
    fig.add_hrect(y0=45, y1=55, fillcolor="#c1533f", opacity=0.12, line_width=0, row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#8894a3", width=0.8, dash="dot"), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#8894a3", width=0.8, dash="dot"), row=2, col=1)

    fig.add_trace(go.Scatter(x=ind_win.index, y=ind_win["atr_ratio"], mode="lines",
                              line=dict(color="#5b6b7a", width=1.3), name="ATR ratio", showlegend=False),
                  row=3, col=1)
    fig.add_hrect(y0=0, y1=0.7, fillcolor="#c1533f", opacity=0.12, line_width=0, row=3, col=1)
    fig.add_hline(y=1.0, line=dict(color="#8894a3", width=0.8, dash="dot"), row=3, col=1)

    fig.update_layout(
        shapes=shapes,
        xaxis3=dict(rangeslider=dict(visible=True, thickness=0.06), type="date", range=[win_start, win_end]),
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="ui-monospace, 'SF Mono', 'Roboto Mono', monospace", size=11, color="#10151c"),
        margin=dict(l=55, r=25, t=40, b=10), height=780, hovermode="closest", dragmode="pan", showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(136,148,163,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(136,148,163,0.15)")
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

    html = fig.to_html(full_html=True, include_plotlyjs="cdn",
                        config={"scrollZoom": True, "displaylogo": False,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    with open(f"{OUT}/{PREFIX}_day_chart.html", "w") as f:
        f.write(html)
    print(f"[{LABEL}] saved day chart for {day_start.date()}, {len(day_trades)} trades touching that window", flush=True)
else:
    print(f"[{LABEL}] NO DATA available near {DAY.date()}, skipping day chart", flush=True)

print(f"[{LABEL}] DONE", flush=True)
