import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dtfx_eurusd_backtest import CACHE_DIR
from dtfx_indicator_diagnostics import build_indicators

LABEL = sys.argv[1]
CSV_PATH = sys.argv[2]
PREFIX = sys.argv[3]
DAY = pd.Timestamp(sys.argv[4])

OUT = CACHE_DIR


def load(path):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["open", "high", "low", "close"]]


print(f"[{LABEL}] loading price data", flush=True)
df = load(CSV_PATH)
ind = build_indicators(df)

t = pd.read_csv(f"{OUT}/{PREFIX}_full_trades.csv", parse_dates=["entry_time", "exit_time", "created_at"])

day_start = DAY
for _ in range(5):
    day_end = day_start + pd.Timedelta(days=1)
    day_df = df[(df.index >= day_start) & (df.index < day_end)]
    if len(day_df) > 10:
        break
    day_start -= pd.Timedelta(days=1)

pad_before = pd.Timedelta(hours=6)
pad_after = pd.Timedelta(hours=1)
win_start, win_end = day_start - pad_before, day_end + pad_after
ind_win = ind[(ind.index >= win_start) & (ind.index <= win_end)]
plot_df = df[(df.index >= win_start) & (df.index <= win_end)]
day_trades = t[(t["created_at"] >= win_start) & (t["created_at"] <= win_end)]
print(f"[{LABEL}] {day_start.date()}: {len(day_trades)} trades touching window", flush=True)

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

html = fig.to_html(full_html=True, include_plotlyjs=True,
                    config={"scrollZoom": True, "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
with open(f"{OUT}/{PREFIX}_day_chart.html", "w") as f:
    f.write(html)
print(f"[{LABEL}] saved self-contained day chart, {len(html)/1024:.0f} KB", flush=True)
