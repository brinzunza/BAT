import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import json
import random
import pandas as pd
import plotly.graph_objects as go
from dtfx_eurusd_backtest import run_structure_fsm, simulate_zone_original, r_multiple
from dtfx_eurusd_backtest_strict import find_pivots_strict

CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"
PAD_BEFORE = pd.Timedelta(hours=3)
PAD_AFTER = pd.Timedelta(hours=1)

parts = []
for yr in ["2024", "2025", "2026"]:
    d = pd.read_csv(f"{CACHE_DIR}/polygon_EURUSD_{yr}.csv", parse_dates=["date"]).set_index("date").sort_index()
    parts.append(d[["open", "high", "low", "close"]])
df = pd.concat(parts).sort_index()
df = df[~df.index.duplicated(keep="first")]
print(f"Loaded {len(df)} 1min bars, {df.index.min()} to {df.index.max()}", flush=True)

pivots = find_pivots_strict(df, min_retracement=3)
print(f"Strict-validated swings: {len(pivots)}", flush=True)
events, zones = run_structure_fsm(pivots)
print(f"Zones formed (structure continuation only): {len(zones)}", flush=True)

times, lows, highs = df.index.values, df["low"].values, df["high"].values
all_trades = []
for z in zones:
    tr = simulate_zone_original(z, times, lows, highs)
    if tr["status"] in ("take_profit", "stopped_out"):
        tr["r"] = r_multiple(tr, "original")
        all_trades.append(tr)
print(f"Completed trades: {len(all_trades)}", flush=True)

same_day = [t for t in all_trades if t["entry_time"].date() == t["exit_time"].date()]
print(f"Trades with entry+exit on same calendar day: {len(same_day)}", flush=True)
random.seed(11)
day_counts = {}
for t in same_day:
    day_counts.setdefault(t["entry_time"].date(), 0)
    day_counts[t["entry_time"].date()] += 1
# pick a random day among those with at least 2 same-day completed trades, for a richer demo
candidates = [d for d, n in day_counts.items() if n >= 2]
DAY = pd.Timestamp(random.choice(candidates) if candidates else random.choice(list(day_counts.keys())))
print(f"Picked day: {DAY.date()}  ({day_counts.get(DAY.date())} same-day completed trades)", flush=True)

day_start, day_end = DAY, DAY + pd.Timedelta(days=1)
win_start, win_end = day_start - PAD_BEFORE, day_end + PAD_AFTER

day_df = df[(df.index >= day_start) & (df.index < day_end)].copy()
print(f"Day bars: {len(day_df)}  range {day_df.index.min()} .. {day_df.index.max()}")

confirmed_events = [e for e in events if e[3] in ("bootstrap", "confirmed", "flip_up", "flip_down")
                     and win_start <= e[0] <= win_end]
print(f"Confirmed structure points in window: {len(confirmed_events)}")

day_zones = [z for z in zones if z["created_at"] >= win_start and z["created_at"] <= day_end]
print(f"Zones created in window: {len(day_zones)}")

day_trades = [t for t in all_trades if (t["entry_time"] is not None and day_start <= t["entry_time"] < day_end)
              or (t["created_at"] >= win_start and t["created_at"] <= day_end)]
day_trades = [t for t in day_trades if t["created_at"] <= day_end]
print(f"Trades touching this day: {len(day_trades)}")
for t in day_trades:
    print(" ", t["side"], t["status"], "created", t["created_at"], "entry", t["entry_time"], "exit", t["exit_time"], "r=", round(t["r"], 2) if t.get("r") else None)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=day_df.index, open=day_df["open"], high=day_df["high"], low=day_df["low"], close=day_df["close"],
    increasing_line_color="#4f9d6e", decreasing_line_color="#c1533f",
    increasing_fillcolor="#4f9d6e", decreasing_fillcolor="#c1533f",
    name="EUR/USD 1m", showlegend=False,
))

if confirmed_events:
    ex = [pd.Timestamp(e[0]) for e in confirmed_events]
    ey = [e[1] for e in confirmed_events]
    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines+markers", name="Confirmed structure",
        line=dict(color="#8894a3", width=1.4, dash="dot"),
        marker=dict(size=7, color=["#c9963c" if e[2] == "L" else "#2f7d6b" for e in confirmed_events],
                    symbol=["triangle-up" if e[2] == "L" else "triangle-down" for e in confirmed_events],
                    line=dict(width=1, color="#10151c")),
        text=[f"{'Higher Low' if e[2]=='L' else 'Higher High'} confirmed<br>{e[0]}<br>{e[1]:.5f}" for e in confirmed_events],
        hoverinfo="text",
    ))

def clip(ts):
    return min(ts, win_end) if ts is not None else win_end

shapes = []
for z in day_zones:
    is_buy = z["side"] == "buy"
    color = "#c9963c" if is_buy else "#2f7d6b"
    x1 = win_end
    for t in day_trades:
        if t is not None and t["created_at"] == z["created_at"]:
            x1 = clip(t["exit_time"])
    shapes.append(dict(type="rect", xref="x", yref="y", x0=z["created_at"], x1=x1, y0=z["bottom"], y1=z["top"],
                        fillcolor=color, opacity=0.10, line=dict(color=color, width=1, dash="dot")))

for t in day_trades:
    is_buy = t["side"] == "buy"
    color = "#c9963c" if is_buy else "#2f7d6b"
    exit_x = clip(t["exit_time"])
    stop_level = t["bottom"] if is_buy else t["top"]
    shapes.append(dict(type="line", xref="x", yref="y", x0=t["entry_time"], x1=exit_x,
                        y0=stop_level, y1=stop_level, line=dict(color="#c1533f", width=1.3, dash="dash")))
    shapes.append(dict(type="line", xref="x", yref="y", x0=t["entry_time"], x1=exit_x,
                        y0=t["tp"], y1=t["tp"], line=dict(color="#4f9d6e", width=1.3, dash="dash")))
    fig.add_trace(go.Scatter(
        x=[t["entry_time"]], y=[t["mid"]], mode="markers",
        marker=dict(symbol="triangle-up" if is_buy else "triangle-down", size=13, color=color,
                    line=dict(width=1.5, color="#10151c")),
        name=f"{'Buy' if is_buy else 'Sell'} entry",
        hovertext=[f"{'BUY' if is_buy else 'SELL'} @ 50%<br>{t['entry_time']}<br>{t['mid']:.5f}"],
        hoverinfo="text", showlegend=False,
    ))
    if t["exit_time"] and t["exit_time"] <= win_end:
        won = t["status"] == "take_profit"
        fig.add_trace(go.Scatter(
            x=[t["exit_time"]], y=[t["exit_price"]], mode="markers",
            marker=dict(symbol="star" if won else "x", size=12,
                        color="#4f9d6e" if won else "#c1533f", line=dict(width=1, color="#10151c")),
            name="Exit", hovertext=[f"{'TAKE PROFIT' if won else 'STOPPED OUT'}<br>{t['exit_time']}<br>{t['exit_price']:.5f}<br>{t['r']:+.2f}R"],
            hoverinfo="text", showlegend=False,
        ))

fig.update_layout(
    shapes=shapes,
    xaxis=dict(rangeslider=dict(visible=True, thickness=0.08), type="date", range=[win_start, win_end],
               showgrid=True, gridcolor="rgba(136,148,163,0.15)"),
    yaxis=dict(title="EUR/USD", showgrid=True, gridcolor="rgba(136,148,163,0.15)", tickformat=".4f"),
    template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="ui-monospace, 'SF Mono', 'Cascadia Code', 'Roboto Mono', monospace", size=12, color="#10151c"),
    margin=dict(l=60, r=30, t=20, b=10), height=620, hovermode="closest", dragmode="pan", showlegend=False,
)

html_div = fig.to_html(full_html=False, include_plotlyjs=True, config={
    "scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
})
with open(f"{CACHE_DIR}/day_chart_strict_div.html", "w") as f:
    f.write(html_div)

meta = dict(
    day=str(DAY.date()), n_bars=len(day_df), n_zones=len(day_zones), n_trades=len(day_trades),
    trades=[dict(side=t["side"], status=t["status"], created_at=str(t["created_at"]),
                 entry_time=str(t["entry_time"]) if t["entry_time"] else None,
                 exit_time=str(t["exit_time"]) if t["exit_time"] else None,
                 entry_price=t["mid"], stop_price=(t["bottom"] if t["side"] == "buy" else t["top"]),
                 tp_price=t["tp"], exit_price=t["exit_price"], r=t.get("r")) for t in day_trades],
)
with open(f"{CACHE_DIR}/day_chart_strict_meta.json", "w") as f:
    json.dump(meta, f, indent=2, default=str)
print("Saved div and meta", flush=True)
