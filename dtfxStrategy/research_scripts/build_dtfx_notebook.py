import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }

cells = []

cells.append(md("""\
# DTFX Strategy: Market-Structure Trend Confirmation + Retracement Zone Entries

**DTFX** combines two ideas developed and tested iteratively in this repo's research sessions:

1. **Trend determination via confirmed market structure** — not a moving average or a drawn
   trendline, but the classic price-action definition: an uptrend is higher highs *and* higher
   lows; a downtrend is lower highs *and* lower lows. Critically, a swing low only counts as a
   genuine "higher low" once a *subsequent* higher high validates it (and symmetrically for lower
   highs in a downtrend) — an unconfirmed swing is just a candidate, not yet part of the trend.
2. **Retracement-zone entries** — every time structure confirms a new extreme (a new higher high in
   an uptrend, a new lower low in a downtrend), the leg that produced it defines a zone. Price is
   expected to retrace partway into that zone before the trend resumes; DTFX buys/sells at the 50%
   level of the zone, stops at the zone's far edge, and targets the high/low that created the zone.

This notebook is the canonical record of the strategy: precise rules, the reference implementation,
a synthetic idealized demonstration, a real-data walkthrough, and an honest multi-ticker backtest.

> ⚠️ Research only, not investment advice. No transaction costs or slippage modeled anywhere in this
> notebook — see the backtest section for why that matters.
"""))

cells.append(md("""\
## 1. Exact Rules

### 1a. Market structure (trend determination)

State tracked at all times: `trend` (up/down), `confirmed_high`, `confirmed_low`.

**While in an uptrend:**
- Any new swing high that exceeds `confirmed_high` is accepted **immediately** as the new
  `confirmed_high` — no separate confirmation step.
- A swing low does **not** update `confirmed_low` on its own. It sits as a *pending* candidate.
  It only becomes the new `confirmed_low` — a validated "higher low" — at the moment a
  **subsequent** high beats `confirmed_high`. The pending low that came right before that high is
  what gets promoted (not any older, already-confirmed low).
- **Trend flip**: the instant any swing low prints *below* `confirmed_low`, the trend flips to
  down. The low that broke structure is auto-validated as the first confirmed low of the new
  downtrend; the old `confirmed_high` carries over unchanged, now reinterpreted as the downtrend's
  ceiling.

**While in a downtrend:** the exact mirror image (lows confirm immediately, highs need a
subsequent lower low to confirm, a high breaking above `confirmed_high` flips back to up).

### 1b. Retracement zones

Every time a new confirmed **higher high** forms in an uptrend:
- **Zone top** = the *previous* confirmed high (not the one just made — the one before it).
- **Zone bottom** = the higher low actually being validated by this breakout (the pending low
  that's about to be promoted — not a stale, already-confirmed low from an earlier cycle).
- **Buy** signal fires when price retraces back down and touches the zone's 50% midpoint.
- **Stop-loss** = zone bottom (this exactly coincides with the level that would flip the trend to
  down — if the stop is hit, the uptrend thesis is invalidated anyway).
- **Take-profit** = the new higher high that triggered the zone's creation.

Every time a new confirmed **lower low** forms in a downtrend: the exact mirror (zone top = the
recent bounce high about to be validated, zone bottom = the previous confirmed low, sell at 50%
from below, stop at zone top, target the low that triggered the zone).

If price reaches the zone's *far* edge before ever touching the 50% level, the zone is invalidated
— no trade (this coincides with a structure flip, so the setup's premise is gone anyway).
"""))

cells.append(code("""\
# If needed, install dependencies (uncomment to run):
# %pip install pandas numpy matplotlib requests

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
plt.rcParams["figure.figsize"] = (14, 7)

OKX_BASE = "https://www.okx.com/api/v5/market"
"""))

cells.append(md("""\
## 2. Reference Implementation

Three pieces: (1) fetch real 1-minute OHLC from OKX's free public API, (2) the structure/zone
engine — pivot detection, alternating zigzag, the confirmation FSM described above, zone creation
on both sides — (3) bar-by-bar simulation of each zone into an entry/stop/target outcome.
"""))

cells.append(code("""\
def fetch_okx_1m(inst_id, target_bars):
    \"\"\"Paginate OKX's public candles + history-candles endpoints to build a target-length
    1-minute OHLC history. No API key required.\"\"\"
    all_rows = []
    r = requests.get(f"{OKX_BASE}/candles", params={"instId": inst_id, "bar": "1m", "limit": 300}, timeout=20)
    r.raise_for_status()
    data = r.json()["data"]
    all_rows.extend(data)
    after = min(int(row[0]) for row in data)
    while len(all_rows) < target_bars:
        r = requests.get(
            f"{OKX_BASE}/history-candles",
            params={"instId": inst_id, "bar": "1m", "limit": 100, "after": after},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()["data"]
        if not data:
            break
        all_rows.extend(data)
        after = min(int(row[0]) for row in data)
        time.sleep(0.2)
    cols = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
    df = pd.DataFrame(all_rows, columns=cols)
    df = df[df["confirm"] == "1"]
    df["date"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms")
    df = df.drop_duplicates(subset="date").set_index("date").sort_index()
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "vol"]]
"""))

cells.append(code("""\
def find_pivots(high, low, window=3):
    \"\"\"A bar is a swing high/low if it's the max/min over a centered window*2+1 bar span
    (needs `window` bars on both sides to confirm — this is inherently backward-looking).\"\"\"
    is_high = high == high.rolling(window * 2 + 1, center=True).max()
    is_low = low == low.rolling(window * 2 + 1, center=True).min()
    return high[is_high].dropna(), low[is_low].dropna()


def build_zigzag(swing_highs, swing_lows):
    \"\"\"Collapse raw pivots into a strictly alternating H,L,H,L,... sequence, keeping the more
    extreme point whenever two same-type pivots occur back to back.\"\"\"
    raw = [(t, v, "H") for t, v in swing_highs.items()] + [(t, v, "L") for t, v in swing_lows.items()]
    raw.sort(key=lambda x: x[0])
    zigzag = []
    for t, v, typ in raw:
        if zigzag and zigzag[-1][2] == typ:
            _, prev_v, _ = zigzag[-1]
            more_extreme = (typ == "H" and v > prev_v) or (typ == "L" and v < prev_v)
            if more_extreme:
                zigzag[-1] = (t, v, typ)
            continue
        zigzag.append((t, v, typ))
    return zigzag
"""))

cells.append(code("""\
def run_structure_fsm(zigzag):
    \"\"\"The confirmation state machine from Section 1a/1b. Returns (events, zones).
    events: [time, value, 'H'/'L', role] with role in
        {'bootstrap','confirmed','pending','flip_up','flip_down'}.
    zones: list of dicts with top/bottom/mid/tp/created_at/side ('buy' or 'sell').\"\"\"
    events, zones = [], []
    trend = confirmed_high = confirmed_low = None
    pending_low_idx = pending_high_idx = None

    i = 0
    seen_low, seen_high = None, None
    while i < len(zigzag) and (seen_low is None or seen_high is None):
        t, v, typ = zigzag[i]
        if typ == "L" and seen_low is None:
            seen_low = (t, v)
        if typ == "H" and seen_high is None:
            seen_high = (t, v)
        i += 1
    if not (seen_low and seen_high):
        return events, zones
    confirmed_low, confirmed_high = seen_low[1], seen_high[1]
    trend = "up"
    events.append([seen_low[0], seen_low[1], "L", "bootstrap"])
    events.append([seen_high[0], seen_high[1], "H", "bootstrap"])

    for t, v, typ in zigzag[i:]:
        if typ == "L":
            if trend == "up":
                if v < confirmed_low:
                    confirmed_high, confirmed_low = confirmed_high, v
                    trend = "down"
                    pending_high_idx = None
                    events.append([t, v, "L", "flip_down"])
                else:
                    pending_low_idx = len(events)
                    events.append([t, v, "L", "pending"])
            else:  # trend == 'down'
                if confirmed_low is None or v < confirmed_low:
                    old_low = confirmed_low
                    zone_top = events[pending_high_idx][1] if pending_high_idx is not None else confirmed_high
                    confirmed_low = v
                    if pending_high_idx is not None:
                        confirmed_high = events[pending_high_idx][1]
                        events[pending_high_idx][3] = "confirmed"
                        pending_high_idx = None
                    events.append([t, v, "L", "confirmed"])
                    if old_low is not None and zone_top is not None and zone_top > old_low:
                        zones.append(dict(top=zone_top, bottom=old_low, mid=(zone_top + old_low) / 2,
                                           tp=v, created_at=t, side="sell"))
                else:
                    events.append([t, v, "L", "pending"])
        else:  # typ == 'H'
            if trend == "down":
                if v > confirmed_high:
                    confirmed_low, confirmed_high = confirmed_low, v
                    trend = "up"
                    pending_low_idx = None
                    events.append([t, v, "H", "flip_up"])
                else:
                    pending_high_idx = len(events)
                    events.append([t, v, "H", "pending"])
            else:  # trend == 'up'
                if confirmed_high is None or v > confirmed_high:
                    old_high = confirmed_high
                    zone_bottom = events[pending_low_idx][1] if pending_low_idx is not None else confirmed_low
                    confirmed_high = v
                    if pending_low_idx is not None:
                        confirmed_low = events[pending_low_idx][1]
                        events[pending_low_idx][3] = "confirmed"
                        pending_low_idx = None
                    events.append([t, v, "H", "confirmed"])
                    if old_high is not None and zone_bottom is not None and old_high > zone_bottom:
                        zones.append(dict(top=old_high, bottom=zone_bottom, mid=(old_high + zone_bottom) / 2,
                                           tp=v, created_at=t, side="buy"))
                else:
                    events.append([t, v, "H", "pending"])
    return events, zones
"""))

cells.append(code("""\
def simulate_zone(zone, bars):
    \"\"\"Walk bars forward from zone creation: invalidate if the far edge breaks before the 50%
    level is touched; otherwise enter at 50%, then resolve to stop-loss or take-profit.\"\"\"
    is_buy = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy else zone["top"]
    stop_level = invalidation_level

    future = bars.loc[bars.index > zone["created_at"]]
    entry_time = None
    for t, row in future.iterrows():
        breached = row["low"] <= invalidation_level if is_buy else row["high"] >= invalidation_level
        if breached:
            return dict(**zone, status="invalidated", entry_time=None, exit_time=t, exit_price=invalidation_level)
        reached_mid = row["low"] <= zone["mid"] if is_buy else row["high"] >= zone["mid"]
        if reached_mid:
            entry_time = t
            break
    if entry_time is None:
        return dict(**zone, status="pending", entry_time=None, exit_time=None, exit_price=None)

    after_entry = future.loc[future.index > entry_time]
    for t, row in after_entry.iterrows():
        hit_sl = row["low"] <= stop_level if is_buy else row["high"] >= stop_level
        hit_tp = row["high"] >= zone["tp"] if is_buy else row["low"] <= zone["tp"]
        if hit_sl:
            return dict(**zone, status="stopped_out", entry_time=entry_time, exit_time=t, exit_price=stop_level)
        if hit_tp:
            return dict(**zone, status="take_profit", entry_time=entry_time, exit_time=t, exit_price=zone["tp"])
    return dict(**zone, status="open", entry_time=entry_time, exit_time=None, exit_price=None)


def run_dtfx(df, pivot_window=3):
    \"\"\"Full pipeline: OHLC dataframe -> (events, zones, trades).\"\"\"
    swing_highs, swing_lows = find_pivots(df["high"], df["low"], window=pivot_window)
    zigzag = build_zigzag(swing_highs, swing_lows)
    events, zones = run_structure_fsm(zigzag)
    trades = [simulate_zone(z, df) for z in zones]
    return events, zones, trades


def r_multiple(tr):
    \"\"\"Risk-normalized outcome: +reward/risk if take-profit, exactly -1 if stopped out,
    None if the zone was never entered or is still open.\"\"\"
    if tr["entry_time"] is None:
        return None
    is_buy = tr["side"] == "buy"
    risk = (tr["mid"] - tr["bottom"]) if is_buy else (tr["top"] - tr["mid"])
    if risk <= 0:
        return None
    if tr["status"] == "take_profit":
        reward = (tr["tp"] - tr["mid"]) if is_buy else (tr["mid"] - tr["tp"])
        return reward / risk
    if tr["status"] == "stopped_out":
        return -1.0
    return None
"""))

cells.append(md("""\
## 3. Synthetic Idealized Demonstration

Before looking at noisy real data, a hand-constructed price path where every pullback retraces to
*exactly* 50% of the zone it created, then continues to a new extreme — the "if the rules play out
exactly as designed" reference case. Clean staircase up, a deliberate structure break, clean
staircase down.
"""))

cells.append(code("""\
def build_synthetic_ideal_path(n_up_legs=4, n_down_legs=4, seed=42):
    rng = np.random.default_rng(seed)
    keypoints = [100.0]
    price = 100.0

    H0 = price + 15
    keypoints.append(H0)
    L0 = H0 - 7
    keypoints.append(L0)
    price = L0
    H_prev, L_prev = H0, L0

    for i in range(n_up_legs):
        leg_size = 12 + i * 3
        H_new = price + leg_size
        keypoints.append(H_new)
        zone_top, zone_bottom = H_prev, L_prev
        mid = (zone_top + zone_bottom) / 2
        if i < n_up_legs - 1:
            L_new = zone_bottom + (mid - zone_bottom) * 0.35
            keypoints.append(L_new)
            price = L_new
            H_prev, L_prev = H_new, L_new
        else:
            breakdown = zone_bottom - 9
            keypoints.append(breakdown)
            price = breakdown
            flip_high, flip_low = H_new, breakdown

    H_prev_dt, L_prev_dt = flip_high, flip_low
    bounce0 = price + 7
    keypoints.append(bounce0)
    price = bounce0
    H_prev_dt = bounce0

    for i in range(n_down_legs):
        leg_size = 12 + i * 3
        L_new = price - leg_size
        keypoints.append(L_new)
        zone_top, zone_bottom = H_prev_dt, L_prev_dt
        mid = (zone_top + zone_bottom) / 2
        if i < n_down_legs - 1:
            H_new = zone_top - (zone_top - mid) * 0.35
            keypoints.append(H_new)
            price = H_new
            L_prev_dt, H_prev_dt = L_new, H_new
        else:
            keypoints.append(L_new + 5)

    bars_per_leg = 25
    closes = []
    for a, b in zip(keypoints[:-1], keypoints[1:]):
        seg = np.linspace(a, b, bars_per_leg)
        noise = rng.normal(0, abs(b - a) * 0.01, bars_per_leg)
        noise[0] = noise[-1] = 0
        closes.append(seg + noise)
    closes = np.concatenate(closes)

    idx = pd.date_range("2026-01-01", periods=len(closes), freq="min")
    close = pd.Series(closes, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + rng.uniform(0, 0.15, len(close))
    low = pd.concat([open_, close], axis=1).min(axis=1) - rng.uniform(0, 0.15, len(close))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


synthetic_df = build_synthetic_ideal_path()
_, synthetic_zones, synthetic_trades = run_dtfx(synthetic_df)
print(f"Synthetic zones: {len(synthetic_zones)}")
for tr in synthetic_trades:
    print(f"  [{tr['side']:4s}] created {tr['created_at']}  status={tr['status']}")
"""))

cells.append(code("""\
def plot_dtfx(df, trades, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    bar_width_days = (df.index[1] - df.index[0]).total_seconds() / 86400 * 0.7
    for t, row in df.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        t_num = mdates.date2num(t)
        ax.plot([t_num, t_num], [row["low"], row["high"]], color=color, linewidth=0.6, zorder=2, alpha=0.8)
        body_low = min(row["open"], row["close"])
        body_height = abs(row["close"] - row["open"]) or 0.01
        ax.add_patch(Rectangle((t_num - bar_width_days / 2, body_low), bar_width_days, body_height,
                                facecolor=color, edgecolor=color, zorder=3, alpha=0.8))

    status_color = {"take_profit": "#1b5e20", "stopped_out": "#b71c1c", "invalidated": "#9e9e9e",
                     "pending": "#f9a825", "open": "#1565c0"}
    for tr in trades:
        t0 = mdates.date2num(tr["created_at"])
        t_end = mdates.date2num(tr["exit_time"]) if tr["exit_time"] is not None else mdates.date2num(df.index[-1])
        color = status_color[tr["status"]]
        ax.add_patch(Rectangle((t0, tr["bottom"]), t_end - t0, tr["top"] - tr["bottom"],
                                facecolor=color, alpha=0.15, edgecolor=color, linewidth=1, zorder=1))
        ax.plot([t0, t_end], [tr["mid"], tr["mid"]], color=color, linestyle="--", linewidth=1, alpha=0.8, zorder=4)
        if tr["entry_time"] is not None:
            t_entry = mdates.date2num(tr["entry_time"])
            is_buy = tr["side"] == "buy"
            stop_level = tr["bottom"] if is_buy else tr["top"]
            ax.scatter([t_entry], [tr["mid"]], marker="^" if is_buy else "v", s=100,
                        color="#1565c0" if is_buy else "#6a1b9a", edgecolor="black", zorder=6)
            ax.plot([t_entry, t_end], [stop_level, stop_level], color="#b71c1c", linewidth=1.3, alpha=0.8, zorder=4)
            ax.plot([t_entry, t_end], [tr["tp"], tr["tp"]], color="#1b5e20", linewidth=1.3, alpha=0.8, zorder=4)
            if tr["status"] == "take_profit":
                ax.scatter([mdates.date2num(tr["exit_time"])], [tr["exit_price"]], marker="*", s=240,
                            color="#1b5e20", edgecolor="black", zorder=7)
            elif tr["status"] == "stopped_out":
                ax.scatter([mdates.date2num(tr["exit_time"])], [tr["exit_price"]], marker="X", s=130,
                            color="#b71c1c", edgecolor="black", zorder=7)

    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1565c0", markeredgecolor="black",
               markersize=10, label="buy signal"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#6a1b9a", markeredgecolor="black",
               markersize=10, label="sell signal"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1b5e20", markeredgecolor="black",
               markersize=15, label="take profit hit"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#b71c1c", markeredgecolor="black",
               markersize=11, label="stopped out"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")
    ax.set_title(title)
    ax.set_ylabel("Price")
    plt.tight_layout()
    plt.show()


plot_dtfx(synthetic_df, synthetic_trades, "DTFX on Synthetic Idealized Uptrend -> Downtrend")
"""))

cells.append(md("""\
## 4. Real-Data Walkthrough (ETH-USDT, 1-Minute)

Same engine, real market data, no cherry-picking the window.
"""))

cells.append(code("""\
eth_df = fetch_okx_1m("ETH-USDT", target_bars=1500)
print(f"Fetched {len(eth_df)} bars: {eth_df.index.min()} to {eth_df.index.max()}")

_, eth_zones, eth_trades = run_dtfx(eth_df)
print(f"Zones: {len(eth_zones)}")
plot_dtfx(eth_df, eth_trades, "DTFX on Real ETH-USDT 1-Minute Data")
"""))

cells.append(code("""\
eth_completed = [tr for tr in eth_trades if tr["status"] in ("take_profit", "stopped_out")]
eth_r = [r for r in (r_multiple(tr) for tr in eth_completed) if r is not None]
wins = sum(1 for tr in eth_completed if tr["status"] == "take_profit")
print(f"Trades taken: {len(eth_completed)}  Wins: {wins}  Losses: {len(eth_completed) - wins}")
print(f"Win rate: {wins / len(eth_completed) * 100:.1f}%" if eth_completed else "No trades taken")
print(f"Sum of R multiples: {sum(eth_r):.2f}   Average R per trade: {np.mean(eth_r):.3f}" if eth_r else "")
"""))

cells.append(md("""\
## 5. Multi-Ticker, Multi-Period Backtest

The real test of whether DTFX has a stable edge: 4 different instruments, 2 non-overlapping
~25-hour periods each (not resamples of the same window — genuinely different market stretches),
all on 1-minute bars, all fee-free. Performance is compared using **R-multiples** (each trade's
outcome normalized by its own risk) since raw price P&L isn't comparable across BTC (tens of
thousands) vs SOL (tens) vs ETH vs XAUT.
"""))

cells.append(code("""\
TICKERS = [\"BTC-USDT\", \"ETH-USDT\", \"SOL-USDT\", \"XAUT-USDT\"]
BARS_PER_TICKER = 3000  # split into two ~25h periods per ticker

results = []
for ticker in TICKERS:
    print(f\"Fetching {ticker}...\")
    df = fetch_okx_1m(ticker, BARS_PER_TICKER)
    half = len(df) // 2
    periods = {\"period_A_earlier\": df.iloc[:half], \"period_B_recent\": df.iloc[half:]}

    for period_name, sub_df in periods.items():
        _, zones, trades = run_dtfx(sub_df)
        completed = [tr for tr in trades if tr[\"status\"] in (\"take_profit\", \"stopped_out\")]
        r_values = [r for r in (r_multiple(tr) for tr in completed) if r is not None]
        wins = sum(1 for tr in completed if tr[\"status\"] == \"take_profit\")

        results.append(dict(
            ticker=ticker, period=period_name, bars=len(sub_df), zones_created=len(zones),
            trades_taken=len(completed), wins=wins, losses=len(completed) - wins,
            win_rate=wins / len(completed) if completed else np.nan,
            sum_R=sum(r_values) if r_values else 0.0,
            avg_R=np.mean(r_values) if r_values else np.nan,
        ))

summary = pd.DataFrame(results)
summary
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

pivot_sumR = summary.pivot(index=\"ticker\", columns=\"period\", values=\"sum_R\")
pivot_sumR.plot(kind=\"bar\", ax=axes[0], color=[\"#90a4ae\", \"#1565c0\"])
axes[0].axhline(0, color=\"black\", linewidth=0.8)
axes[0].set_title(\"Total R-Multiple Sum by Ticker & Period\")
axes[0].set_ylabel(\"Sum of R multiples (risk-normalized)\")

pivot_wr = summary.pivot(index=\"ticker\", columns=\"period\", values=\"win_rate\") * 100
pivot_wr.plot(kind=\"bar\", ax=axes[1], color=[\"#90a4ae\", \"#1565c0\"])
axes[1].axhline(50, color=\"black\", linewidth=0.8, linestyle=\"--\", label=\"50%\")
axes[1].set_title(\"Win Rate by Ticker & Period\")
axes[1].set_ylabel(\"Win rate (%)\")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
## 6. Takeaways

**Rules recap** (the permanent record — refer back here, not to any single backtest run):
- Trend = confirmed HH+HL (up) or LH+LL (down); a swing only confirms once the *opposite* extreme
  breaks structure in the trend's favor (Section 1a).
- Zone = [previous confirmed extreme on the far side, the extreme just being validated] on every
  new confirmed high (uptrend) or low (downtrend) (Section 1b).
- Entry at zone 50%, stop at the zone's far edge (= the structure-break level), target the
  high/low that created the zone.

**Honest performance read, from Section 5**: across 8 backtests (4 tickers x 2 periods), win rate
never exceeded ~33% and was often much lower. More importantly, **3 of 4 tickers flipped sign
entirely between the two periods** (e.g. SOL: +8.0R then -5.7R; XAUT: -10.5R then +10.1R) — the
same instrument, back-to-back time windows, opposite results. Only ETH-USDT was consistent, and it
was consistently negative in both windows. That pattern — sign flipping across adjacent periods
for most instruments — looks much more like noise than a real, extractable edge specific to any of
these tickers.

**What this notebook does not model**: transaction costs, slippage, funding/borrow costs on the
short (sell) side, or execution latency between the 50%-touch signal and actual fill. At a
20-30-trade sample per run with 1-minute holding periods, real costs would likely turn several of
the recorded "wins" negative.

**Before trusting this strategy with real capital**, at minimum: (1) test on much longer combined
windows per ticker (25 hours is a small sample at this win rate), (2) re-run with realistic fee/
slippage assumptions, (3) test out-of-sample on data collected *after* this notebook was written,
not just different historical slices of data already seen during development.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/dtfx_strategy.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Wrote", out_path)
