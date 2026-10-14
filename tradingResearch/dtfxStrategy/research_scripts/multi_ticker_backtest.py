import time
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

OKX_BASE = "https://www.okx.com/api/v5/market"
TICKERS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XAUT-USDT"]
BARS_PER_TICKER = 3000  # split into two ~25h periods per ticker


def fetch_okx_1m(inst_id, target_bars):
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


def find_pivots(high, low, window=3):
    is_high = high == high.rolling(window * 2 + 1, center=True).max()
    is_low = low == low.rolling(window * 2 + 1, center=True).min()
    return high[is_high].dropna(), low[is_low].dropna()


def run_zone_strategy(df, window=3):
    """Full pipeline: pivots -> zigzag -> structure FSM (+ zone creation) -> simulate each zone.
    Returns (zones, trades) — identical logic to the real-data / synthetic-data scripts."""
    swing_highs, swing_lows = find_pivots(df["high"], df["low"], window=window)
    raw = [(t, v, "H") for t, v in swing_highs.items()] + [(t, v, "L") for t, v in swing_lows.items()]
    raw.sort(key=lambda x: x[0])
    zigzag = []
    for t, v, typ in raw:
        if zigzag and zigzag[-1][2] == typ:
            prev_t, prev_v, prev_typ = zigzag[-1]
            more_extreme = (typ == "H" and v > prev_v) or (typ == "L" and v < prev_v)
            if more_extreme:
                zigzag[-1] = (t, v, typ)
            continue
        zigzag.append((t, v, typ))

    events = []
    zones = []
    trend = None
    confirmed_high = None
    confirmed_low = None
    pending_low_idx = None
    pending_high_idx = None

    i = 0
    seen_low, seen_high = None, None
    while i < len(zigzag) and (seen_low is None or seen_high is None):
        t, v, typ = zigzag[i]
        if typ == "L" and seen_low is None:
            seen_low = (t, v)
        if typ == "H" and seen_high is None:
            seen_high = (t, v)
        i += 1
    if seen_low and seen_high:
        confirmed_low, confirmed_high = seen_low[1], seen_high[1]
        trend = "up"
        events.append([seen_low[0], seen_low[1], "L", "bootstrap"])
        events.append([seen_high[0], seen_high[1], "H", "bootstrap"])
    else:
        return [], []

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
            else:
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
        else:
            if trend == "down":
                if v > confirmed_high:
                    confirmed_low, confirmed_high = confirmed_low, v
                    trend = "up"
                    pending_low_idx = None
                    events.append([t, v, "H", "flip_up"])
                else:
                    pending_high_idx = len(events)
                    events.append([t, v, "H", "pending"])
            else:
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

    def simulate_zone(zone, bars):
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

    trades = [simulate_zone(z, df) for z in zones]
    return zones, trades


def r_multiple(tr):
    """Risk-normalized outcome: +reward/risk if TP, exactly -1 if stopped out, else None."""
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
    return None  # still open at end of sample


# ---- run across all tickers x 2 periods each, staying on 1-minute bars throughout ----
results = []
detail_rows = []
for ticker in TICKERS:
    print(f"Fetching {ticker}...")
    df = fetch_okx_1m(ticker, BARS_PER_TICKER)
    half = len(df) // 2
    periods = {"period_A_earlier": df.iloc[:half], "period_B_recent": df.iloc[half:]}

    for period_name, sub_df in periods.items():
        zones, trades = run_zone_strategy(sub_df)
        completed = [tr for tr in trades if tr["status"] in ("take_profit", "stopped_out")]
        r_values = [r_multiple(tr) for tr in completed]
        r_values = [r for r in r_values if r is not None]
        wins = sum(1 for tr in completed if tr["status"] == "take_profit")

        for tr in trades:
            detail_rows.append(dict(ticker=ticker, period=period_name, side=tr["side"],
                                     created_at=tr["created_at"], status=tr["status"], r=r_multiple(tr)))

        results.append(dict(
            ticker=ticker, period=period_name,
            bars=len(sub_df), zones_created=len(zones),
            trades_taken=len(completed), wins=wins,
            losses=len(completed) - wins,
            win_rate=wins / len(completed) if completed else np.nan,
            sum_R=sum(r_values) if r_values else 0.0,
            avg_R=np.mean(r_values) if r_values else np.nan,
        ))

summary = pd.DataFrame(results)
detail = pd.DataFrame(detail_rows)
print("\n" + "=" * 100)
print(summary.to_string(index=False))

summary.to_csv("/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/multi_ticker_summary.csv", index=False)
detail.to_csv("/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/multi_ticker_detail.csv", index=False)

# ---- plot: sum_R per ticker/period, and win rate ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

pivot_sumR = summary.pivot(index="ticker", columns="period", values="sum_R")
pivot_sumR.plot(kind="bar", ax=axes[0], color=["#90a4ae", "#1565c0"])
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("Total R-Multiple Sum by Ticker & Period")
axes[0].set_ylabel("Sum of R multiples (risk-normalized)")
axes[0].legend(fontsize=8)

pivot_wr = summary.pivot(index="ticker", columns="period", values="win_rate") * 100
pivot_wr.plot(kind="bar", ax=axes[1], color=["#90a4ae", "#1565c0"])
axes[1].axhline(50, color="black", linewidth=0.8, linestyle="--", label="50%")
axes[1].set_title("Win Rate by Ticker & Period")
axes[1].set_ylabel("Win rate (%)")
axes[1].legend(fontsize=8)

plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/multi_ticker_backtest.png"
plt.savefig(out_path, dpi=140)
print("\nWrote", out_path)
