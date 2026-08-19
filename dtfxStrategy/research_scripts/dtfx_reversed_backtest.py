import time
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

OKX_BASE = "https://www.okx.com/api/v5/market"
TICKERS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XAUT-USDT"]
BARS_PER_TICKER = 3000


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


def build_zigzag(swing_highs, swing_lows):
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


def run_structure_fsm(zigzag):
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
    return events, zones


def simulate_zone(zone, bars, reversed_signal=False):
    """reversed_signal=False: original DTFX (long a buy-zone at 50%, short a sell-zone at 50%).
    reversed_signal=True: same entry trigger/invalidation, but take the OPPOSITE position,
    with stop/target swapped between the zone's far edge and the zone's trigger price
    (e.g. a reversed buy-zone trade goes SHORT at 50%, stop at the recent high that created the
    zone, target the zone's bottom -- betting the pullback fails and structure breaks, instead
    of betting the pullback holds and the trend resumes)."""
    is_buy_zone = zone["side"] == "buy"
    invalidation_level = zone["bottom"] if is_buy_zone else zone["top"]

    # entry direction actually taken
    going_long = is_buy_zone if not reversed_signal else not is_buy_zone

    if not reversed_signal:
        stop_level = zone["bottom"] if is_buy_zone else zone["top"]
        target_level = zone["tp"]
    else:
        stop_level = zone["tp"]
        target_level = zone["bottom"] if is_buy_zone else zone["top"]

    future = bars.loc[bars.index > zone["created_at"]]
    entry_time = None
    for t, row in future.iterrows():
        breached = row["low"] <= invalidation_level if is_buy_zone else row["high"] >= invalidation_level
        if breached:
            return dict(**zone, status="invalidated", entry_time=None, exit_time=t, exit_price=invalidation_level)
        reached_mid = row["low"] <= zone["mid"] if is_buy_zone else row["high"] >= zone["mid"]
        if reached_mid:
            entry_time = t
            break
    if entry_time is None:
        return dict(**zone, status="pending", entry_time=None, exit_time=None, exit_price=None)

    after_entry = future.loc[future.index > entry_time]
    for t, row in after_entry.iterrows():
        hit_sl = row["low"] <= stop_level if going_long else row["high"] >= stop_level
        hit_tp = row["high"] >= target_level if going_long else row["low"] <= target_level
        if hit_sl:
            return dict(**zone, status="stopped_out", entry_time=entry_time, exit_time=t, exit_price=stop_level)
        if hit_tp:
            return dict(**zone, status="take_profit", entry_time=entry_time, exit_time=t, exit_price=target_level)
    return dict(**zone, status="open", entry_time=entry_time, exit_time=None, exit_price=None)


def run_dtfx(df, reversed_signal=False, pivot_window=3):
    swing_highs, swing_lows = find_pivots(df["high"], df["low"], window=pivot_window)
    zigzag = build_zigzag(swing_highs, swing_lows)
    events, zones = run_structure_fsm(zigzag)
    trades = [simulate_zone(z, df, reversed_signal=reversed_signal) for z in zones]
    return zones, trades


def r_multiple(tr, reversed_signal=False):
    if tr["entry_time"] is None:
        return None
    is_buy_zone = tr["side"] == "buy"
    if not reversed_signal:
        risk = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
        reward_level = tr["tp"]
        going_long = is_buy_zone
    else:
        risk = (tr["tp"] - tr["mid"]) if is_buy_zone else (tr["mid"] - tr["tp"])
        reward_level = tr["bottom"] if is_buy_zone else tr["top"]
        going_long = not is_buy_zone
    if risk <= 0:
        return None
    if tr["status"] == "take_profit":
        reward = (reward_level - tr["mid"]) if going_long else (tr["mid"] - reward_level)
        return reward / risk
    if tr["status"] == "stopped_out":
        return -1.0
    return None


# ---- run BOTH original and reversed on the SAME fetched data for a clean paired comparison ----
results = []
for ticker in TICKERS:
    print(f"Fetching {ticker}...")
    df = fetch_okx_1m(ticker, BARS_PER_TICKER)
    half = len(df) // 2
    periods = {"period_A_earlier": df.iloc[:half], "period_B_recent": df.iloc[half:]}

    for period_name, sub_df in periods.items():
        zones, trades_orig = run_dtfx(sub_df, reversed_signal=False)
        trades_rev = [simulate_zone(z, sub_df, reversed_signal=True) for z in zones]

        for label, trades, rev_flag in [("original", trades_orig, False), ("reversed", trades_rev, True)]:
            completed = [tr for tr in trades if tr["status"] in ("take_profit", "stopped_out")]
            r_values = [r for r in (r_multiple(tr, reversed_signal=rev_flag) for tr in completed) if r is not None]
            wins = sum(1 for tr in completed if tr["status"] == "take_profit")
            results.append(dict(
                ticker=ticker, period=period_name, variant=label,
                zones_created=len(zones), trades_taken=len(completed), wins=wins,
                losses=len(completed) - wins,
                win_rate=wins / len(completed) if completed else np.nan,
                sum_R=sum(r_values) if r_values else 0.0,
                avg_R=np.mean(r_values) if r_values else np.nan,
            ))

summary = pd.DataFrame(results)
print("\n" + "=" * 110)
print(summary.to_string(index=False))
summary.to_csv("/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/dtfx_reversed_summary.csv", index=False)

# ---- comparison chart ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sumR_pivot = summary.pivot_table(index="ticker", columns=["variant", "period"], values="sum_R")
sumR_pivot.columns = [f"{v}_{p.replace('period_','')}" for v, p in sumR_pivot.columns]
sumR_pivot.plot(kind="bar", ax=axes[0], color=["#90caf9", "#1565c0", "#ef9a9a", "#b71c1c"])
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("Sum R: Original vs. Reversed, by Ticker & Period")
axes[0].set_ylabel("Sum of R multiples")
axes[0].legend(fontsize=7)

wr_pivot = summary.pivot_table(index="ticker", columns=["variant", "period"], values="win_rate") * 100
wr_pivot.columns = [f"{v}_{p.replace('period_','')}" for v, p in wr_pivot.columns]
wr_pivot.plot(kind="bar", ax=axes[1], color=["#90caf9", "#1565c0", "#ef9a9a", "#b71c1c"])
axes[1].axhline(50, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title("Win Rate: Original vs. Reversed, by Ticker & Period")
axes[1].set_ylabel("Win rate (%)")
axes[1].legend(fontsize=7)

plt.tight_layout()
out_path = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad/dtfx_reversed_comparison.png"
plt.savefig(out_path, dpi=140)
print("\nWrote", out_path)
