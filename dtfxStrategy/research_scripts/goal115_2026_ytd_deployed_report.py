"""Direct user request (2026-09-07): full 2026 YTD backtest of the currently-deployed
strategy (5-pip zone, 3% risk, symmetric-exit-spread cost model, production-accurate
1-slot-per-side-globally concurrency), showing: a day-by-day table (trades, win rate, avg
pips per win/loss), overall % return, the $200 two-account compounding equity curve, and
data for a chart of the last 10 trades.

Data: the static full-history CSVs (through ~2026-09-03) merged with a fresh direct-from-
OANDA pull (through 2026-09-07, this session's own goal114 fetch) so 2026 YTD runs all the
way through today, not stopping a few days short. Zones are built over the FULL merged
history (needed for correct trend continuity into 2026), then everything reported is
filtered to entry_time >= 2026-01-01.
"""
import os
import sys
import time
import heapq
import json

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
MERGED_DIR = "/tmp/dtfx_fresh_data"

import goal95_every_swing_zones as g95
import goal99_dtfx_base_backtest as g99
import goal79_two_account_portfolio as g79

SYMBOLS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}
MIN_ZONE_PIPS = 5.0
MIN_STOP_PIPS = 3.0
SPREAD_PIPS = 1.6
MAX_WINDOW = 1440
RISK_PCT = 0.03
YTD_START = pd.Timestamp("2026-01-01")


def load_and_build(symbol):
    path = os.path.join(MERGED_DIR, f"{symbol}_merged.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.tz_localize(None)
    df = df.drop_duplicates(subset="date").set_index("date").sort_index()
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high, low, close)
    (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
     ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)
    return dict(times=df.index, high=high, low=low, close=close, z_created=z_created,
                z_top=z_top, z_bottom=z_bottom, z_is_buy=z_is_buy, pip=PIP[symbol])


def find_next_touch(high, low, is_buy, top, bottom, mid, from_bar, deadline):
    n = len(high)
    end = min(deadline + 1, n)
    for j in range(from_bar, end):
        if is_buy:
            if low[j] <= bottom:
                return None
            if low[j] <= mid:
                return j
        else:
            if high[j] >= top:
                return None
            if high[j] >= mid:
                return j
    return None


def resolve_exit_symmetric(high, low, is_buy, top, bottom, entry_price, half_spread, from_bar, deadline):
    n = len(high)
    end = min(deadline + 1, n)
    if is_buy:
        stop_price = bottom - half_spread
        target_price = top - half_spread
    else:
        stop_price = top + half_spread
        target_price = bottom + half_spread
    eff_risk = abs(entry_price - stop_price)
    for j in range(from_bar, end):
        hit_stop = (low[j] < stop_price) if is_buy else (high[j] > stop_price)
        hit_target = (high[j] > target_price) if is_buy else (low[j] < target_price)
        if hit_stop:
            return j, 2, -1.0, eff_risk, stop_price, target_price
        elif hit_target:
            win_r = abs(target_price - entry_price) / eff_risk if eff_risk > 0 else 0.0
            return j, 1, win_r, eff_risk, stop_price, target_price
    return None, 0, 0.0, eff_risk, stop_price, target_price


def simulate_side(is_buy_side, per_symbol):
    heap = []
    zone_data = {}
    for symbol, d in per_symbol.items():
        high, low, times_ns = d["high"], d["low"], d["times_ns"]
        zones = d["zones"]
        zdata = []
        for created_idx, top, bottom in zones:
            if top - bottom <= 0:
                zdata.append(None)
                continue
            mid = (top + bottom) / 2.0
            deadline = min(created_idx + MAX_WINDOW, len(high) - 1)
            overshoot_idx = None
            if is_buy_side:
                if high[created_idx] > top:
                    overshoot_idx = created_idx
            else:
                if low[created_idx] < bottom:
                    overshoot_idx = created_idx
            if overshoot_idx is None:
                end = min(deadline + 1, len(high))
                for j in range(created_idx + 1, end):
                    if is_buy_side and high[j] > top:
                        overshoot_idx = j; break
                    if (not is_buy_side) and low[j] < bottom:
                        overshoot_idx = j; break
            zi = len(zdata)
            zdata.append(dict(created_idx=created_idx, top=top, bottom=bottom, mid=mid, deadline=deadline))
            if overshoot_idx is not None:
                nt = find_next_touch(high, low, is_buy_side, top, bottom, mid, overshoot_idx, deadline)
                if nt is not None:
                    heapq.heappush(heap, (int(times_ns[nt]), symbol, zi))
        zone_data[symbol] = zdata

    slot_free_at = -1
    trades = []
    while heap:
        ts, symbol, zi = heapq.heappop(heap)
        d = per_symbol[symbol]
        high, low, times_ns = d["high"], d["low"], d["times_ns"]
        half_spread = d["half_spread"]
        z = zone_data[symbol][zi]
        bar_idx = int(np.searchsorted(times_ns, ts))
        if ts <= slot_free_at:
            nt = find_next_touch(high, low, is_buy_side, z["top"], z["bottom"], z["mid"],
                                  bar_idx + 1, z["deadline"])
            if nt is not None:
                heapq.heappush(heap, (int(times_ns[nt]), symbol, zi))
            continue
        mid, top, bottom = z["mid"], z["top"], z["bottom"]
        entry_price = mid + half_spread if is_buy_side else mid - half_spread
        exit_idx, outcome, net_r, eff_risk, stop_price, target_price = resolve_exit_symmetric(
            high, low, is_buy_side, top, bottom, entry_price, half_spread, bar_idx, z["deadline"])
        if eff_risk < d["min_stop_price"]:
            continue
        if exit_idx is None:
            continue
        trades.append(dict(instrument=symbol, entry_time=times_ns[bar_idx], exit_time=times_ns[exit_idx],
                            entry_idx=bar_idx, exit_idx=exit_idx, zone_top=top, zone_bottom=bottom,
                            entry_price=entry_price, stop_price=stop_price, target_price=target_price,
                            stop_risk=eff_risk, net_r=net_r, is_buy=is_buy_side, outcome=outcome))
        slot_free_at = int(times_ns[exit_idx])
    return trades


def main():
    t0 = time.time()
    print("Loading merged data (static + fresh gap-fill through 2026-09-07) + building zones...")
    cache = {}
    for sym in SYMBOLS:
        cache[sym] = load_and_build(sym)
        print(f"  {sym}: {len(cache[sym]['z_created']):,} raw zones, "
              f"{cache[sym]['times'].min()} -> {cache[sym]['times'].max()}  ({time.time()-t0:.1f}s)")

    per_symbol_buy, per_symbol_sell = {}, {}
    for sym, d in cache.items():
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = MIN_ZONE_PIPS * pip
        times_ns = d["times"].values.astype("datetime64[ns]").astype(np.int64)
        keep = (d["z_top"] - d["z_bottom"]) >= min_zone_height
        base = dict(high=d["high"], low=d["low"], times_ns=times_ns,
                    half_spread=spread_price / 2.0, min_stop_price=min_stop_price)
        buy_zones = [(int(c), float(t), float(b)) for c, t, b, ib in
                     zip(d["z_created"][keep], d["z_top"][keep], d["z_bottom"][keep], d["z_is_buy"][keep]) if ib]
        sell_zones = [(int(c), float(t), float(b)) for c, t, b, ib in
                      zip(d["z_created"][keep], d["z_top"][keep], d["z_bottom"][keep], d["z_is_buy"][keep]) if not ib]
        per_symbol_buy[sym] = dict(base, zones=buy_zones)
        per_symbol_sell[sym] = dict(base, zones=sell_zones)

    print("\nSimulating BUY side (production-accurate, global concurrency, full history)...")
    t1 = time.time()
    buy_trades = simulate_side(True, per_symbol_buy)
    print(f"  {len(buy_trades):,} trades  ({time.time()-t1:.1f}s)")
    print("Simulating SELL side...")
    t1 = time.time()
    sell_trades = simulate_side(False, per_symbol_sell)
    print(f"  {len(sell_trades):,} trades  ({time.time()-t1:.1f}s)")

    all_trades = pd.DataFrame(buy_trades + sell_trades)
    all_trades["entry_time"] = pd.to_datetime(all_trades["entry_time"])
    all_trades["exit_time"] = pd.to_datetime(all_trades["exit_time"])
    all_trades = all_trades.sort_values("entry_time").reset_index(drop=True)
    all_trades["pip_size"] = all_trades["instrument"].map(PIP)
    all_trades["is_win"] = all_trades["net_r"] > 0
    all_trades["pips"] = np.where(all_trades["is_win"],
                                   all_trades["net_r"] * all_trades["stop_risk"] / all_trades["pip_size"],
                                   -all_trades["stop_risk"] / all_trades["pip_size"])

    ytd = all_trades[all_trades["entry_time"] >= YTD_START].reset_index(drop=True)
    print(f"\n2026 YTD: {len(ytd):,} accepted trades ({ytd['entry_time'].min()} -> {ytd['entry_time'].max()})")

    # ---- Day-by-day table ----
    ytd["day"] = ytd["entry_time"].dt.date
    rows = []
    for day, g in ytd.groupby("day"):
        wins = g[g["is_win"]]
        losses = g[~g["is_win"]]
        n = len(g)
        wr = 100.0 * len(wins) / n if n else float("nan")
        avg_pips_win = wins["pips"].mean() if len(wins) else float("nan")
        avg_pips_loss = losses["pips"].mean() if len(losses) else float("nan")
        rows.append(dict(day=str(day), trades=n, win_rate=wr, avg_pips_win=avg_pips_win,
                          avg_pips_loss=avg_pips_loss, sum_r=g["net_r"].sum()))
    daily = pd.DataFrame(rows)
    daily.to_csv(os.path.join(DATA_DIR, "goal115_2026_ytd_daily.csv"), index=False)

    # ---- Two-account $200 compounding equity curve, 2026 YTD ----
    result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
        ytd, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
        max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=True)
    acc = result[result["accepted"]]
    final = eq_buy + eq_sell
    pct_return = (final / 200.0 - 1.0) * 100.0
    trace["day"] = trace["close_time"].dt.date
    daily_equity = trace.groupby("day")["equity_combined"].last().reset_index()
    daily_equity.to_csv(os.path.join(DATA_DIR, "goal115_2026_ytd_equity.csv"), index=False)

    n = len(acc)
    wr_all = 100.0 * (acc["net_r"] > 0).mean()
    gw, gl = acc.loc[acc["net_r"] > 0, "net_r"].sum(), -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    span_days = (acc["entry_time"].max() - acc["entry_time"].min()).total_seconds() / 86400

    print(f"\n{'='*78}\n2026 YTD SUMMARY (production-accurate, 5-pip, 3% risk, symmetric-exit-spread)\n{'='*78}")
    print(f"  trades sized/filled: {n:,}  ({n/span_days:.2f}/day over {span_days:.0f} days)")
    print(f"  win rate: {wr_all:.2f}%   PF: {pf:.3f}")
    print(f"  final balance: ${final:,.2f}  (buy=${eq_buy:,.2f} sell=${eq_sell:,.2f})")
    print(f"  return: {pct_return:+.2f}%   max drawdown: {max_dd:.2f}%")

    # ---- Last 10 trades, with local OHLC context for a chart ----
    last10 = all_trades.tail(10).reset_index(drop=True)
    chart_data = []
    for _, t in last10.iterrows():
        d = cache[t["instrument"]]
        lo = max(0, int(t["entry_idx"]) - 120)
        hi = min(len(d["high"]), int(t["exit_idx"]) + 30)
        bars = pd.DataFrame({"time": d["times"][lo:hi].astype(str),
                              "high": d["high"][lo:hi], "low": d["low"][lo:hi],
                              "close": d["close"][lo:hi]})
        chart_data.append(dict(
            instrument=t["instrument"], side="buy" if t["is_buy"] else "sell",
            entry_time=str(t["entry_time"]), exit_time=str(t["exit_time"]),
            zone_top=float(t["zone_top"]), zone_bottom=float(t["zone_bottom"]),
            entry_price=float(t["entry_price"]), stop_price=float(t["stop_price"]),
            target_price=float(t["target_price"]), outcome="win" if t["is_win"] else "loss",
            net_r=float(t["net_r"]), pips=float(t["pips"]), bars=bars.to_dict("records")))
    with open(os.path.join(DATA_DIR, "goal115_last10_trades_chart_data.json"), "w") as f:
        json.dump(chart_data, f)

    print(f"\nLast 10 trades:")
    print(last10[["instrument", "is_buy", "entry_time", "outcome", "pips", "net_r"]].to_string(index=False))

    all_trades.to_csv(os.path.join(DATA_DIR, "goal115_all_2026_trades.csv"), index=False)
    print(f"\nSaved -> goal115_2026_ytd_daily.csv, goal115_2026_ytd_equity.csv, "
          f"goal115_all_2026_trades.csv, goal115_last10_trades_chart_data.json")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
