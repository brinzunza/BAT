"""Direct user request (2026-09-07): "assume that i start the $200 account start of 2025
and show growth" -- a SINGLE continuous two-account run starting fresh $100+$100 on
2025-01-01, compounding straight through to today (2026-09-07) with no reset at the
2025/2026 year boundary (unlike goal115/goal116, which each started fresh on their own
Jan 1). Same currently-deployed config: 5-pip zone, 3% risk, 4 pairs, symmetric-exit-spread
cost model, production-accurate concurrency.

Reuses the merged (static history + fresh OANDA gap-fill through 2026-09-07) data already
built for goal115, so the endpoint is real and current, not stopping a few days short.
Produces the same month -> week -> day -> trades drill-down shape as goal116, just spanning
~21 months (Jan 2025 - Sep 2026) instead of 12.
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
import goal79_two_account_portfolio as g79

SYMBOLS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}
MIN_ZONE_PIPS = 5.0
MIN_STOP_PIPS = 3.0
SPREAD_PIPS = 1.6
MAX_WINDOW = 1440
RISK_PCT = 0.03
START = pd.Timestamp("2025-01-01")


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
    return dict(times=df.index, high=high, low=low, z_created=z_created, z_top=z_top,
                z_bottom=z_bottom, z_is_buy=z_is_buy, pip=PIP[symbol])


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
                            entry_price=entry_price, stop_price=stop_price, target_price=target_price,
                            stop_risk=eff_risk, net_r=net_r, is_buy=is_buy_side, outcome=outcome))
        slot_free_at = int(times_ns[exit_idx])
    return trades


def main():
    t0 = time.time()
    print("Loading merged data (static + fresh through 2026-09-07) + building zones...")
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
    all_trades["is_win"] = all_trades["outcome"] == 1
    all_trades["pips"] = np.where(all_trades["is_win"],
                                   all_trades["net_r"] * all_trades["stop_risk"] / all_trades["pip_size"],
                                   -all_trades["stop_risk"] / all_trades["pip_size"])

    span = all_trades[all_trades["entry_time"] >= START].reset_index(drop=True)
    print(f"\nContinuous span (2025-01-01 -> {span['entry_time'].max()}): {len(span):,} accepted trades")

    # ---- ONE continuous two-account $200 run, no reset at the 2025/2026 boundary ----
    result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
        span, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
        max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=True)
    acc_ids = set(result.index[result["accepted"]])
    span["sized"] = span.index.isin(acc_ids)
    final = eq_buy + eq_sell
    pct_return = (final / 200.0 - 1.0) * 100.0
    trace["date"] = trace["close_time"].dt.date
    equity_by_date = trace.groupby("date")["equity_combined"].last()

    n_sized = int(span["sized"].sum())
    wr_all = 100.0 * (span["net_r"] > 0).mean()
    gw, gl = span.loc[span["net_r"] > 0, "net_r"].sum(), -span.loc[span["net_r"] <= 0, "net_r"].sum()
    pf = gw / gl if gl > 0 else float("inf")
    span_days = (span["entry_time"].max() - span["entry_time"].min()).total_seconds() / 86400

    print(f"\n{'='*78}\nCONTINUOUS $200 GROWTH: 2025-01-01 -> {span['entry_time'].max().date()} ({span_days:.0f} days)\n{'='*78}")
    print(f"  trades taken: {len(span):,}  ({len(span)/span_days:.2f}/day)   trades sized: {n_sized:,}")
    print(f"  win rate: {wr_all:.2f}%   PF: {pf:.3f}")
    print(f"  final balance: ${final:,.2f}  (buy=${eq_buy:,.2f} sell=${eq_sell:,.2f})")
    print(f"  return: {pct_return:+.2f}%   max drawdown: {max_dd:.2f}%")

    all_dates = pd.date_range(START, span["entry_time"].max().normalize(), freq="D").date
    running = 200.0
    equity_series = {}
    for d in all_dates:
        if d in equity_by_date.index:
            running = float(equity_by_date.loc[d])
        equity_series[str(d)] = round(running, 2)

    # ---- month -> week -> day -> trades, continuous across the whole span ----
    span["date"] = span["entry_time"].dt.date
    span["month"] = span["entry_time"].dt.strftime("%Y-%m")
    span["week_start"] = (span["entry_time"] - pd.to_timedelta(span["entry_time"].dt.weekday, unit="D")).dt.date

    def agg_block(g, key_val, key_name, prev_equity, last_equity):
        wins = g[g["is_win"]]; losses = g[~g["is_win"]]
        n = len(g)
        return {
            key_name: str(key_val), "trades": int(n),
            "win_rate": round(100.0 * len(wins) / n, 2) if n else None,
            "avg_pips_win": round(float(wins["pips"].mean()), 2) if len(wins) else None,
            "avg_pips_loss": round(float(losses["pips"].mean()), 2) if len(losses) else None,
            "sum_r": round(float(g["net_r"].sum()), 2),
            "equity_end": round(last_equity, 2),
            "return_pct": round((last_equity / prev_equity - 1.0) * 100.0, 2) if prev_equity else None,
        }

    months_out, weeks_out, days_out, trades_out = [], {}, {}, {}
    prev_month_eq = 200.0
    for month, gmon in span.groupby("month"):
        month_end_date = str(gmon["date"].max())
        month_eq = equity_series.get(month_end_date, prev_month_eq)
        months_out.append(agg_block(gmon, month, "month", prev_month_eq, month_eq))
        prev_month_eq = month_eq

        prev_week_eq = months_out[-2]["equity_end"] if len(months_out) > 1 else 200.0
        week_list = []
        for week_start, gweek in gmon.groupby("week_start"):
            week_end_date = str(gweek["date"].max())
            week_eq = equity_series.get(week_end_date, prev_week_eq)
            week_list.append(agg_block(gweek, week_start, "week", prev_week_eq, week_eq))
            prev_week_eq = week_eq

            prev_day_eq = week_list[-2]["equity_end"] if len(week_list) > 1 else (
                months_out[-2]["equity_end"] if len(months_out) > 1 else 200.0)
            day_list = []
            for day, gday in gweek.groupby("date"):
                day_eq = equity_series.get(str(day), prev_day_eq)
                day_list.append(agg_block(gday, day, "day", prev_day_eq, day_eq))
                prev_day_eq = day_eq
                trades_out[str(day)] = [
                    dict(i=r.instrument, s="b" if r.is_buy else "s",
                         t=r.entry_time.strftime("%H:%M"), x=r.exit_time.strftime("%H:%M"),
                         o="w" if r.is_win else "l", p=round(float(r.pips), 1), r=round(float(r.net_r), 2))
                    for r in gday.itertuples()]
            days_out[str(week_start)] = day_list
        weeks_out[str(month)] = week_list

    out = dict(summary=dict(trades=len(span), trades_sized=n_sized, win_rate=round(wr_all, 2),
                             profit_factor=round(pf, 3), final_balance=round(final, 2),
                             return_pct=round(pct_return, 2), max_drawdown_pct=round(max_dd, 2),
                             eq_buy=round(eq_buy, 2), eq_sell=round(eq_sell, 2),
                             start_date="2025-01-01", end_date=str(span["entry_time"].max().date()),
                             span_days=round(span_days, 1)),
               equity_series=equity_series,
               months=months_out, weeks=weeks_out, days=days_out, trades=trades_out)

    out_path = os.path.join(DATA_DIR, "goal117_2025_to_now_continuous.json")
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"\nSaved -> {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB)")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
