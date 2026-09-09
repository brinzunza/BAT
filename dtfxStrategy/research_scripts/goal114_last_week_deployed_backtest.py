"""Direct user request (2026-09-07): "with the new backtester, backtest the past week with
this deployed strat and show results profit factors, trades, trades per day, trades
accepted, total trades detected, max drawdown, etc."

Uses FRESH data pulled directly from OANDA (not the static historical CSVs, which end
2026-09-03) so "the past week" means the actual last 7 days relative to right now, and the
EXACT config currently running in PRODLIVETRADING/bot.py:
  - 4 pairs: EURUSD, GBPUSD, EURGBP, USDJPY
  - MIN_ZONE_PIPS=5, MIN_STOP_PIPS=3.0, MAX_WINDOW_MIN=1440
  - fixed 1:1R target (variant A)
  - symmetric_exit_spread=True (2026-09-07's cost-model correction: half the spread on
    entry AND both exits, not the whole spread on entry alone) -- see dtfx.md
  - 1.6-pip spread assumption (the project's standing baseline; no historical bid/ask
    series is available for these fresh bars, only OHLC, so the live spread gate itself
    isn't simulated here -- this is a cost ASSUMPTION, same as every other backtest)
  - "new backtester" = the production-accurate concurrency model
    (goal110_production_concurrency_simulator.py's simulate_side, ported from
    PRODLIVETRADING/strategy/engine.py::resolve_multi_symbol_history): one global resting
    slot per side across all 4 symbols, walked in true chronological order, rejected (not
    discarded) and rechecked on every subsequent bar when concurrency-blocked, updated
    here for the symmetric-spread convention.

Zone detection replays a ~45-day warmup (fetched fresh) so trend/pivot state is genuine
going into the reporting window, then reports are filtered to entries in the last 7 days
only. "Total trades detected" = every zone that resolved to a valid entry+exit
independently (pre-concurrency, matches resolve_base_strategy's own count). "Trades
accepted" = what actually survives the one-global-slot-per-side concurrency model, i.e.
what bot.py would actually have placed.
"""
import os
import sys
import time
import heapq

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FRESH_DIR = "/tmp/dtfx_fresh_data"

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
LOOKBACK_DAYS = 7


def load_and_build(symbol):
    path = os.path.join(FRESH_DIR, f"{symbol}.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.tz_localize(None)
    df = df.set_index("date").sort_index()
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
    """Symmetric-exit-spread convention (dtfx.md, 2026-09-07): both stop and target are
    offset by half_spread against the position, matching bot.py's actual marketable-limit
    order pricing exactly."""
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
            return j, 2, -1.0, eff_risk
        elif hit_target:
            win_r = abs(target_price - entry_price) / eff_risk if eff_risk > 0 else 0.0
            return j, 1, win_r, eff_risk
    return None, 0, 0.0, eff_risk


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
        exit_idx, outcome, net_r, eff_risk = resolve_exit_symmetric(
            high, low, is_buy_side, top, bottom, entry_price, half_spread, bar_idx, z["deadline"])
        if eff_risk < d["min_stop_price"]:
            continue
        if exit_idx is None:
            continue
        trades.append(dict(instrument=symbol, entry_time=times_ns[bar_idx], exit_time=times_ns[exit_idx],
                            entry_price=entry_price, stop_risk=eff_risk, net_r=net_r, is_buy=is_buy_side))
        slot_free_at = int(times_ns[exit_idx])
    return trades


def main():
    t0 = time.time()
    print("Loading fresh OANDA data + building zones (5-pip min, symmetric-exit-spread convention)...")
    cache = {}
    for sym in SYMBOLS:
        cache[sym] = load_and_build(sym)
        print(f"  {sym}: {len(cache[sym]['z_created']):,} raw zones over "
              f"{cache[sym]['times'].min()} -> {cache[sym]['times'].max()}")

    now = max(d["times"].max() for d in cache.values())
    window_start = now - pd.Timedelta(days=LOOKBACK_DAYS)
    print(f"\n'Past week' reporting window: {window_start} -> {now}")

    # ---- 1. Total trades detected: independent per-zone resolution, no concurrency ----
    detected_frames = []
    per_symbol_buy, per_symbol_sell = {}, {}
    for sym, d in cache.items():
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = MIN_ZONE_PIPS * pip
        entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            np.full(len(d["high"]), np.nan), np.full(len(d["high"]), np.nan),
            spread_price, min_stop_price, MAX_WINDOW, False, min_zone_height, True)
        resolved = outcome > 0
        ei = entry_idx[resolved]
        detected_frames.append(pd.DataFrame({
            "instrument": sym, "entry_time": d["times"].values[ei],
            "net_r": net_r[resolved], "is_win": outcome[resolved] == 1}))

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

    detected = pd.concat(detected_frames, ignore_index=True)
    detected_week = detected[detected["entry_time"] >= window_start]

    # ---- 2. Trades accepted: production-accurate global concurrency model ----
    print("\nRunning production-accurate concurrency simulation (BUY side)...")
    buy_trades = simulate_side(True, per_symbol_buy)
    print("Running production-accurate concurrency simulation (SELL side)...")
    sell_trades = simulate_side(False, per_symbol_sell)
    accepted = pd.DataFrame(buy_trades + sell_trades)
    accepted["entry_time"] = pd.to_datetime(accepted["entry_time"])
    accepted["exit_time"] = pd.to_datetime(accepted["exit_time"])
    accepted = accepted.sort_values("entry_time").reset_index(drop=True)
    accepted_week = accepted[accepted["entry_time"] >= window_start].reset_index(drop=True)

    def report(df, label):
        n = len(df)
        if n == 0:
            print(f"{label}: 0 trades"); return
        r = df["net_r"].values
        wr = 100.0 * (r > 0).mean()
        gw, gl = r[r > 0].sum(), -r[r <= 0].sum()
        pf = gw / gl if gl > 0 else float("inf")
        print(f"{label}: n={n}  WR={wr:.2f}%  PF={pf:.3f}  avgR={r.mean():+.4f}  "
              f"avgR(win)={r[r>0].mean() if (r>0).any() else float('nan'):+.4f}")

    print(f"\n{'='*78}\nTOTAL TRADES DETECTED (independent, pre-concurrency) -- past {LOOKBACK_DAYS} days\n{'='*78}")
    report(detected_week, "  detected")
    print(f"  by instrument: {detected_week['instrument'].value_counts().to_dict()}")
    print(f"  trades/day: {len(detected_week)/LOOKBACK_DAYS:.2f}")

    print(f"\n{'='*78}\nTRADES ACCEPTED (production-accurate 1-per-side-globally concurrency) -- past {LOOKBACK_DAYS} days\n{'='*78}")
    report(accepted_week, "  accepted")
    if len(accepted_week):
        print(f"  by instrument: {accepted_week['instrument'].value_counts().to_dict()}")
        print(f"  by side: {accepted_week['is_buy'].value_counts().to_dict()}  (True=buy, False=sell)")
    print(f"  trades/day: {len(accepted_week)/LOOKBACK_DAYS:.2f}")
    print(f"  acceptance rate: {100.0*len(accepted_week)/len(detected_week):.1f}% of detected" if len(detected_week) else "")

    # ---- 3. Two-account $ simulation on the accepted week's trades, fresh $100/$100 ----
    if len(accepted_week):
        accepted_week_sized = accepted_week.rename(columns={"stop_risk": "stop_risk"})
        result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
            accepted_week_sized, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
            max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=True)
        acc = result[result["accepted"]]
        final = eq_buy + eq_sell
        print(f"\n{'='*78}\nTWO-ACCOUNT $ SIMULATION (fresh $100 buy + $100 sell, {RISK_PCT:.0%} risk) -- past {LOOKBACK_DAYS} days\n{'='*78}")
        print(f"  trades sized/filled: {len(acc)} of {len(accepted_week)} (rest blocked by "
              f"per-account margin/leverage, not the strategy's own concurrency cap)")
        print(f"  final balance: ${final:,.2f}  (buy=${eq_buy:,.2f} sell=${eq_sell:,.2f})")
        print(f"  max drawdown: {max_dd:.2f}%")
        print(f"  P&L: {'+'if final>=200 else ''}{final-200:,.2f} ({(final/200-1)*100:+.2f}%)")
    else:
        print("\nNo accepted trades this week -- nothing to size/simulate.")

    detected_week.to_csv(os.path.join(DATA_DIR, "goal114_last_week_detected.csv"), index=False)
    accepted_week.to_csv(os.path.join(DATA_DIR, "goal114_last_week_accepted.csv"), index=False)
    print(f"\nSaved -> data/goal114_last_week_detected.csv, data/goal114_last_week_accepted.csv")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
