"""Production-accurate concurrency simulator for the dtfx.md base strategy (variant A,
fixed 1:1R), ported from PRODLIVETRADING/strategy/engine.py::resolve_multi_symbol_history
-- the ACTUAL logic the live bot uses (confirmed by reading it directly, not guessed):

  - A single GLOBAL open-position count per side, shared across all four symbols (one real
    buy-only and one real sell-only OANDA sub-account -- "how many positions are open on
    this side" is a property of the ACCOUNT, not any one symbol).
  - Walked in TRUE chronological order across all four symbols' bars merged together, not
    each symbol resolved independently and merely accept/reject-filtered afterward (that
    was goal79_two_account_portfolio.py's simplification, confirmed against this reference
    to be the gap that mattered).
  - "Rejected, not queued" at the exact moment a touch is concurrency-blocked -- but the
    zone stays alive and gets rechecked on every subsequent bar (not discarded), so it can
    still fill later once a slot frees AND price is still (or again) at a fillable level.
  - A pending (not-yet-filled) zone is invalidated outright if price ever breaches its own
    stop side before it fills -- entering a trade whose stop already triggered before entry
    makes no sense, exactly as engine.py's `breached` check enforces.

Implemented here as a per-side (buy/sell are independent -- separate sub-accounts, separate
slots) heap-based event simulation: each zone's OWN independent "natural" touch time is
precomputed (fast, no concurrency dependency, reuses goal99's overshoot-then-tag logic);
if the slot is free at that moment, fill immediately (the common case); if blocked, a small
forward re-scan finds this zone's next touch opportunity (or its invalidation/expiry) and
re-queues it -- avoiding the need to visit every one of tens of millions of raw bars, since
most zones never get blocked at all.

Restricted to PRE-2021 data only (dtfx.md's "Ground Truth" finding: 2021-01 through
2026-03 is a data-feed artifact, excluded from anything meant to be trustworthy).
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

import goal104_optimal_risk_two_account as g104
import goal79_two_account_portfolio as g79

PRE_END = pd.Timestamp("2021-01-01")
MIN_ZONE_PIPS = 10
MIN_STOP_PIPS = 3.0
SPREAD_PIPS = 1.6
MAX_WINDOW = 1440
RISK_PCT = 0.03


def find_next_touch(high, low, is_buy, top, bottom, mid, from_bar, deadline):
    """Scan forward from `from_bar` (inclusive) for the next bar where this zone's touch
    condition holds, returning that bar index -- or None if the zone dies first (price
    breaches its own stop side, i.e. engine.py's `breached`) or expires (passes deadline)."""
    n = len(high)
    end = min(deadline + 1, n)
    for j in range(from_bar, end):
        if is_buy:
            if low[j] <= bottom:
                return None  # breached its own stop before ever filling -- dead
            if low[j] <= mid:
                return j
        else:
            if high[j] >= top:
                return None
            if high[j] >= mid:
                return j
    return None


def resolve_exit(high, low, is_buy, top, bottom, entry_price, from_bar, deadline):
    n = len(high)
    end = min(deadline + 1, n)
    stop_price = bottom if is_buy else top
    target_price = top if is_buy else bottom
    for j in range(from_bar, end):
        hit_stop = (low[j] < stop_price) if is_buy else (high[j] > stop_price)
        hit_target = (high[j] > target_price) if is_buy else (low[j] < target_price)
        if hit_stop:
            return j, 2, -1.0
        elif hit_target:
            eff_risk = abs(entry_price - stop_price)
            win_r = abs(target_price - entry_price) / eff_risk if eff_risk > 0 else 0.0
            return j, 1, win_r
    return None, 0, 0.0


def simulate_side(is_buy_side, per_symbol):
    """per_symbol: dict[symbol] -> dict(high, low, times_ns, pip, zones=[(created_idx, top,
    bottom)], spread_price, min_stop_price). Returns list of trade dicts."""
    heap = []  # (touch_ts_ns, symbol, zone_idx)
    zone_data = {}
    for symbol, d in per_symbol.items():
        high, low, times_ns = d["high"], d["low"], d["times_ns"]
        spread_price, min_stop_price = d["spread_price"], d["min_stop_price"]
        zones = d["zones"]
        zdata = []
        for created_idx, top, bottom in zones:
            height = top - bottom
            if height <= 0:
                zdata.append(None)
                continue
            mid = (top + bottom) / 2.0
            deadline = created_idx + MAX_WINDOW
            # overshoot: independent of concurrency, find first bar >= created where price
            # exceeds the far edge (dtfx.md #6.1)
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
            zdata.append(dict(created_idx=created_idx, top=top, bottom=bottom, mid=mid,
                               deadline=deadline, overshoot_idx=overshoot_idx))
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
        z = zone_data[symbol][zi]
        bar_idx = int(np.searchsorted(times_ns, ts))
        # strict: a fill that opens AND closes within one bar must not free the slot for
        # another new entry on that SAME bar -- the slot becomes available starting the
        # NEXT bar, not at the exact instant (still ambiguous sub-minute) it closed
        if ts <= slot_free_at:
            # concurrency-blocked at this exact moment -- rejected, not queued, but the
            # zone stays alive: find its NEXT opportunity (matches engine.py exactly)
            nt = find_next_touch(high, low, is_buy_side, z["top"], z["bottom"], z["mid"],
                                  bar_idx + 1, z["deadline"])
            if nt is not None:
                heapq.heappush(heap, (int(times_ns[nt]), symbol, zi))
            continue
        # slot free: fill now
        mid, top, bottom = z["mid"], z["top"], z["bottom"]
        spread_price = d["spread_price"]
        if is_buy_side:
            deep = low[bar_idx] <= mid - spread_price
            entry_price = mid if deep else mid + spread_price
        else:
            deep = high[bar_idx] >= mid + spread_price
            entry_price = mid if deep else mid - spread_price
        eff_risk = abs(entry_price - (bottom if is_buy_side else top))
        if eff_risk < d["min_stop_price"]:
            continue  # too tight post-spread, matches dtfx.md #8 -- skip, slot stays free
        exit_idx, outcome, net_r = resolve_exit(high, low, is_buy_side, top, bottom,
                                                 entry_price, bar_idx, z["deadline"])
        if exit_idx is None:
            continue  # never resolves within window -- not counted, slot stays free
        trades.append(dict(instrument=symbol, entry_time=times_ns[bar_idx], exit_time=times_ns[exit_idx],
                            entry_price=entry_price, stop_risk=eff_risk, net_r=net_r, is_buy=is_buy_side))
        slot_free_at = int(times_ns[exit_idx])

    return trades


def main():
    t0 = time.time()
    print("Loading + building zones once...")
    cache = g104.load_and_build_zones()

    print(f"\nBuilding PRE-2021 zone lists, min_zone={MIN_ZONE_PIPS}pips...")
    per_symbol_buy, per_symbol_sell = {}, {}
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = MIN_ZONE_PIPS * pip
        times_ns = d["times"].values.astype("datetime64[ns]").astype(np.int64)
        pre_cut = int(np.searchsorted(times_ns, PRE_END.value))

        z_created, z_top, z_bottom, z_is_buy = d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"]
        keep = (z_created < pre_cut) & ((z_top - z_bottom) >= min_zone_height)
        base = dict(high=d["high"], low=d["low"], times_ns=times_ns,
                    spread_price=spread_price, min_stop_price=min_stop_price)

        buy_zones = [(int(c), float(t), float(b)) for c, t, b, ib in
                     zip(z_created[keep], z_top[keep], z_bottom[keep], z_is_buy[keep]) if ib]
        sell_zones = [(int(c), float(t), float(b)) for c, t, b, ib in
                      zip(z_created[keep], z_top[keep], z_bottom[keep], z_is_buy[keep]) if not ib]
        per_symbol_buy[inst] = dict(base, zones=buy_zones)
        per_symbol_sell[inst] = dict(base, zones=sell_zones)
        print(f"  {inst}: {len(buy_zones):,} buy zones, {len(sell_zones):,} sell zones (PRE-2021, >={MIN_ZONE_PIPS}pips)")

    print("\nSimulating BUY side (global concurrency across all 4 pairs)...")
    t1 = time.time()
    buy_trades = simulate_side(True, per_symbol_buy)
    print(f"  {len(buy_trades):,} filled trades  ({time.time()-t1:.1f}s)")

    print("Simulating SELL side (global concurrency across all 4 pairs)...")
    t1 = time.time()
    sell_trades = simulate_side(False, per_symbol_sell)
    print(f"  {len(sell_trades):,} filled trades  ({time.time()-t1:.1f}s)")

    all_trades = pd.DataFrame(buy_trades + sell_trades)
    all_trades["entry_time"] = pd.to_datetime(all_trades["entry_time"])
    all_trades["exit_time"] = pd.to_datetime(all_trades["exit_time"])
    all_trades = all_trades.sort_values("entry_time").reset_index(drop=True)
    all_trades.to_csv(os.path.join(DATA_DIR, "goal110_production_accurate_trades.csv"), index=False)

    n = len(all_trades)
    wr = 100.0 * (all_trades["net_r"] > 0).mean()
    gl = -all_trades.loc[all_trades["net_r"] <= 0, "net_r"].sum()
    pf = all_trades.loc[all_trades["net_r"] > 0, "net_r"].sum() / gl if gl > 0 else float("inf")
    print(f"\n=== Production-accurate concurrency: {n:,} trades, WR={wr:.2f}%, PF={pf:.3f} ===")

    print(f"\n{'='*70}\nCompare: goal79's simplified independent-resolve-then-filter model\n{'='*70}")
    # rebuild the same PRE-2021, 10-pip trade candidates the OLD way for a fair comparison
    import goal99_dtfx_base_backtest as g99
    frames = []
    for inst, d in cache.items():
        pip = d["pip"]
        spread_price = SPREAD_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        min_zone_height = MIN_ZONE_PIPS * pip
        entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
            d["z_created"], d["z_top"], d["z_bottom"], d["z_is_buy"], d["high"], d["low"],
            d["last_high_of_bar"], d["last_low_of_bar"], spread_price, min_stop_price,
            MAX_WINDOW, False, min_zone_height)
        resolved = outcome > 0
        ei = entry_idx[resolved]; xi = exit_idx[resolved]
        is_buy = d["z_is_buy"][resolved]
        stop_price = np.where(is_buy, d["z_bottom"][resolved], d["z_top"][resolved])
        stop_risk = np.abs(entry_price[resolved] - stop_price)
        frames.append(pd.DataFrame({
            "instrument": inst, "entry_time": d["times"].values[ei], "exit_time": d["times"].values[xi],
            "entry_price": entry_price[resolved], "stop_risk": stop_risk, "net_r": net_r[resolved], "is_buy": is_buy,
        }))
    old_trades = pd.concat(frames, ignore_index=True)
    old_trades = old_trades[old_trades["entry_time"] < PRE_END].sort_values("entry_time").reset_index(drop=True)

    g79.RISK_PCT = RISK_PCT
    old_result, old_eq_buy, old_eq_sell, old_dd, _ = g79.simulate_two_account(
        old_trades, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
        max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=False)
    old_accepted = old_result[old_result["accepted"]]
    old_wr = 100.0 * (old_accepted["net_r"] > 0).mean()
    print(f"OLD (independent-resolve + accept/reject filter): {len(old_accepted):,} accepted trades, "
          f"WR={old_wr:.2f}%, final=${old_eq_buy+old_eq_sell:,.2f}, maxDD={old_dd:.2f}%")

    g79.RISK_PCT = RISK_PCT
    new_result, new_eq_buy, new_eq_sell, new_dd, _ = g79.simulate_two_account(
        all_trades, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
        max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=False)
    print(f"NEW (production-accurate global concurrency):     {n:,} filled trades, "
          f"WR={wr:.2f}%, final=${new_eq_buy+new_eq_sell:,.2f}, maxDD={new_dd:.2f}%")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
