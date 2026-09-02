"""What would capping concurrency to 1 open position PER SUB-ACCOUNT (max 1 buy + 1 sell,
never 2 stacked on the same side) actually do to Speed-King's results, vs. the currently
deployed rule (margin-constrained + a flat global cap of 3, no side distinction)?

Both variants share the same margin-constrained core (goal20's simulate_margin_constrained)
-- this only adds an extra headcount gate on top, same as bot.py's MAX_CONCURRENT_POSITIONS
check sits on top of the live margin pre-check. One real simplification carried over from
goal20 itself: this still treats margin as a single shared pool, not the two genuinely
separate long/short sub-account pools OANDA actually has -- the per-side headcount cap here
answers "how many positions can be open on each side," which is independent of that
simplification, but the margin _capacity_ numbers underneath are still the same
single-pool approximation the rest of this project's margin research uses.
"""
import os
import sys
import heapq
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal20_speed_king_margin_report as g20

ROBUST_RISK_PCT = 0.03


def build_speed_king_full_with_side():
    """Same as goal20.build_speed_king_full(), except it calls simulate_generic_ex
    separately per side (buy-only zones, then sell-only zones) instead of once with both
    mixed together -- simulate_generic_ex's output is per-TRADE, not per-input-zone (a zone
    can produce 0, 1, or several trades via touches/re-arm), with no zone-index back-
    reference, so there's no way to recover which side a given output trade came from when
    both sides go in as one mixed call. Splitting the input by side sidesteps that: each
    call's output is unambiguously all-buy or all-sell by construction. Trade order/values
    are otherwise identical to goal20's version (same engine, same params, just split)."""
    rs, pg, TRADEABLE = g20.rs, g20.pg, g20.TRADEABLE
    frames = []
    for inst in TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (g20.SK_N * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        risk_ok = risk_pct >= pg.MIN_RISK_PCT

        for side_is_buy in (True, False):
            mask = risk_ok & (z["is_buy"] == side_is_buy)
            top, bottom, mid = z["top"][mask], z["bottom"][mask], z["mid"][mask]
            is_buy_arr, start_idx, c_ns = z["is_buy"][mask], z["start_idx"][mask], created_ns[mask]
            if len(top) == 0:
                continue
            touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
                top, bottom, mid, is_buy_arr, start_idx, c_ns, times_ns, lows, highs,
                int(g20.SK_AGE_H * 3600 * 1e9), g20.SK_N, g20.SK_N * g20.SK_RR, g20.SK_BOUNCE,
                pg.PROBE_DEPTH, 1.0, g20.SK_TOUCHES, pg.MAX_WINDOW_BARS)
            pip = pg.PIP_SIZE[inst]
            spread_price = pg.SPREAD_PIPS_TYPICAL * pip
            r_gross = np.where(is_win, g20.SK_RR, -1.0)
            cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
            net_r = r_gross - cost_in_r
            frames.append(pd.DataFrame({
                "instrument": inst,
                "entry_time": pd.to_datetime(times[entry_idx]),
                "exit_time": pd.to_datetime(times[exit_idx]),
                "entry_price": entry_mid,
                "stop_risk": stop_risk,
                "net_r": net_r,
                "is_win": is_win,
                "is_buy": side_is_buy,
            }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def simulate_with_concurrency_rule(trades, start=g20.START, risk_pct=ROBUST_RISK_PCT,
                                    max_leverage=g20.MAX_LEVERAGE, cap=g20.CAP,
                                    global_max=None, per_side_max=None):
    """Same event-driven margin walk as goal20's simulate_margin_constrained, with an
    optional extra headcount gate layered on top: global_max caps total open positions
    (any mix of sides), per_side_max caps open positions on EACH side independently.
    Both can be set at once; either can be None (no extra gate beyond margin)."""
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    entry_time = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_time = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values
    is_buy = t.is_buy.values

    equity = start
    margin_used = 0.0
    open_buy, open_sell = 0, 0
    heap = []  # (exit_time_ns, risk_dollars, margin, net_r, is_buy)

    accepted = np.zeros(n, dtype=bool)
    close_times, close_equities, close_net_r = [], [], []
    peak_equity = start
    max_dd_pct = 0.0

    for i in range(n):
        et = entry_time[i]
        while heap and heap[0][0] <= et:
            xt, r_dollars, margin, r, buy_side = heapq.heappop(heap)
            equity += r_dollars * r
            margin_used -= margin
            if buy_side:
                open_buy -= 1
            else:
                open_sell -= 1
            if equity > peak_equity:
                peak_equity = equity
            dd = 100 * (equity - peak_equity) / peak_equity
            if dd < max_dd_pct:
                max_dd_pct = dd
            close_times.append(xt); close_equities.append(equity); close_net_r.append(r)
        margin_used = max(margin_used, 0.0)

        units, risk_dollars, margin = g20.position_size_and_margin(
            instrument[i], equity, entry_price[i], stop_risk[i], risk_pct, max_leverage, cap)
        available_capacity = min(equity, cap) if cap is not None else equity

        fits_margin = margin_used + margin <= available_capacity
        total_open = open_buy + open_sell
        fits_global = (global_max is None) or (total_open < global_max)
        side_open = open_buy if is_buy[i] else open_sell
        fits_side = (per_side_max is None) or (side_open < per_side_max)

        if fits_margin and fits_global and fits_side:
            accepted[i] = True
            heapq.heappush(heap, (exit_time[i], risk_dollars, margin, net_r[i], bool(is_buy[i])))
            margin_used += margin
            if is_buy[i]:
                open_buy += 1
            else:
                open_sell += 1

    while heap:
        xt, r_dollars, margin, r, buy_side = heapq.heappop(heap)
        equity += r_dollars * r
        if equity > peak_equity:
            peak_equity = equity
        dd = 100 * (equity - peak_equity) / peak_equity
        if dd < max_dd_pct:
            max_dd_pct = dd
        close_times.append(xt); close_equities.append(equity); close_net_r.append(r)

    trace = pd.DataFrame({"close_time": pd.to_datetime(np.array(close_times, dtype="int64")),
                           "equity": close_equities, "net_r": close_net_r})
    t["accepted"] = accepted
    return t, equity, max_dd_pct, trace


def report(label, t, final_eq, max_dd, start, n_days=7):
    acc = t[t.accepted]
    n_sig, n_acc = len(t), len(acc)
    wr = 100 * acc.is_win.mean() if n_acc else float("nan")
    w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    print(f"\n--- {label} ---")
    print(f"Signals: {n_sig:,}  Accepted: {n_acc:,} ({100*n_acc/max(n_sig,1):.1f}%)")
    print(f"Accepted WR={wr:.2f}%  PF={pf:.3f}  trades/day={n_acc/max(n_days,1):.1f}")
    print(f"Start ${start:.2f} -> End ${final_eq:,.2f} ({100*(final_eq/start-1):+.1f}%)  maxDD={max_dd:.1f}%")


def main():
    for inst in g20.TRADEABLE:
        g20.rs.load_instrument(inst)
        g20.rs._prep_zones(inst, g20.pg.MIN_RETRACEMENT)

    full = build_speed_king_full_with_side()
    data_end = full.entry_time.max()
    START = 100.0  # matches the $100/$100 sub-account funding actually in place now

    for label_suffix, days in [("7 days", 7), ("FULL HISTORY", None)]:
        if days is not None:
            window_start = data_end - pd.Timedelta(days=days)
            sub = full[(full.entry_time >= window_start) & (full.entry_time <= data_end)].copy()
        else:
            window_start = full.entry_time.min()
            sub = full
        n_days = (data_end - window_start).days
        print(f"\n{'='*90}\nWindow: {window_start} -> {data_end}  ({len(sub):,} signal candidates, "
              f"{label_suffix}, {n_days} days)\n{'='*90}")

        t1, eq1, dd1, _ = simulate_with_concurrency_rule(sub, start=START, global_max=3, per_side_max=None)
        report(f"CURRENT (deployed): margin + global cap=3, no side distinction [{n_days}d]",
               t1, eq1, dd1, START, n_days)

        t2, eq2, dd2, _ = simulate_with_concurrency_rule(sub, start=START, global_max=None, per_side_max=1)
        report(f"PROPOSED: margin + max 1 open PER SIDE (<=2 total, never 2 same-side) [{n_days}d]",
               t2, eq2, dd2, START, n_days)

    run_mc_comparison(full)


YEAR_NS = 365.25 * 86400 * 1e9


def mc_path_concurrency(entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r, is_buy,
                         risk_pct, rng, global_max, per_side_max, block=g20.BLOCK,
                         max_trades=300_000, start=100.0, max_leverage=g20.MAX_LEVERAGE,
                         cap=g20.CAP, target1=g20.T1, ruin_floor=g20.RUIN_FLOOR):
    """goal21's real-timing block-bootstrap MC path, extended with the same
    global/per-side headcount gate as simulate_with_concurrency_rule. Returns
    (years_to_1m, max_dd_pct, ruined, processed, accepted_n) -- stops the moment $1M is
    crossed or the account is ruined (this project's standard early-stop convention)."""
    n = len(entry_ns)
    equity = start
    peak = start
    dd_worst = 0.0
    margin_used = 0.0
    open_buy, open_sell = 0, 0
    heap = []  # (exit_time_ns, risk_dollars, margin, net_r, is_buy)
    clock = 0
    t1 = -1.0
    ruined = False
    processed = 0
    accepted_n = 0

    while processed < max_trades and t1 < 0 and not ruined:
        start_idx = rng.integers(0, n)
        end_idx = min(start_idx + block, n)
        offset = clock - entry_ns[start_idx]
        block_exit_max = clock

        for idx in range(start_idx, end_idx):
            et = entry_ns[idx] + offset
            xt = exit_ns[idx] + offset
            if xt > block_exit_max:
                block_exit_max = xt

            while heap and heap[0][0] <= et:
                _, r_dollars, margin, r, buy_side = heapq.heappop(heap)
                equity += r_dollars * r
                margin_used -= margin
                if buy_side:
                    open_buy -= 1
                else:
                    open_sell -= 1
                if equity > peak:
                    peak = equity
                dd = 100 * (equity - peak) / peak
                if dd < dd_worst:
                    dd_worst = dd
                if equity <= ruin_floor:
                    ruined = True
            margin_used = max(margin_used, 0.0)
            if ruined:
                break

            _, risk_dollars, margin = g20.position_size_and_margin(
                instrument[idx], equity, entry_price[idx], stop_risk[idx], risk_pct,
                max_leverage, cap)
            avail = min(equity, cap) if cap is not None else equity
            side_is_buy = bool(is_buy[idx])
            total_open = open_buy + open_sell
            fits_global = (global_max is None) or (total_open < global_max)
            side_open = open_buy if side_is_buy else open_sell
            fits_side = (per_side_max is None) or (side_open < per_side_max)

            if margin_used + margin <= avail and fits_global and fits_side:
                heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx], side_is_buy))
                margin_used += margin
                if side_is_buy:
                    open_buy += 1
                else:
                    open_sell += 1
                accepted_n += 1

            processed += 1
            if t1 < 0 and equity >= target1:
                t1 = et / YEAR_NS
                break
            if processed >= max_trades:
                break

        if ruined or t1 >= 0:
            break
        clock = block_exit_max

    return t1, dd_worst, ruined, processed, accepted_n


def run_mc_comparison(full, risk_pct=ROBUST_RISK_PCT, n_paths=500, start=100.0, seed=2000):
    full = full.sort_values("entry_time").reset_index(drop=True)
    entry_ns = full.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = full.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    instrument = full.instrument.values
    entry_price = full.entry_price.values
    stop_risk = full.stop_risk.values
    net_r = full.net_r.values
    is_buy = full.is_buy.values

    print(f"\n{'='*90}\nMONTE CARLO: quickest to $1M, ${start:.0f} start, {100*risk_pct:.1f}% risk, "
          f"{n_paths} paths per rule\n{'='*90}")

    for label, global_max, per_side_max in [
        ("CURRENT: global cap=3", 3, None),
        ("PROPOSED: max 1 per side", None, 1),
    ]:
        rng = np.random.default_rng(seed)
        t1s = np.empty(n_paths)
        dds = np.empty(n_paths)
        ruins = np.zeros(n_paths, dtype=bool)
        accept_rates = np.empty(n_paths)
        for p in range(n_paths):
            t1, dd, ruined, processed, accepted_n = mc_path_concurrency(
                entry_ns, exit_ns, instrument, entry_price, stop_risk, net_r, is_buy,
                risk_pct, rng, global_max, per_side_max, start=start)
            t1s[p] = t1
            dds[p] = dd
            ruins[p] = ruined
            accept_rates[p] = accepted_n / processed if processed else np.nan
        reached = t1s >= 0
        print(f"\n--- {label} ---")
        print(f"Reached $1M: {100*reached.mean():.1f}% of paths  |  Ruined: {100*ruins.mean():.1f}%")
        if reached.any():
            yrs = t1s[reached]
            print(f"Years to $1M: P5={np.percentile(yrs,5):.2f}  P50(median)={np.percentile(yrs,50):.2f}  "
                  f"P95={np.percentile(yrs,95):.2f}")
        print(f"Median max drawdown: {np.median(dds):.1f}%  |  Median accept rate: {100*np.nanmedian(accept_rates):.1f}%")


if __name__ == "__main__":
    main()
