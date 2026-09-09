"""BEID.md verification (2026-09-09).

FINDING (decisive, confirmed by direct instrumentation of the real
optimize_beid_timeframe_risk_touches_mc.py::backtest_beid_multitouch code -- not a
reimplementation): the pullback-tracking state machine has an unbounded wait with no
timeout. `in_pullback` only clears when price closes back past the frozen pHL/pLH extreme
(`cl < cLL` in a downtrend, `cl > cHH` in an uptrend). If price never revisits that exact
extreme again, `in_pullback` -- and the whole state machine -- freezes PERMANENTLY: no more
zones, no more trades, for the rest of the dataset, no matter how long it runs or how much
the market moves afterward. Verified directly:
  - Full EURUSD 2000-2026 (3-min bars): last zone/flip created 2002-05-29, in_pullback=True,
    cLL=0.9300 frozen forever after -- EURUSD never closes below 0.9300 again in the
    dataset (it ranged 0.93-1.60 for the next 24 years), so the machine locks up for good
    after ~2 years of a 26-year run.
  - Within the EXACT 500,000-bar (~16-month) window optimize_beid_timeframe_risk_touches_mc.py
    actually uses for BEID.md's Table 5.1: EURUSD is only active 2025-04-21->2025-05-20 (1
    of 16 months), GBPUSD 2025-04-16->2025-11-25 (7 of 16), EURGBP 2025-05-01->2025-05-30
    (1 of 16), and USDJPY 2025-04-22->2025-04-23 -- ONE DAY of an already-short window.
  - So BEID.md's "trades_per_year": 453.7 (and every downstream Monte Carlo number built
    on it) is computed as total_trades / the FULL window's span_years, while almost all of
    that span actually produced zero signal on 3 of 4 pairs. It is not a real, sustained
    trading rate; it's an artifact of dividing by the wrong denominator on top of a
    silently-broken generator.

This script: (1) fixes the bug with a pullback timeout (scaled by timeframe, matching the
project's own 48h zone-expiry convention rather than the original's un-scaled 2880-BAR
constant, which was itself a second, smaller bug -- 2880 bars is 48h at 1-minute but 6 DAYS
at 3-minute and 60 days at 30-minute); (2) re-runs Rank #1's exact config (3-min, 5 touches,
>=10p, 1.6p spread) on genuine FULL history, regime-split PRE-2021 (design) / POST-2026-04
(never-touched holdout) per goal108's boundaries -- not the pooled, mostly-frozen window;
(3) runs BOTH BEID's own simplistic single-curve i.i.d Monte Carlo AND this project's own
validated two-account, margin-and-concurrency-aware block-bootstrap engine
(goal79_two_account_portfolio.py), since BEID's MC models neither -- to see how much of the
gap is the data problem vs. the unrealistic execution model.
"""
import os
import sys
import time
import heapq
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from optimize_beid_timeframe_risk_touches_mc import resample_ohlc, get_pip_size, fast_monte_carlo
import goal79_two_account_portfolio as g79

PRE_END = pd.Timestamp("2021-01-01")
POST_START = pd.Timestamp("2026-04-01")

PATHS = {
    "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
    "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
    "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
    "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv",
}


def backtest_beid_fixed(df, ticker_name, timeframe_min, max_touches=5, spread_pips=1.6,
                         pullback_timeout_hours=48.0, zone_expiry_hours=48.0):
    """Same mechanism as the original backtest_beid_multitouch, with the lockup fixed:
    in_pullback now times out (scaled to the actual bar duration, like the zone-fill
    expiry already claimed to do but didn't -- 2880 was a bar count, not an hour count)."""
    pip_size = get_pip_size(ticker_name)
    time_col = "time" if "time" in df.columns else "date"
    dates = df[time_col].values
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    n = len(opens)

    pullback_timeout_bars = max(1, int(pullback_timeout_hours * 60 / timeframe_min))
    zone_expiry_bars = max(1, int(zone_expiry_hours * 60 / timeframe_min))

    trend = 1
    pHH = opens[0]; pHH_idx = 0; cHH = np.nan
    pHL = opens[0]; pHL_idx = 0; cHL = np.nan; last_cHL = np.nan
    pLL = opens[0]; pLL_idx = 0; cLL = np.nan
    pLH = opens[0]; pLH_idx = 0; cLH = np.nan; last_cLH = np.nan
    in_pullback = False
    pullback_started_idx = -1
    prev_green_open = np.nan
    prev_red_open = np.nan

    active_zones, trades = [], []
    max_active = 25

    for i in range(n):
        o, h, l, cl = opens[i], highs[i], lows[i], closes[i]
        is_green = cl >= o
        is_red = cl < o

        retained = []
        for z in active_zones:
            side, entry, tp, sl, half_pips = z["side"], z["entry"], z["tp"], z["sl"], z["half_pips"]
            if not z["filled"]:
                if side == 1:
                    if l <= sl:
                        continue
                    if l <= entry:
                        z["filled"] = True; z["fill_idx"] = i; z["touch_count"] += 1; retained.append(z)
                    elif i - z["created_idx"] <= zone_expiry_bars:
                        retained.append(z)
                else:
                    if h >= sl:
                        continue
                    if h >= entry:
                        z["filled"] = True; z["fill_idx"] = i; z["touch_count"] += 1; retained.append(z)
                    elif i - z["created_idx"] <= zone_expiry_bars:
                        retained.append(z)
            else:
                if i > z["fill_idx"]:
                    resolved, raw_result = False, ""
                    if side == 1:
                        hit_sl = (l <= sl) or (cl < sl)
                        hit_tp = (h >= tp) or (cl > tp)
                        if hit_sl and hit_tp:
                            raw_result = "WIN" if cl >= entry else "LOSS"; resolved = True
                        elif hit_tp:
                            raw_result = "WIN"; resolved = True
                        elif hit_sl:
                            raw_result = "LOSS"; resolved = True
                    else:
                        hit_sl = (h >= sl) or (cl > sl)
                        hit_tp = (l <= tp) or (cl < tp)
                        if hit_sl and hit_tp:
                            raw_result = "WIN" if cl <= entry else "LOSS"; resolved = True
                        elif hit_tp:
                            raw_result = "WIN"; resolved = True
                        elif hit_sl:
                            raw_result = "LOSS"; resolved = True
                    if resolved:
                        net_pips = (half_pips - spread_pips) if raw_result == "WIN" else -(half_pips + spread_pips)
                        net_r = net_pips / max(1e-4, half_pips)
                        trades.append(dict(
                            ticker=ticker_name, side="BUY" if side == 1 else "SELL", type=z["type"],
                            fill_time=dates[z["fill_idx"]], exit_time=dates[i],
                            entry=entry, tp=tp, sl=sl, zone_pips=z["zone_pips"], half_pips=half_pips,
                            touch_num=z["touch_count"], raw_result=raw_result, net_pips=net_pips, net_r=net_r,
                        ))
                        if raw_result == "WIN" and z["touch_count"] < max_touches:
                            z["filled"] = False; z["fill_idx"] = -1; z["created_idx"] = i; retained.append(z)
                    else:
                        retained.append(z)
        active_zones = retained

        # FIX: pullback timeout, scaled to this timeframe's actual bar duration. Resetting
        # only in_pullback isn't enough -- pLL/pHH (the primary extreme the NEXT pullback
        # attempt re-anchors from) stays frozen at the ancient value forever too (it only
        # advances via "if l<pLL" / "if h>pHH", which never fires again once price has
        # moved away from that extreme for good), so a bare in_pullback reset just lets the
        # very next bar restart an identical, equally-doomed pullback attempt with the same
        # stale reference -- an infinite retry loop instead of a one-time freeze. Confirmed
        # by direct instrumentation: EURUSD hit this reset 20+ times a few months apart
        # before still permanently locking up in 2002. Fix: abandon the stale swing
        # entirely on timeout and re-anchor from the current bar, same as giving up on a
        # zone that's expired unfilled.
        if in_pullback and (i - pullback_started_idx > pullback_timeout_bars):
            in_pullback = False
            if trend == 1:
                pHH = h; pHH_idx = i
            else:
                pLL = l; pLL_idx = i

        if trend == 1:
            if h > pHH or np.isnan(pHH):
                pHH = h; pHH_idx = i
            if not np.isnan(last_cHL) and cl < last_cHL:
                old_floor = last_cHL; top_peak = pHH
                trend = -1; cLH = top_peak; last_cLH = np.nan
                pLL = l; pLL_idx = i; cLL = np.nan; in_pullback = False
                top, bottom = top_peak, old_floor
                mid = (top + bottom) / 2.0
                zone_pips = (top - bottom) / pip_size; half_pips = zone_pips / 2.0
                if top > bottom and len(active_zones) < max_active:
                    active_zones.append(dict(side=-1, type="SELL_FLIP", top=top, bottom=bottom, entry=mid,
                                              tp=bottom, sl=top, zone_pips=zone_pips, half_pips=half_pips,
                                              created_idx=i, filled=False, fill_idx=-1, touch_count=0))
            else:
                if not in_pullback and is_red and not np.isnan(prev_green_open) and cl < prev_green_open:
                    in_pullback = True; pullback_started_idx = i
                    cHH = pHH; pHL = l; pHL_idx = i
                elif in_pullback:
                    if l < pHL:
                        pHL = l; pHL_idx = i
                    if cl > cHH:
                        in_pullback = False
                        cHL = pHL; last_cHL = cHL
                        top, bottom = cHH, cHL
                        mid = (top + bottom) / 2.0
                        zone_pips = (top - bottom) / pip_size; half_pips = zone_pips / 2.0
                        pHH = h; pHH_idx = i
                        if top > bottom and len(active_zones) < max_active:
                            active_zones.append(dict(side=1, type="BUY", top=top, bottom=bottom, entry=mid,
                                                      tp=top, sl=bottom, zone_pips=zone_pips, half_pips=half_pips,
                                                      created_idx=i, filled=False, fill_idx=-1, touch_count=0))
        elif trend == -1:
            if l < pLL or np.isnan(pLL):
                pLL = l; pLL_idx = i
            if not np.isnan(last_cLH) and cl > last_cLH:
                old_ceiling = last_cLH; bottom_trough = pLL
                trend = 1; cHL = bottom_trough; last_cHL = np.nan
                pHH = h; pHH_idx = i; cHH = np.nan; in_pullback = False
                top, bottom = old_ceiling, bottom_trough
                mid = (top + bottom) / 2.0
                zone_pips = (top - bottom) / pip_size; half_pips = zone_pips / 2.0
                if top > bottom and len(active_zones) < max_active:
                    active_zones.append(dict(side=1, type="BUY_FLIP", top=top, bottom=bottom, entry=mid,
                                              tp=top, sl=bottom, zone_pips=zone_pips, half_pips=half_pips,
                                              created_idx=i, filled=False, fill_idx=-1, touch_count=0))
            else:
                if not in_pullback and is_green and not np.isnan(prev_red_open) and cl > prev_red_open:
                    in_pullback = True; pullback_started_idx = i
                    cLL = pLL; pLH = h; pLH_idx = i
                elif in_pullback:
                    if h > pLH:
                        pLH = h; pLH_idx = i
                    if cl < cLL:
                        in_pullback = False
                        cLH = pLH; last_cLH = cLH
                        top, bottom = cLH, cLL
                        mid = (top + bottom) / 2.0
                        zone_pips = (top - bottom) / pip_size; half_pips = zone_pips / 2.0
                        pLL = l; pLL_idx = i
                        if top > bottom and len(active_zones) < max_active:
                            active_zones.append(dict(side=-1, type="SELL", top=top, bottom=bottom, entry=mid,
                                                      tp=bottom, sl=top, zone_pips=zone_pips, half_pips=half_pips,
                                                      created_idx=i, filled=False, fill_idx=-1, touch_count=0))
        if is_green:
            prev_green_open = o
        elif is_red:
            prev_red_open = o

    return pd.DataFrame(trades)


def run_two_account_mc(trades_sub, risk_pct, n_sims=1000, seed=42):
    """Proper concurrency/margin-aware MC, reusing this project's own validated engine
    instead of BEID's single-curve i.i.d bootstrap. Adapts BEID trade columns
    (entry/sl/half_pips) into the shape simulate_two_account/position_size_and_margin
    expect (entry_price/stop_risk/is_buy/instrument)."""
    t = trades_sub.copy()
    t["entry_price"] = t["entry"]
    t["stop_risk"] = (t["entry"] - t["sl"]).abs()
    t["is_buy"] = t["side"] == "BUY"
    t["instrument"] = t["ticker"]
    t["entry_time"] = pd.to_datetime(t["fill_time"])
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t = t.sort_values("entry_time").reset_index(drop=True)

    n = len(t)
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values
    is_buy = t.is_buy.values

    START_PER_SIDE, MAX_LEVERAGE, ACCOUNT_CAP, RUIN_FLOOR = 100.0, 50.0, 10_000_000.0, 20.0
    BLOCK = 50
    rng = np.random.default_rng(seed)
    days_to_target, max_dds, ruin_count = [], [], 0

    for sim_i in range(n_sims):
        equity = {True: START_PER_SIDE, False: START_PER_SIDE}
        margin_used = {True: 0.0, False: 0.0}
        open_count = {True: 0, False: 0}
        heap = []
        clock = 0
        peak = START_PER_SIDE * 2.0
        worst_dd = 0.0
        ruined = False
        hit_target = False
        t_hit_days = -1.0
        trades_count = 0

        while trades_count < 50_000 and not hit_target and not ruined:
            start_idx = rng.integers(0, n)
            end_idx = min(start_idx + BLOCK, n)
            offset = clock - entry_ns[start_idx]
            block_exit_max = clock
            for idx in range(start_idx, end_idx):
                et = entry_ns[idx] + offset
                xt = exit_ns[idx] + offset
                if xt > block_exit_max:
                    block_exit_max = xt
                while heap and heap[0][0] <= et:
                    _, r_dollars, margin, r, side = heapq.heappop(heap)
                    equity[side] += r_dollars * r
                    margin_used[side] -= margin
                    open_count[side] -= 1
                    comb = equity[True] + equity[False]
                    if comb > peak:
                        peak = comb
                    dd = 100.0 * (comb - peak) / peak
                    if dd < worst_dd:
                        worst_dd = dd
                    if comb <= RUIN_FLOOR or equity[True] <= 1.0 or equity[False] <= 1.0:
                        ruined = True
                if ruined:
                    break
                side = bool(is_buy[idx])
                sizing_base = min(equity[True], equity[False])
                _, risk_dollars, margin = g79.position_size_and_margin(
                    instrument[idx], sizing_base, entry_price[idx], stop_risk[idx], risk_pct,
                    MAX_LEVERAGE, ACCOUNT_CAP)
                avail = min(equity[side], ACCOUNT_CAP) - margin_used[side]
                if margin <= avail and open_count[side] < 1:
                    heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx], side))
                    margin_used[side] += margin
                    open_count[side] += 1
                    trades_count += 1
                comb = equity[True] + equity[False]
                if comb >= 1_000_000.0:
                    hit_target = True
                    t_hit_days = et / 86400e9
                    break
            clock = block_exit_max
        if hit_target:
            days_to_target.append(t_hit_days)
        elif ruined:
            ruin_count += 1
        max_dds.append(worst_dd)

    med_days = float(np.median(days_to_target)) if days_to_target else float("nan")
    prob = 100.0 * len(days_to_target) / n_sims
    med_dd = float(np.median(max_dds))
    ruin_rate = 100.0 * ruin_count / n_sims
    return med_days, prob, med_dd, ruin_rate


def main():
    t0 = time.time()
    TF = 3
    TOUCHES = 5
    MIN_PIP = 10.0
    SPREAD = 1.6
    RISK = 0.06

    all_trades = []
    for name, path in PATHS.items():
        tt0 = time.time()
        df = pd.read_csv(path)
        df3 = resample_ohlc(df, TF)
        tdf = backtest_beid_fixed(df3, name, timeframe_min=TF, max_touches=TOUCHES, spread_pips=SPREAD)
        all_trades.append(tdf)
        print(f"  {name}: {len(df3):,} {TF}-min bars -> {len(tdf):,} trades  ({time.time()-tt0:.1f}s)")

    trades = pd.concat(all_trades, ignore_index=True)
    trades["fill_time"] = pd.to_datetime(trades["fill_time"])
    trades = trades.sort_values("fill_time").reset_index(drop=True)
    print(f"\nTotal trades, all pairs, unfiltered: {len(trades):,}")
    print(f"Active date range: {trades['fill_time'].min()} -> {trades['fill_time'].max()}")

    sub = trades[trades["zone_pips"] >= MIN_PIP].reset_index(drop=True)
    print(f"Trades with zone_pips >= {MIN_PIP}: {len(sub):,}")

    pre = sub[sub["fill_time"] < PRE_END]
    post = sub[sub["fill_time"] >= POST_START]
    print(f"\nPRE-2021 (<2021-01-01): {len(pre):,} trades")
    print(f"POST-2026-04 (>=2026-04-01, genuine holdout): {len(post):,} trades")

    def stats(d):
        n = len(d)
        if n == 0:
            return None
        wr = 100.0 * (d["raw_result"] == "WIN").mean()
        gw = d.loc[d["net_pips"] > 0, "net_pips"].sum()
        gl = -d.loc[d["net_pips"] <= 0, "net_pips"].sum()
        pf = gw / gl if gl > 0 else float("inf")
        return dict(n=n, wr=wr, pf=pf)

    for label, d in (("PRE-2021", pre), ("POST-2026-04", post), ("FULL (fixed engine)", sub)):
        s = stats(d)
        if s:
            span_years = max(1e-6, (d["fill_time"].max() - d["fill_time"].min()).total_seconds() / (365.25*86400))
            print(f"  {label}: n={s['n']:,} WR={s['wr']:.1f}% PF={s['pf']:.3f} trades/yr={s['n']/span_years:.0f} (active span {span_years:.2f}yr)")

    pre_csv = os.path.join(SCRIPT_DIR, "..", "research_results", "beid_fixed_pre2021_trades.csv")
    sub.to_csv(pre_csv, index=False)
    print(f"\nSaved all >= {MIN_PIP}p trades -> {pre_csv}")

    print(f"\n{'='*100}\nMonte Carlo comparison at {RISK*100:.0f}% risk, PRE-2021 trades only (n={len(pre):,})\n{'='*100}")
    if len(pre) >= 30:
        span_years_pre = max(1e-6, (pre["fill_time"].max() - pre["fill_time"].min()).total_seconds() / (365.25*86400))
        tpy_pre = len(pre) / span_years_pre
        med_days_naive, prob_naive, dd_naive, ruin_naive = fast_monte_carlo(
            pre["net_r"].values, tpy_pre, risk_pct=RISK, start_equity=200.0, target=1_000_000.0, n_sims=1000)
        print(f"BEID's own single-curve i.i.d MC (no concurrency/margin model): "
              f"median {med_days_naive:.0f}d ({med_days_naive/30.42:.1f}mo), prob={prob_naive:.1f}%, "
              f"medDD={dd_naive:.1f}%, ruin={ruin_naive:.1f}%")

        med_days_real, prob_real, dd_real, ruin_real = run_two_account_mc(pre, risk_pct=RISK, n_sims=1000)
        if not np.isnan(med_days_real):
            print(f"Proper two-account, margin+concurrency-aware MC (goal79 engine): "
                  f"median {med_days_real:.0f}d ({med_days_real/30.42:.1f}mo), prob={prob_real:.1f}%, "
                  f"medDD={dd_real:.1f}%, ruin={ruin_real:.1f}%")
        else:
            print(f"Proper two-account MC: never reached $1M in {1000} sims (prob={prob_real:.1f}%, ruin={ruin_real:.1f}%, medDD={dd_real:.1f}%)")
    else:
        print("Not enough PRE-2021 trades for a meaningful MC.")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
