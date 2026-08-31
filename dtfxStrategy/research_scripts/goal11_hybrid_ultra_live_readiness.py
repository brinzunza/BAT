"""Rigorous live-trading-readiness audit for HYBRID-ULTRA -- the #1 pick in
STRATEGY_RANKINGS.md v8 (N=6.0 stop, R:R=0.4, 2.0h fill-age, unconditional re-arm,
touches<=4, no session filter, tradeable 4-pair universe).

Every prior validation on this strategy was a SINGLE 50/50 split plus a single coarse
(N,R:R) grid. Before recommending this for real deployment, this script runs 8
independent checks a single train/test split cannot catch:

  0. CODE AUDIT: confirms and quantifies a real bug found while starting this audit --
     `_simulate_generic`'s `entry_idx` output is actually the RESOLUTION (exit) bar,
     not the entry bar, mislabeled as "entry_time" everywhere downstream. Quantifies
     how much this actually matters for Hybrid-Ultra specifically (no session filter,
     so hour-bucket effects don't apply -- but the walk-forward split boundary and
     MC/Sharpe chronological ordering both use this mislabeled timestamp).
  1. FINER (N, R:R) GRID: the original grid capped at N=6 (the edge of the search
     space) -- extends N upward and refines R:R to check whether N=6/R:R=0.4 is a
     genuine local peak or the search just stopped at the wall.
  2. MULTI-FOLD WALK-FORWARD: one 50/50 split is one data point. Splits history into
     4 chronological quarters and re-selects/re-tests on 3 independent expanding-window
     folds instead of just one.
  3. PARAMETER SENSITIVITY: +/-10/20/30% perturbations of N and R:R independently,
     checked on the TRUE holdout, to find cliffs vs. plateaus near the chosen point.
  4. YEAR-BY-YEAR BREAKDOWN: no session filter here, so this is a clean read on
     whether the edge is concentrated in any one regime/year.
  5. REAL DRAWDOWN / LOSING-STREAK ANALYSIS: deterministic $200@5%-risk equity curve
     max drawdown, longest losing streak, and time-underwater -- not just the MC
     bootstrap's ruin floor.
  6. SPREAD/SLIPPAGE STRESS TEST: R:R=0.4 sits closer to the failure cliff than any
     other candidate in the file -- finds the breakeven spread multiplier where PF
     crosses 1.0.
  7. TRADE CONCURRENCY: with ~4,100 trades/yr across 4 pairs, checks how many
     positions are realistically open at once -- relevant to whether "5% risk per
     trade" is also secretly "5% x N concurrent" in practice.
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal5_rr_sweep as g5
import goal10_ultra_family as g10
import prod_filter_grid_search as pg
import dtfx_robust_search as rs

TRADEABLE = g5.TRADEABLE
SPLIT = pd.Timestamp("2018-03-09")
HU_KW = dict(age_h=2.0, require_bounce=False, touches=4)  # Hybrid-Ultra's fixed axes
HU_N, HU_RR = 6.0, 0.4


def warm_up():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)


# ============================================================================
# 0. CODE AUDIT -- quantify the entry_idx/exit_idx mislabeling
# ============================================================================

@njit(cache=True)
def _simulate_with_true_entry(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                               zone_created_ns, times_ns, lows, highs, min_fill_age_ns,
                               stop_mult, target_mult, require_bounce, probe_depth, exc_mult,
                               max_touches, max_window):
    """Exact copy of prod_filter_grid_search._simulate_generic, PLUS an extra output
    array recording the TRUE entry bar index (when in_trade first became True for that
    touch) alongside the existing (mislabeled) resolution-bar index."""
    n = len(zone_top)
    n_times = len(lows)
    max_out = n * max_touches
    out_touch = np.zeros(max_out, dtype=np.int64)
    out_resolve_idx = np.full(max_out, -1, dtype=np.int64)
    out_true_entry_idx = np.full(max_out, -1, dtype=np.int64)
    out_is_win = np.zeros(max_out, dtype=np.bool_)
    out_stop_risk = np.zeros(max_out, dtype=np.float64)
    out_count = 0

    for i in range(n):
        top, bottom, mid = zone_top[i], zone_bottom[i], zone_mid[i]
        is_buy = zone_is_buy[i]
        zone_width = top - bottom
        half_width = zone_width / 2.0
        stop_risk = stop_mult * half_width
        target_risk = target_mult * half_width
        invalidation = bottom if is_buy else top
        probe_lvl = (top - probe_depth * zone_width) if is_buy else (bottom + probe_depth * zone_width)
        exc_target = (top + exc_mult * zone_width) if is_buy else (bottom - exc_mult * zone_width)
        stop_level = mid - stop_risk if is_buy else mid + stop_risk
        target_level = mid + target_risk if is_buy else mid - target_risk
        created_ns = zone_created_ns[i]

        j = zone_start_idx[i]
        if j >= n_times:
            continue
        window_end = min(j + max_window, n_times)

        zone_dead = False
        in_trade = False
        touch_count = 0
        in_probe = False
        had_bounce = False
        cur_entry_j = -1

        while j < window_end and touch_count < max_touches:
            lo = lows[j]
            hi = highs[j]

            if not zone_dead:
                breached = (lo <= invalidation) if is_buy else (hi >= invalidation)
                if breached:
                    zone_dead = True
                elif require_bounce:
                    probed = (lo <= probe_lvl) if is_buy else (hi >= probe_lvl)
                    if probed:
                        in_probe = True
                    if in_probe:
                        bounced = (hi >= exc_target) if is_buy else (lo <= exc_target)
                        if bounced:
                            had_bounce = True

            if in_trade:
                tp_hit = (hi >= target_level) if is_buy else (lo <= target_level)
                sl_hit = (lo <= stop_level) if is_buy else (hi >= stop_level)
                if tp_hit or sl_hit:
                    is_win = tp_hit and not sl_hit
                    out_touch[out_count] = touch_count
                    out_resolve_idx[out_count] = j
                    out_true_entry_idx[out_count] = cur_entry_j
                    out_is_win[out_count] = is_win
                    out_stop_risk[out_count] = stop_risk
                    out_count += 1
                    in_trade = False
                    if is_win and touch_count < max_touches and not zone_dead:
                        in_probe = False
                        had_bounce = False
                    else:
                        break
            else:
                if zone_dead:
                    break
                touched_mid = (lo <= mid) if is_buy else (hi >= mid)
                if touch_count == 0:
                    if touched_mid:
                        age_ns = times_ns[j] - created_ns
                        if age_ns < min_fill_age_ns:
                            break
                        in_trade = True
                        touch_count = 1
                        cur_entry_j = j
                else:
                    if (not require_bounce) or had_bounce:
                        if touched_mid:
                            in_trade = True
                            touch_count += 1
                            cur_entry_j = j

            j += 1

    return (out_touch[:out_count], out_true_entry_idx[:out_count], out_resolve_idx[:out_count],
            out_is_win[:out_count], out_stop_risk[:out_count])


def build_hybrid_ultra_with_true_times(n=HU_N, rr=HU_RR):
    """Rebuild Hybrid-Ultra's trade log with BOTH true entry_time and true exit_time,
    to quantify how much the entry/exit mislabeling actually matters."""
    frames = []
    for inst in TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (n * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        touch_num, true_entry_idx, resolve_idx, is_win, stop_risk = _simulate_with_true_entry(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(HU_KW["age_h"] * 3600 * 1e9), n, n * rr, HU_KW["require_bounce"],
            pg.PROBE_DEPTH, 1.0, HU_KW["touches"], pg.MAX_WINDOW_BARS)
        pip = pg.PIP_SIZE[inst]; spread_price = pg.SPREAD_PIPS_TYPICAL * pip
        r_gross = np.where(is_win, rr, -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        frames.append(pd.DataFrame({
            "true_entry_time": pd.to_datetime(times[true_entry_idx]),
            "exit_time": pd.to_datetime(times[resolve_idx]),
            "mislabeled_entry_time": pd.to_datetime(times[resolve_idx]),  # what every prior script called "entry_time"
            "net_r": net_r, "is_win": is_win, "instrument": inst,
        }))
    return pd.concat(frames, ignore_index=True).sort_values("mislabeled_entry_time").reset_index(drop=True)


def test0_code_audit():
    print("\n" + "=" * 90)
    print("TEST 0: entry_idx/exit_idx mislabeling -- quantify impact on Hybrid-Ultra")
    print("=" * 90)
    t = build_hybrid_ultra_with_true_times()
    gap_min = (t.exit_time - t.true_entry_time).dt.total_seconds() / 60.0
    print(f"  Trade resolution delay (true entry -> exit), n={len(t):,}:")
    print(f"    median={gap_min.median():.1f} min, mean={gap_min.mean():.1f} min, "
          f"p90={gap_min.quantile(0.9):.1f} min, max={gap_min.max():.1f} min")

    same_hour = (t.true_entry_time.dt.hour == t.exit_time.dt.hour).mean()
    same_day = (t.true_entry_time.dt.date == t.exit_time.dt.date).mean()
    print(f"    same UTC hour: {100*same_hour:.1f}%   same calendar day: {100*same_day:.1f}%")
    print("    (Hybrid-Ultra has NO session filter, so hour-bucket mislabeling doesn't")
    print("     change *which* trades are counted -- but it does apply to every other")
    print("     session-filtered row in STRATEGY_RANKINGS.md: SOMEGA, SOMEGA Robust,")
    print("     DTFX Robust, CANA+, goal70_final_winner all filter populations by this")
    print("     same mislabeled 'entry_hour', which is really the exit hour.)")

    # Does the walk-forward split boundary reclassify any trades?
    true_first = t[t.true_entry_time < SPLIT]
    true_second = t[t.true_entry_time >= SPLIT]
    mis_first = t[t.mislabeled_entry_time < SPLIT]
    mis_second = t[t.mislabeled_entry_time >= SPLIT]
    flipped = (t.true_entry_time < SPLIT) != (t.mislabeled_entry_time < SPLIT)
    print(f"\n  TRUE-holdout split reclassification at {SPLIT.date()}:")
    print(f"    trades that flip sides of the split (true vs mislabeled): {flipped.sum()} of {len(t):,} "
          f"({100*flipped.sum()/len(t):.3f}%)")
    print(f"    first-half size:  true={len(true_first):,}  mislabeled={len(mis_first):,}")
    print(f"    second-half size: true={len(true_second):,}  mislabeled={len(mis_second):,}")

    def summarize(sub, label):
        yrs = (sub.iloc[:, 0].max() - sub.iloc[:, 0].min()).days / 365.25 if len(sub) > 1 else float("nan")
        wr = 100 * sub.is_win.mean()
        w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        print(f"    {label}: n={len(sub):,} ({len(sub)/yrs:,.0f}/yr) WR={wr:.2f}% PF={pf:.3f}")

    print("\n  Second-half (holdout) scored with TRUE entry_time vs the mislabeled one used everywhere else:")
    summarize(true_second[["true_entry_time", "is_win", "net_r"]], "true entry_time   ")
    summarize(mis_second[["mislabeled_entry_time", "is_win", "net_r"]].rename(
        columns={"mislabeled_entry_time": "true_entry_time"}), "mislabeled (as reported)")
    return t


# ============================================================================
# 1. FINER (N, R:R) GRID -- was N=6/R:R=0.4 a genuine peak or the search wall?
# ============================================================================

def test1_finer_grid():
    print("\n" + "=" * 90)
    print("TEST 1: finer (N, R:R) grid -- extend past the original N<=6 wall")
    print("=" * 90)
    N_GRID = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
    RR_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    rows = []
    for n in N_GRID:
        for rr in RR_GRID:
            trades = g10.build(n, rr, **HU_KW)
            first = trades[trades.entry_time < SPLIT]
            if len(first) < 100:
                continue
            glog = np.log1p(g5.RISK_FRAC * first.net_r)
            years = (first.entry_time.max() - first.entry_time.min()).days / 365.25
            g_score = (len(first) / years) * glog.mean()
            rows.append(dict(n=n, rr=rr, g_first_half=g_score))
    df = pd.DataFrame(rows).sort_values("g_first_half", ascending=False)
    print(df.head(10).to_string(index=False))
    best = df.iloc[0]
    print(f"\n  Best in finer grid: N={best.n}, R:R={best.rr} (g={best.g_first_half:.3f})")
    print(f"  Original v8 pick:   N={HU_N}, R:R={HU_RR}")
    if best.n != HU_N or best.rr != HU_RR:
        print("  *** The published (N,R:R) was NOT the true local optimum on this finer grid. ***")
    else:
        print("  Confirmed: N=6.0/R:R=0.4 still wins even on a finer, wider grid -- genuine local peak.")
    df.to_csv(os.path.join(DATA_DIR, "goal11_finer_grid.csv"), index=False)
    return df


# ============================================================================
# 2. MULTI-FOLD WALK-FORWARD
# ============================================================================

def test2_multifold_wf():
    print("\n" + "=" * 90)
    print("TEST 2: multi-fold expanding-window walk-forward (not just one 50/50 split)")
    print("=" * 90)
    N_GRID = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    RR_GRID = [0.35, 0.4, 0.45, 0.5, 0.6, 0.7]

    full_all_rr1 = g10.build(HU_N, 1.0, **HU_KW)  # cheap call just to get full time range
    tmin, tmax = full_all_rr1.entry_time.min(), full_all_rr1.entry_time.max()
    total_days = (tmax - tmin).days
    bounds = [tmin + pd.Timedelta(days=int(total_days * f)) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    print(f"  History: {tmin.date()} -> {tmax.date()}  ({total_days/365.25:.1f} yrs), split into 4 quarters")

    results = []
    for k in range(1, 4):  # test folds = quarters 2,3,4 (fold 0 has no prior data to select on)
        train_end = bounds[k]
        test_start, test_end = bounds[k], bounds[k + 1]
        best = (None, None, -1e9)
        for n in N_GRID:
            for rr in RR_GRID:
                trades = g10.build(n, rr, **HU_KW)
                train = trades[trades.entry_time < train_end]
                if len(train) < 200:
                    continue
                glog = np.log1p(g5.RISK_FRAC * train.net_r)
                years = (train.entry_time.max() - train.entry_time.min()).days / 365.25
                g_score = (len(train) / years) * glog.mean()
                if g_score > best[2]:
                    best = (n, rr, g_score)
        n_sel, rr_sel, _ = best
        full = g10.build(n_sel, rr_sel, **HU_KW)
        test = full[(full.entry_time >= test_start) & (full.entry_time < test_end)]
        yrs = (test_end - test_start).days / 365.25
        wr = 100 * test.is_win.mean()
        w = test.loc[test.net_r > 0, "net_r"].sum(); l = -test.loc[test.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        sharpe = g5.weekly_sharpe(test) if len(test) > 20 else float("nan")
        row = dict(fold=k, test_start=test_start.date(), test_end=test_end.date(),
                   n_sel=n_sel, rr_sel=rr_sel, trades=len(test), tpy=len(test) / yrs, wr=wr, pf=pf, sharpe=sharpe)
        results.append(row)
        print(f"  Fold {k} [{test_start.date()} -> {test_end.date()}]: selected N={n_sel}, R:R={rr_sel} on prior data -> "
              f"OOS: n={len(test):,} ({len(test)/yrs:,.0f}/yr) WR={wr:.2f}% PF={pf:.3f} Sharpe={sharpe:.2f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, "goal11_multifold_wf.csv"), index=False)
    all_positive_pf = (df.pf > 1.0).all()
    print(f"\n  All {len(df)} OOS folds profitable (PF>1.0)? {'YES' if all_positive_pf else 'NO -- see above'}")
    print(f"  Selected N stable across folds? {sorted(df.n_sel.unique())}")
    print(f"  Selected R:R stable across folds? {sorted(df.rr_sel.unique())}")
    return df


# ============================================================================
# 3. PARAMETER SENSITIVITY
# ============================================================================

def test3_param_sensitivity():
    print("\n" + "=" * 90)
    print("TEST 3: parameter sensitivity (+/-10/20/30%) around N=6.0, R:R=0.4, on TRUE holdout")
    print("=" * 90)

    def score_holdout(n, rr):
        trades = g10.build(n, rr, **HU_KW)
        second = trades[trades.entry_time >= SPLIT]
        yrs = (second.entry_time.max() - second.entry_time.min()).days / 365.25
        wr = 100 * second.is_win.mean()
        w = second.loc[second.net_r > 0, "net_r"].sum(); l = -second.loc[second.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        sharpe = g5.weekly_sharpe(second)
        return dict(tpy=len(second) / yrs, wr=wr, pf=pf, sharpe=sharpe)

    print("\n  A. N sensitivity (R:R fixed at 0.4):")
    rows_n = []
    for pct in [-30, -20, -10, 0, 10, 20, 30]:
        n = HU_N * (1 + pct / 100)
        r = score_holdout(n, HU_RR)
        rows_n.append(dict(pct=pct, n=n, **r))
        tag = " <- baseline" if pct == 0 else ""
        print(f"    {pct:+4d}%  N={n:5.2f}  tpy={r['tpy']:6.0f}  WR={r['wr']:5.2f}%  PF={r['pf']:.3f}  Sharpe={r['sharpe']:5.2f}{tag}")

    print("\n  B. R:R sensitivity (N fixed at 6.0):")
    rows_rr = []
    for pct in [-30, -20, -10, 0, 10, 20, 30]:
        rr = HU_RR * (1 + pct / 100)
        r = score_holdout(HU_N, rr)
        rows_rr.append(dict(pct=pct, rr=rr, **r))
        tag = " <- baseline" if pct == 0 else ""
        print(f"    {pct:+4d}%  R:R={rr:5.3f}  tpy={r['tpy']:6.0f}  WR={r['wr']:5.2f}%  PF={r['pf']:.3f}  Sharpe={r['sharpe']:5.2f}{tag}")

    pd.DataFrame(rows_n).to_csv(os.path.join(DATA_DIR, "goal11_sensitivity_n.csv"), index=False)
    pd.DataFrame(rows_rr).to_csv(os.path.join(DATA_DIR, "goal11_sensitivity_rr.csv"), index=False)

    pf_n = pd.DataFrame(rows_n).pf
    pf_rr = pd.DataFrame(rows_rr).pf
    cliff_n = (pf_n < 1.0).any()
    cliff_rr = (pf_rr < 1.0).any()
    print(f"\n  PF ever drops below 1.0 within +/-30% of N?    {'YES -- cliff nearby' if cliff_n else 'No -- plateau'}")
    print(f"  PF ever drops below 1.0 within +/-30% of R:R?  {'YES -- cliff nearby' if cliff_rr else 'No -- plateau'}")
    return rows_n, rows_rr


# ============================================================================
# 4. YEAR-BY-YEAR BREAKDOWN
# ============================================================================

def test4_year_by_year(full_trades):
    print("\n" + "=" * 90)
    print("TEST 4: year-by-year breakdown (no session filter -> clean regime read)")
    print("=" * 90)
    t = full_trades.copy()
    t["year"] = t.entry_time.dt.year
    rows = []
    for yr, sub in t.groupby("year"):
        if len(sub) < 20:
            continue
        wr = 100 * sub.is_win.mean()
        w = sub.loc[sub.net_r > 0, "net_r"].sum(); l = -sub.loc[sub.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        total_r = sub.net_r.sum()
        rows.append(dict(year=yr, trades=len(sub), wr=wr, pf=pf, total_r=total_r))
        print(f"    {yr}: n={len(sub):5,}  WR={wr:5.2f}%  PF={pf:.3f}  sum(net_r)={total_r:+8.1f}R")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal11_year_by_year.csv"), index=False)
    losing_years = df[df.pf < 1.0]
    print(f"\n  Years with PF<1.0: {len(losing_years)} of {len(df)} "
          f"({', '.join(str(y) for y in losing_years.year) if len(losing_years) else 'none'})")
    print(f"  PF range across years: {df.pf.min():.3f} - {df.pf.max():.3f}")
    return df


# ============================================================================
# 5. REAL DRAWDOWN / LOSING-STREAK ANALYSIS
# ============================================================================

def test5_drawdown(full_trades):
    print("\n" + "=" * 90)
    print("TEST 5: deterministic $200@5%-risk equity curve -- real drawdown & streaks")
    print("=" * 90)
    t = full_trades.sort_values("entry_time").reset_index(drop=True)
    eq = np.empty(len(t) + 1)
    eq[0] = g5.START
    for i, r in enumerate(t.net_r.values):
        risk = eq[i] * g5.RISK_FRAC
        eq[i + 1] = max(eq[i] + risk * r, 0.01)
    equity = eq[1:]
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    max_dd = dd.min()
    max_dd_idx = dd.argmin()

    # longest losing streak (consecutive losers)
    is_loss = (t.net_r.values <= 0).astype(int)
    streaks = []
    cur = 0
    for v in is_loss:
        if v:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    longest_streak = max(streaks) if streaks else 0

    # time underwater from the worst drawdown point: bars until equity re-exceeds prior peak
    peak_before = running_max[max_dd_idx]
    recovered_idx = None
    for i in range(max_dd_idx, len(equity)):
        if equity[i] >= peak_before:
            recovered_idx = i
            break
    if recovered_idx is not None:
        recovery_time = t.entry_time.iloc[recovered_idx] - t.entry_time.iloc[max_dd_idx]
    else:
        recovery_time = None

    print(f"  Deterministic path (no MC, single real historical sequence, $200 start, 5% risk/trade):")
    print(f"    Final equity after {len(t):,} trades: ${equity[-1]:,.0f}")
    print(f"    Max drawdown: {100*max_dd:.1f}% (at trade #{max_dd_idx:,}, {t.entry_time.iloc[max_dd_idx].date()})")
    print(f"    Longest losing streak: {longest_streak} consecutive losses")
    worst_streak_dd = 1 - (1 - g5.RISK_FRAC * 1.0) ** longest_streak
    print(f"    Equity impact of a {longest_streak}-loss streak alone, from any peak: "
          f"~{100*worst_streak_dd:.1f}% (compounding 5% losses back-to-back)")
    if recovery_time is not None:
        print(f"    Time to recover from worst drawdown: {recovery_time.days} days "
              f"({recovery_time.days/365.25:.2f} yrs)")
    else:
        print(f"    Never recovered to prior peak before end of data (ends at trade #{len(t)-1:,})")

    print(f"\n  NOTE: since R:R=0.4, a loss costs -1.0R + spread while a win pays only +0.4R --")
    print(f"  {longest_streak} straight losses need roughly {longest_streak/0.4:.0f} straight wins to fully")
    print(f"  recover in R-terms (before compounding effects), which is exactly the mechanism")
    print(f"  the file's own caveat about R:R<1 already flags.")
    return dict(max_dd_pct=100 * max_dd, longest_streak=longest_streak, final_equity=equity[-1])


# ============================================================================
# 6. SPREAD / SLIPPAGE STRESS TEST
# ============================================================================

def build_hybrid_ultra_spread_mult(spread_mult):
    frames = []
    for inst in TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (HU_N * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        touch_num, entry_idx, is_win, entry_mid, stop_risk = pg._simulate_generic(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(HU_KW["age_h"] * 3600 * 1e9), HU_N, HU_N * HU_RR, HU_KW["require_bounce"],
            pg.PROBE_DEPTH, 1.0, HU_KW["touches"], pg.MAX_WINDOW_BARS)
        pip = pg.PIP_SIZE[inst]
        spread_price = pg.SPREAD_PIPS_TYPICAL * spread_mult * pip
        r_gross = np.where(is_win, HU_RR, -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        frames.append(pd.DataFrame({"entry_time": pd.to_datetime(times[entry_idx]),
                                     "net_r": net_r, "is_win": is_win}))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def test6_spread_stress():
    print("\n" + "=" * 90)
    print("TEST 6: spread/slippage stress test -- how far can real-world costs drift?")
    print("=" * 90)
    print(f"  Baseline model assumes {pg.SPREAD_PIPS_TYPICAL} pips, charged only on stop-outs.")
    rows = []
    for mult in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        trades = build_hybrid_ultra_spread_mult(mult)
        wr = 100 * trades.is_win.mean()
        w = trades.loc[trades.net_r > 0, "net_r"].sum(); l = -trades.loc[trades.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        rows.append(dict(spread_mult=mult, spread_pips=pg.SPREAD_PIPS_TYPICAL * mult, wr=wr, pf=pf))
        print(f"    spread x{mult:.1f} ({pg.SPREAD_PIPS_TYPICAL*mult:.1f} pips): WR={wr:.2f}%  PF={pf:.3f}"
              f"{'  <- BREAKS EVEN OR WORSE' if pf <= 1.0 else ''}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA_DIR, "goal11_spread_stress.csv"), index=False)
    breakeven = df[df.pf <= 1.0]
    if len(breakeven):
        print(f"\n  Breakeven spread: between {df[df.pf>1.0].spread_pips.max():.1f} and "
              f"{breakeven.spread_pips.min():.1f} pips "
              f"({breakeven.spread_mult.min():.1f}x - {df[df.pf>1.0].spread_mult.max():.1f}x the modeled 1.6 pips)")
    else:
        print(f"\n  PF stays above 1.0 even at {df.spread_mult.max():.1f}x modeled spread "
              f"({df.spread_pips.max():.1f} pips) -- did not find breakeven in this range.")
    return df


# ============================================================================
# 7. TRADE CONCURRENCY
# ============================================================================

def test7_concurrency(t_true):
    print("\n" + "=" * 90)
    print("TEST 7: trade concurrency -- how many positions open at once?")
    print("=" * 90)
    t = t_true.sort_values("true_entry_time").reset_index(drop=True)
    starts = t.true_entry_time.values.astype("datetime64[ns]")
    ends = t.exit_time.values.astype("datetime64[ns]")
    order = np.argsort(starts)
    starts_s = starts[order]
    ends_s = ends[order]

    # sweep-line: events (+1 at start, -1 at end), max concurrent
    events = np.concatenate([starts_s, ends_s])
    deltas = np.concatenate([np.ones(len(starts_s), dtype=int), -np.ones(len(ends_s), dtype=int)])
    ev_order = np.argsort(events, kind="stable")
    events_sorted = events[ev_order]
    deltas_sorted = deltas[ev_order]
    concurrent = np.cumsum(deltas_sorted)
    max_concurrent = concurrent.max()

    # time-weighted average concurrency
    dt_seconds = np.diff(events_sorted).astype("timedelta64[s]").astype(float)
    avg_concurrent = np.average(concurrent[:-1], weights=np.maximum(dt_seconds, 0)) if len(dt_seconds) else 0

    dur_min = (ends_s - starts_s).astype("timedelta64[s]").astype(float) / 60.0
    print(f"  Trade duration (true entry -> exit): median={np.median(dur_min):.1f} min, "
          f"mean={np.mean(dur_min):.1f} min, p90={np.percentile(dur_min, 90):.1f} min, "
          f"max={np.max(dur_min):.1f} min")
    print(f"  Max simultaneous open positions (across all 4 pairs): {max_concurrent}")
    print(f"  Time-weighted average open positions: {avg_concurrent:.2f}")
    frac_ge2 = (concurrent[:-1] >= 2).astype(float)
    frac_ge2_time = np.average(frac_ge2, weights=np.maximum(dt_seconds, 0)) if len(dt_seconds) else 0
    print(f"  Fraction of time with >=2 positions open simultaneously: {100*frac_ge2_time:.1f}%")
    print(f"\n  With 5% fixed-fractional risk PER TRADE and {max_concurrent} positions open at once,")
    print(f"  worst-case simultaneous capital-at-risk is ~{5*max_concurrent:.0f}% of equity, not 5% --")
    print(f"  this is a real capacity/margin consideration for live sizing, not modeled in the")
    print(f"  Monte Carlo (which treats trades as a single sequential stream).")
    return dict(max_concurrent=int(max_concurrent), avg_concurrent=avg_concurrent, frac_time_ge2=100 * frac_ge2_time)


def main():
    print("#" * 90)
    print("  HYBRID-ULTRA LIVE-TRADING READINESS AUDIT")
    print("  N=6.0 stop, R:R=0.4, 2.0h fill-age, unconditional re-arm, touches<=4, no session filter")
    print("#" * 90)
    warm_up()

    t_true = test0_code_audit()
    test1_finer_grid()
    test2_multifold_wf()
    test3_param_sensitivity()

    full_trades = g10.build(HU_N, HU_RR, **HU_KW)
    test4_year_by_year(full_trades)
    test5_drawdown(full_trades)
    test6_spread_stress()
    test7_concurrency(t_true)

    print("\n" + "#" * 90)
    print("  AUDIT COMPLETE")
    print("#" * 90)


if __name__ == "__main__":
    main()
