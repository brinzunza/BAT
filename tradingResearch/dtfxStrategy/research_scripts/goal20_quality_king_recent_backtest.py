"""Quality-King (N=7.0, R:R=0.35, age_h=0.40, bounce-gated, touches=8) backtest over
the RECENT months only -- not the full 17-year history -- requested after Speed-King
was ruled out for a real reason: 50:1 leverage still needs margin per open position,
and Speed-King's real historical concurrency (unaudited, but almost certainly higher
than Hybrid-Ultra's already-high max of 9, given ~4x the frequency at an even
shorter fill-age) would need more simultaneous margin than is actually available.

Reports, for the last 1 and 3 months of data (through 2026-08-21):
  1. Deterministic $200-start, 5%-risk equity curve over the real recent trade
     sequence -- what $200 would actually have become.
  2. The same win rate / PF / Sharpe stats used throughout this project.
  3. A block-bootstrap Monte Carlo (same 4,000-sim, 20-trade-block, $1M/$10M
     methodology as everywhere else in this file) sampled from the RECENT window's
     own trades, not the full-history pool -- a "if the market keeps behaving like
     it recently has" read, distinct from the published full TRUE-holdout number.
  4. A concurrency check (same technique as Hybrid-Ultra's live-readiness audit) --
     directly answering whether Quality-King has the same margin problem Speed-King
     did.
"""
import os
import sys
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal10_ultra_family as g10
import goal5_rr_sweep as g5
import prod_filter_grid_search as pg
import dtfx_robust_search as rs

QK = dict(n=7.0, rr=0.35, age_h=0.40, require_bounce=True, touches=8)


def warm_up():
    for inst in g5.TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)


@njit(cache=True)
def _simulate_with_true_entry(zone_top, zone_bottom, zone_mid, zone_is_buy, zone_start_idx,
                               zone_created_ns, times_ns, lows, highs, min_fill_age_ns,
                               stop_mult, target_mult, require_bounce, probe_depth, exc_mult,
                               max_touches, max_window):
    """Same as the fixed _simulate_generic, plus the resolution bar too (for duration
    / concurrency analysis) -- entry_idx alone (now correctly the fill bar) isn't
    enough to know how long each trade stays open."""
    n = len(zone_top)
    n_times = len(lows)
    max_out = n * max_touches
    out_touch = np.zeros(max_out, dtype=np.int64)
    out_entry_idx = np.full(max_out, -1, dtype=np.int64)
    out_exit_idx = np.full(max_out, -1, dtype=np.int64)
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
                    out_entry_idx[out_count] = cur_entry_j
                    out_exit_idx[out_count] = j
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

    return (out_touch[:out_count], out_entry_idx[:out_count], out_exit_idx[:out_count],
            out_is_win[:out_count], out_stop_risk[:out_count])


def build_qk_with_exit():
    frames = []
    for inst in g5.TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (QK["n"] * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        touch_num, entry_idx, exit_idx, is_win, stop_risk = _simulate_with_true_entry(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(QK["age_h"] * 3600 * 1e9), QK["n"], QK["n"] * QK["rr"], QK["require_bounce"],
            pg.PROBE_DEPTH, 1.0, QK["touches"], pg.MAX_WINDOW_BARS)
        pip = pg.PIP_SIZE[inst]; spread_price = pg.SPREAD_PIPS_TYPICAL * pip
        r_gross = np.where(is_win, QK["rr"], -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        frames.append(pd.DataFrame({
            "entry_time": pd.to_datetime(times[entry_idx]),
            "exit_time": pd.to_datetime(times[exit_idx]),
            "net_r": net_r, "is_win": is_win, "instrument": inst,
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def deterministic_equity(trades, start=200.0, risk_frac=0.05):
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    eq = np.empty(len(trades) + 1)
    eq[0] = start
    for i, r in enumerate(trades.net_r.values):
        risk = eq[i] * risk_frac
        eq[i + 1] = max(eq[i] + risk * r, 0.01)
    equity = eq[1:]
    running_max = np.maximum.accumulate(np.concatenate([[start], equity]))[1:]
    dd = (equity - running_max) / running_max
    return equity, dd.min() * 100


def report_window(full, months, label):
    end = full.entry_time.max()
    start_cut = end - pd.DateOffset(months=months)
    win = full[full.entry_time >= start_cut].reset_index(drop=True)
    print(f"\n{'='*90}\n{label}: {start_cut.date()} -> {end.date()} ({len(win):,} trades)\n{'='*90}")
    if len(win) < 20:
        print("  Too few trades in this window.")
        return None

    wr = 100 * win.is_win.mean()
    w = win.loc[win.net_r > 0, "net_r"].sum(); l = -win.loc[win.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    equity, max_dd = deterministic_equity(win)
    days = (win.entry_time.max() - win.entry_time.min()).total_seconds() / 86400
    tpy_window = len(win) / max(days / 365.25, 1e-6)

    print(f"  Trades: {len(win):,}  (~{tpy_window:,.0f}/yr pace)   Win rate: {wr:.2f}%   PF: {pf:.3f}")
    print(f"  $200 start, 5% risk/trade, deterministic real sequence:")
    print(f"    Final equity: ${equity[-1]:,.2f}   Return: {100*(equity[-1]/200 - 1):+.1f}%")
    print(f"    Max drawdown within window: {max_dd:.1f}%")

    weekly = win.set_index("entry_time")["net_r"]
    if len(win) > 200:
        sharpe = g5.weekly_sharpe(win)
        print(f"    Weekly Sharpe: {sharpe:.2f}")

    r = g5.score_row(win, f"{label} (recent-window MC pool)")
    g5.print_row(r)
    return dict(label=label, trades=win, final_equity=equity[-1], max_dd=max_dd, wr=wr, pf=pf, mc=r)


def concurrency_check(t_exit):
    print(f"\n{'='*90}\nQuality-King concurrency check (full history, vs. Hybrid-Ultra's audited max of 9)\n{'='*90}")
    t = t_exit.sort_values("entry_time").reset_index(drop=True)
    starts = t.entry_time.values.astype("datetime64[ns]")
    ends = t.exit_time.values.astype("datetime64[ns]")
    events = np.concatenate([starts, ends])
    deltas = np.concatenate([np.ones(len(starts), dtype=int), -np.ones(len(ends), dtype=int)])
    order = np.argsort(events, kind="stable")
    events_sorted = events[order]
    deltas_sorted = deltas[order]
    concurrent = np.cumsum(deltas_sorted)
    max_concurrent = concurrent.max()
    dt_seconds = np.diff(events_sorted).astype("timedelta64[s]").astype(float)
    avg_concurrent = np.average(concurrent[:-1], weights=np.maximum(dt_seconds, 0)) if len(dt_seconds) else 0
    frac_ge2 = (concurrent[:-1] >= 2).astype(float)
    frac_ge2_time = np.average(frac_ge2, weights=np.maximum(dt_seconds, 0)) if len(dt_seconds) else 0

    dur_min = (ends - starts).astype("timedelta64[s]").astype(float) / 60.0
    print(f"  Trade duration: median={np.median(dur_min):.1f} min, mean={np.mean(dur_min):.1f} min, "
          f"p90={np.percentile(dur_min, 90):.1f} min")
    print(f"  Max simultaneous open positions: {max_concurrent}  (Hybrid-Ultra's audited max: 9)")
    print(f"  Time-weighted average open positions: {avg_concurrent:.2f}")
    print(f"  Fraction of time with >=2 positions open: {100*frac_ge2_time:.1f}%")
    print(f"  With 5% risk/trade, worst-case simultaneous capital-at-risk: ~{5*max_concurrent:.0f}% of equity")
    return max_concurrent


def main():
    warm_up()

    print("Building Quality-King full trade log (N=7.0, R:R=0.35, age_h=0.40, bounce-gated, touches=8)...")
    full = g10.build(QK["n"], QK["rr"], age_h=QK["age_h"], require_bounce=QK["require_bounce"],
                      touches=QK["touches"])
    full.to_csv(os.path.join(DATA_DIR, "goal20_quality_king_full_recheck.csv"), index=False)

    report_window(full, 1, "Quality-King: LAST 1 MONTH")
    report_window(full, 3, "Quality-King: LAST 3 MONTHS")

    print("\nBuilding true entry/exit times for concurrency check (full history)...")
    t_exit = build_qk_with_exit()
    t_exit.to_csv(os.path.join(DATA_DIR, "goal20_quality_king_with_exit.csv"), index=False)
    concurrency_check(t_exit)


if __name__ == "__main__":
    main()
