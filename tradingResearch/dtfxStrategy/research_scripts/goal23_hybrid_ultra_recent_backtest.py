"""Hybrid-Ultra (N=6.0, R:R=0.4, age_h=2.0, unconditional re-arm, touches=4) --
same recent-months + concurrency analysis just run for Quality-King, for direct
comparison. Hybrid-Ultra already has a full live-trading-readiness audit
(LIVE_TRADING_READINESS_AUDIT.md, full-history concurrency max=9), but that audit
never broke out the most recent 1-3 months specifically the way this does, and
never showed a $200-start deterministic equity curve over just that recent window.

ALWAYS uses the C++ dtfx_engine extension, not the numba fallback: zone detection
via dtfx_engine.detect_zones, the main trade log via _simulate_generic's own C++
dispatch (already automatic inside prod_filter_grid_search.py / goal10_ultra_family
whenever dtfx_engine is importable), and the concurrency check via
dtfx_engine.simulate_generic_ex -- the C++ engine's own purpose-built variant that
additionally returns exit_idx (the resolution bar), replacing what was previously a
hand-written numba reimplementation just to get exit times. Both dtfx_engine
functions are validated byte-identical to their Python/numba counterparts by
cpp_engine/validate.py, re-run immediately before this script to confirm.

Reports, for the last 1 and 3 months of data (through 2026-08-21):
  1. Deterministic $200-start, 5%-risk equity curve over the real recent trade
     sequence.
  2. Win rate / PF / Sharpe over that window.
  3. A block-bootstrap Monte Carlo sampled from the RECENT window's own trades.
  4. A concurrency check (same technique as the original audit, restated here for
     side-by-side comparison against Quality-King's numbers).
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal10_ultra_family as g10
import goal5_rr_sweep as g5
import prod_filter_grid_search as pg
import dtfx_robust_search as rs

try:
    import dtfx_engine
except ImportError:
    raise RuntimeError(
        "dtfx_engine (C++ extension) is not importable -- this script requires it "
        "per 'always use C++ engines'. Build cpp_engine/ first (see its README)."
    )

HU = dict(n=6.0, rr=0.40, age_h=2.0, require_bounce=False, touches=4)


def warm_up():
    for inst in g5.TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)
    assert pg._HAS_CPP_ENGINE, "prod_filter_grid_search fell back to numba -- C++ engine not active"


def build_hu_with_exit():
    """Same population as g10.build(**HU), but via dtfx_engine.simulate_generic_ex
    directly, to also get exit_idx (resolution bar) for the concurrency check."""
    frames = []
    for inst in g5.TRADEABLE:
        z = rs._prep_zones(inst, pg.MIN_RETRACEMENT)
        times, lows, highs, df = rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (HU["n"] * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(HU["age_h"] * 3600 * 1e9), HU["n"], HU["n"] * HU["rr"], HU["require_bounce"],
            pg.PROBE_DEPTH, 1.0, HU["touches"], pg.MAX_WINDOW_BARS)
        entry_idx = np.asarray(entry_idx); exit_idx = np.asarray(exit_idx)
        is_win = np.asarray(is_win); stop_risk = np.asarray(stop_risk)
        pip = pg.PIP_SIZE[inst]; spread_price = pg.SPREAD_PIPS_TYPICAL * pip
        r_gross = np.where(is_win, HU["rr"], -1.0)
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

    if len(win) > 200:
        sharpe = g5.weekly_sharpe(win)
        print(f"    Weekly Sharpe: {sharpe:.2f}")

    r = g5.score_row(win, f"{label} (recent-window MC pool)")
    g5.print_row(r)
    return dict(label=label, trades=win, final_equity=equity[-1], max_dd=max_dd, wr=wr, pf=pf, mc=r)


def concurrency_check(t_exit):
    print(f"\n{'='*90}\nHybrid-Ultra concurrency check (full history)\n{'='*90}")
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
    print(f"  Max simultaneous open positions: {max_concurrent}")
    print(f"  Time-weighted average open positions: {avg_concurrent:.2f}")
    print(f"  Fraction of time with >=2 positions open: {100*frac_ge2_time:.1f}%")
    print(f"  With 5% risk/trade, worst-case simultaneous capital-at-risk: ~{5*max_concurrent:.0f}% of equity")
    return max_concurrent


def main():
    warm_up()

    print("Building Hybrid-Ultra full trade log (N=6.0, R:R=0.4, age_h=2.0, unconditional, touches=4)...")
    full = g10.build(HU["n"], HU["rr"], age_h=HU["age_h"], require_bounce=HU["require_bounce"],
                      touches=HU["touches"])
    full.to_csv(os.path.join(DATA_DIR, "goal23_hybrid_ultra_full_recheck.csv"), index=False)

    report_window(full, 1, "Hybrid-Ultra: LAST 1 MONTH")
    report_window(full, 3, "Hybrid-Ultra: LAST 3 MONTHS")

    print("\nBuilding true entry/exit times for concurrency check (full history)...")
    t_exit = build_hu_with_exit()
    t_exit.to_csv(os.path.join(DATA_DIR, "goal23_hybrid_ultra_with_exit.csv"), index=False)
    concurrency_check(t_exit)


if __name__ == "__main__":
    main()
