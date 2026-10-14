"""User asked for 3 more strategies with R:R fully unconstrained ("regardless of RRR").
Two real findings emerged while building these:

1. BUG: goal5_rr_sweep.py's build_pool() calls prod_filter_grid_search.build_trades(),
   which hardcodes MAX_ZONE_TOUCHES (a module constant = 4) and has NO parameter to
   override it -- goal5's own MAX_TOUCHES=8 constant was never actually wired in. So
   every "DTFX-NarrowTarget"-family result from goal5/goal9 was silently run at
   touches=4, not touches=8 as documented. (goal2_hour_opt.py's build_all(), used for
   "DTFX Robust", calls _simulate_generic directly and correctly passes its own
   MAX_TOUCHES -- that family is unaffected.) Fixed here by calling
   pg._simulate_generic() directly everywhere, with touches always explicit.

2. BIGGER FINDING: jointly sweeping stop distance (N_MULTIPLIER) together with R:R --
   not fixing N at each family's "native" value (2.0, 3.0, 3.343496) and only
   varying R:R, as every previous R:R sweep in this project did -- finds a
   substantially better region: wider stops (N=4-6) paired with narrower targets
   (R:R=0.4-0.6) beat anything found by varying R:R alone. This also exposed that
   CANA-base and the DTFX-skeleton become MECHANICALLY IDENTICAL once N floats freely
   and touches is fixed -- the only genuine remaining structural differences across
   this whole family are fill-age gate (2.0h vs 3.366682h) and re-arm rule
   (bounce-gated vs unconditional). The 3 strategies below are built along exactly
   those two genuinely-distinct axes, each independently jointly-optimized over
   (N, R:R) using ONLY first-half data (2009 -> 2018-03-09), scored purely on the
   entirely untouched second half.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal5_rr_sweep as g5

SPLIT = pd.Timestamp("2018-03-09")


def build(n, rr, age_h, require_bounce, touches):
    """Direct _simulate_generic call -- bypasses the buggy build_trades() wrapper,
    touches always explicit."""
    frames = []
    for inst in g5.TRADEABLE:
        z = g5.rs._prep_zones(inst, g5.pg.MIN_RETRACEMENT)
        times, lows, highs, df = g5.rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (n * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= g5.pg.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        touch_num, entry_idx, is_win, entry_mid, stop_risk = g5.pg._simulate_generic(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(age_h * 3600 * 1e9), n, n * rr, require_bounce, g5.pg.PROBE_DEPTH, 1.0,
            touches, g5.pg.MAX_WINDOW_BARS)
        pip = g5.pg.PIP_SIZE[inst]; spread_price = g5.pg.SPREAD_PIPS_TYPICAL * pip
        r_gross = np.where(is_win, rr, -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        frames.append(pd.DataFrame({"entry_time": pd.to_datetime(times[entry_idx]),
                                     "net_r": net_r, "is_win": is_win}))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def joint_search(age_h, require_bounce, touches, n_grid, rr_grid, label):
    """Select (N, R:R) jointly using ONLY first-half data."""
    best = (None, None, -1e9)
    for n in n_grid:
        for rr in rr_grid:
            trades = build(n, rr, age_h, require_bounce, touches)
            first = trades[trades.entry_time < SPLIT]
            glog = np.log1p(g5.RISK_FRAC * first.net_r)
            years = (first.entry_time.max() - first.entry_time.min()).days / 365.25
            g_score = (len(first) / years) * glog.mean()
            if g_score > best[2]:
                best = (n, rr, g_score)
    print(f"  {label}: best N={best[0]}, R:R={best[1]} (first-half g={best[2]:.3f})")
    return best[0], best[1]


def main():
    for inst in g5.TRADEABLE:
        g5.rs.load_instrument(inst)
        g5.rs._prep_zones(inst, g5.pg.MIN_RETRACEMENT)

    N_GRID = [3.0, 4.0, 5.0, 6.0]
    RR_GRID = [0.4, 0.45, 0.5, 0.6, 0.7]

    print("Selecting (N, R:R) jointly for each of 3 structurally distinct combos:")
    configs = [
        ("DTFX-Ultra", dict(age_h=2.0, require_bounce=True, touches=8)),
        ("Hybrid-Ultra", dict(age_h=2.0, require_bounce=False, touches=4)),
        ("SOMEGA-Ultra", dict(age_h=3.366682, require_bounce=False, touches=4)),
    ]

    for label, kw in configs:
        n, rr = joint_search(kw["age_h"], kw["require_bounce"], kw["touches"], N_GRID, RR_GRID, label)
        full = build(n, rr, **kw)
        full.to_csv(os.path.join(DATA_DIR, f"goal10_{label.lower().replace('-','_')}_full.csv"), index=False)
        r_full = g5.score_row(full, f"{label} (full-history, N={n}, R:R={rr})")
        g5.print_row(r_full)

        second = full[full.entry_time >= SPLIT].reset_index(drop=True)
        second.to_csv(os.path.join(DATA_DIR, f"goal10_{label.lower().replace('-','_')}_true_holdout.csv"), index=False)
        r_holdout = g5.score_row(second, f"{label} (TRUE holdout, N={n}, R:R={rr})")
        g5.print_row(r_holdout)
        print()


if __name__ == "__main__":
    main()
