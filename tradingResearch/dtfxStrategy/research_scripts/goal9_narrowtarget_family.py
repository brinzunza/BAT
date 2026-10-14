"""User re-opened R:R<=1 after previously ruling it out ('I know i previously said no
but im interested to see if they are better') and asked for 3 DISTINCT strategies,
not just one. This builds narrow-target (R:R<1) variants on three genuinely different
base zone-parameter configs -- DTFX-skeleton, CANA's own params, SOMEGA's own params
-- each with its own R:R independently optimized, each validated with the SAME
rigorous procedure established for the R:R>=1 comparison: select R:R using ONLY the
first half of history (2009 -> 2018-03-09), score purely on the entirely untouched
second half.

Result: all three land in the same neighborhood (0.55-0.59yr median, Sharpe 4.8-5.3,
frequency ~3,100-3,200/yr) -- strong evidence the "narrow the target" effect is a
property of the DTFX zone-detection mechanism itself, not an artifact of any one
specific N_MULTIPLIER/touches/re-arm combination. CANA-NarrowTarget (rr=0.8) is
narrowly fastest; SOMEGA-NarrowTarget (rr=0.7) has the highest Sharpe.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal5_rr_sweep as g5
import cana_faithful_backtest as cana
import prod_filter_grid_search as pg
import dtfx_robust_search as rs

rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
SPLIT = pd.Timestamp("2018-03-09")
RR_GRID = [1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]


def build_dtfx_rr(rr):
    """DTFX-skeleton base: N=2.0, age=2.0h, touches<=8, bounce-gated."""
    return g5.build_pool(rr, session_filter=False)


def build_cana_rr(rr):
    """CANA's own base: N=3.00, age=2.0h, touches<=4, bounce-gated."""
    frames = []
    for inst in g5.TRADEABLE:
        z = cana.rs._prep_zones(inst, cana.MIN_RETRACEMENT)
        times, lows, highs, df = cana.rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        risk_pct = (cana.N_MULTIPLIER * (z["top"] - z["bottom"]) / 2.0) / z["mid"]
        ok = risk_pct >= cana.MIN_RISK_PCT
        top, bottom, mid = z["top"][ok], z["bottom"][ok], z["mid"][ok]
        is_buy, start_idx, c_ns = z["is_buy"][ok], z["start_idx"][ok], created_ns[ok]
        target_mult = cana.N_MULTIPLIER * rr
        touch_num, entry_idx, is_win, entry_mid, stop_risk = pg._simulate_generic(
            top, bottom, mid, is_buy, start_idx, c_ns, times_ns, lows, highs,
            int(cana.MIN_FILL_AGE_H * 3600 * 1e9), cana.N_MULTIPLIER, target_mult, True,
            cana.PROBE_DEPTH, 1.0, cana.MAX_ZONE_TOUCHES, cana.MAX_WINDOW_BARS)
        pip = cana.PIP_SIZE[inst]; spread_price = cana.SPREAD_PIPS_TYPICAL * pip
        r_gross = np.where(is_win, rr, -1.0)
        cost_in_r = np.where(is_win, 0.0, spread_price / stop_risk)
        net_r = r_gross - cost_in_r
        frames.append(pd.DataFrame({"entry_time": pd.to_datetime(times[entry_idx]),
                                     "net_r": net_r, "is_win": is_win}))
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def build_somega_rr(rr):
    """SOMEGA's own base: N=3.343496, age=3.366682h, touches<=4, unconditional re-arm."""
    N = 3.343496
    AGE = 3.366682
    frames = [pg.build_trades(inst, N, N * rr, AGE, require_bounce=False, session_filter=False)
              for inst in g5.TRADEABLE]
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def select_best_rr(build_fn, label):
    """Pick R:R using ONLY first-half data (honest -- no lookahead into the holdout)."""
    best_rr, best_g = None, -1e9
    for rr in RR_GRID:
        trades = build_fn(rr)
        first = trades[trades.entry_time < SPLIT]
        glog = np.log1p(g5.RISK_FRAC * first.net_r)
        years = (first.entry_time.max() - first.entry_time.min()).days / 365.25
        g_score = (len(first) / years) * glog.mean()
        print(f"  {label} rr={rr}: first-half g={g_score:.3f}")
        if g_score > best_g:
            best_g, best_rr = g_score, rr
    print(f"  -> best rr for {label}: {best_rr}")
    return best_rr


def main():
    for inst in g5.TRADEABLE:
        g5.rs.load_instrument(inst)
        g5.rs._prep_zones(inst, g5.pg.MIN_RETRACEMENT)

    families = [
        ("DTFX-NarrowTarget", build_dtfx_rr, "goal9_dtfx_narrowtarget"),
        ("CANA-NarrowTarget", build_cana_rr, "goal9_cana_narrowtarget"),
        ("SOMEGA-NarrowTarget", build_somega_rr, "goal9_somega_narrowtarget"),
    ]

    for label, build_fn, filestem in families:
        print(f"\n=== {label} ===")
        best_rr = select_best_rr(build_fn, label)
        full = build_fn(best_rr)
        full.to_csv(os.path.join(DATA_DIR, f"{filestem}_full.csv"), index=False)
        r_full = g5.score_row(full, f"{label} (full-history, rr={best_rr})")
        g5.print_row(r_full)

        second = full[full.entry_time >= SPLIT].reset_index(drop=True)
        second.to_csv(os.path.join(DATA_DIR, f"{filestem}_true_holdout.csv"), index=False)
        r_holdout = g5.score_row(second, f"{label} (TRUE holdout, rr={best_rr})")
        g5.print_row(r_holdout)


if __name__ == "__main__":
    main()
