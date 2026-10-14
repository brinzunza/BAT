"""Re-runs DTFX Robust and goal70_final_winner WITHOUT the XAUUSD leg -- gold is not
tradeable on the live account, so it must be excluded from any strategy actually
under consideration for deployment, not just from the live SOMEGA config.

Removing gold cost DTFX Robust its lead over CANA: g drops from 8.836 (with gold) to
7.889 (without) full-history, and the walk-forward holdout drops from 8.497 to 8.042
-- both now BELOW CANA's g=8.304. MAX_ZONE_TOUCHES was re-swept (8/12/16/25) as a
possible compensating lever and, as before, saturates almost immediately (g only
reaches 8.033 at touches=25) -- not enough to close the gap. CANA is the honest
winner of the tradeable-only race; DTFX Robust (no gold) is a close, higher-quality
second (WR 61.47% vs CANA's 59.10%, PF 1.237 vs 1.177) at nearly the same speed.
"""
import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal2_hour_opt as h
import goal2_robust_hours as rh
import goal3_expanding_wf as wf
import goal70_final_winner as g70

TRADEABLE = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]  # XAUUSD excluded: not tradeable on this account


def build_dtfx_robust_no_gold():
    for inst in TRADEABLE:
        h.rs.load_instrument(inst)
        h.rs._prep_zones(inst, 2)

    pools = {}
    tmin = tmax = None
    for inst in TRADEABLE:
        t = h.build_all(inst)
        t["glog"] = np.log1p(h.RISK_FRAC * t.net_r)
        t["hour"] = t.entry_time.dt.hour // 6
        pools[inst] = t
        tmin = t.entry_time.min() if tmin is None else min(tmin, t.entry_time.min())
        tmax = t.entry_time.max() if tmax is None else max(tmax, t.entry_time.max())

    quarters4 = rh.make_quarters(tmin, tmax, k=4)
    final_blocks, _ = rh.select_hours(pools, quarters4)
    BLOCK_LABELS = {0: "00-05", 1: "06-11", 2: "12-17", 3: "18-23"}
    for inst in TRADEABLE:
        print(f"  {inst}: blocks kept: {[BLOCK_LABELS[b] for b in sorted(final_blocks[inst])]}")

    frames = [t[t.hour.isin(final_blocks[inst])] for inst, t in pools.items()]
    full = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    rh.print_score(rh.score(full, "DTFX Robust (no gold, full-history)"))

    folds5 = wf.make_folds(tmin, tmax, k=5)
    holdout = wf.run_expanding_wf(pools, folds5, "DTFX Robust (no gold)", test_from=2)

    full.to_csv(os.path.join(DATA_DIR, "goal4_dtfx_nogold_full_trades.csv"), index=False)
    holdout.to_csv(os.path.join(DATA_DIR, "goal4_dtfx_nogold_wf_holdout_trades.csv"), index=False)
    return full, holdout


def build_goal70_no_gold():
    for inst in TRADEABLE:
        g70.rs.load_instrument(inst)
        g70.rs._prep_zones(inst, 2)

    core_parts = [g70.session_mask(g70.build_with_stoppips(i, 1.6, 1e-4, g70.CORE_AGE_H), i)
                  for i in ["EURGBP", "GBPUSD"]]
    core_parts.append(g70.session_mask(g70.build_with_stoppips("USDJPY", 1.6, 1e-2, g70.CORE_AGE_H), "USDJPY"))
    core = pd.concat(core_parts, ignore_index=True)

    eurusd = g70.build_with_stoppips("EURUSD", 1.6, 1e-4, 3.366682)
    eurusd = eurusd[eurusd.entry_time.dt.hour.isin([19, 20, 21, 22])]

    full = pd.concat([core, eurusd], ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    years = (full.entry_time.max() - full.entry_time.min()).days / 365.25
    w = full.loc[full.net_r > 0, "net_r"].sum(); l = -full.loc[full.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    print(f"goal70 (no gold): n={len(full):,} ({len(full)/years:.0f}/yr) "
          f"WR={100*full.is_win.mean():.2f}% PF={pf:.3f}")

    full.to_csv(os.path.join(DATA_DIR, "goal4_goal70_nogold_trades.csv"), index=False)
    return full


if __name__ == "__main__":
    print("=== DTFX Robust, no gold ===")
    build_dtfx_robust_no_gold()
    print("\n=== goal70_final_winner, no gold ===")
    build_goal70_no_gold()
