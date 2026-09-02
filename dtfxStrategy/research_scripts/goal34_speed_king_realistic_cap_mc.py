"""Speed-King Monte Carlo under explicitly more conservative, more realistic assumptions
than this project's standing defaults: $200 start, 2.0-pip spread (vs the usual 1.6),
3.00% risk, 50x leverage, and -- the key change -- a $500,000 POSITION-SIZING CAP instead
of the usual $2,000,000. This directly tests the capacity concern raised in the prior
robustness audit (goal39/The Robustness Docket): position sizing is risk_pct * min(equity,
cap), so once equity passes the cap, growth stops compounding exponentially and becomes
roughly linear (fixed dollar risk per trade) -- a much more realistic stand-in for "this is
about where a retail/prop account's real execution capacity runs out," even though the
true ceiling is unmeasured and could be lower.

Reports:
  1. Standard forward Monte Carlo (25,000 sims, dtfx_engine.run_mc_capped): time to reach
     the $500k cap itself, and time to $1M despite the cap (i.e. how much the cap alone
     slows the trip to $1M), plus the drawdown distribution and ruin rate under these
     assumptions.
  2. A single-path margin-constrained backtest (goal20's exact model: real position
     sizing/margin checked against currently-available margin, not just the sizing
     formula) at the same $500k cap, for direct context.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal20_speed_king_margin_report as g20
from goal29_speed_king_spread_sensitivity import build_raw, with_spread
from goal10_cana_nt_mc_capped import BLOCK, MAX_TRADES, RUIN_FLOOR

rs, pg, TRADEABLE = g20.rs, g20.pg, g20.TRADEABLE

START = 200.0
SPREAD_PIPS = 2.0
RISK_PCT = 0.03
MAX_LEVERAGE = 50.0
CAP = 500_000.0
TARGET_CAP = 500_000.0
TARGET_1M = 1_000_000.0
N_SIMS = 25_000


def main():
    for inst in TRADEABLE:
        rs.load_instrument(inst)
        rs._prep_zones(inst, pg.MIN_RETRACEMENT)

    print("Building Speed-King's spread-independent population and applying 2.0-pip spread...", flush=True)
    raw = build_raw()
    t = with_spread(raw, SPREAD_PIPS)
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    tpy = len(t) / years
    wr = 100 * t.is_win.mean()
    w = t.loc[t.net_r > 0, "net_r"].sum(); l = -t.loc[t.net_r <= 0, "net_r"].sum()
    pf = w / l if l > 0 else float("inf")
    print(f"  {len(t):,} trades ({tpy:,.0f}/yr), WR={wr:.2f}%, PF={pf:.3f} at {SPREAD_PIPS}-pip spread\n")

    print("=" * 94)
    print(f"PART 1: Monte Carlo -- {N_SIMS:,} sims, ${START:.0f} start, {SPREAD_PIPS}-pip spread, "
          f"{100*RISK_PCT:.1f}% risk, ${CAP:,.0f} sizing cap")
    print("=" * 94)
    net_r = t.net_r.values.astype(np.float64)
    t1_raw, t10_raw, mc_dd, ruined = dtfx_engine.run_mc_capped(
        net_r, N_SIMS, BLOCK, MAX_TRADES, 30, start=START, risk_frac=RISK_PCT,
        cap=CAP, target1=TARGET_CAP, target2=TARGET_1M, ruin_floor=RUIN_FLOOR)
    dd_pct = mc_dd * 100

    hit_cap = t1_raw[t1_raw > 0]
    y_cap = hit_cap / tpy
    hit_1m = t10_raw[t10_raw > 0]
    y_1m = hit_1m / tpy

    print(f"\nTime to reach the ${TARGET_CAP:,.0f} sizing cap itself:")
    print(f"  {100*len(hit_cap)/N_SIMS:.2f}% of sims reach it at all")
    if len(hit_cap):
        for p in [10, 25, 50, 75, 90]:
            print(f"    P{p}: {np.percentile(y_cap, p):.3f} yr")

    print(f"\nTime to reach ${TARGET_1M:,.0f} (growth continues past the cap, just no longer "
          f"exponential once equity > cap):")
    print(f"  {100*len(hit_1m)/N_SIMS:.2f}% of sims reach it within the simulation's trade budget")
    if len(hit_1m):
        for p in [10, 25, 50, 75, 90]:
            print(f"    P{p}: {np.percentile(y_1m, p):.3f} yr")
    else:
        print(f"  0 of {N_SIMS:,} sims reached $1M within the simulated trade budget "
              f"(MAX_TRADES={MAX_TRADES:,}) -- the $500k cap makes the final stretch to $1M "
              f"far slower than the exponential-growth phase below it.")

    print(f"\nMax drawdown distribution:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(dd_pct, p):.1f}%")
    print(f"  Worst of {N_SIMS:,}: {dd_pct.min():.1f}%")
    print(f"\nRuin rate: {100*ruined.mean():.3f}%")

    out = pd.DataFrame({
        "years_to_cap": np.where(t1_raw > 0, t1_raw / tpy, np.nan),
        "years_to_1m": np.where(t10_raw > 0, t10_raw / tpy, np.nan),
        "max_drawdown_pct": dd_pct, "ruined": ruined,
    })
    out.to_csv(os.path.join(DATA_DIR, "goal34_realistic_cap_mc.csv"), index=False)

    # ---- Part 2: single-path margin-constrained backtest, same assumptions -----------
    print(f"\n{'='*94}")
    print(f"PART 2: Margin-constrained single-path backtest (goal20's model), same assumptions")
    print("=" * 94)
    tm, final_eq, max_dd_pct, accepted, trace = g20.simulate_margin_constrained(
        t, start=START, risk_pct=RISK_PCT, max_leverage=MAX_LEVERAGE, cap=CAP, track_detail=True)
    acc = tm[tm.accepted]
    accept_pct = 100 * len(acc) / len(tm)
    wr_acc = 100 * acc.is_win.mean()
    wA = acc.loc[acc.net_r > 0, "net_r"].sum(); lA = -acc.loc[acc.net_r <= 0, "net_r"].sum()
    pf_acc = wA / lA if lA > 0 else float("inf")
    print(f"Accepted {len(acc):,}/{len(tm):,} signals ({accept_pct:.1f}%), WR={wr_acc:.2f}%, PF={pf_acc:.3f}")
    print(f"${START:.0f} start -> ${final_eq:,.2f} over {years:.1f}yr (single continuous path, "
          f"${CAP:,.0f} sizing cap)")
    print(f"Realized max drawdown: {max_dd_pct:.1f}%")

    # time (in the realized path) at which equity first crosses $500k and $1M, if ever
    crossing_500k = trace[trace.equity >= TARGET_CAP]
    crossing_1m = trace[trace.equity >= TARGET_1M]
    t0 = t.entry_time.min()
    if len(crossing_500k):
        first = crossing_500k.iloc[0]
        print(f"Realized path first crosses ${TARGET_CAP:,.0f} at {first.close_time} "
              f"({(first.close_time - t0).days/365.25:.2f}yr in)")
    else:
        print(f"Realized path never reaches ${TARGET_CAP:,.0f}")
    if len(crossing_1m):
        first = crossing_1m.iloc[0]
        print(f"Realized path first crosses ${TARGET_1M:,.0f} at {first.close_time} "
              f"({(first.close_time - t0).days/365.25:.2f}yr in)")
    else:
        print(f"Realized path never reaches ${TARGET_1M:,.0f}")


if __name__ == "__main__":
    main()
