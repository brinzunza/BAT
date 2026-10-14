"""/goal: find a strategy that is net profitable AND has the quickest time to $1M, under
every rule established this session:
  - Realistic fill model (goal51): a touch only fills free at mid if it reaches a full
    1.6-pip depth beyond it; shallower touches are priced as a market fill 1.6 pips worse.
    Every trade pays the spread now, not just stop-outs.
  - Two genuinely separate $100 sub-accounts (buy-only, sell-only), sizing off
    min(buy_equity, sell_equity), 50x leverage cap (goal30::simulate_two_account /
    run_mc, reused unmodified).
  - Concurrency: at most 1 open position per side. For a single symbol this is exactly
    what MAX_CONCURRENT_PER_SIDE enforces already; goal51's resolve_closest_price handles
    "which zone gets the one resting slot" -- not re-derived here.
  - EUR/GBP only, matching PRODLIVETRADING's current deployment scope.

goal53/54/55 already established PF as the primary gate (net profitable, non-negotiable)
and found one validated survivor (Bedrock-King, N=22/RR=0.75/MIN_RISK_PCT=0.020) -- but
it's slow (31 trades/yr, MC-to-$1M didn't converge). This search adds SPEED as an
explicit second objective: screen broadly by a fast, single deterministic-path years-to-$1M
(goal30's own historical-trace convention, not a full bootstrap -- cheap enough to screen
hundreds of configs), then validate the fastest still-profitable candidates properly
(train/holdout PF, then a real bootstrap MC for the finalist).
"""
import os
import sys
import itertools
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal30_apex_king_deployment as g30
import goal51_single_symbol_closest_price_backtest as g51

INST = "EURGBP"
AGE_H, TOUCHES = 0.08, 20
SPLIT = pd.Timestamp("2018-01-01")


def build_trades(n, rr, min_risk_pct, age_h=AGE_H, touches=TOUCHES):
    parts = [g51.build_touches_with_metadata(INST, n, rr, age_h, False, touches, side,
                                              min_risk_pct=min_risk_pct)
             for side in (True, False)]
    return pd.concat(parts, ignore_index=True)


def pf_of(t):
    if len(t) < 20:
        return None
    gw, gl = t[t.net_r > 0].net_r.sum(), -t[t.net_r <= 0].net_r.sum()
    return gw / gl if gl > 0 else float("inf")


def years_to_1m(t, risk_pct):
    """Fast, single deterministic-path estimate (goal30's own trace convention) -- NOT a
    bootstrap; cheap enough to screen a wide grid with, refined with a real MC later for
    the finalist only."""
    _, eq_buy, eq_sell, max_dd, trace = g30.simulate_two_account(t, risk_pct)
    if trace.empty:
        return None, eq_buy + eq_sell, max_dd
    hit = trace[trace.equity_combined >= 1_000_000]
    if hit.empty:
        return None, eq_buy + eq_sell, max_dd
    start = t.entry_time.min()
    years = (hit.close_time.iloc[0] - start).days / 365.25
    return years, eq_buy + eq_sell, max_dd


def screen(n_grid, rr_grid, mrp_grid, risk_pct=0.06):
    combos = list(itertools.product(n_grid, rr_grid, mrp_grid))
    rows = []
    t0 = time.time()
    for i, (n, rr, mrp) in enumerate(combos):
        t = build_trades(n, rr, mrp)
        pf = pf_of(t)
        if pf is None or pf <= 1.0:
            continue
        years, final_eq, dd = years_to_1m(t, risk_pct)
        rows.append(dict(n=n, rr=rr, min_risk_pct=mrp, n_trades=len(t), pf=pf,
                          wr=100 * (t.net_r > 0).mean(), freq_tpy=len(t) / max((t.entry_time.max() - t.entry_time.min()).days / 365.25, 0.01),
                          years_to_1m=years, final_equity=final_eq, max_dd=dd))
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(combos)} [{time.time()-t0:.0f}s], "
                  f"{len(rows)} profitable so far")
    return pd.DataFrame(rows)


def train_holdout(n, rr, mrp):
    t = build_trades(n, rr, mrp)
    train, holdout = t[t.entry_time < SPLIT], t[t.entry_time >= SPLIT]
    return pf_of(train), len(train), pf_of(holdout), len(holdout)


def main():
    # Same core grid goal54/55 already explored (profitability lever = MIN_RISK_PCT), but
    # scored here by speed-to-$1M among the profitable subset, not just PF.
    n_grid = [4, 5, 6, 8, 10, 12, 15, 18, 20, 22, 25, 30]
    rr_grid = [0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    mrp_grid = [0.000170, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.013, 0.016, 0.020, 0.025]

    print(f"Screening {len(n_grid)*len(rr_grid)*len(mrp_grid)} configs for PF>1, "
          f"then ranking by fastest single-path years-to-$1M...")
    df = screen(n_grid, rr_grid, mrp_grid)
    df_reachable = df[df.years_to_1m.notna()].sort_values("years_to_1m")
    df_unreachable = df[df.years_to_1m.isna()]
    print(f"\n{len(df)} profitable configs found; {len(df_reachable)} reached $1M on the "
          f"single historical path, {len(df_unreachable)} did not (too slow/low-frequency).")

    out = os.path.join(DATA_DIR, "goal56_fastest_to_1m_screen.csv")
    df.sort_values("years_to_1m", na_position="last").to_csv(out, index=False)
    print(f"Saved: {out}")

    cols = ["n", "rr", "min_risk_pct", "n_trades", "pf", "wr", "freq_tpy", "years_to_1m", "max_dd"]
    print("\nFastest profitable configs (single historical path to $1M):")
    print(df_reachable[cols].head(15).to_string(index=False))

    print("\nValidating top 8 fastest via 2018 train/holdout split...")
    validated = []
    for _, row in df_reachable.head(8).iterrows():
        pf_tr, n_tr, pf_ho, n_ho = train_holdout(row.n, row.rr, row.min_risk_pct)
        robust = (pf_tr is not None and pf_ho is not None and pf_tr > 1.0 and pf_ho > 1.0)
        print(f"  N={row.n:5.1f} RR={row.rr:.2f} mrp={row.min_risk_pct:.4f} "
              f"years_to_1m={row.years_to_1m:.2f} | train(pre-2018) n={n_tr} PF={pf_tr} | "
              f"holdout(2018+) n={n_ho} PF={pf_ho} | {'ROBUST' if robust else 'NOT ROBUST -- rejected'}")
        if robust:
            validated.append(row)

    if not validated:
        print("\nNo candidate in the fastest-8 survived train/holdout. Widening search "
              "needed -- see next iteration.")
        return df, None

    winner = validated[0]
    print(f"\n=== WINNER (fastest-to-$1M AND train/holdout-robust): "
          f"N={winner.n} RR={winner.rr} MIN_RISK_PCT={winner.min_risk_pct} ===")
    return df, winner


if __name__ == "__main__":
    main()
