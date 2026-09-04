"""Re-evaluates the trading-data/STRATEGY_RANKINGS.md leaderboard (its own repo has no
scripts -- data/results only -- so each strategy is reproduced here from its published
(N, R:R) parameters, run through this project's own dtfx_engine) under the corrected
fill model the user specified after a live EUR/GBP order sat resting past when the old
backtest would have called it filled:

1. A resting limit only counts as genuinely filled if the entry bar's low (buy) / high
   (sell) reaches a full SPREAD_PIPS_TYPICAL (1.6 pips) beyond the zone's own mid -- not
   just touching mid, which the old model treated as a free, guaranteed fill. A touch
   that reaches mid but not the full 1.6 pips beyond it still happened (kept, not
   discarded) but is priced as a market-order fill 1.6 pips worse, in the adverse
   direction (see goal51_single_symbol_closest_price_backtest.py's own docstring for the
   full derivation).
2. EVERY trade now pays the 1.6-pip spread, not just stop-outs (2026-09-04, at the
   user's explicit direction -- the old "passive-fill target" exemption doesn't hold once
   entry itself isn't assumed free either).

Scope: EUR/GBP only, matching PRODLIVETRADING's current single-symbol deployment --
this is what the user is actually trading, and the corrected fill model has already been
confirmed to reverse the edge on every recently-deployed strategy tested (Apex-Ultra,
Velocity-King, Apex-King all flip from PF>1.4 to PF<0.5).

"Legit" filter: excludes Causal-Confluence (its own name implies a causal-discovery
signal beyond pure zone geometry -- no source for that filter exists in this repo, so it
can't be honestly reproduced here, and an ML-discovered signal not built causally is
exactly the kind of lookahead risk this filter is meant to catch) and goal70_final_winner
(an internal placeholder name in the leaderboard's own CSV, not a named, described
strategy). Steadfast-King's own leaderboard entry itself flags a labeling inconsistency
between two different rows in its source document (N=5.0/age=3min/75-touch vs a
separately-described N=4.0/age=0.08h/20-touch row also called "Steadfast-King", which is
actually Apex-Ultra's own parameterization) -- (N, R:R)=(5.0, 0.15) from its own CSV row
plus age=3min/75-touch from its more detailed prose description is used here as the
authoritative definition, not the conflicting one.

age_h/touches for every OTHER leaderboard strategy are not independently recorded in
trading-data (CSV only has n/rr) -- standardized to Apex-Ultra's own (age_h=0.08,
touches=20) for the reproduction pass, explicitly flagged in the output. This is a
best-effort reproduction of a leaderboard whose own generating code isn't available to
audit, not a byte-for-byte replay of it.
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
TRADING_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "trading-data")

import goal29_rerank_1_per_side as g29
import goal51_single_symbol_closest_price_backtest as g51

pg = g29.pg
INST = "EURGBP"

# Precisely known (age_h, touches) overrides; everything else standardized below.
KNOWN_PARAMS = {
    "Velocity-King": dict(age_h=0.20, touches=16),
    "Apex-King": dict(age_h=0.30, touches=8),
    "Apex-Ultra": dict(age_h=0.08, touches=20),
    "Steadfast-King": dict(age_h=0.05, touches=75),  # 3 min = 0.05h, per its own prose
}
STANDARD_AGE_H, STANDARD_TOUCHES = 0.08, 20
EXCLUDED = {"Causal-Confluence", "goal70_final_winner"}


def evaluate(n, rr, age_h, touches, bounce=False, min_risk_pct=None, fill_depth_pips=None):
    rows = [g51.build_touches_with_metadata(INST, n, rr, age_h, bounce, touches, side,
                                             fill_depth_pips=fill_depth_pips,
                                             min_risk_pct=min_risk_pct)
            for side in (True, False)]
    t = pd.concat(rows, ignore_index=True)
    if t.empty:
        return None
    n_trades = len(t)
    wins = t[t.net_r > 0]
    losses = t[t.net_r <= 0]
    wr = 100 * len(wins) / n_trades
    gross_win, gross_loss = wins.net_r.sum(), -losses.net_r.sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    freq_tpy = n_trades / years if years > 0 else 0.0
    return dict(n_trades=n_trades, wr=wr, pf=pf, mean_r=t.net_r.mean(), freq_tpy=freq_tpy,
                shallow_pct=100 * t.shallow_touch.mean(), years=years)


def composite(row):
    """Simple, transparent 3-factor composite for THIS comparison: normalized PF, WR, and
    log-frequency, equally weighted, each min-max scaled across the compared set. Not a
    reproduction of trading-data's own evolving 3/4/5-factor formula (which depends on MC
    drawdown and audited-consistency data this pass doesn't compute) -- documented here as
    its own simpler thing, not claimed to be that one."""
    return row


def load_leaderboard():
    df = pd.read_csv(os.path.join(TRADING_DATA_DIR, "strategy_rankings.csv"))
    return df[["strategy", "n", "rr"]].drop_duplicates("strategy")


def main():
    leaderboard = load_leaderboard()
    print(f"Leaderboard entries: {len(leaderboard)}")
    results = []
    t0 = time.time()
    for _, row in leaderboard.iterrows():
        name, n, rr = row["strategy"], row["n"], row["rr"]
        if name in EXCLUDED:
            print(f"  [{name}] excluded -- {('placeholder name' if name == 'goal70_final_winner' else 'non-reproducible causal-discovery filter')}")
            continue
        params = KNOWN_PARAMS.get(name, dict(age_h=STANDARD_AGE_H, touches=STANDARD_TOUCHES))
        r = evaluate(n, rr, params["age_h"], params["touches"])
        if r is None:
            print(f"  [{name}] N={n} RR={rr}: no trades at this config, skipping")
            continue
        r.update(strategy=name, n=n, rr=rr, **params, standardized_params=name not in KNOWN_PARAMS)
        results.append(r)
        print(f"  [{name:22s}] N={n:.2f} RR={rr:.2f} age={params['age_h']:.2f}h "
              f"touches={params['touches']:3d}{'*' if name not in KNOWN_PARAMS else ' '} "
              f"-> n={r['n_trades']:>7,} WR={r['wr']:5.2f}% PF={r['pf']:.3f} "
              f"mean_r={r['mean_r']:+.4f} freq/yr={r['freq_tpy']:,.0f} shallow={r['shallow_pct']:.1f}% "
              f"[{time.time()-t0:.0f}s]")

    res_df = pd.DataFrame(results).sort_values("pf", ascending=False).reset_index(drop=True)
    out_path = os.path.join(DATA_DIR, "goal53_leaderboard_realistic_fill.csv")
    res_df.to_csv(out_path, index=False)
    print(f"\n(* = standardized age_h/touches, not independently confirmed for that strategy)")
    print(f"Saved: {out_path}")
    print(f"\nAny strategy with PF > 1.0 under the corrected model: "
          f"{res_df[res_df.pf > 1.0].strategy.tolist() or 'NONE'}")
    return res_df


if __name__ == "__main__":
    main()
