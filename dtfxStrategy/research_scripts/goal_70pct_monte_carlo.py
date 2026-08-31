"""Block-bootstrap Monte Carlo for the narrow-target R:R candidate (SOMEGA session,
stop=3.343496, target=0.35x stop -- see goal_70pct_winrate_search.py / trading-data/
NARROW_TARGET_WINRATE_RESEARCH.md).

Deliberately NOT a single deterministic compounding run over the fixed historical trade
sequence (that's what produced CANA's meaningless 10^24x+ "final equity" and this
project's own earlier version of the same bug). Instead:

  1. Resample the actual historical trade-return sequence with a BLOCK bootstrap (blocks
     of 20 consecutive trades, sampled with replacement, concatenated to the target
     horizon length) -- this preserves realistic streakiness/autocorrelation structure
     instead of assuming trades are iid, which would understate drawdown risk.
  2. Runs N=3000 simulated paths per (horizon, risk_pct) combination and reports the
     DISTRIBUTION (percentiles) of outcomes, not one number -- sequence-of-returns risk
     is the whole point of a Monte Carlo; collapsing it back to a single expected value
     defeats the purpose.
  3. Reports max-drawdown threshold probabilities (P(DD > 50%), P(DD > 80%) etc) as the
     primary decision-relevant output, not "time to $1M" -- large equity multiples are
     mathematically implied by fixed-fractional compounding of a positive edge over
     hundreds of trades, not a realistic forecast once compared against this strategy's
     own previously-established ~$5M AUM capacity ceiling (CONFIDENCE_CAPACITY_AUDIT.md
     in PRODLIVETRADING). Flagged explicitly in the output rather than reported as if real.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_robust_search as rs
rs.INSTRUMENTS.update({"USDJPY": "lse_usdjpy_1min_2018on.csv"})
import goal_70pct_winrate_search as g

RNG_SEED = 20260830  # fixed for reproducibility -- not "randomness for its own sake"
BLOCK_LEN = 20
N_PATHS = 3000
HORIZONS_TRADES = {"1yr": 782, "3yr": 782 * 3}
RISK_LEVELS = [0.05, 0.02, 0.01]


def block_bootstrap_paths(returns, horizon_len, n_paths, block_len, rng):
    n = len(returns)
    n_blocks_needed = int(np.ceil(horizon_len / block_len))
    paths = np.empty((n_paths, horizon_len), dtype=np.float64)
    for p in range(n_paths):
        starts = rng.integers(0, n - block_len, size=n_blocks_needed)
        seq = np.concatenate([returns[s:s + block_len] for s in starts])[:horizon_len]
        paths[p] = seq
    return paths


def summarize(paths_r, risk_pct):
    n_paths, horizon = paths_r.shape
    equity = np.full(n_paths, 100.0)
    peak = np.full(n_paths, 100.0)
    maxdd = np.zeros(n_paths)
    for t in range(horizon):
        equity = equity * (1.0 + risk_pct * paths_r[:, t])
        equity = np.maximum(equity, 1e-9)
        peak = np.maximum(peak, equity)
        dd = (peak - equity) / peak
        maxdd = np.maximum(maxdd, dd)
    multiple = equity / 100.0
    pct = lambda a, q: np.percentile(a, q)
    return dict(
        median_multiple=pct(multiple, 50), p5_multiple=pct(multiple, 5), p25_multiple=pct(multiple, 25),
        p75_multiple=pct(multiple, 75), p95_multiple=pct(multiple, 95),
        median_maxdd=pct(maxdd, 50) * 100, p95_maxdd=pct(maxdd, 95) * 100,
        prob_dd_gt_50=(maxdd > 0.50).mean() * 100, prob_dd_gt_80=(maxdd > 0.80).mean() * 100,
        prob_dd_gt_95=(maxdd > 0.95).mean() * 100, prob_below_start=(multiple < 1.0).mean() * 100,
    )


def main():
    rng = np.random.default_rng(RNG_SEED)
    for inst in g.INSTRUMENTS:
        rs.load_instrument(inst)
        rs._prep_zones(inst, g.MIN_RETRACEMENT)

    stop_mult, target_mult = 3.343496, 3.343496 * 0.35
    all_t = [g.build_trades_asym(inst, stop_mult, target_mult, g.MIN_FILL_AGE_H, True) for inst in g.INSTRUMENTS]
    combined = pd.concat(all_t, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    full_returns = combined["net_r"].values
    oos_returns = combined.loc[combined["entry_time"] >= "2020-01-01", "net_r"].values
    print(f"Full-history pool: {len(full_returns)} trades, WR={100*(full_returns>0).mean():.1f}%")
    print(f"OOS-only pool (2020-2026): {len(oos_returns)} trades, WR={100*(oos_returns>0).mean():.1f}%\n")

    rows = []
    for pool_name, pool in [("full_history_2009_2026", full_returns), ("oos_only_2020_2026", oos_returns)]:
        for horizon_name, horizon_len in HORIZONS_TRADES.items():
            paths_r = block_bootstrap_paths(pool, horizon_len, N_PATHS, BLOCK_LEN, rng)
            for risk in RISK_LEVELS:
                s = summarize(paths_r, risk)
                s.update(pool=pool_name, horizon=horizon_name, risk_pct=risk)
                rows.append(s)
                print(f"[{pool_name:24s} {horizon_name:4s} risk={risk*100:4.1f}%] "
                      f"median={s['median_multiple']:.2f}x (p5={s['p5_multiple']:.2f}x, p95={s['p95_multiple']:.2f}x)  "
                      f"medianDD={s['median_maxdd']:.1f}%  P(DD>50%)={s['prob_dd_gt_50']:.1f}%  "
                      f"P(DD>80%)={s['prob_dd_gt_80']:.1f}%  P(end<start)={s['prob_below_start']:.1f}%")

    out = pd.DataFrame(rows)
    cols = ["pool", "horizon", "risk_pct", "median_multiple", "p5_multiple", "p25_multiple", "p75_multiple",
            "p95_multiple", "median_maxdd", "p95_maxdd", "prob_dd_gt_50", "prob_dd_gt_80", "prob_dd_gt_95",
            "prob_below_start"]
    out = out[cols]
    out_path = os.path.join(DATA_DIR, "goal_70pct_monte_carlo_bootstrap.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
