"""Canonical, mathematically correct Monte Carlo evaluator -- promoted to its canonical home
from `research_scripts/beid_goal_hunt/mc_true_median.py` (kept there as a one-line re-export
shim so nothing that already imports it breaks).

Written in response to an external review that caught a serious, pervasive error: computing
`np.median(results[reached])` -- the median of ONLY the paths that eventually reach the
target, however long that takes -- as if it were the median time-to-reach. It is not: it's the
50th percentile of a subset that excludes every ruined/non-reaching path, which sits at a
percentile of the FULL distribution well above 50% whenever P(reach) < 100%, since ruined paths
sort as +infinity and land above every reached path. The true median requires sorting ALL
simulated outcomes (ruined/non-reaching counted as +infinity) and taking the 50th percentile of
THAT list -- equivalently, the smallest T such that P(reach within T years) >= 50%.

This must be the ONLY statistics path any Monte Carlo engine in this codebase uses.
`combined_fx_gold_mc.py` still computes the old, invalidated statistic directly and should not
be used as reference logic for anything margin-aware.
"""
from __future__ import annotations

import numpy as np


def true_median_and_stats(results, ruins, n_sims: int) -> dict:
    """results: array of years-to-reach (>=0) or -1.0 (ruined or ran out of simulated trades
    without reaching or ruining). Returns the mathematically correct all-path statistics."""
    results = np.asarray(results)
    reached = results >= 0
    p_reach_ever = 100 * reached.mean()
    p_ruin = 100 * np.asarray(ruins).mean()
    allvals = np.where(reached, results, np.inf)
    allvals_sorted = np.sort(allvals)
    true_median = allvals_sorted[len(allvals_sorted) // 2]
    p_reach_within_1yr = 100 * (reached & (results <= 1.0)).mean()
    return dict(
        true_median_years=true_median,
        p_reach_ever=p_reach_ever,
        p_reach_within_1yr=p_reach_within_1yr,
        p_ruin=p_ruin,
        median_of_reached_only=(np.median(results[reached]) if reached.any() else float("inf")),
    )
