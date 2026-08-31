"""Composite strategy score balancing 5 factors: speed to $1M, drawdown, win-rate
robustness, trade frequency, and R:R. Every input uses each strategy's TRUE
out-of-sample holdout trade log (selection rule chosen on first-half data only,
scored on the untouched second half) -- CANA/SOMEGA/goal70 have no fitted selection
rule, so their full 16.9-year history serves as their "holdout" (nothing to hold out
from). This keeps the score consistent with the rest of this project's discipline:
no factor gets to look good just because it was fit to the data it's being judged on.

Each factor is min-max normalized to 0-100 across the 9 candidates (100 = best),
then combined with configurable weights (default: equal 20% each).
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from goal10_cana_nt_mc_capped import run_mc_capped, BLOCK, MAX_TRADES

N_SIMS = 4000

# label -> (holdout trade file, R:R for this config)
STRATEGIES = {
    "CANA-NarrowTarget":   ("goal9_cana_narrowtarget_true_holdout.csv", 0.80),
    "SOMEGA-NarrowTarget": ("goal9_somega_narrowtarget_true_holdout.csv", 0.70),
    "DTFX-NarrowTarget":   ("goal9_dtfx_narrowtarget_true_holdout.csv", 0.70),
    "CANA+":               ("goal8_cana_plus_true_holdout.csv", 1.00),
    "DTFX Robust":         ("goal8_dtfxrobust_true_holdout.csv", 1.00),
    "SOMEGA Robust":       ("goal8_somegarobust_true_holdout.csv", 1.00),
    "CANA (faithful)":     ("cana_faithful_trades.csv", 1.00),          # no selection -> full history
    "SOMEGA (live)":       ("somega_faithful_trades.csv", 1.00),        # no selection -> full history
    "goal70_final_winner": ("goal4_goal70_nogold_trades.csv", 1.00),    # no selection -> full history
}

DEFAULT_WEIGHTS = dict(speed=0.20, drawdown=0.20, wr=0.20, freq=0.20, rrr=0.20)


def minmax_norm(values, higher_is_better):
    """0-100, 100=best. values: dict label->float"""
    vals = np.array(list(values.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {k: 100.0 for k in values}
    out = {}
    for k, v in values.items():
        frac = (v - lo) / (hi - lo)
        out[k] = 100 * frac if higher_is_better else 100 * (1 - frac)
    return out


def main():
    raw = {}
    for label, (fname, rr) in STRATEGIES.items():
        t = pd.read_csv(os.path.join(DATA_DIR, fname), parse_dates=["entry_time"]).sort_values("entry_time")
        net_r = t.net_r.values.astype(np.float64)
        years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
        tpy = len(t) / years
        wr = 100 * (net_r > 0).mean()

        t1_raw, t10_raw, max_dd, ruined = run_mc_capped(net_r, N_SIMS, BLOCK, MAX_TRADES, seed=abs(hash(label)) % 100000)
        t1_hit = t1_raw[t1_raw > 0]
        median_yrs_1m = np.median(t1_hit / tpy) if len(t1_hit) else float("inf")
        median_dd = np.median(max_dd) * 100  # negative number, e.g. -75.3

        raw[label] = dict(trades=len(t), tpy=tpy, wr=wr, rr=rr,
                           median_yrs_1m=median_yrs_1m, median_dd=median_dd,
                           pct_ruined=100 * ruined.mean())
        print(f"{label:20s} n={len(t):6,} ({tpy:6.0f}/yr) WR={wr:5.2f}% RR={rr:.2f} "
              f"yrs->1M={median_yrs_1m:.3f} medDD={median_dd:6.1f}% ruin={100*ruined.mean():.2f}%")

    # ---------- Normalize each factor ----------
    speed_n = minmax_norm({k: v["median_yrs_1m"] for k, v in raw.items()}, higher_is_better=False)
    dd_n = minmax_norm({k: v["median_dd"] for k, v in raw.items()}, higher_is_better=True)  # less negative = better
    wr_n = minmax_norm({k: v["wr"] for k, v in raw.items()}, higher_is_better=True)
    freq_n = minmax_norm({k: v["tpy"] for k, v in raw.items()}, higher_is_better=True)
    rrr_n = minmax_norm({k: v["rr"] for k, v in raw.items()}, higher_is_better=True)

    def composite(weights):
        scores = {}
        for k in raw:
            scores[k] = (weights["speed"] * speed_n[k] + weights["drawdown"] * dd_n[k] +
                         weights["wr"] * wr_n[k] + weights["freq"] * freq_n[k] +
                         weights["rrr"] * rrr_n[k])
        return scores

    default_scores = composite(DEFAULT_WEIGHTS)

    rows = []
    for k in raw:
        rows.append(dict(
            strategy=k, **raw[k],
            speed_score=round(speed_n[k], 1), dd_score=round(dd_n[k], 1),
            wr_score=round(wr_n[k], 1), freq_score=round(freq_n[k], 1),
            rrr_score=round(rrr_n[k], 1), composite_score=round(default_scores[k], 1),
        ))
    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(os.path.join(DATA_DIR, "goal12_strategy_scores.csv"), index=False)

    print("\n" + "=" * 100)
    print("COMPOSITE STRATEGY SCORE (equal 20% weights: speed, drawdown, WR, frequency, RRR)")
    print("=" * 100)
    print(df[["rank", "strategy", "composite_score", "speed_score", "dd_score", "wr_score",
              "freq_score", "rrr_score"]].to_string(index=False))
    print(f"\nSaved: goal12_strategy_scores.csv")

    # sensitivity: a "safety-weighted" alternative (drawdown 40%, rest 15% each)
    safety_weights = dict(speed=0.15, drawdown=0.40, wr=0.15, freq=0.15, rrr=0.15)
    safety_scores = composite(safety_weights)
    print("\n--- Sensitivity check: drawdown-weighted (40% DD, 15% each other) ---")
    for k, v in sorted(safety_scores.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:.1f}  (equal-weight was {default_scores[k]:.1f})")


if __name__ == "__main__":
    main()
