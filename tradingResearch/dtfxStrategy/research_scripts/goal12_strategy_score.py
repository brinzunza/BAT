"""Composite strategy score balancing 6 factors: speed to $1M, drawdown, win-rate
robustness, trade frequency, margin capacity, and R:R. Every input uses each
strategy's TRUE out-of-sample holdout trade log (selection rule chosen on first-half
data only, scored on the untouched second half) -- CANA/SOMEGA/goal70 have no fitted
selection rule, so their full 16.9-year history serves as their "holdout" (nothing to
hold out from). This keeps the score consistent with the rest of this project's
discipline: no factor gets to look good just because it was fit to the data it's
being judged on.

Margin capacity (added after the Speed-King margin-realism investigation): at $200
start / 5% risk / 50x leverage, most of a strategy's DESIGNED trade population gets
rejected because concurrent positions run out of margin before an open position
closes -- every OTHER factor here (speed, frequency, WR, drawdown) is computed on the
full unconstrained trade stream the backtest generates, silently assuming unlimited
concurrency the whole time. Margin capacity is the % of that population that
actually survives goal20's margin-constrained walk (see
goal23_margin_capacity_all_strategies.py) -- correlates moderately with speed
(r=0.585) and inversely with raw frequency (r=-0.496), but only weakly with drawdown
(r=0.137): real, mostly-independent information, not a redundant restatement of
frequency (a strategy with more overlap opportunities loses more edge to margin, but
not in strict proportion to its raw trade count -- stop width and instrument mix
matter too).

Each factor is min-max normalized to 0-100 across the 14 candidates (100 = best),
then combined with configurable weights (default weighting below).
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

from goal10_cana_nt_mc_capped import run_mc_capped, BLOCK, MAX_TRADES

N_SIMS = 50000

# label -> (holdout trade file, R:R for this config)
STRATEGIES = {
    "Speed-King":          ("goal19_speed_king_true_holdout.csv", 0.20),
    "Quality-King":        ("goal19_quality_king_true_holdout.csv", 0.35),
    "Hybrid-Ultra":        ("goal10_hybrid_ultra_true_holdout.csv", 0.40),
    "DTFX-Ultra":          ("goal10_dtfx_ultra_true_holdout.csv", 0.40),
    "SOMEGA-Ultra":        ("goal10_somega_ultra_true_holdout.csv", 0.40),
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

# FINALIZED weights (was equal 20% each, then speed=35/wr=30/freq=20/dd=10/rrr=5
# before margin capacity was added). Original correlation analysis across the first
# 12 candidates showed frequency is the dominant driver of speed (r=-0.867 vs
# median-years-to-$1M) while win rate is the dominant driver of drawdown-safety
# (r=+0.801 vs median drawdown) -- two genuinely different jobs, not redundant with
# each other. Speed is kept as its own factor because it captures the interaction
# between frequency, WR, and R:R that neither raw input sees alone.
#
# Margin capacity is conceptually closest to frequency (both measure "how much can
# you actually trade"), so it's carved out of frequency's own budget rather than
# added on top or taken from WR/speed/drawdown: freq 20%->12%, margin 8% (new),
# keeping the combined "how much trading can actually happen" weight at 20%, now
# split between the naive (raw signal count) and realistic (margin-survives) views
# of it. Kept deliberately smaller than frequency's own share since margin capacity
# is a correction/refinement of frequency, not an equally-fundamental, independent
# axis the way WR or drawdown are.
DEFAULT_WEIGHTS = dict(speed=0.35, wr=0.30, freq=0.12, margin=0.08, drawdown=0.10, rrr=0.05)
EQUAL_WEIGHTS = dict(speed=1 / 6, drawdown=1 / 6, wr=1 / 6, freq=1 / 6, margin=1 / 6, rrr=1 / 6)


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
    margin_capacity = pd.read_csv(os.path.join(DATA_DIR, "goal23_margin_capacity.csv")).set_index("strategy")["accept_pct"]

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
        margin_pct = margin_capacity[label]

        raw[label] = dict(trades=len(t), tpy=tpy, wr=wr, rr=rr,
                           median_yrs_1m=median_yrs_1m, median_dd=median_dd,
                           pct_ruined=100 * ruined.mean(), margin_pct=margin_pct)
        print(f"{label:20s} n={len(t):6,} ({tpy:6.0f}/yr) WR={wr:5.2f}% RR={rr:.2f} "
              f"yrs->1M={median_yrs_1m:.3f} medDD={median_dd:6.1f}% margin={margin_pct:5.1f}% "
              f"ruin={100*ruined.mean():.2f}%")

    # ---------- Normalize each factor ----------
    speed_n = minmax_norm({k: v["median_yrs_1m"] for k, v in raw.items()}, higher_is_better=False)
    dd_n = minmax_norm({k: v["median_dd"] for k, v in raw.items()}, higher_is_better=True)  # less negative = better
    wr_n = minmax_norm({k: v["wr"] for k, v in raw.items()}, higher_is_better=True)
    freq_n = minmax_norm({k: v["tpy"] for k, v in raw.items()}, higher_is_better=True)
    margin_n = minmax_norm({k: v["margin_pct"] for k, v in raw.items()}, higher_is_better=True)
    rrr_n = minmax_norm({k: v["rr"] for k, v in raw.items()}, higher_is_better=True)

    def composite(weights):
        scores = {}
        for k in raw:
            scores[k] = (weights["speed"] * speed_n[k] + weights["drawdown"] * dd_n[k] +
                         weights["wr"] * wr_n[k] + weights["freq"] * freq_n[k] +
                         weights["margin"] * margin_n[k] + weights["rrr"] * rrr_n[k])
        return scores

    final_scores = composite(DEFAULT_WEIGHTS)
    equal_scores = composite(EQUAL_WEIGHTS)

    rows = []
    for k in raw:
        rows.append(dict(
            strategy=k, **raw[k],
            speed_score=round(speed_n[k], 1), dd_score=round(dd_n[k], 1),
            wr_score=round(wr_n[k], 1), freq_score=round(freq_n[k], 1),
            margin_score=round(margin_n[k], 1), rrr_score=round(rrr_n[k], 1),
            final_score=round(final_scores[k], 1), equal_weight_score=round(equal_scores[k], 1),
        ))
    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(os.path.join(DATA_DIR, "goal12_strategy_scores.csv"), index=False)

    w = DEFAULT_WEIGHTS
    print("\n" + "=" * 100)
    print(f"FINAL STRATEGY SCORE  (weights: speed={w['speed']:.0%}, WR={w['wr']:.0%}, "
          f"freq={w['freq']:.0%}, margin={w['margin']:.0%}, drawdown={w['drawdown']:.0%}, RRR={w['rrr']:.0%})")
    print("=" * 100)
    print(df[["rank", "strategy", "final_score", "speed_score", "wr_score", "freq_score",
              "margin_score", "dd_score", "rrr_score", "equal_weight_score"]].to_string(index=False))
    print(f"\nSaved: goal12_strategy_scores.csv")


if __name__ == "__main__":
    main()
