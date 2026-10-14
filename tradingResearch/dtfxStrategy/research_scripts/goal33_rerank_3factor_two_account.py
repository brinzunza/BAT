"""Simplified 3-factor composite score: win rate + frequency + speed-to-$1M only
(drops drawdown, margin/concurrency-as-its-own-factor, and R:R from the prior 6-factor
formula). Reranks all 15 STRATEGY_RANKINGS.md candidates under this deployment's exact
realistic setup: EURUSD/GBPUSD/EURGBP/USDJPY, two SEPARATE $100 sub-accounts (buy-only,
sell-only, never pooled -- goal30's two-account model, not goal29's single-shared-pool
approximation), 50x leverage, 1.6-pip spread, max 1 open position per side, and EACH
STRATEGY'S OWN MC-robust-optimal risk_pct (not one fixed risk applied to all 15 -- what
counts as "optimal" genuinely differs by strategy, as Apex-King's 6% vs. this project's
prior 3%-for-Speed-King already showed).

"Frequency" here means ACCEPTED trades/year under this exact restriction set (margin +
1-per-side cap), not the raw unconstrained signal count every other factor in this
project's history has used -- the user's own framing ("frequency, still following the
restrictions shown below") makes this explicit: a strategy doesn't get credit for
signals it structurally can't take.

Composite weights: renormalizes the prior 6-factor formula's speed/wr/freq proportions
(0.35/0.30/0.12) to sum to 1 now that margin/dd/rrr are gone -- 0.4545/0.3896/0.1558 --
rather than picking new numbers arbitrarily; keeps the same established priority order
(speed > WR > frequency) the project has used throughout.
"""
import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal30_apex_king_deployment as g30

STRATEGIES = g29.STRATEGIES  # already all 15 (goal23's 14 + Apex-King)
RISK_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
WEIGHTS = dict(speed=0.35 / 0.77, wr=0.30 / 0.77, freq=0.12 / 0.77)  # renormalized to sum 1


def find_optimal_risk(trades, n_paths, max_trades, seed_base):
    rows = []
    for i, risk in enumerate(RISK_GRID):
        t1s, t10s, dds, ruins, accs = g30.run_mc(trades, risk, n_paths=n_paths,
                                                  seed=seed_base + i, max_trades=max_trades)
        reached = t1s >= 0
        med_yrs = np.median(t1s[reached]) if reached.any() else float("inf")
        rows.append(dict(risk_pct=risk, median_yrs_1m=med_yrs, pct_ruin=100 * ruins.mean(),
                          pct_reached=100 * reached.mean(), median_dd=np.median(dds)))
    df = pd.DataFrame(rows)
    safe = df[df.pct_ruin == 0.0]
    pick_from = safe if len(safe) else df
    best = pick_from.loc[pick_from.median_yrs_1m.idxmin()]
    return float(best.risk_pct), float(best.median_yrs_1m), float(best.median_dd), df


def main():
    for inst in g29.TRADEABLE:
        g29.rs.load_instrument(inst)
        g29.rs._prep_zones(inst, g29.pg.MIN_RETRACEMENT)

    print(f"Rebuilding TRUE trade streams (with side) and sweeping risk for all {len(STRATEGIES)} strategies...")
    raw = {}
    all_sweeps = []
    for label, cfg in STRATEGIES.items():
        t0 = time.perf_counter()
        trades = g29.get_true_trades(label, cfg)
        years = (trades.entry_time.max() - trades.entry_time.min()).days / 365.25
        raw_tpy = len(trades) / years

        n_paths = 60 if raw_tpy >= 1000 else 40
        max_trades = 250_000 if raw_tpy >= 1000 else 500_000
        best_risk, best_med_yrs, best_med_dd, sweep_df = find_optimal_risk(
            trades, n_paths, max_trades, seed_base=abs(hash(label)) % 100000)
        sweep_df["strategy"] = label
        all_sweeps.append(sweep_df)

        t, eq_buy, eq_sell, max_dd, _ = g30.simulate_two_account(trades, best_risk, track_detail=False)
        acc = t[t.accepted]
        wr = 100 * acc.is_win.mean() if len(acc) else float("nan")
        acc_tpy = len(acc) / years

        raw[label] = dict(raw_trades=len(trades), raw_tpy=raw_tpy, wr=wr, acc_tpy=acc_tpy,
                           optimal_risk=best_risk, median_yrs_1m=best_med_yrs, median_dd=best_med_dd,
                           n_accepted=len(acc))
        dt = time.perf_counter() - t0
        print(f"{label:22s} risk*={100*best_risk:4.1f}%  WR={wr:5.2f}%  acc_tpy={acc_tpy:7.0f}/yr  "
              f"(raw={raw_tpy:7.0f}/yr)  yrs->1M={best_med_yrs:6.3f}  medDD={best_med_dd:6.1f}%  [{dt:5.1f}s]")

    pd.concat(all_sweeps, ignore_index=True).to_csv(
        os.path.join(DATA_DIR, "goal33_risk_sweeps_all_strategies.csv"), index=False)

    def minmax_norm(values, higher_is_better):
        vals = np.array(list(values.values()), dtype=float)
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            return {k: 100.0 for k in values}
        out = {}
        for k, v in values.items():
            frac = (v - lo) / (hi - lo)
            out[k] = 100 * frac if higher_is_better else 100 * (1 - frac)
        return out

    speed_n = minmax_norm({k: v["median_yrs_1m"] for k, v in raw.items()}, higher_is_better=False)
    wr_n = minmax_norm({k: v["wr"] for k, v in raw.items()}, higher_is_better=True)
    freq_n = minmax_norm({k: v["acc_tpy"] for k, v in raw.items()}, higher_is_better=True)

    final_scores = {k: (WEIGHTS["speed"] * speed_n[k] + WEIGHTS["wr"] * wr_n[k] +
                         WEIGHTS["freq"] * freq_n[k]) for k in raw}

    rows = []
    for k in raw:
        rows.append(dict(strategy=k, **raw[k], speed_score=round(speed_n[k], 1),
                          wr_score=round(wr_n[k], 1), freq_score=round(freq_n[k], 1),
                          final_score=round(final_scores[k], 1)))
    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(os.path.join(DATA_DIR, "goal33_rerank_3factor.csv"), index=False)

    print("\n" + "=" * 100)
    print("3-FACTOR RERANK (win rate + frequency + speed-to-$1M), two-account model, per-strategy optimal risk")
    print("=" * 100)
    print(df[["rank", "strategy", "final_score", "speed_score", "wr_score", "freq_score",
              "optimal_risk", "wr", "acc_tpy", "median_yrs_1m", "median_dd"]].to_string(index=False))
    print(f"\nSaved: goal33_rerank_3factor.csv, goal33_risk_sweeps_all_strategies.csv")


if __name__ == "__main__":
    main()
