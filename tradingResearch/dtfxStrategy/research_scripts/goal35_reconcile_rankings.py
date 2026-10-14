"""Reconciles two parallel, conflicting STRATEGY_RANKINGS.md updates into one consistent
ranking: this session's v16 (16 candidates incl. Velocity-King, weights renormalized from
the project's historical 6-factor proportions -- 45.45/38.96/15.58) and a second,
concurrent session's uncommitted working-tree edit (different weights -- 40/35/25, no
stated derivation found -- and a 17th candidate, Titan-76, not present in v16).

Titan-76 (ret=1, N=2.0, R:R=0.76, age_h=0.20h, touches=6, probe=0.75) uses two dimensions
(MIN_RETRACEMENT, PROBE_DEPTH) no strategy in this project's history has varied before --
both fixed project-wide constants (2 and 0.50) in every other search including this
session's own goal34. Its own discovery script (goal40_tight_n_high_rr_sweep.py) screened
directly against entry_time >= SPLIT -- the TRUE HOLDOUT SET ITSELF -- the exact same
methodology error this session caught and fixed in its own discarded rounds 1-2 before
finding Velocity-King properly. That error isn't fixable after the fact by re-scoring the
same fixed candidate on the same holdout (the contamination is in which parameters got
chosen, not in how the final number gets computed) -- so Titan-76's numbers below carry a
real, unresolved risk of being optimistic that Velocity-King and every other named
strategy in this table (all originally selected on training data, matching this project's
own long-standing discipline) do not carry. Included and directly comparable on the SAME
methodology as everything else, but flagged, not silently treated as equally trustworthy.

Keeps this session's weight derivation (renormalized from the historical 6-factor
proportions, a stated rationale) rather than the other session's 40/35/25 (no derivation
found in any script) -- reconciliation needs ONE formula, and "keep the one with a
documented reason" is the least arbitrary tiebreak available.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal30_apex_king_deployment as g30
import goal33_rerank_3factor_two_account as g33
import goal42_titan76_two_account_backtest as g42

SPLIT = g29.SPLIT
WEIGHTS = g33.WEIGHTS  # 45.45/38.96/15.58, this session's derivation


def evaluate_titan76():
    for inst in g29.TRADEABLE:
        g29.rs.load_instrument(inst)
    all_trades = g42.build_titan_trades()  # full history, min_retracement=1 (not project-standard 2)

    train = all_trades[all_trades.entry_time < SPLIT].reset_index(drop=True)
    holdout = all_trades[all_trades.entry_time >= SPLIT].reset_index(drop=True)
    print(f"Titan-76: {len(all_trades):,} total, {len(train):,} train, {len(holdout):,} holdout")

    # Same risk sweep discipline as every other strategy in goal33 -- selected on TRAIN.
    best_risk, best_med_yrs_train, best_med_dd_train, sweep_df = g33.find_optimal_risk(
        train, n_paths=60, max_trades=250_000, seed_base=abs(hash("Titan-76")) % 100000)
    print(f"  optimal risk (from train): {100*best_risk:.1f}%")

    # Final reported numbers: TRUE holdout only, at that train-selected risk.
    t, eq_buy, eq_sell, max_dd, _ = g30.simulate_two_account(holdout, best_risk, track_detail=False)
    acc = t[t.accepted]
    years = (holdout.entry_time.max() - holdout.entry_time.min()).days / 365.25
    wr = 100 * acc.is_win.mean() if len(acc) else float("nan")
    acc_tpy = len(acc) / years

    # Speed (median days to $1M) on the true holdout via the real-timing MC, matching
    # every other strategy's own reported figure.
    t1s, t10s, dds, ruins, accs = g33.g30.run_mc(holdout, best_risk, n_paths=200, seed=9191, max_trades=250_000)
    reached = t1s >= 0
    med_yrs = np.median(t1s[reached]) if reached.any() else float("inf")
    med_dd = np.median(dds)

    print(f"  HOLDOUT: WR={wr:.2f}%  acc_tpy={acc_tpy:.0f}/yr  days->1M={med_yrs*365.25:.1f}  medDD={med_dd:.1f}%")
    return dict(strategy="Titan-76", wr=wr, acc_tpy=acc_tpy, optimal_risk=best_risk,
                median_yrs_1m=med_yrs, median_dd=med_dd)


def main():
    titan = evaluate_titan76()

    pool = pd.read_csv(os.path.join(DATA_DIR, "goal34_final_rerank_16.csv")).drop(columns=["rank"])
    keep_cols = ["strategy", "wr", "acc_tpy", "optimal_risk", "median_yrs_1m", "median_dd"]
    pool = pool[keep_cols]
    pool = pd.concat([pool, pd.DataFrame([{k: titan[k] for k in keep_cols}])], ignore_index=True)

    def minmax_norm(values, higher_is_better):
        vals = np.array(list(values.values()), dtype=float)
        lo, hi = vals.min(), vals.max()
        out = {}
        for k, v in values.items():
            frac = (v - lo) / (hi - lo)
            out[k] = 100 * frac if higher_is_better else 100 * (1 - frac)
        return out

    speed_n = minmax_norm(dict(zip(pool.strategy, pool.median_yrs_1m)), higher_is_better=False)
    wr_n = minmax_norm(dict(zip(pool.strategy, pool.wr)), higher_is_better=True)
    freq_n = minmax_norm(dict(zip(pool.strategy, pool.acc_tpy)), higher_is_better=True)
    final = {k: WEIGHTS["speed"] * speed_n[k] + WEIGHTS["wr"] * wr_n[k] + WEIGHTS["freq"] * freq_n[k]
             for k in speed_n}

    pool["final_score"] = pool.strategy.map(final)
    pool["speed_score"] = pool.strategy.map(speed_n)
    pool["wr_score"] = pool.strategy.map(wr_n)
    pool["freq_score"] = pool.strategy.map(freq_n)
    pool["days_to_1m"] = pool.median_yrs_1m * 365.25
    pool = pool.sort_values("final_score", ascending=False).reset_index(drop=True)
    pool.insert(0, "rank", range(1, len(pool) + 1))
    pool.to_csv(os.path.join(DATA_DIR, "goal35_reconciled_rerank_17.csv"), index=False)

    print("\n" + "=" * 100)
    print("RECONCILED 17-CANDIDATE RANKING")
    print("=" * 100)
    for _, r in pool.iterrows():
        flag = "  <-- discovery methodology unresolved (see docstring)" if r.strategy == "Titan-76" else ""
        print(f"{r['rank']:2.0f} | {r['strategy']:20s} | final={r['final_score']:5.1f} "
              f"speed={r['speed_score']:5.1f} wr={r['wr_score']:5.1f} freq={r['freq_score']:5.1f} | "
              f"risk={100*r['optimal_risk']:4.1f}% WR={r['wr']:5.2f}% acc_tpy={r['acc_tpy']:7.0f} "
              f"days1M={r['days_to_1m']:7.1f}{flag}")


if __name__ == "__main__":
    main()
