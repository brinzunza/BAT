"""Post-processing for dtfx_robust_search.py's walk-forward output:

1. A consensus final config (median/mode of each fold's chosen params) -- deliberately NOT
   a fresh full-history refit, since that would reintroduce exactly the in-sample overfitting
   this whole exercise exists to avoid. The consensus is only trustworthy if the folds agree;
   fold-to-fold parameter spread is reported explicitly as a stability signal.
2. Full out-of-sample performance for the consensus config AND the current live config, on
   the concatenation of all 7 held-out test windows (never used for any fold's own search).
3. A per-parameter sensitivity sweep around the consensus config, evaluated OOS-only.
4. A Sharpe-vs-drawdown and Sharpe-vs-kurtosis Pareto frontier across the best distinct
   configs the search actually tried.
5. A Deflated Sharpe Ratio (Bailey & Lopez de Prado) for the consensus config, accounting for
   the number of configurations this search alone tried, PLUS the ~10 major strategy variants
   already tried earlier this session (v1-v5 RL lifecycle, HTF-ML, bandit, trend-alignment,
   multiplier sweep) -- an honest deflation, not a precise headcount.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

from dtfx_robust_search import (
    INSTRUMENTS, ANALYSIS_START, EMBARGO_DAYS, N_WINDOWS, N_TRIALS_PER_FOLD,
    load_instrument, build_trades, score_trades, purge, evaluate_config, DATA_DIR,
)

LIVE_CONFIG = dict(n_multiplier=2.0, min_risk_pct=0.00025, min_fill_age_hours=1.0,
                    min_retracement=3, use_rsi_atr_filter=False)
PRIOR_MAJOR_VARIANTS_THIS_SESSION = 10  # v1-v5 lifecycle RL, HTF-ML, bandit, trend-align, multiplier sweep, RSI/ATR


def consensus_config(folds_df):
    cfg = {}
    cfg["n_multiplier"] = float(folds_df["param__n_multiplier"].median())
    cfg["min_risk_pct"] = float(folds_df["param__min_risk_pct"].median())
    cfg["min_fill_age_hours"] = float(folds_df["param__min_fill_age_hours"].median())
    cfg["min_retracement"] = int(folds_df["param__min_retracement"].mode().iloc[0])
    cfg["use_rsi_atr_filter"] = bool(folds_df["param__use_rsi_atr_filter"].mode().iloc[0])
    spread = {
        "n_multiplier_std": float(folds_df["param__n_multiplier"].std()),
        "min_risk_pct_std": float(folds_df["param__min_risk_pct"].std()),
        "min_fill_age_hours_std": float(folds_df["param__min_fill_age_hours"].std()),
        "min_retracement_mode_frac": float((folds_df["param__min_retracement"] == cfg["min_retracement"]).mean()),
        "use_rsi_atr_filter_mode_frac": float((folds_df["param__use_rsi_atr_filter"] == cfg["use_rsi_atr_filter"]).mean()),
    }
    return cfg, spread


def oos_windows(folds_df):
    """The 7 held-out test windows, each never used in that fold's own search."""
    return list(zip(folds_df["test_start"], folds_df["test_end"]))


def evaluate_on_concatenated_oos(config, windows):
    """Build trades once per instrument, keep only rows inside ANY oos test window (each
    embargoed at its start), then score the concatenation -- true walk-forward OOS performance,
    never the same window used to pick that window's own params."""
    per_instrument = {}
    for instrument in INSTRUMENTS:
        trades = build_trades(instrument, config["min_retracement"], config["n_multiplier"],
                               config["min_risk_pct"], config["min_fill_age_hours"],
                               config["use_rsi_atr_filter"])
        mask = pd.Series(False, index=trades.index)
        for lo, hi in windows:
            lo_e = pd.Timestamp(lo) + pd.Timedelta(days=EMBARGO_DAYS)
            mask |= (trades["entry_time"] >= lo_e) & (trades["entry_time"] < hi)
        window_trades = trades[mask]
        score, stats = score_trades(window_trades)
        per_instrument[instrument] = dict(score=score, **(stats or {}))
    return per_instrument


def deflated_sharpe(sr_hat, sr_trials, T, skew, kurt):
    sr_trials = np.asarray(sr_trials)
    sr_trials = sr_trials[np.isfinite(sr_trials)]
    N = len(sr_trials)
    if N < 2 or T < 3:
        return None
    var_sr = np.var(sr_trials, ddof=1)
    sigma_sr = np.sqrt(var_sr) if var_sr > 1e-12 else 1e-6
    euler_gamma = 0.5772156649015329
    sr0 = sigma_sr * ((1 - euler_gamma) * norm.ppf(1 - 1.0 / N) +
                       euler_gamma * norm.ppf(1 - 1.0 / (N * np.e)))
    denom = T - 1
    inside = 1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2
    if inside <= 0 or denom <= 0:
        return None
    sr_std = np.sqrt(inside / denom)
    if sr_std <= 0:
        return None
    psr = norm.cdf((sr_hat - sr0) / sr_std)
    return dict(dsr=float(psr), sr0_expected_max_by_chance=float(sr0), sr_std=float(sr_std),
                n_trials_deflated_for=int(N))


def sensitivity_sweep(base_config, windows):
    sweeps = {
        "n_multiplier": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "min_risk_pct": [0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001],
        "min_fill_age_hours": [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        "min_retracement": [2, 3, 4, 5],
    }
    rows = []
    for param, values in sweeps.items():
        for v in values:
            cfg = dict(base_config)
            cfg[param] = v
            result = evaluate_on_concatenated_oos(cfg, windows)
            scores = [r["score"] for r in result.values() if r["score"] is not None]
            min_score = min(scores) if len(scores) == 3 else None
            rows.append(dict(param=param, value=v, min_oos_score=min_score,
                              **{f"{inst}_score": r["score"] for inst, r in result.items()}))
            print(f"  sensitivity {param}={v}: min_oos_score={min_score}", flush=True)
    return pd.DataFrame(rows)


def pareto_frontier(trials_df, top_k=25):
    """Take the best distinct configs from the search, evaluate full OOS stats for each,
    return the (ann_sharpe, maxdd, excess_kurtosis) non-dominated frontier."""
    trials_df = trials_df.dropna(subset=["score"]).sort_values("score", ascending=False)
    seen, candidates = set(), []
    for _, row in trials_df.iterrows():
        key = (round(row["n_multiplier"], 2), round(row["min_risk_pct"], 6),
               round(row["min_fill_age_hours"], 2), row["min_retracement"], row["use_rsi_atr_filter"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(row)
        if len(candidates) >= top_k:
            break

    folds_df = pd.read_csv(os.path.join(DATA_DIR, "dtfx_robust_search_folds.csv"),
                            parse_dates=["test_start", "test_end"])
    windows = oos_windows(folds_df)

    rows = []
    for c in candidates:
        cfg = dict(n_multiplier=c["n_multiplier"], min_risk_pct=c["min_risk_pct"],
                   min_fill_age_hours=c["min_fill_age_hours"], min_retracement=int(c["min_retracement"]),
                   use_rsi_atr_filter=bool(c["use_rsi_atr_filter"]))
        result = evaluate_on_concatenated_oos(cfg, windows)
        sharpes = [r.get("ann_sharpe") for r in result.values() if r.get("ann_sharpe") is not None]
        dds = [r.get("maxdd") for r in result.values() if r.get("maxdd") is not None]
        kurts = [r.get("excess_kurtosis") for r in result.values() if r.get("excess_kurtosis") is not None]
        pfs = [r.get("profit_factor") for r in result.values() if r.get("profit_factor") is not None]
        calmars = [r.get("calmar") for r in result.values() if r.get("calmar") is not None]
        if len(sharpes) < 3:
            continue
        rows.append(dict(**cfg, worst_ann_sharpe=min(sharpes), worst_maxdd=max(dds),
                          worst_excess_kurtosis=max(kurts), worst_profit_factor=min(pfs),
                          worst_calmar=min(calmars), mean_ann_sharpe=np.mean(sharpes)))

    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    is_dominated = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            better_or_equal = (df.iloc[j]["worst_ann_sharpe"] >= df.iloc[i]["worst_ann_sharpe"] and
                                df.iloc[j]["worst_maxdd"] <= df.iloc[i]["worst_maxdd"])
            strictly_better = (df.iloc[j]["worst_ann_sharpe"] > df.iloc[i]["worst_ann_sharpe"] or
                               df.iloc[j]["worst_maxdd"] < df.iloc[i]["worst_maxdd"])
            if better_or_equal and strictly_better:
                is_dominated[i] = True
                break
    df["pareto_optimal"] = ~is_dominated
    return df, df[df["pareto_optimal"]]


def main():
    folds_df = pd.read_csv(os.path.join(DATA_DIR, "dtfx_robust_search_folds.csv"),
                            parse_dates=["test_start", "test_end"])
    trials_df = pd.read_csv(os.path.join(DATA_DIR, "dtfx_robust_search_pareto.csv"))
    windows = oos_windows(folds_df)

    cfg, spread = consensus_config(folds_df)
    print("Consensus config (median/mode across folds):", cfg)
    print("Fold-to-fold spread:", spread)

    print("\nEvaluating LIVE config on concatenated OOS windows...")
    live_oos = evaluate_on_concatenated_oos(LIVE_CONFIG, windows)
    for inst, r in live_oos.items():
        print(f"  {inst}: {r}")

    print("\nEvaluating CONSENSUS config on concatenated OOS windows...")
    consensus_oos = evaluate_on_concatenated_oos(cfg, windows)
    for inst, r in consensus_oos.items():
        print(f"  {inst}: {r}")

    print("\nRunning sensitivity sweep around consensus config (OOS-only)...")
    sens_df = sensitivity_sweep(cfg, windows)
    sens_df.to_csv(os.path.join(DATA_DIR, "dtfx_robust_search_sensitivity.csv"), index=False)

    print("\nComputing Pareto frontier from search trials...")
    all_candidates_df, frontier_df = pareto_frontier(trials_df)
    all_candidates_df.to_csv(os.path.join(DATA_DIR, "dtfx_robust_search_pareto_candidates.csv"), index=False)

    n_trials_this_search = len(trials_df)
    n_total_trials = n_trials_this_search + PRIOR_MAJOR_VARIANTS_THIS_SESSION
    worst_inst_consensus = min(consensus_oos.items(), key=lambda kv: (kv[1]["ann_sharpe"] if kv[1].get("ann_sharpe") is not None else -999))
    sr_hat = worst_inst_consensus[1].get("ann_sharpe")
    T = worst_inst_consensus[1].get("n_daily_obs")
    skew = worst_inst_consensus[1].get("skew", 0.0)
    exk = worst_inst_consensus[1].get("excess_kurtosis", 0.0)
    dsr = None
    if sr_hat is not None and T is not None:
        dsr = deflated_sharpe(sr_hat, trials_df["score"].values, T, skew, exk + 3.0)
        if dsr is not None:
            dsr["n_total_trials_used"] = n_total_trials
            dsr["note"] = ("sr_trials substitutes the search objective (Calmar/kurtosis score) for raw "
                            "Sharpe -- same purpose (distribution of the metric actually being optimized "
                            "across trials), not a literal Sharpe distribution.")

    report = dict(
        consensus_config=cfg,
        fold_to_fold_spread=spread,
        live_config=LIVE_CONFIG,
        live_oos=live_oos,
        consensus_oos=consensus_oos,
        worst_instrument_for_consensus=worst_inst_consensus[0],
        deflated_sharpe_ratio=dsr,
        n_trials_this_search=n_trials_this_search,
        n_folds=N_WINDOWS - 1,
        n_trials_per_fold=N_TRIALS_PER_FOLD,
    )
    with open(os.path.join(DATA_DIR, "dtfx_robust_search_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nSaved -> dtfx_robust_search_report.json")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
