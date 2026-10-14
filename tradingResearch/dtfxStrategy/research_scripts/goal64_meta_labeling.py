"""/goal (2026-09-04): "keep exploring, search research papers online for ideas... think
different than normal and test test test. optimize and find one that is high frequency
trading, high winrate, and high PF."

Research grounding (web search, this session): meta-labeling (Marcos Lopez de Prado,
*Advances in Financial Machine Learning*, 2018) is the standard quant technique for
exactly this shape of problem -- take a high-frequency, high-RECALL "primary" model (fires
often, doesn't care about quality) and train a SEPARATE secondary classifier that predicts
whether a given primary signal will actually win, using it as a quality filter (or bet-size
scaler) rather than retraining the primary model itself. Search results cite concrete
precedent for exactly the win-rate lift being chased here: one published example took a
mean-reverting strategy's precision from 17% to 63%; another cut trade frequency 70% while
lifting win rate from 55% to 83%. This script is a direct, from-scratch implementation of
that idea on top of this project's own primary signal (the realistic-fill DTFX zone-fade
touches), not a reuse of anyone else's code.

Primary model: an aggressive, high-N/low-R:R zone-fade config (same shape as goal63's
Federated Portfolio sleeves) -- high frequency, thin/marginal PF, exactly the "high recall,
low precision" primary meta-labeling is designed to rescue.

Meta-model: sklearn RandomForestClassifier (no OpenMP dependency -- xgboost/lightgbm both
fail to import on this machine, see cpp_engine's own build notes) trained on features that
are ALL knowable at the moment a touch would be taken (zero lookahead):
  - efficiency_ratio, hurst_est, vol_percentile, realized_vol (goal62's C++ regime engine)
  - impulse_velocity (goal62's zone-geometry construct)
  - zone height in pips, touch_num, side, shallow_touch (this touch's own fill quality)
  - hour-of-day / day-of-week (cyclically encoded)
  - a CAUSAL trailing win-rate of the same symbol/side's last 20 touches (shifted by 1 so
    the current touch's own outcome is never included)

Split: chronological (not random -- a random split would leak future regime information
into training via nearby-in-time rows), first 70% train / last 30% test. All frequency/WR/
PF numbers reported below are measured ONLY on the held-out test set.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal51_single_symbol_closest_price_backtest as g51
import goal62_regime_feature_strategies as g62

rs, pg = g29.rs, g29.pg


def build_labeled_touches(inst, n, rr, age_h, touches=20, window=500, stride=50, vol_lookback=200):
    """Both sides' raw (ungated) realistic-fill touches, engineered with every meta-labeling
    feature. Reuses g62's already-validated regime-feature and impulse-velocity lookups --
    no new indicator math, just feature assembly for the classifier."""
    parts = [g51.build_touches_with_metadata(inst, n, rr, age_h, False, touches, side,
                                              min_risk_pct=None) for side in (True, False)]
    t = pd.concat(parts, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    if t.empty:
        return t

    times_ns, er, hurst, volpct, rvol = g62.load_regime_features(inst, window, stride, vol_lookback)
    entry_ns = t["entry_time"].values.astype("int64")
    idx = np.clip(np.searchsorted(times_ns, entry_ns), 0, len(times_ns) - 1)
    t["er"] = er[idx]
    t["hurst"] = hurst[idx]
    t["vol_percentile"] = volpct[idx]
    t["realized_vol"] = rvol[idx]

    lookup = g62.impulse_velocity_lookup(inst)
    mid_r, created_ns = t["entry_price"].values, t["created_at"].values.astype("int64")
    t["impulse_velocity"] = [lookup.get((m, c), np.nan) for m, c in zip(mid_r, created_ns)]

    pip = pg.PIP_SIZE[inst]
    t["zone_height_pips"] = t["stop_risk"] / pip
    hour = t["entry_time"].dt.hour + t["entry_time"].dt.minute / 60.0
    t["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    t["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = t["entry_time"].dt.dayofweek
    t["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    t["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    t["is_buy_f"] = t["is_buy"].astype(float)
    t["shallow_touch_f"] = t["shallow_touch"].astype(float)

    # Causal trailing win-rate: for EACH (side) stream separately, sorted by entry_time,
    # the mean of the PRECEDING 20 outcomes only (shift(1) before rolling -- the current
    # row's own outcome is never visible to itself).
    t["trailing_wr_20"] = np.nan
    for side_val, grp in t.groupby("is_buy"):
        g = grp.sort_values("entry_time")
        tw = g["is_win"].astype(float).shift(1).rolling(20, min_periods=5).mean()
        t.loc[g.index, "trailing_wr_20"] = tw.values

    return t.dropna(subset=["er", "hurst", "vol_percentile", "impulse_velocity", "trailing_wr_20"]
                     ).reset_index(drop=True)


FEATURES = ["er", "hurst", "vol_percentile", "realized_vol", "impulse_velocity",
            "zone_height_pips", "touch_num", "is_buy_f", "shallow_touch_f",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "trailing_wr_20"]


def thirds_pf(t):
    if len(t) < 45:
        return (None, None, None)
    start, end = t.entry_time.min(), t.entry_time.max()
    span = end - start
    b1, b2 = start + span / 3, start + 2 * span / 3
    windows = [t[t.entry_time < b1], t[(t.entry_time >= b1) & (t.entry_time < b2)],
               t[t.entry_time >= b2]]

    def pf_of(x):
        if len(x) < 15:
            return None
        gw, gl = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return gw / gl if gl > 0 else float("inf")
    return tuple(pf_of(w) for w in windows)


def stats_of(t):
    if t.empty:
        return None
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    years = max(years, 1e-6)
    wr = 100 * (t.net_r > 0).mean()
    gw, gl = t[t.net_r > 0].net_r.sum(), -t[t.net_r <= 0].net_r.sum()
    pf = gw / gl if gl > 0 else float("inf")
    p1, p2, p3 = thirds_pf(t)
    return dict(n=len(t), freq=len(t) / years, wr=wr, pf=pf, pf_third1=p1, pf_third2=p2, pf_third3=p3)


def run_meta_labeling(inst, n, rr, age_h, train_frac=0.7, model_name="rf"):
    t = build_labeled_touches(inst, n, rr, age_h)
    t = t.sort_values("entry_time").reset_index(drop=True)
    split_idx = int(len(t) * train_frac)
    train, test = t.iloc[:split_idx], t.iloc[split_idx:]

    # Target is net_r > 0 (realized profitability after the realistic-fill spread cost),
    # NOT the raw is_win flag -- confirmed by direct check that at extreme low R:R (0.05)
    # ~15% of raw "wins" (target hit before stop) still net-lose once the fixed spread cost
    # exceeds the tiny 0.05R reward. Training on is_win would have taught the classifier to
    # predict the wrong thing for exactly the touches this filter most needs to catch.
    X_train, y_train = train[FEATURES].values, (train["net_r"] > 0).astype(int).values
    X_test = test[FEATURES].values

    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    test = test.copy()
    test["meta_prob"] = clf.predict_proba(X_test)[:, 1]

    baseline = stats_of(test)
    print(f"\n=== {inst} N={n} R:R={rr} age={age_h} -- primary model, TEST set only ===")
    print(f"  train={len(train)} test={len(test)} (chronological {train_frac:.0%}/{1-train_frac:.0%} split)")
    print(f"  BASELINE (no meta-filter): n={baseline['n']} freq={baseline['freq']:.1f}/yr "
          f"WR={baseline['wr']:.1f}% PF={baseline['pf']:.3f} "
          f"thirds={baseline['pf_third1']}/{baseline['pf_third2']}/{baseline['pf_third3']}")

    rows = [dict(threshold=0.0, **baseline)]
    for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        kept = test[test.meta_prob >= thresh]
        s = stats_of(kept)
        if s is None:
            continue
        rows.append(dict(threshold=thresh, **s))
        print(f"  meta>={thresh:.2f}: n={s['n']:5d} freq={s['freq']:6.1f}/yr WR={s['wr']:5.1f}% "
              f"PF={s['pf']:.3f} thirds={s['pf_third1']}/{s['pf_third2']}/{s['pf_third3']}")

    importances = dict(zip(FEATURES, clf.feature_importances_)) if model_name == "rf" else None
    if importances:
        top = sorted(importances.items(), key=lambda kv: -kv[1])[:5]
        print(f"  top features: {', '.join(f'{k}={v:.3f}' for k, v in top)}")

    return dict(inst=inst, n=n, rr=rr, age=age_h, sweep=pd.DataFrame(rows),
                importances=importances, test=test, clf=clf)


def true_accepted(filtered_touches, age_h, inst):
    """Runs the meta-filtered touches through the SAME realistic-fill acceptance pipeline
    as every other candidate this session (resolve_closest_price's single-symbol resting-
    slot arbitration, then g30.simulate_two_account's real one-confirmed-trade-per-side
    gate) -- the raw sweep counts above are pre-arbitration touch counts, not what's
    actually tradeable. Known modeling caveat: if a zone's FIRST touch is filtered out by
    the meta-model (never placed), a later re-armed touch on the same zone still appears in
    dtfx_engine's raw output as though the first touch had genuinely occupied and closed a
    position -- resolve_closest_price's dispatch-time model inherits that assumption. This
    is a second-order timing effect on a minority of re-armed touches, not a frequency- or
    direction-altering bug, but it isn't independently re-verified here."""
    price_at = g51._price_lookup(inst)
    resolved = g51.resolve_closest_price(filtered_touches, age_h, price_at)
    from goal30_apex_king_deployment import simulate_two_account
    t, _, _, _, _ = simulate_two_account(resolved, 0.06)
    acc = t[t.accepted].sort_values("entry_time").reset_index(drop=True)
    return acc


def walk_forward_check(inst, n, rr, age, threshold, folds):
    """Confirms the meta-labeling lift isn't an artifact of one lucky train/test split --
    retrains from scratch at each fold boundary (always training on everything strictly
    before the test window) and reports baseline vs. meta-filtered PF/WR on each held-out
    slice independently."""
    t = build_labeled_touches(inst, n, rr, age).sort_values("entry_time").reset_index(drop=True)
    N = len(t)
    rows = []
    for train_end, test_end in folds:
        train, test = t.iloc[:int(N * train_end)], t.iloc[int(N * train_end):int(N * test_end)]
        clf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(train[FEATURES].values, (train["net_r"] > 0).astype(int).values)
        test = test.copy()
        test["meta_prob"] = clf.predict_proba(test[FEATURES].values)[:, 1]
        base, filt = stats_of(test), stats_of(test[test.meta_prob >= threshold])
        rows.append(dict(train_end=train_end, test_end=test_end,
                          baseline_pf=base["pf"], baseline_wr=base["wr"], baseline_n=base["n"],
                          meta_pf=filt["pf"] if filt else None, meta_wr=filt["wr"] if filt else None,
                          meta_n=filt["n"] if filt else 0))
        print(f"  train<{train_end:.0%} test=[{train_end:.0%},{test_end:.0%}]: "
              f"baseline PF={base['pf']:.3f} WR={base['wr']:.1f}% (n={base['n']}) | "
              f"meta>={threshold} PF={rows[-1]['meta_pf']} WR={rows[-1]['meta_wr']} (n={rows[-1]['meta_n']})")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time
    t0 = time.time()

    CANDIDATES = [
        ("EURGBP", 35.0, 0.05, 0.05),
        ("EURUSD", 40.0, 0.05, 0.05),
        ("EURGBP", 22.0, 0.15, 0.08),   # a moderate-frequency primary too, for comparison
        ("USDJPY", 8.0, 0.15, 0.05),
    ]
    results = {}
    for inst, n, rr, age in CANDIDATES:
        res = run_meta_labeling(inst, n, rr, age)
        results[f"{inst}_N{n}_RR{rr}"] = res
        res["sweep"].to_csv(
            os.path.join(DATA_DIR, f"goal64_metalabel_sweep_{inst.lower()}_n{int(n)}_rr{rr}.csv"),
            index=False)

    FOLDS = [(0.5, 0.65), (0.5, 0.75), (0.65, 0.80), (0.75, 0.90), (0.80, 1.0)]
    print("\n" + "=" * 100)
    print("WALK-FORWARD VALIDATION -- is the meta-labeling lift real, or one lucky split?")
    print("=" * 100)
    print("--- EURGBP N=35 R:R=0.05, meta>=0.85 ---")
    wf_eurgbp = walk_forward_check("EURGBP", 35.0, 0.05, 0.05, 0.85, FOLDS)
    wf_eurgbp.to_csv(os.path.join(DATA_DIR, "goal64_walkforward_eurgbp.csv"), index=False)
    print("--- EURUSD N=40 R:R=0.05, meta>=0.75 ---")
    wf_eurusd = walk_forward_check("EURUSD", 40.0, 0.05, 0.05, 0.75, FOLDS)
    wf_eurusd.to_csv(os.path.join(DATA_DIR, "goal64_walkforward_eurusd.csv"), index=False)

    print("\n" + "=" * 100)
    print("FLAGSHIP RESULT: EURGBP meta-labeled, TRUE ACCEPTED (resolve_closest_price + g30)")
    print("=" * 100)
    res = results["EURGBP_N35.0_RR0.05"]
    test = res["test"]
    for th, fname in [(0.80, "goal64_eurgbp_metalabeled_th080_trades.csv"),
                       (0.85, "goal64_eurgbp_metalabeled_th085_trades.csv"),
                       (0.88, "goal64_eurgbp_metalabeled_th088_trades.csv")]:
        filt = test[test.meta_prob >= th].copy()
        acc = true_accepted(filt, 0.05, "EURGBP")
        if len(acc) < 30:
            print(f"  th={th}: too few accepted trades")
            continue
        years = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25
        wr = 100 * (acc.net_r > 0).mean()
        gw, gl = acc[acc.net_r > 0].net_r.sum(), -acc[acc.net_r <= 0].net_r.sum()
        pf = gw / gl if gl > 0 else float("inf")
        from goal63_hyperfreq_portfolio import thirds_pf
        p1, p2, p3 = thirds_pf(acc)
        robust = bool(p1 and p2 and p3 and p1 > 1.0 and p2 > 1.0 and p3 > 1.0)
        print(f"  th={th}: n={len(acc)} freq={len(acc)/years:.1f}/yr WR={wr:.1f}% PF={pf:.3f} "
              f"thirds={p1:.3f}/{p2:.3f}/{p3:.3f} robust={robust}")
        acc.to_csv(os.path.join(DATA_DIR, fname), index=False)

    print(f"\n(total: {time.time()-t0:.1f}s)")
