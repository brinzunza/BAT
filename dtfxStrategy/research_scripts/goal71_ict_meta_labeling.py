"""/goal (2026-09-04) continued: applies this session's own validated meta-labeling
technique (GradientBoostingClassifier filtering a high-recall/low-precision primary,
goal64/goal68) to the new ICT liquidity-sweep + fair-value-gap signal (goal70), which on
its own is marginal (PF 0.77-1.00 depending on filter strictness) -- exactly the shape
meta-labeling is for. Reasonable R:R only (this goal's own explicit instruction): the
primary here uses R:R=2.5, not the <=0.15 extreme the prior /goal's era converged on.
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal62_regime_feature_strategies as g62
import goal70_ict_liquidity_fvg as g70

FEATURES = ["er", "hurst", "vol_percentile", "realized_vol", "fvg_width_pips",
            "bars_sweep_to_fvg", "bars_fvg_to_entry", "is_buy_f", "shallow_touch_f",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "trailing_wr_20"]


def build_ict_labeled_touches(inst, rr, swing_lookback, fvg_lookahead=15, stop_buffer_pips=1.0,
                               min_stop_pips=5.0, window=500, stride=50, vol_lookback=200):
    t = g70.build_ict_touches(inst, rr, swing_lookback, fvg_lookahead, stop_buffer_pips,
                               min_stop_pips=min_stop_pips)
    if t.empty:
        return t
    times_ns, er, hurst, volpct, rvol = g62.load_regime_features(inst, window, stride, vol_lookback)
    entry_ns = t["entry_time"].values.astype("int64")
    idx = np.clip(np.searchsorted(times_ns, entry_ns), 0, len(times_ns) - 1)
    t["er"] = er[idx]
    t["hurst"] = hurst[idx]
    t["vol_percentile"] = volpct[idx]
    t["realized_vol"] = rvol[idx]

    hour = t["entry_time"].dt.hour + t["entry_time"].dt.minute / 60.0
    t["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    t["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = t["entry_time"].dt.dayofweek
    t["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    t["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    t["is_buy_f"] = t["is_buy"].astype(float)
    t["shallow_touch_f"] = t["shallow_touch"].astype(float)

    t["trailing_wr_20"] = np.nan
    for side_val, grp in t.groupby("is_buy"):
        g = grp.sort_values("entry_time")
        tw = g["is_win"].astype(float).shift(1).rolling(20, min_periods=5).mean()
        t.loc[g.index, "trailing_wr_20"] = tw.values

    return t.dropna(subset=["er", "hurst", "vol_percentile", "trailing_wr_20"]).reset_index(drop=True)


def stats_of(t):
    if t is None or len(t) == 0:
        return None
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    years = max(years, 1e-6)
    wr = 100 * (t.net_r > 0).mean()
    gw, gl = t[t.net_r > 0].net_r.sum(), -t[t.net_r <= 0].net_r.sum()
    pf = gw / gl if gl > 0 else float("inf")
    return dict(n=len(t), freq=len(t) / years, wr=wr, pf=pf)


def thirds_pf(t):
    if len(t) < 45:
        return (None, None, None)
    start, end = t.entry_time.min(), t.entry_time.max()
    span = end - start
    b1, b2 = start + span / 3, start + 2 * span / 3
    windows = [t[t.entry_time < b1], t[(t.entry_time >= b1) & (t.entry_time < b2)],
               t[t.entry_time >= b2]]

    def pf_of(x):
        if len(x) < 10:
            return None
        gw, gl = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return gw / gl if gl > 0 else float("inf")
    return tuple(pf_of(w) for w in windows)


if __name__ == "__main__":
    import time
    from sklearn.ensemble import GradientBoostingClassifier

    t0 = time.time()
    t = build_ict_labeled_touches("EURUSD", rr=2.5, swing_lookback=60)
    print(f"built {len(t)} labeled ICT touches in {time.time()-t0:.0f}s")

    t = t.sort_values("entry_time").reset_index(drop=True)
    split_idx = int(len(t) * 0.7)
    train, test = t.iloc[:split_idx], t.iloc[split_idx:]
    print(f"train={len(train)} test={len(test)}")

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42)
    clf.fit(train[FEATURES].values, (train["net_r"] > 0).astype(int).values)
    test = test.copy()
    test["meta_prob"] = clf.predict_proba(test[FEATURES].values)[:, 1]

    base = stats_of(test)
    print(f"BASELINE (test set, no filter): n={base['n']} freq={base['freq']:.1f}/yr "
          f"WR={base['wr']:.1f}% PF={base['pf']:.3f}")

    for th in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        kept = test[test.meta_prob >= th]
        s = stats_of(kept)
        if s is None:
            continue
        p1, p2, p3 = thirds_pf(kept)
        print(f"  th={th}: n={s['n']:4d} freq={s['freq']:6.1f}/yr WR={s['wr']:5.1f}% "
              f"PF={s['pf']:.3f} thirds={p1}/{p2}/{p3}")

    importances = dict(zip(FEATURES, clf.feature_importances_))
    top = sorted(importances.items(), key=lambda kv: -kv[1])[:6]
    print(f"top features: {', '.join(f'{k}={v:.3f}' for k, v in top)}")
