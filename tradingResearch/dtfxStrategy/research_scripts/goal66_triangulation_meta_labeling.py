"""/goal (2026-09-04): "keep researching, developing, and inventing around the factors of
dtfx to find a better strategy."

New, genuinely-invented factor: **triangulation mispricing**. EURGBP, EURUSD, and GBPUSD are
not three independent prices -- EUR/GBP = (EUR/USD) / (GBP/USD) by construction (triangular
arbitrage keeps real market prices tightly bound to this identity; classic FX microstructure
result, not something this project has used anywhere before). This project's three pairs are
each pulled from the same vendor as separate feeds, so tiny timing/quoting differences between
them can let the ACTUAL EURGBP price drift briefly away from what EURUSD and GBPUSD jointly
imply -- a transient cross-pair "mispricing" that (per triangular arbitrage) has real economic
pressure to close, making it a plausible mean-reversion-flavored quality signal for exactly
the kind of zone-fade entry this whole project trades.

Built as a rolling z-score (not a raw level, to strip out any constant feed-offset bias):
  implied_eurgbp[t]  = EURUSD_close[t] / GBPUSD_close[t]
  mispricing[t]      = (actual_eurgbp[t] - implied_eurgbp[t]) / actual_eurgbp[t]
  triangulation_z[t] = (mispricing[t] - rolling_mean(mispricing, W)) / rolling_std(mispricing, W)

Tested here as an ADDITIONAL feature bolted onto goal64's meta-labeling pipeline for EURGBP
(the flagship sleeve) -- does it meaningfully improve on the already-strong meta>=0.85 result
(PF=2.286, 98.9% WR, 1,331.7 trades/yr), or is it noise the classifier ignores?
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal29_rerank_1_per_side as g29
import goal51_single_symbol_closest_price_backtest as g51
import goal63_hyperfreq_portfolio as g63
import goal64_meta_labeling as g64
from sklearn.ensemble import RandomForestClassifier

rs = g29.rs


def build_triangulation_z(window=500):
    """Rolling z-scored EURGBP-vs-(EURUSD/GBPUSD) mispricing, aligned to EURGBP's own bar
    grid via merge_asof (the three feeds are ~99% timestamp-aligned already, since they're
    the same vendor, but asof-merging is the honest way to handle the small remainder)."""
    times_g, _, _, df_g = rs.load_instrument("EURGBP")
    times_u, _, _, df_u = rs.load_instrument("EURUSD")
    times_p, _, _, df_p = rs.load_instrument("GBPUSD")

    g = pd.DataFrame({"time": times_g, "eurgbp": df_g["close"].values}).sort_values("time")
    u = pd.DataFrame({"time": times_u, "eurusd": df_u["close"].values}).sort_values("time")
    p = pd.DataFrame({"time": times_p, "gbpusd": df_p["close"].values}).sort_values("time")

    merged = pd.merge_asof(g, u, on="time", direction="nearest", tolerance=pd.Timedelta("2min"))
    merged = pd.merge_asof(merged, p, on="time", direction="nearest", tolerance=pd.Timedelta("2min"))
    merged = merged.dropna(subset=["eurusd", "gbpusd"])

    implied = merged["eurusd"] / merged["gbpusd"]
    mispricing = (merged["eurgbp"] - implied) / merged["eurgbp"]
    roll_mean = mispricing.rolling(window, min_periods=window // 2).mean()
    roll_std = mispricing.rolling(window, min_periods=window // 2).std()
    z = (mispricing - roll_mean) / roll_std.replace(0, np.nan)

    out = pd.DataFrame({"time": merged["time"].values, "triangulation_z": z.values}).dropna()
    return out.sort_values("time").reset_index(drop=True)


def attach_triangulation_feature(touches, tri_df):
    times_ns = tri_df["time"].values.astype("int64")
    vals = tri_df["triangulation_z"].values
    entry_ns = touches["entry_time"].values.astype("int64")
    idx = np.clip(np.searchsorted(times_ns, entry_ns, side="right") - 1, 0, len(times_ns) - 1)
    touches = touches.copy()
    touches["triangulation_z"] = vals[idx]
    return touches


if __name__ == "__main__":
    import time
    t0 = time.time()

    print("Building triangulation mispricing series...")
    tri = build_triangulation_z()
    print(f"  {len(tri):,} points, z range [{tri.triangulation_z.min():.2f}, "
          f"{tri.triangulation_z.max():.2f}], std={tri.triangulation_z.std():.3f}")

    print("\nRebuilding EURGBP labeled touches with the new feature added...")
    t = g64.build_labeled_touches("EURGBP", 35.0, 0.05, 0.05)
    t = attach_triangulation_feature(t, tri)
    t = t.sort_values("entry_time").reset_index(drop=True)

    FEATURES_PLUS = g64.FEATURES + ["triangulation_z"]
    split_idx = int(len(t) * 0.7)
    train, test = t.iloc[:split_idx], t.iloc[split_idx:]

    clf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=50,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(train[FEATURES_PLUS].values, (train["net_r"] > 0).astype(int).values)
    test = test.copy()
    test["meta_prob"] = clf.predict_proba(test[FEATURES_PLUS].values)[:, 1]

    importances = dict(zip(FEATURES_PLUS, clf.feature_importances_))
    top = sorted(importances.items(), key=lambda kv: -kv[1])[:6]
    print(f"top features (with triangulation_z): {', '.join(f'{k}={v:.3f}' for k, v in top)}")
    print(f"triangulation_z importance rank: "
          f"{sorted(importances.items(), key=lambda kv: -kv[1]).index(('triangulation_z', importances['triangulation_z'])) + 1} "
          f"of {len(FEATURES_PLUS)} (importance={importances['triangulation_z']:.4f})")

    print("\n=== EURGBP + triangulation_z: TRUE ACCEPTED comparison at th=0.85 ===")
    filt = test[test.meta_prob >= 0.85].copy()
    acc = g64.true_accepted(filt, 0.05, "EURGBP")
    if len(acc) >= 30:
        years = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25
        wr = 100 * (acc.net_r > 0).mean()
        gw, gl = acc[acc.net_r > 0].net_r.sum(), -acc[acc.net_r <= 0].net_r.sum()
        pf = gw / gl if gl > 0 else float("inf")
        p1, p2, p3 = g63.thirds_pf(acc)
        print(f"WITH triangulation_z: n={len(acc)} freq={len(acc)/years:.1f}/yr WR={wr:.1f}% "
              f"PF={pf:.3f} thirds={p1}/{p2}/{p3}")
        print("(compare to WITHOUT it: n=8251 freq=1331.7/yr WR=98.9% PF=2.286 "
              "thirds=1.900/2.336/2.842 -- goal64's original result)")
    else:
        print("too few accepted trades")

    print(f"\n(total: {time.time()-t0:.1f}s)")
