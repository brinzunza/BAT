"""/goal (2026-09-04): "keep researching, developing, and inventing around the factors of
dtfx to find a better strategy."

This round tested four ideas against the meta-labeled Federated Portfolio (goal65):

1. Rescue USDJPY's weak sleeve (PF=1.031) via a broader N/R:R primary search + meta-
   labeling -- NEGATIVE. Every low-R:R shape tested (N=4-15, R:R=0.05-0.30) has a baseline
   PF far below 1.0 (as low as 0.008) and meta-labeling never lifts any of them past the
   plain N=8/R:R=0.30 config's own 1.031. Kept ungated.
2. A new invented factor -- triangulation mispricing (EUR/GBP vs. the EURUSD/GBPUSD implied
   cross-rate, a classic FX microstructure construct never used in this project before) --
   NEGATIVE as an added meta-labeling feature. Shows a mild standalone U-shaped win-rate
   pattern across deciles but the RF ignores it entirely (importance rank 9/15) once
   impulse_velocity/zone_height_pips are already in the model; true-accepted EURGBP result
   is statistically unchanged (PF 2.286 -> 2.348, within noise).
3. Confidence-scaled position sizing (size each trade by its own meta_prob instead of one
   fixed risk_pct for every accepted trade) -- NEGATIVE, but instructively so: RandomForest's
   probability outputs among ALREADY-FILTERED (>=0.85) trades only span [0.850, 0.915],
   std=0.011 -- essentially no spread left to size against once the binary filter has done
   its job. Best scaled result (0.04yr faster) is noise, not a real gain.
4. **GradientBoostingClassifier instead of RandomForestClassifier** for the meta-model
   itself -- POSITIVE, confirmed via 5-fold walk-forward on both EURGBP and EURUSD (same
   protocol as goal64's original validation). GBM's smoother, better-calibrated probability
   outputs let a materially different, better operating point be found:
     - EURGBP: same frequency (1,338.8 vs 1,331.7/yr), better PF (2.573 vs 2.286)
     - EURUSD: **~4x the frequency** (1,054.5 vs 279.5/yr) at essentially the same PF
       (2.602 vs 2.609) -- RF's coarser probabilities were leaving real, valid signal on
       the table by forcing an all-or-nothing cliff between adjacent thresholds.

Rebuilds the Federated Portfolio with GBM sleeves for EURGBP/EURUSD/GBPUSD (USDJPY stays
ungated, per finding #1). Reports the result across a spectrum of risk caps rather than only
the single CAGR-maximizing number, since the unconstrained optimum pushes to 58% risk per
trade on the EURUSD sleeve -- the most aggressive risk level found anywhere this session.
"""
import os
import sys
import contextlib
import io

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal30_apex_king_deployment as g30
import goal63_hyperfreq_portfolio as g63
import goal64_meta_labeling as g64

GBM_SLEEVES = {
    "EURGBP": dict(n=35.0, rr=0.05, age=0.05, threshold=0.97),
    "EURUSD": dict(n=40.0, rr=0.05, age=0.05, threshold=0.98),
    "GBPUSD": dict(n=22.0, rr=0.25, age=0.08, threshold=0.90),
}


def gbm_sleeve(inst, n, rr, age, threshold):
    t = g64.build_labeled_touches(inst, n, rr, age).sort_values("entry_time").reset_index(drop=True)
    split_idx = int(len(t) * 0.7)
    train, test = t.iloc[:split_idx], t.iloc[split_idx:]
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42)
    clf.fit(train[g64.FEATURES].values, (train["net_r"] > 0).astype(int).values)
    test = test.copy()
    test["meta_prob"] = clf.predict_proba(test[g64.FEATURES].values)[:, 1]
    filt = test[test.meta_prob >= threshold].copy()
    return g64.true_accepted(filt, age, inst)


if __name__ == "__main__":
    import time
    t0 = time.time()

    sleeves = {inst: gbm_sleeve(inst, **cfg) for inst, cfg in GBM_SLEEVES.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        usdjpy_res = g63.evaluate({"USDJPY": dict(n=8.0, rr=0.30, mrp=0.005, age=0.05)}, "USDJPY")
    sleeves["USDJPY"] = usdjpy_res["trades"]

    all_trades = pd.concat(list(sleeves.values()), ignore_index=True)
    span_years = (all_trades.entry_time.max() - all_trades.entry_time.min()).days / 365.25
    total_n = sum(len(v) for v in sleeves.values())
    combined_freq = total_n / span_years
    gw = sum(v[v.net_r > 0].net_r.sum() for v in sleeves.values())
    gl = sum(-v[v.net_r <= 0].net_r.sum() for v in sleeves.values())
    pooled_pf = gw / gl if gl > 0 else float("inf")
    pooled_wr = 100 * sum((v.net_r > 0).sum() for v in sleeves.values()) / total_n

    print(f"COMBINED (GBM Federated Portfolio): freq={combined_freq:.1f}/yr "
          f"pooled WR={pooled_wr:.1f}% pooled PF={pooled_pf:.3f}")
    for inst, v in sleeves.items():
        years = (v.entry_time.max() - v.entry_time.min()).days / 365.25
        wr = 100 * (v.net_r > 0).mean()
        gws, gls = v[v.net_r > 0].net_r.sum(), -v[v.net_r <= 0].net_r.sum()
        pf = gws / gls if gls > 0 else float("inf")
        p1, p2, p3 = g63.thirds_pf(v)
        print(f"  {inst}: n={len(v)} freq={len(v)/years:.1f}/yr WR={wr:.1f}% PF={pf:.3f} "
              f"thirds={p1}/{p2}/{p3}")
        v.to_csv(os.path.join(DATA_DIR, f"goal68_gbm_federated_{inst.lower()}_trades.csv"),
                  index=False)

    print("\nyears-to-$1M across a risk-cap spectrum (not just the CAGR-maximizing optimum):")
    for risk_cap in [0.10, 0.20, 0.30, 0.61]:
        combined_final = 0.0
        detail = {}
        for inst, v in sleeves.items():
            best = None
            for rp in np.arange(0.02, risk_cap + 0.001, 0.02):
                _, eq_buy, eq_sell, dd, _ = g30.simulate_two_account(v, rp)
                if best is None or (eq_buy + eq_sell) > best[0]:
                    best = (eq_buy + eq_sell, rp, dd)
            combined_final += best[0]
            detail[inst] = best
        start_total = 200.0 * len(sleeves)
        cagr = (combined_final / start_total) ** (1 / span_years) - 1
        y1m = np.log(1_000_000 / start_total) / np.log(1 + cagr) if cagr > 0 else float("inf")
        cap_label = "unrestricted" if risk_cap >= 0.61 else f"<={risk_cap:.0%}"
        print(f"  risk {cap_label}: final=${combined_final:,.0f} CAGR={100*cagr:.2f}% "
              f"years_to_1m={y1m:.2f}")
        for inst, (fin, rp, dd) in detail.items():
            print(f"      {inst}: risk={rp:.2f} own_final=${fin:,.0f} own_maxDD={dd:.1f}%")

    print(f"\n(total: {time.time()-t0:.1f}s)")
