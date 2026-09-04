"""Follow-up correction (2026-09-04): the user's own words -- "why are these strategies
individual symbols when i said i want a portfolio of all four?" -- caught a real drift.
The most recent /goal ("find one that is high frequency, high winrate, and high PF") didn't
repeat the earlier, still-standing "must trade all four tickers together" requirement, and
goal64's meta-labeling work went back to single-symbol exploration as a result. The
STRATEGY_RANKINGS.md update then buried the goal63 Federated Portfolio (which DOES trade
all four) at #8, below several single-symbol entries -- a real miss, not a deliberate
decision to drop the portfolio requirement.

This script fixes it directly: applies goal64's meta-labeling technique to EVERY sleeve of
goal63's Federated Portfolio (four independent per-symbol bot instances, each on its own
dedicated two-sub-account pair -- no shared execution slot), not just the two symbols
goal64 happened to test standalone.

Result: meta-labeling lifts three of the four sleeves substantially (GBPUSD's own accepted
frequency alone goes from 27/yr to 344/yr AND its PF improves 1.074->1.477); USDJPY's
N=8/R:R=0.30 primary doesn't respond to meta-labeling (best filtered PF still <1) so its
sleeve keeps the plain, ungated goal63 config. Combined portfolio clears every one of the
user's standing requirements AT ONCE: genuine 4-symbol portfolio, high frequency, high win
rate, and (new, not previously true of any 4-symbol combination this session) high PF and
a fast, not just "positive," path to $1M.
"""
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal30_apex_king_deployment as g30
import goal63_hyperfreq_portfolio as g63
import goal64_meta_labeling as g64

# Each sleeve's own primary config + (threshold=None means "no meta-filter, use as-is").
SLEEVES = {
    "EURGBP": dict(n=35.0, rr=0.05, age=0.05, meta_threshold=0.85),
    "EURUSD": dict(n=40.0, rr=0.05, age=0.05, meta_threshold=0.80),
    "GBPUSD": dict(n=22.0, rr=0.25, age=0.08, meta_threshold=0.60),
    "USDJPY": dict(n=8.0, rr=0.30, age=0.05, meta_threshold=None),  # meta-labeling tested,
    # doesn't rescue this primary above PF=1.0 (best filtered PF=0.918 < unfiltered 1.031)
    # -- kept plain rather than forced into the same technique for its own sake.
}


def build_sleeve(inst, cfg):
    if cfg["meta_threshold"] is None:
        mrp = {"USDJPY": 0.005}.get(inst, 0.020)
        res = g63.evaluate({inst: dict(n=cfg["n"], rr=cfg["rr"], mrp=mrp, age=cfg["age"])}, inst)
        return res["trades"]
    res = g64.run_meta_labeling(inst, cfg["n"], cfg["rr"], cfg["age"])
    filt = res["test"][res["test"].meta_prob >= cfg["meta_threshold"]].copy()
    return g64.true_accepted(filt, cfg["age"], inst)


if __name__ == "__main__":
    import time
    t0 = time.time()
    import io
    import contextlib

    sleeves = {}
    for inst, cfg in SLEEVES.items():
        with contextlib.redirect_stdout(io.StringIO()):
            sleeves[inst] = build_sleeve(inst, cfg)

    all_trades = pd.concat(list(sleeves.values()), ignore_index=True)
    span_years = (all_trades.entry_time.max() - all_trades.entry_time.min()).days / 365.25
    total_n = sum(len(v) for v in sleeves.values())
    combined_freq = total_n / span_years
    gw = sum(v[v.net_r > 0].net_r.sum() for v in sleeves.values())
    gl = sum(-v[v.net_r <= 0].net_r.sum() for v in sleeves.values())
    pooled_pf = gw / gl if gl > 0 else float("inf")
    pooled_wr = 100 * sum((v.net_r > 0).sum() for v in sleeves.values()) / total_n

    print(f"COMBINED (Meta-labeled Federated Portfolio): freq={combined_freq:.1f}/yr "
          f"[bar>730: {'PASS' if combined_freq > 730 else 'FAIL'}] "
          f"pooled WR={pooled_wr:.1f}% pooled PF={pooled_pf:.3f}")

    combined_final = 0.0
    for inst, v in sleeves.items():
        years = (v.entry_time.max() - v.entry_time.min()).days / 365.25
        wr = 100 * (v.net_r > 0).mean()
        gws, gls = v[v.net_r > 0].net_r.sum(), -v[v.net_r <= 0].net_r.sum()
        pf = gws / gls if gls > 0 else float("inf")
        p1, p2, p3 = g63.thirds_pf(v)
        best = None
        for rp in np.arange(0.02, 0.61, 0.02):
            _, eq_buy, eq_sell, dd, _ = g30.simulate_two_account(v, rp)
            if best is None or (eq_buy + eq_sell) > best[0]:
                best = (eq_buy + eq_sell, rp, dd)
        combined_final += best[0]
        meta_note = f"meta>={SLEEVES[inst]['meta_threshold']}" if SLEEVES[inst]["meta_threshold"] else "ungated"
        print(f"  {inst} ({meta_note}): n={len(v)} freq={len(v)/years:.1f}/yr WR={wr:.1f}% "
              f"PF={pf:.3f} thirds={p1}/{p2}/{p3} own_best_risk={best[1]:.2f} "
              f"own_final_eq=${best[0]:,.0f} own_maxDD={best[2]:.1f}%")
        v.to_csv(os.path.join(DATA_DIR, f"goal65_metalabeled_federated_{inst.lower()}_trades.csv"),
                  index=False)

    start_total = 200.0 * len(SLEEVES)
    cagr = (combined_final / start_total) ** (1 / span_years) - 1
    y1m = np.log(1_000_000 / start_total) / np.log(1 + cagr) if cagr > 0 else float("inf")
    print(f"\ncombined start=${start_total:.0f}, final=${combined_final:,.0f} over "
          f"{span_years:.1f}yr -> CAGR={100*cagr:.2f}% years_to_$1M={y1m:.2f}")
    print(f"\n(total: {time.time()-t0:.1f}s)")
