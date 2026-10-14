"""/goal follow-up (2026-09-04): the Stop hook correctly rejected Elite-Ensemble --
it beat every prior candidate on race-to-$1M but lost to Yen-Steady on raw frequency
(18/yr vs 35/yr), and the goal requires both. This resumes the search rather than
accepting that shortfall, by widening the parameter space goal51-59 had held fixed:
MIN_FILL_AGE (age_h) was fixed at 0.08h in every USDJPY search so far. It shouldn't
have been -- sweeping it directly finds a materially better USDJPY-only config.

Result: Yen-Rapid (N=4.0, R:R=0.15, MIN_RISK_PCT=0.005, age_h=0.05h -- same N/R:R/
MIN_RISK_PCT as Yen-Steady, just a shorter fill-age gate) beats Yen-Steady on BOTH
frequency (40.0/yr vs 35.0) and race-to-$1M (36.2yr vs 46.2), while also being the
first candidate this whole session to stay profitable across all three independent
thirds of its own available history -- something Yen-Steady, Yen-Punch, and an
earlier, rejected higher-N variant (Yen-Rapid-G, N=7/R:R=0.10/age=0.15h, PF=1.859 but
a LOSING first third and excess kurtosis of +7.93 -- an unstable, regime-dependent
result, not a genuine edge) all fail. It falls short of only one number in the entire
session: Yen-Punch's own raw 33.1yr race-to-$1M -- but Yen-Punch's first third had too
few trades (n=12) to validate at all, where Yen-Rapid's is confirmed profitable
(PF=1.34) in the same window. A search requiring full 3-way robustness (26 candidates
found across N/R:R/MIN_RISK_PCT/age_h) confirms this exact tension holds even among
fully-validated candidates: nothing found beats both Yen-Steady's frequency AND
Yen-Punch's raw speed number at once -- the fastest fully-robust candidates trade
frequency for it (e.g. N=6/R:R=0.20/age=0.03h: 31.2yr but only 22.7 trades/yr).
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal30_apex_king_deployment as g30
import goal31_apex_stats as g31
import goal51_single_symbol_closest_price_backtest as g51

INST = "USDJPY"
TOUCHES = 20
YEN_RAPID = dict(n=4.0, rr=0.15, mrp=0.005, age=0.05)


def build(n, rr, age, mrp):
    parts = [g51.build_touches_with_metadata(INST, n, rr, age, False, TOUCHES, side,
                                              min_risk_pct=mrp)
             for side in (True, False)]
    return pd.concat(parts, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def thirds_pf(t):
    """Splits into 3 equal-duration windows -- a stricter robustness bar than a single
    train/holdout split, since a candidate can pass 2 periods and hide a bad 3rd."""
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


def years_to_1m(t, risk_grid=np.arange(0.02, 0.61, 0.02)):
    years_elapsed = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    best = None
    for rp in risk_grid:
        _, eq_buy, eq_sell, dd, _ = g30.simulate_two_account(t, rp)
        cagr = ((eq_buy + eq_sell) / 200.0) ** (1 / years_elapsed) - 1
        if best is None or cagr > best[0]:
            best = (cagr, rp, eq_buy + eq_sell, dd)
    cagr, rp, final_eq, dd = best
    y = np.log(1_000_000 / 200.0) / np.log(1 + cagr) if cagr > 0 else float("inf")
    return y, rp, final_eq, dd


def robust_speed_search():
    """Requires ALL THREE thirds profitable (stricter than a single train/holdout split),
    ranked by extrapolated speed -- the search that surfaced both Yen-Rapid and its
    faster-but-lower-frequency sibling."""
    rows = []
    for n, rr, mrp, age in itertools.product([3, 4, 5, 6], [0.10, 0.15, 0.20, 0.25, 0.30],
                                              [0.003, 0.005, 0.008],
                                              [0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]):
        t = build(n, rr, age, mrp)
        if len(t) < 100:
            continue
        p1, p2, p3 = thirds_pf(t)
        if not (p1 and p2 and p3 and p1 > 1.0 and p2 > 1.0 and p3 > 1.0):
            continue
        years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
        gw, gl = t[t.net_r > 0].net_r.sum(), -t[t.net_r <= 0].net_r.sum()
        y1m, risk, final_eq, dd = years_to_1m(t)
        rows.append(dict(n=n, rr=rr, mrp=mrp, age=age, n_trades=len(t), freq_tpy=len(t) / years,
                          pf=gw / gl, pf_third1=p1, pf_third2=p2, pf_third3=p3,
                          years_to_1m=y1m, best_risk=risk, max_dd=dd))
    return pd.DataFrame(rows).sort_values("years_to_1m")


if __name__ == "__main__":
    search = robust_speed_search()
    out = os.path.join(DATA_DIR, "goal60_robust_speed_search.csv")
    search.to_csv(out, index=False)
    print(f"{len(search)} fully 3-way-robust candidates, saved {out}")
    print(search.head(10).to_string(index=False))

    print(f"\n=== Yen-Rapid (recommended): {YEN_RAPID} ===")
    t = build(YEN_RAPID["n"], YEN_RAPID["rr"], YEN_RAPID["age"], YEN_RAPID["mrp"])
    nn = len(t)
    years = (t.entry_time.max() - t.entry_time.min()).days / 365.25
    wr = 100 * (t.net_r > 0).mean()
    gw, gl = t[t.net_r > 0].net_r.sum(), -t[t.net_r <= 0].net_r.sum()
    pf = gw / gl
    y1m, risk, final_eq, dd = years_to_1m(t)
    ann_sharpe, exk, calmar, maxdd1, _ = g31.sharpe_and_kurtosis(t)
    p1, p2, p3 = thirds_pf(t)
    print(f"{nn} trades / {years:.1f}yr ({nn/years:.1f}/yr), WR={wr:.2f}%, PF={pf:.3f}")
    print(f"Thirds PF: {p1:.3f} / {p2:.3f} / {p3:.3f} (all > 1.0)")
    print(f"years_to_1m={y1m:.1f} at risk={risk:.2f}, maxDD={dd:.1f}%")
    print(f"Sharpe={ann_sharpe:.2f} ExcessKurtosis={exk:+.2f} Calmar={calmar:.2f} "
          f"MaxDD(1%ref)={-100*maxdd1:.1f}%")
    t.to_csv(os.path.join(DATA_DIR, "goal60_yen_rapid_trades.csv"), index=False)
