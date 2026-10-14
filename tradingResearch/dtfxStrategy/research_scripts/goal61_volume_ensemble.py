"""/goal (2026-09-04): "do research and work with this base strategy and portfolio but
make it profitable. Keep the amount of trades high." Follows the literal original-DTFX
4-symbol portfolio (N=2.0/R:R=1.00, uniform across all 4 symbols) blowing up completely
under the corrected fill model (PF=0.402, -100% drawdown, confirmed in the prior turn).

Rather than abandon the "high-frequency 4-symbol portfolio" shape entirely (as
Bedrock-King/Yen-Rapid/Elite-Ensemble effectively did by trading away frequency for
edge), this searches for per-symbol configs that are simultaneously high-frequency AND
individually profitable enough to survive the shared-slot cross-symbol competition
(goal59's own finding: that competition doesn't add frequency, but it does select
somewhat blindly by whichever entry-time arrives first, not by quality -- so a config
needs real per-trade margin, not just a barely-positive PF, to survive dilution).

Result: Volume-Ensemble, 94 trades/yr -- more than 2x Yen-Rapid (40/yr) and over 5x
Elite-Ensemble (18/yr) -- while staying genuinely profitable and validated: PF 1.102
full-history, robust across a 2018 train/holdout split (1.114 / 1.088) AND a stricter
three-way historical split (1.143 / 1.075 / 1.080, all positive). This is NOT the
fastest or highest-Sharpe candidate this session found (Sharpe 0.63, ~353yr extrapolated
to $1M -- a thin per-trade edge spread across many trades compounds slower than a
strong edge on fewer trades, the same tension found throughout this whole session) --
it directly answers "make the high-frequency portfolio profitable while keeping trade
count high," not "find the fastest path to $1M."
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
import goal31_apex_stats as g31
import goal51_single_symbol_closest_price_backtest as g51

AGE_H, TOUCHES = 0.08, 20

# Volume-Ensemble: each symbol's own tuned config, chosen for high frequency WITHOUT
# sacrificing enough per-trade margin to lose money once diluted by the cross-symbol
# first-entry-wins competition.
VOLUME_ENSEMBLE = {
    "EURUSD": (28.0, 0.50, 0.020),
    "GBPUSD": (30.0, 0.15, 0.016),
    "EURGBP": (22.0, 0.20, 0.025),
    "USDJPY": (4.0, 0.15, 0.005),   # Yen-Steady's own config
}


def portfolio_trades(configs):
    frames = []
    for inst, (n, rr, mrp) in configs.items():
        g29.rs.load_instrument(inst)
        price_at = g51._price_lookup(inst)
        for side_is_buy in (True, False):
            touches = g51.build_touches_with_metadata(inst, n, rr, AGE_H, False, TOUCHES,
                                                       side_is_buy, min_risk_pct=mrp)
            frames.append(g51.resolve_closest_price(touches, AGE_H, price_at))
    return pd.concat(frames, ignore_index=True)


def thirds_pf(t):
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


if __name__ == "__main__":
    cand = portfolio_trades(VOLUME_ENSEMBLE)
    t, _, _, _, _ = g30.simulate_two_account(cand, 0.06)
    acc = t[t.accepted].sort_values("entry_time").reset_index(drop=True)
    n = len(acc)
    years = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25
    wr = 100 * (acc.net_r > 0).mean()
    gw, gl = acc[acc.net_r > 0].net_r.sum(), -acc[acc.net_r <= 0].net_r.sum()
    pf = gw / gl if gl > 0 else float("inf")
    print(f"Volume-Ensemble: {n} trades / {years:.1f}yr ({n/years:.1f}/yr), "
          f"WR={wr:.2f}%, PF={pf:.3f}")

    for inst in VOLUME_ENSEMBLE:
        sub = acc[acc.instrument == inst]
        if len(sub):
            gws, gls = sub[sub.net_r > 0].net_r.sum(), -sub[sub.net_r <= 0].net_r.sum()
            print(f"  {inst}: {len(sub)} trades, PF={gws/gls if gls>0 else float('inf'):.3f}")

    split = pd.Timestamp("2018-01-01")
    train, holdout = acc[acc.entry_time < split], acc[acc.entry_time >= split]

    def pf_of(x):
        gwx, glx = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return gwx / glx if glx > 0 else float("inf")
    print(f"Train(pre-2018) PF={pf_of(train):.3f}(n={len(train)}) | "
          f"Holdout(2018+) PF={pf_of(holdout):.3f}(n={len(holdout)})")

    p1, p2, p3 = thirds_pf(acc)
    print(f"Thirds PF: {p1:.3f} / {p2:.3f} / {p3:.3f}")

    ann_sharpe, exk, calmar, maxdd1, _ = g31.sharpe_and_kurtosis(acc)
    print(f"Sharpe={ann_sharpe:.2f} ExcessKurtosis={exk:+.2f} Calmar={calmar:.2f} "
          f"MaxDD(1%ref)={-100*maxdd1:.1f}%")

    best = None
    for rp in np.arange(0.02, 0.61, 0.02):
        _, eq_buy, eq_sell, dd, _ = g30.simulate_two_account(acc, rp)
        cagr = ((eq_buy + eq_sell) / 200.0) ** (1 / years) - 1
        if best is None or cagr > best[0]:
            best = (cagr, rp, eq_buy + eq_sell, dd)
    cagr, rp, final_eq, dd = best
    y1m = np.log(1_000_000 / 200.0) / np.log(1 + cagr) if cagr > 0 else float("inf")
    print(f"Best risk={rp:.2f}: CAGR={100*cagr:.2f}% years_to_1m={y1m:.1f} maxDD={dd:.1f}%")

    acc.to_csv(os.path.join(DATA_DIR, "goal61_volume_ensemble_trades.csv"), index=False)
