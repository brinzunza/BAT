"""/goal (2026-09-04) continued: user asked to increase trades/year on the winning
goal72 confluence strategy, first to >730 (interrupted before acted on), then revised to
>365. This characterizes the actual frequency/PF tradeoff frontier for a 4-symbol
federated version of the same recipe (independent two-sub-account pair per symbol, so
each symbol keeps its own one-open-trade-per-side ceiling rather than sharing one slot),
using the SAME sweep-confluence filter as goal72 -- just with each symbol's own N/R:R/MRP
pushed toward higher raw zone frequency than Velocity-Cable's original N=15/R:R=1.5.

Finding: frequency and PF trade off close to monotonically once MRP is pushed this far.
No config found clears 365 trades/yr while holding pooled PF>=1.0 -- the best PF>=1.0
point tops out near 280-290/yr; clearing 365/yr costs pooled PF down to ~0.98 (net
roughly breakeven-to-slightly-negative after realistic costs, not a real edge). Reported
here as an honest frontier, not a single "the" answer -- see ICT_LIQUIDITY_FVG_STRATEGY.md.
"""
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal30_apex_king_deployment as g30
import goal51_single_symbol_closest_price_backtest as g51
import goal72_dtfx_ict_confluence as g72

# (inst, N, R:R, fill-age hours, MIN_RISK_PCT) -- all with require_sweep=True (the same
# confirmation filter goal72 validated), N=45 fixed, tuned per-symbol for frequency.
FREQUENCY_POINTS = {
    "pf_priority (freq~284/yr)": [
        ("GBPUSD", 45.0, 0.3, 0.08, 0.020),
        ("EURUSD", 45.0, 0.2, 0.08, 0.008),
        ("EURGBP", 45.0, 0.2, 0.08, 0.012),
        ("USDJPY", 45.0, 0.2, 0.05, 0.004),
    ],
    "over_365 (freq~392/yr)": [
        ("GBPUSD", 45.0, 0.2, 0.08, 0.012),
        ("EURUSD", 45.0, 0.2, 0.08, 0.003),
        ("EURGBP", 45.0, 0.2, 0.08, 0.008),
        ("USDJPY", 45.0, 0.2, 0.05, 0.0008),
    ],
}


def evaluate_point(configs, require_sweep=True):
    all_acc = []
    per_symbol = {}
    for inst, n, rr, age, mrp in configs:
        t = g72.build_confluence_touches(inst, n, rr, age, 20, dict(mrp=mrp), lookback_bars=60,
                                          require_sweep=require_sweep)
        if len(t) < 15:
            per_symbol[inst] = None
            continue
        price_at = g51._price_lookup(inst)
        resolved = g51.resolve_closest_price(t, age, price_at)
        acc_t, _, _, _, _ = g30.simulate_two_account(resolved, 0.06)
        acc = acc_t[acc_t.accepted].sort_values("entry_time").reset_index(drop=True)
        if len(acc) < 15:
            per_symbol[inst] = None
            continue
        years = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25
        wr = 100 * (acc.net_r > 0).mean()
        gw, gl = acc[acc.net_r > 0].net_r.sum(), -acc[acc.net_r <= 0].net_r.sum()
        pf = gw / gl if gl > 0 else float("inf")
        per_symbol[inst] = dict(n=len(acc), freq=len(acc) / years, wr=wr, pf=pf)
        all_acc.append(acc)
    if not all_acc:
        return per_symbol, None
    pooled = pd.concat(all_acc, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    years = (pooled.entry_time.max() - pooled.entry_time.min()).days / 365.25
    wr = 100 * (pooled.net_r > 0).mean()
    gw, gl = pooled[pooled.net_r > 0].net_r.sum(), -pooled[pooled.net_r <= 0].net_r.sum()
    pf = gw / gl if gl > 0 else float("inf")
    split = pd.Timestamp("2018-01-01")
    train, hold = pooled[pooled.entry_time < split], pooled[pooled.entry_time >= split]

    def pf_of(x):
        if len(x) < 5:
            return None, len(x)
        gwx, glx = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return (gwx / glx if glx > 0 else float("inf")), len(x)

    pooled_stats = dict(n=len(pooled), freq=len(pooled) / years, wr=wr, pf=pf,
                         train_pf=pf_of(train)[0], train_n=pf_of(train)[1],
                         hold_pf=pf_of(hold)[0], hold_n=pf_of(hold)[1])
    return per_symbol, (pooled, pooled_stats)


if __name__ == "__main__":
    import time
    t0 = time.time()
    for label, configs in FREQUENCY_POINTS.items():
        print(f"\n=== {label} ===")
        per_symbol, result = evaluate_point(configs)
        for inst, s in per_symbol.items():
            if s is None:
                print(f"  {inst}: too few")
                continue
            print(f"  {inst}: n={s['n']} freq={s['freq']:.1f}/yr WR={s['wr']:.1f}% PF={s['pf']:.3f}")
        if result is not None:
            pooled, stats = result
            print(f"  POOLED: n={stats['n']} freq={stats['freq']:.1f}/yr WR={stats['wr']:.1f}% "
                  f"PF={stats['pf']:.3f}")
            print(f"    train (pre-2018): PF={stats['train_pf']} n={stats['train_n']}")
            print(f"    holdout (2018+): PF={stats['hold_pf']} n={stats['hold_n']}")
            if "over_365" in label:
                pooled.to_csv(os.path.join(DATA_DIR, "goal73_high_frequency_federated_trades.csv"),
                               index=False)
    print(f"\n(total: {time.time()-t0:.0f}s)")
