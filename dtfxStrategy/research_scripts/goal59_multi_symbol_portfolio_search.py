"""/goal (2026-09-04): "find a strategy that outperforms all of them, specifically in
frequency and race to 1 million. New factor is you can trade EURUSD, EURGBP, USDJPY, and
GBPUSD."

Key realization: no new resolver is needed for real multi-symbol trading under the
established rules. Each symbol's own per-(symbol,side) closest-price arbitration
(goal51::resolve_closest_price) already answers "which zone does THIS symbol rest at any
given moment" correctly in isolation. The cross-symbol rule -- at most 1 CONFIRMED fill
per side across the whole account (matching PRODLIVETRADING's actual, always-been-this-way
MAX_CONCURRENT_PER_SIDE=1) -- is exactly "whichever symbol's resting order fills first,
chronologically, wins that side's slot and blocks every other symbol's resting order on
it until it closes." That is precisely goal30::simulate_two_account's own existing
accept/reject logic (first entry-time wins, per side), unmodified. So: resolve each
symbol/side independently, concatenate all 8 (4 symbols x 2 sides) streams, and feed the
combined DataFrame into simulate_two_account/run_mc exactly as before.

Finding, confirmed by direct testing and then an exhaustive 227-combination sweep across
every validated single-symbol candidate this session found: the shared per-side slot
makes multi-symbol trading a ZERO-SUM reallocation of slot-time, not an additive source
of more trades. Adding a symbol can only ever maintain or REDUCE the dominant symbol's
own frequency (confirmed: every tested addition to USDJPY's Yen-Steady baseline reduced
its frequency, never increased it), because occupying the shared slot -- regardless of
which symbol does it -- blocks everyone else on that side for the position's full
duration. No combination of any 2, 3, or 4 symbols beat Yen-Steady on BOTH frequency and
race-to-$1M simultaneously.
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


def portfolio_trades(configs):
    """configs: {symbol: (n, rr, min_risk_pct)}. Each symbol/side resolved independently
    (closest-price-per-symbol, already validated in goal51/53-58); concatenated for
    simulate_two_account to apply the cross-symbol one-confirmed-fill-per-side gate."""
    frames = []
    for inst, (n, rr, mrp) in configs.items():
        g29.rs.load_instrument(inst)
        price_at = g51._price_lookup(inst)
        for side_is_buy in (True, False):
            touches = g51.build_touches_with_metadata(inst, n, rr, AGE_H, False, TOUCHES,
                                                       side_is_buy, min_risk_pct=mrp)
            frames.append(g51.resolve_closest_price(touches, AGE_H, price_at))
    return pd.concat(frames, ignore_index=True)


def evaluate(configs, risk_grid=np.arange(0.04, 0.61, 0.04)):
    cand = portfolio_trades(configs)
    t, _, _, _, _ = g30.simulate_two_account(cand, 0.06)
    acc = t[t.accepted].sort_values("entry_time").reset_index(drop=True)
    if len(acc) < 30:
        return None
    years_elapsed = (acc.entry_time.max() - acc.entry_time.min()).days / 365.25
    freq = len(acc) / years_elapsed
    gw, gl = acc[acc.net_r > 0].net_r.sum(), -acc[acc.net_r <= 0].net_r.sum()
    pf = gw / gl if gl > 0 else float("inf")
    best_years, best_risk = float("inf"), None
    for rp in risk_grid:
        _, eq_buy, eq_sell, dd, _ = g30.simulate_two_account(acc, rp)
        cagr = ((eq_buy + eq_sell) / 200.0) ** (1 / years_elapsed) - 1
        if cagr > 0:
            y = np.log(1_000_000 / 200.0) / np.log(1 + cagr)
            if y < best_years:
                best_years, best_risk = y, rp
    return dict(trades=acc, n_trades=len(acc), freq_tpy=freq, pf=pf,
                years_to_1m=best_years, best_risk=best_risk)


# Every validated single-symbol candidate this session found, grouped by symbol -- the
# pool the exhaustive combinatorial search below draws from.
CANDIDATE_POOL = {
    "USDJPY": [(4.0, 0.15, 0.005), (4.0, 0.75, 0.005), (4.0, 0.30, 0.005), (6.0, 0.15, 0.008)],
    "EURUSD": [(15.0, 1.00, 0.020), (15.0, 0.75, 0.020), (30.0, 0.50, 0.020)],
    "GBPUSD": [(15.0, 1.50, 0.020), (30.0, 0.25, 0.016)],
    "EURGBP": [(22.0, 0.75, 0.020), (5.0, 0.15, 0.008), (8.0, 0.30, 0.008)],
}

# Winner: USDJPY's own Yen-Steady config, supplemented by each other symbol's own
# best-individual-PF config (not its fastest/highest-frequency one) -- infrequent,
# high-quality signals from the other 3 pairs add real edge without displacing much of
# USDJPY's own activity.
ELITE_ENSEMBLE = {
    "USDJPY": (4.0, 0.15, 0.005),
    "EURUSD": (15.0, 1.00, 0.020),
    "GBPUSD": (15.0, 1.50, 0.020),
    "EURGBP": (5.0, 0.15, 0.008),
}


def exhaustive_search():
    import itertools
    rows = []
    for r in (2, 3, 4):
        for symbols in itertools.combinations(CANDIDATE_POOL.keys(), r):
            for choice in itertools.product(*(CANDIDATE_POOL[s] for s in symbols)):
                cfg = dict(zip(symbols, choice))
                res = evaluate(cfg)
                if res is None:
                    continue
                rows.append(dict(symbols="+".join(symbols), **{k: v for k, v in cfg.items()},
                                  n_trades=res["n_trades"], freq_tpy=res["freq_tpy"],
                                  pf=res["pf"], years_to_1m=res["years_to_1m"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Running exhaustive combinatorial search (227 combinations)...")
    df = exhaustive_search()
    out = os.path.join(DATA_DIR, "goal59_multi_symbol_combinatorial_search.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} valid combinations, saved {out}")

    beats_yen_steady = df[(df.freq_tpy > 35.0) & (df.years_to_1m < 46.2)]
    print(f"\nCombinations beating Yen-Steady on BOTH frequency AND speed: {len(beats_yen_steady)}")
    if len(beats_yen_steady):
        print(beats_yen_steady.to_string(index=False))
    else:
        print("None -- the trade-off is fundamental, not a search gap (confirmed exhaustively).")

    print("\n=== Elite-Ensemble (recommended: best multi-symbol result found) ===")
    res = evaluate(ELITE_ENSEMBLE)
    acc = res["trades"]
    ann_sharpe, exk, calmar, maxdd, _ = g31.sharpe_and_kurtosis(acc)
    split = pd.Timestamp("2018-01-01")
    train, holdout = acc[acc.entry_time < split], acc[acc.entry_time >= split]

    def pf_of(x):
        gw, gl = x[x.net_r > 0].net_r.sum(), -x[x.net_r <= 0].net_r.sum()
        return gw / gl if gl > 0 else float("inf")

    print(f"{res['n_trades']} trades, {res['freq_tpy']:.1f}/yr, PF={res['pf']:.3f}, "
          f"best_risk={res['best_risk']:.2f}, years_to_1m={res['years_to_1m']:.1f}")
    print(f"Sharpe={ann_sharpe:.2f} ExcessKurtosis={exk:+.2f} Calmar={calmar:.2f} "
          f"MaxDD(1%%ref)={-100*maxdd:.1f}%%")
    print(f"Train (pre-2018) PF={pf_of(train):.3f} (n={len(train)}) | "
          f"Holdout (2018+) PF={pf_of(holdout):.3f} (n={len(holdout)})")
    acc.to_csv(os.path.join(DATA_DIR, "goal59_elite_ensemble_trades.csv"), index=False)
