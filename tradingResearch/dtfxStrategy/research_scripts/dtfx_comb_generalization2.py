"""Generalization batch 2: full-history gold (settling the earlier 'no edge' question against
the CURRENT rule, not the older one that finding was originally made against), plus SPY,
AAPL, and ES.F futures -- none of which were touched by COMB's search.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import (
    build_trades as _build_cached, score_trades, DATA_DIR,
    _price_cache, _indicator_cache, _rsi, _atr_ratio,
)

COMB = dict(n_multiplier=4.623254141476141, min_risk_pct=8.837618556064067e-05,
            min_fill_age_hours=3.0892056840204307, min_retracement=2,
            use_rsi_atr_filter=True)
LIVE = dict(n_multiplier=2.0, min_risk_pct=0.00025, min_fill_age_hours=1.0,
            min_retracement=3, use_rsi_atr_filter=False)

GEN_INSTRUMENTS = {
    "XAUUSD_FULL": "lse_xauusd_full_1min.csv",
    "SPY": "lse_spy_1min.csv",
    "AAPL": "lse_aapl_1min.csv",
    "ESF": "lse_esf_1min.csv",
}


def load_gen_instrument(name):
    if name in _price_cache:
        return
    path = os.path.join(DATA_DIR, GEN_INSTRUMENTS[name])
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[["open", "high", "low", "close"]]
    _indicator_cache[name] = (_rsi(df["close"]).values, _atr_ratio(df).values)
    times = df.index.values
    lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    _price_cache[name] = (times, lows, highs, df)
    print(f"  {name}: {len(df):,} bars, {df.index.min()} -> {df.index.max()}", flush=True)


def run_config(name, cfg, label):
    load_gen_instrument(name)
    trades = _build_cached(name, cfg["min_retracement"], cfg["n_multiplier"], cfg["min_risk_pct"],
                            cfg["min_fill_age_hours"], cfg["use_rsi_atr_filter"])
    score, stats = score_trades(trades)
    if stats is None:
        print(f"  [{name}] {label}: insufficient trades/degenerate ({len(trades)} trades)")
        return dict(instrument=name, config=label, score=None, trades=len(trades))
    print(f"  [{name}] {label}: trades={stats['trades']} win={stats['win_rate']*100:.1f}% "
          f"sharpe={stats['ann_sharpe']:.2f} calmar={stats['calmar']:.2f} "
          f"maxdd={stats['maxdd']*100:.1f}% avg_r={stats['avg_r']:.3f} score={score:.3f}")
    return dict(instrument=name, config=label, score=score, **stats)


def main():
    rows = []
    print("=== COMB config ===")
    for name in GEN_INSTRUMENTS:
        rows.append(run_config(name, COMB, "COMB"))

    print("\n=== LIVE config (same instruments, same window) ===")
    for name in GEN_INSTRUMENTS:
        rows.append(run_config(name, LIVE, "LIVE"))

    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA_DIR}/dtfx_comb_generalization2.csv", index=False)
    print(f"\nSaved -> dtfx_comb_generalization2.csv")


if __name__ == "__main__":
    main()
