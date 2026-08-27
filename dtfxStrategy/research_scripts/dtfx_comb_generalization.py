"""Generalization check: does COMB (found by searching only EUR/USD, GBP/USD, EUR/GBP) transfer
to instruments it never saw during the search?

USD/JPY: another FX pair, but a different regional/rate-driver story than the EUR/GBP/USD
triangle -- a moderate test.
BTC/USD: a genuinely uncorrelated asset class -- a hard test.
XAU/USD (gold): the sharpest test available. CHECKPOINT_NOTES.md found the ORIGINAL DTFX rule
has NO edge on gold at any point in 20 years. If COMB also shows no edge here, that's
consistent and expected. If COMB *does* show edge on gold, that's a meaningful, surprising
generalization result worth flagging rather than just reporting the number.

Range: 2018-01-01 onward only (not full history) -- matches the walk-forward search's own
finding that statistical confidence concentrates in 2018+ data, and keeps the fetch tractable.
Also runs the LIVE config on the same instruments/window for a same-vendor, same-window
side-by-side.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import (
    load_instrument as _load_cached, _prep_zones as _prep_cached,
    build_trades as _build_cached, score_trades, DATA_DIR,
    _price_cache, _indicator_cache, _zone_cache, _rsi, _atr_ratio,
)

COMB = dict(n_multiplier=4.623254141476141, min_risk_pct=8.837618556064067e-05,
            min_fill_age_hours=3.0892056840204307, min_retracement=2,
            use_rsi_atr_filter=True)
LIVE = dict(n_multiplier=2.0, min_risk_pct=0.00025, min_fill_age_hours=1.0,
            min_retracement=3, use_rsi_atr_filter=False)

GEN_INSTRUMENTS = {
    "USDJPY": "lse_usdjpy_1min_2018on.csv",
    "BTCUSD": "lse_btcusd_1min_2018on.csv",
    "XAUUSD": "lse_xauusd_1min_2018on.csv",
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
        return None
    print(f"  [{name}] {label}: trades={stats['trades']} win={stats['win_rate']*100:.1f}% "
          f"sharpe={stats['ann_sharpe']:.2f} calmar={stats['calmar']:.2f} "
          f"maxdd={stats['maxdd']*100:.1f}% avg_r={stats['avg_r']:.3f} score={score:.3f}")
    return dict(instrument=name, config=label, score=score, **stats)


def main():
    rows = []
    print("=== COMB config on out-of-search instruments ===")
    for name in GEN_INSTRUMENTS:
        r = run_config(name, COMB, "COMB")
        if r:
            rows.append(r)

    print("\n=== LIVE config, same instruments, same window (for comparison) ===")
    for name in GEN_INSTRUMENTS:
        r = run_config(name, LIVE, "LIVE")
        if r:
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA_DIR}/dtfx_comb_generalization.csv", index=False)
    print(f"\nSaved -> dtfx_comb_generalization.csv")


if __name__ == "__main__":
    main()
