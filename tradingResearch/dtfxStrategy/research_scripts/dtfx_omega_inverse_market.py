"""What would Omega have done in a mirror universe where EUR/USD moved exactly opposite to
what actually happened? Builds a genuine inverted OHLC series -- not just 1/price (that
distorts percentage moves asymmetrically), but a bar-by-bar negation of every log return,
including the intrabar wick structure, so the inverted series has the same volatility and
realistic OHLC ordering as the original, just with every up-move turned into a down-move of
identical magnitude and vice versa.

Per bar t, relative to that bar's own open:
    rel_close = close/open, rel_high = high/open, rel_low = low/open
and relative to the previous close:
    rel_gap = open/prev_close
Inverting means negating the log return, i.e. taking the reciprocal of each ratio. Because
low <= open <= high and low <= close <= high always hold in real OHLC data, and reciprocals
of a chain of positive inequalities reverse the chain, the reconstructed bars automatically
satisfy new_low <= new_open <= new_high and new_low <= new_close <= new_high -- no manual
patching needed, it falls out of the math.

If Omega's edge is a genuine, direction-agnostic structural pattern (retracement then
continuation, which should exist symmetrically in up and down trends), performance here
should land in the same ballpark as the real series. A large gap either way would say the
edge depends on something specific to how EUR/USD actually happened to move, not on the
pattern the strategy claims to detect.
"""
import os
import numpy as np
import pandas as pd

from dtfx_robust_search import (
    build_trades as _build_cached, score_trades, DATA_DIR,
    _price_cache, _indicator_cache, _zone_cache, _rsi, _atr_ratio,
)

OMEGA = dict(n_multiplier=3.343496, min_risk_pct=0.000170, min_fill_age_hours=3.366682,
             min_retracement=2, use_rsi_atr_filter=False)


def build_inverted_ohlc(df):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(df)

    rel_close = c / o
    rel_high = h / o
    rel_low = l / o
    rel_gap = np.empty(n)
    rel_gap[0] = 1.0
    rel_gap[1:] = o[1:] / c[:-1]

    inv_close_r = 1.0 / rel_close
    inv_high_r = 1.0 / rel_low   # old low excursion becomes new high excursion
    inv_low_r = 1.0 / rel_high   # old high excursion becomes new low excursion
    inv_gap_r = 1.0 / rel_gap

    new_o = np.empty(n); new_h = np.empty(n); new_l = np.empty(n); new_c = np.empty(n)
    new_o[0] = o[0]
    new_c[0] = new_o[0] * inv_close_r[0]
    new_h[0] = new_o[0] * inv_high_r[0]
    new_l[0] = new_o[0] * inv_low_r[0]
    for t in range(1, n):
        new_o[t] = new_c[t - 1] * inv_gap_r[t]
        new_c[t] = new_o[t] * inv_close_r[t]
        new_h[t] = new_o[t] * inv_high_r[t]
        new_l[t] = new_o[t] * inv_low_r[t]

    out = pd.DataFrame({"open": new_o, "high": new_h, "low": new_l, "close": new_c}, index=df.index)
    # sanity: OHLC ordering must hold everywhere
    assert (out["high"] >= out[["open", "close", "low"]].max(axis=1) - 1e-9).all(), "high ordering violated"
    assert (out["low"] <= out[["open", "close", "high"]].min(axis=1) + 1e-9).all(), "low ordering violated"
    return out


def main():
    path = os.path.join(DATA_DIR, "lse_eurusd_1min_full.csv")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[["open", "high", "low", "close"]]
    print(f"Loaded EURUSD: {len(df):,} bars, {df.index.min()} -> {df.index.max()}", flush=True)

    print("Building inverted OHLC series...", flush=True)
    inv_df = build_inverted_ohlc(df)
    print(f"Inverted series built and OHLC-ordering verified. "
          f"Original close range [{df['close'].min():.4f}, {df['close'].max():.4f}], "
          f"inverted close range [{inv_df['close'].min():.4f}, {inv_df['close'].max():.4f}]", flush=True)

    out_path = os.path.join(DATA_DIR, "lse_eurusd_1min_inverted.csv")
    inv_df.reset_index().rename(columns={"index": "date"}).to_csv(out_path, index=False)
    print(f"Saved inverted series -> {out_path}", flush=True)

    # run Omega on the REAL series (using the already-cached loader/zone infra)
    real_trades = _build_cached("EURUSD", OMEGA["min_retracement"], OMEGA["n_multiplier"],
                                 OMEGA["min_risk_pct"], OMEGA["min_fill_age_hours"], OMEGA["use_rsi_atr_filter"])
    real_score, real_stats = score_trades(real_trades)

    # run Omega on the INVERTED series -- register it as a new "instrument" in the shared caches
    _indicator_cache["EURUSD_INV"] = (_rsi(inv_df["close"]).values, _atr_ratio(inv_df).values)
    times = inv_df.index.values
    lows = np.ascontiguousarray(inv_df["low"].values, dtype=np.float64)
    highs = np.ascontiguousarray(inv_df["high"].values, dtype=np.float64)
    _price_cache["EURUSD_INV"] = (times, lows, highs, inv_df)
    inv_trades = _build_cached("EURUSD_INV", OMEGA["min_retracement"], OMEGA["n_multiplier"],
                                OMEGA["min_risk_pct"], OMEGA["min_fill_age_hours"], OMEGA["use_rsi_atr_filter"])
    inv_score, inv_stats = score_trades(inv_trades)

    print("\n=== REAL EUR/USD, Omega ===")
    print(real_stats)
    print("\n=== INVERTED EUR/USD (mirror-world price action), Omega ===")
    print(inv_stats)

    rows = [dict(series="real", **real_stats), dict(series="inverted", **inv_stats)]
    pd.DataFrame(rows).to_csv(f"{DATA_DIR}/dtfx_omega_inverse_market_test.csv", index=False)
    print(f"\nSaved -> dtfx_omega_inverse_market_test.csv")


if __name__ == "__main__":
    main()
