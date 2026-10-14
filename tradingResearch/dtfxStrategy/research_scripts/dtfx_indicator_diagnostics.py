import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dtfx_eurusd_backtest import CACHE_DIR
from dtfx_eurusd_backtest_strict import get_trades_strict

OUT = CACHE_DIR


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def agg(g):
    n = len(g)
    return pd.Series({"trades": n, "win_rate": (g["r"] > 0).mean(), "avg_r": g["r"].mean(), "sum_r": g["r"].sum()})


def build_indicators(df):
    ind = pd.DataFrame(index=df.index)
    ind["sma200"] = df["close"].rolling(200).mean()
    ind["sma1000"] = df["close"].rolling(1000).mean()
    ind["rsi14"] = rsi(df["close"], 14)
    ind["atr14"] = atr(df, 14)
    ind["atr_ratio"] = ind["atr14"] / ind["atr14"].rolling(2000).mean()
    ind["mom60"] = df["close"].pct_change(60)
    return ind


def analyze(label, df, trades_csv=None):
    print(f"\n{'='*20} {label} {'='*20}", flush=True)
    if trades_csv:
        t = pd.read_csv(trades_csv, parse_dates=["entry_time"])
    else:
        trades, _ = get_trades_strict(df, "original")
        t = pd.DataFrame([{"entry_time": tr["entry_time"], "side": tr["side"], "r": tr["r"]} for tr in trades])
    ind = build_indicators(df)
    t = t.set_index("entry_time").join(ind, how="left").reset_index().rename(columns={"index": "entry_time"})
    t = t.dropna(subset=["sma200", "sma1000", "rsi14", "atr_ratio"])
    print(f"{len(t)} trades with indicator data", flush=True)

    baseline_sum_r = t["r"].sum()
    print(f"BASELINE: n={len(t)} sum_r={baseline_sum_r:.1f} win_rate={(t['r']>0).mean():.3f}", flush=True)

    # 1. MA trend filter: buy above SMA200, sell below (and same for SMA1000)
    print("\n--- 1. MOVING AVERAGE TREND FILTER ---")
    t = t.set_index("entry_time").join(df["close"], how="left")
    t.index.name = "entry_time"
    t = t.reset_index()
    for sma_col, sma_label in [("sma200", "SMA200"), ("sma1000", "SMA1000")]:
        above = t["close"] > t[sma_col]
        t[f"ma_aligned_{sma_col}"] = np.where(
            ((t["side"] == "buy") & above) | ((t["side"] == "sell") & ~above), "aligned", "against")
        by = t.groupby(f"ma_aligned_{sma_col}", observed=True).apply(agg)
        print(f"  {sma_label}:")
        print(by.to_string().replace("\n", "\n    "))
        aligned_only = t[t[f"ma_aligned_{sma_col}"] == "aligned"]
        print(f"    ALIGNED-ONLY: n={len(aligned_only)} sum_r={aligned_only['r'].sum():.1f}")

    # 2. RSI mean-reversion filter: buy when RSI oversold, sell when overbought
    print("\n--- 2. RSI(14) MEAN-REVERSION FILTER AT ENTRY ---")
    t["rsi_bucket"] = pd.cut(t["rsi14"], bins=[0, 30, 45, 55, 70, 100],
                              labels=["<30 oversold", "30-45", "45-55", "55-70", ">70 overbought"])
    by_rsi = t.groupby("rsi_bucket", observed=True).apply(agg)
    print(by_rsi.to_string())
    buy_deep = t[(t["side"] == "buy") & (t["rsi14"] < 40)]
    sell_deep = t[(t["side"] == "sell") & (t["rsi14"] > 60)]
    confirmed = pd.concat([buy_deep, sell_deep])
    print(f"  RSI-CONFIRMED (buy w/ RSI<40, sell w/ RSI>60): n={len(confirmed)} sum_r={confirmed['r'].sum():.1f} "
          f"win_rate={(confirmed['r']>0).mean():.3f}")

    # 3. ATR volatility regime
    print("\n--- 3. ATR VOLATILITY REGIME (atr14 / its own 2000-bar rolling mean) ---")
    t["vol_regime"] = pd.cut(t["atr_ratio"], bins=[0, 0.7, 1.0, 1.3, 100],
                              labels=["compressed <0.7x", "below-avg 0.7-1x", "above-avg 1-1.3x", "expanding >1.3x"])
    by_vol = t.groupby("vol_regime", observed=True).apply(agg)
    print(by_vol.to_string())
    expanding_only = t[t["atr_ratio"] > 1.0]
    print(f"  EXPANDING-VOL-ONLY (atr_ratio>1.0): n={len(expanding_only)} sum_r={expanding_only['r'].sum():.1f} "
          f"win_rate={(expanding_only['r']>0).mean():.3f}")

    return t


if __name__ == "__main__":
    df_eur = pd.read_csv(f"{CACHE_DIR}/lse_eurusd_1min_full.csv", parse_dates=["date"]).set_index("date").sort_index()
    df_eur.index = df_eur.index.tz_localize(None)
    df_eur = df_eur[["open", "high", "low", "close"]]
    t_eur = analyze("EUR/USD full history (2009-2026)", df_eur, trades_csv=f"{CACHE_DIR}/dtfx_diag_trades.csv")
    t_eur.to_csv(f"{CACHE_DIR}/dtfx_ind_eurusd.csv", index=False)

    t_eur_recent = t_eur[t_eur["entry_time"] >= "2021-01-01"]
    print(f"\n{'='*20} EUR/USD 2021-2026 ONLY (the regime with real edge) {'='*20}")
    print(f"BASELINE: n={len(t_eur_recent)} sum_r={t_eur_recent['r'].sum():.1f}")
    for sma_col in ["sma200", "sma1000"]:
        aligned_only = t_eur_recent[t_eur_recent[f"ma_aligned_{sma_col}"] == "aligned"]
        print(f"  {sma_col} aligned-only: n={len(aligned_only)} sum_r={aligned_only['r'].sum():.1f}")
    expanding_only = t_eur_recent[t_eur_recent["atr_ratio"] > 1.0]
    print(f"  expanding-vol-only: n={len(expanding_only)} sum_r={expanding_only['r'].sum():.1f}")

    parts = []
    for yr in ["2024", "2025", "2026"]:
        d = pd.read_csv(f"{CACHE_DIR}/polygon_GBPUSD_{yr}.csv", parse_dates=["date"]).set_index("date").sort_index()
        parts.append(d[["open", "high", "low", "close"]])
    df_gbp = pd.concat(parts).sort_index()
    df_gbp = df_gbp[~df_gbp.index.duplicated(keep="first")]
    t_gbp = analyze("GBP/USD (2024-2026) -- VALIDATION", df_gbp)
    t_gbp.to_csv(f"{CACHE_DIR}/dtfx_ind_gbpusd.csv", index=False)
