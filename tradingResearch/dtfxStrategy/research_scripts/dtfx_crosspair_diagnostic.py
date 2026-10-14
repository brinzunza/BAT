import numpy as np
import pandas as pd

CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"


def agg(g):
    n = len(g)
    return pd.Series({"trades": n, "win_rate": (g["r"] > 0).mean(), "avg_r": g["r"].mean(), "sum_r": g["r"].sum()})


# EUR/USD trades, restricted to the window where we also have GBP/USD data (2024-08 to 2026-08)
t = pd.read_csv(f"{CACHE_DIR}/dtfx_diag_trades.csv", parse_dates=["entry_time"])
t = t[t["entry_time"] >= "2024-08-18"].copy()
print(f"EUR/USD trades in GBP/USD-overlap window: {len(t)}", flush=True)

parts = []
for yr in ["2024", "2025", "2026"]:
    d = pd.read_csv(f"{CACHE_DIR}/polygon_GBPUSD_{yr}.csv", parse_dates=["date"]).set_index("date").sort_index()
    parts.append(d[["open", "high", "low", "close"]])
gbp = pd.concat(parts).sort_index()
gbp = gbp[~gbp.index.duplicated(keep="first")]
gbp["mom60"] = gbp["close"].pct_change(60)
gbp["mom15"] = gbp["close"].pct_change(15)

t = t.set_index("entry_time").join(gbp[["mom60", "mom15"]], how="left").reset_index()
t = t.dropna(subset=["mom60", "mom15"])
print(f"{len(t)} trades with GBP/USD momentum data at entry", flush=True)

baseline_sum_r = t["r"].sum()
print(f"BASELINE: n={len(t)} sum_r={baseline_sum_r:.1f} win_rate={(t['r']>0).mean():.3f}", flush=True)

for mom_col, label in [("mom60", "60-min"), ("mom15", "15-min")]:
    gbp_bullish = t[mom_col] > 0
    t[f"confirmed_{mom_col}"] = np.where(
        ((t["side"] == "buy") & gbp_bullish) | ((t["side"] == "sell") & ~gbp_bullish),
        "confirmed", "diverging")
    by = t.groupby(f"confirmed_{mom_col}", observed=True).apply(agg)
    print(f"\n--- EUR/USD trade direction vs GBP/USD {label} momentum ---")
    print(by.to_string())
    conf = t[t[f"confirmed_{mom_col}"] == "confirmed"]
    print(f"CONFIRMED-ONLY ({label}): n={len(conf)} sum_r={conf['r'].sum():.1f} win_rate={(conf['r']>0).mean():.3f}")

t.to_csv(f"{CACHE_DIR}/dtfx_crosspair_trades.csv", index=False)
