import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad"
ECON_DIR = f"{CACHE_DIR}/econ_calendar"

events = []
for f, label in [("nfp.json", "NFP"), ("fomc.json", "FOMC"), ("cpi_us.json", "CPI")]:
    d = json.load(open(f"{ECON_DIR}/{f}"))
    for r in d:
        events.append(pd.Timestamp(r["datetime"]))
event_times = np.array(sorted(set(events)), dtype="datetime64[ns]")
print(f"Loaded {len(event_times)} unique high-impact US news timestamps (NFP+FOMC+CPI), "
      f"{event_times.min()} -> {event_times.max()}", flush=True)

t = pd.read_csv(f"{CACHE_DIR}/dtfx_diag_trades.csv", parse_dates=["entry_time"])
# only trades within the news-calendar coverage window (2015+) -- can't classify proximity outside it
t = t[t["entry_time"] >= pd.Timestamp("2015-01-01")].copy()
print(f"{len(t)} trades with entry_time in the news-calendar coverage window (2015-2026)", flush=True)

entry_vals = t["entry_time"].values.astype("datetime64[ns]")
idx = np.searchsorted(event_times, entry_vals)
idx = np.clip(idx, 1, len(event_times) - 1)
dist_after = np.abs(event_times[idx] - entry_vals)
dist_before = np.abs(entry_vals - event_times[idx - 1])
min_dist_hours = np.minimum(dist_after, dist_before).astype("timedelta64[m]").astype(float) / 60.0
t["hours_to_nearest_news"] = min_dist_hours

bins = [0, 1, 4, 24, np.inf]
labels = ["<1h from news", "1-4h from news", "4-24h from news", ">24h from news"]
t["news_bucket"] = pd.cut(t["hours_to_nearest_news"], bins=bins, labels=labels)


def agg(g):
    wins = (g["r"] > 0).sum()
    n = len(g)
    return pd.Series({"trades": n, "win_rate": wins / n, "avg_r": g["r"].mean(), "sum_r": g["r"].sum()})


print("\n=== 4. PROXIMITY TO HIGH-IMPACT NEWS (NFP/FOMC/CPI) ===")
by_news = t.groupby("news_bucket", observed=True).apply(agg).reindex(labels)
print(by_news.to_string())

t["regime"] = np.where(t["entry_time"].dt.year >= 2021, "2021-2026", "2015-2020")
by_news_regime = t.groupby(["regime", "news_bucket"], observed=True).apply(agg)
print("\n--- by regime ---")
print(by_news_regime.to_string())

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#b71c1c" if x < 0 else "#2e7d32" for x in by_news["avg_r"]]
ax.bar(by_news.index.astype(str), by_news["avg_r"], color=colors)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_ylabel("Avg R per trade")
ax.set_title("DTFX EUR/USD: avg R by proximity to NFP/FOMC/CPI release\n(original variant, 2015-2026)")
for i, (wr, n) in enumerate(zip(by_news["win_rate"], by_news["trades"])):
    ax.annotate(f"n={int(n)}\nwin={wr*100:.1f}%", (i, by_news["avg_r"].iloc[i]),
                ha="center", va="bottom" if by_news["avg_r"].iloc[i] >= 0 else "top", fontsize=9)
fig.tight_layout()
fig.savefig(f"{CACHE_DIR}/dtfx_diagnostics3_news.png", dpi=130)
print(f"\nSaved chart -> {CACHE_DIR}/dtfx_diagnostics3_news.png", flush=True)
