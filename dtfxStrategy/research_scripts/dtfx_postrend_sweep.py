import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-brunoinzunza-Documents-GitHub/3b4d2171-ccf5-4c75-8a5d-8dac2a9ede36/scratchpad")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dtfx_eurusd_backtest import CACHE_DIR, run_structure_fsm
from dtfx_eurusd_backtest_strict import find_pivots_strict, simulate_zone_original

SWEEP_DIR = f"{CACHE_DIR}/lse_sweep"

INSTRUMENTS = [
    ("EUR/USD (fx, full)", f"{CACHE_DIR}/lse_eurusd_1min_full.csv"),
    ("XAU/USD (gold, full)", f"{CACHE_DIR}/lse_xauusd_1min_full.csv"),
    ("GBP/USD (fx, 2024-26)", f"{CACHE_DIR}/polygon_GBPUSD_2024.csv"),  # placeholder, replaced below
    ("SPX500 (index)", f"{SWEEP_DIR}/SPX500_pre2020.csv"),
    ("ES.F (SPX futures)", f"{SWEEP_DIR}/ESF_pre2020.csv"),
    ("AAPL (stock)", f"{SWEEP_DIR}/AAPL_pre2020.csv"),
    ("USD/JPY (fx)", f"{SWEEP_DIR}/USDJPY_2009_2016.csv"),
    ("AUD/USD (fx)", f"{SWEEP_DIR}/AUDUSD_2009_2016.csv"),
]


def load_gbpusd():
    parts = []
    for yr in ["2024", "2025", "2026"]:
        d = pd.read_csv(f"{CACHE_DIR}/polygon_GBPUSD_{yr}.csv", parse_dates=["date"]).set_index("date").sort_index()
        parts.append(d[["open", "high", "low", "close"]])
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated(keep="first")]


def get_trades_with_pos(df):
    zigzag = find_pivots_strict(df, 3)
    _, zones = run_structure_fsm(zigzag)
    sides = [z["side"] for z in zones]
    pos_in_trend, prev_side, counter = [], None, 0
    for s in sides:
        if s != prev_side:
            counter = 0
        counter += 1
        pos_in_trend.append(counter)
        prev_side = s

    times, lows, highs = df.index.values, df["low"].values, df["high"].values
    records = []
    for z, pos in zip(zones, pos_in_trend):
        tr = simulate_zone_original(z, times, lows, highs)
        if tr["status"] not in ("take_profit", "stopped_out"):
            continue
        is_buy_zone = tr["side"] == "buy"
        risk = (tr["mid"] - tr["bottom"]) if is_buy_zone else (tr["top"] - tr["mid"])
        if risk <= 0:
            continue
        if tr["status"] == "take_profit":
            reward = (tr["tp"] - tr["mid"]) if is_buy_zone else (tr["mid"] - tr["tp"])
            r = reward / risk
        else:
            r = -1.0
        records.append(dict(entry_time=tr["entry_time"], r=r, pos_in_trend=pos))
    return pd.DataFrame(records), len(zones)


def agg(g):
    n = len(g)
    return pd.Series({"trades": n, "win_rate": (g["r"] > 0).mean(), "avg_r": g["r"].mean(), "sum_r": g["r"].sum()})


all_results = []
filter_results = []
for label, path in INSTRUMENTS:
    if label.startswith("GBP/USD"):
        df = load_gbpusd()
    else:
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df[["open", "high", "low", "close"]]
    span_years = (df.index.max() - df.index.min()).days / 365.25
    print(f"=== {label}: {len(df)} bars, {df.index.min()} -> {df.index.max()} ({span_years:.1f}y) ===", flush=True)

    t, n_zones = get_trades_with_pos(df)
    t["pos_bucket"] = np.where(t["pos_in_trend"] >= 6, "6+", t["pos_in_trend"].astype(str))
    by_pos = t.groupby("pos_bucket", observed=True).apply(agg).reindex(["1", "2", "3", "4", "5", "6+"])
    print(by_pos.to_string(), flush=True)

    baseline_sum_r = t["r"].sum()
    baseline_n = len(t)
    filt = t[t["pos_bucket"].isin(["1", "6+"])]
    filt_sum_r = filt["r"].sum()
    filt_n = len(filt)
    print(f"  BASELINE: n={baseline_n} sum_r={baseline_sum_r:.1f}  |  "
          f"POS 1-or-6+ ONLY: n={filt_n} sum_r={filt_sum_r:.1f}", flush=True)

    for _, row in by_pos.reset_index().iterrows():
        all_results.append(dict(instrument=label, pos_bucket=row["pos_bucket"], **row.drop("pos_bucket").to_dict()))
    filter_results.append(dict(instrument=label, span_years=round(span_years, 1),
                                baseline_trades=baseline_n, baseline_sum_r=baseline_sum_r,
                                filtered_trades=filt_n, filtered_sum_r=filt_sum_r))

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{SWEEP_DIR}/dtfx_postrend_sweep_by_bucket.csv", index=False)
filter_df = pd.DataFrame(filter_results)
filter_df.to_csv(f"{SWEEP_DIR}/dtfx_postrend_sweep_filter.csv", index=False)

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
pd.set_option("display.width", 200)
print("\n\n=== FILTER TEST SUMMARY (position 1-or-6+ only, keep rest baseline) ===")
print(filter_df.to_string(index=False))

# ---- chart: avg_r by position bucket, one line per instrument ----
fig, ax = plt.subplots(figsize=(12, 7))
order = ["1", "2", "3", "4", "5", "6+"]
for label, _ in INSTRUMENTS:
    sub = results_df[results_df["instrument"] == label].set_index("pos_bucket").reindex(order)
    ax.plot(order, sub["avg_r"], marker="o", label=label)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xlabel("Position in trend (swing # since last flip)")
ax.set_ylabel("Avg R per trade")
ax.set_title("Position-in-trend pattern across instruments (original variant, strict pivots, gross)")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{SWEEP_DIR}/dtfx_postrend_sweep_chart.png", dpi=130)
print(f"\nSaved chart -> {SWEEP_DIR}/dtfx_postrend_sweep_chart.png", flush=True)
