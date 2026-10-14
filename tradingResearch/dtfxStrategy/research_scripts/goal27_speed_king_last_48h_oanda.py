"""Speed-King, last 48 hours, freshly pulled straight from OANDA -- the LSE vault fetcher
needs a separately-issued API key that isn't available this session and its cached files
only run through 2026-08-31 21:59, ~27h+ stale. OANDA's InstrumentsCandles feed is current
to the last completed 1-minute bar and PRODLIVETRADING already holds practice-account
credentials for it, so this reuses that connection read-only (fetch_warmup_bars -- the same
call, with the same pagination fix, the live bot itself makes on every restart) instead of
asking for a new key.

Zone-formation context uses the same WARMUP_BARS=60,000 convention the live bot warms up
with (~41 days of M1 bars) plus enough extra bars to cover the 48h analysis window, so zones
active in the window are formed exactly the way they'd be on the real deployed bot -- not
padded out with years of LSE history the live system never sees.

$200 start, 3.00% risk (goal21's MC-robust optimum), 50x leverage, margin-constrained
(goal20's model). Hour-by-hour and day-by-day breakdown.
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PROD_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "PRODLIVETRADING"))
sys.path.insert(0, PROD_DIR)

import goal20_speed_king_margin_report as g20

ROBUST_RISK_PCT = 0.03
WARMUP_BARS = 60_000
WINDOW_HOURS = 48
FETCH_COUNT = WARMUP_BARS + WINDOW_HOURS * 60 + 1_500  # buffer for weekend/holiday gaps


def fetch_fresh():
    import config as prod_config
    from broker.oanda_client import OANDAClient

    print(f"Environment: {prod_config.ENV_LABEL}")
    if prod_config.IS_LIVE:
        sys.exit("Refusing: OANDA_ENVIRONMENT is 'live'. This script only reads practice-account data.")

    client = OANDAClient(prod_config.OANDA_API_TOKEN, prod_config.OANDA_ACCOUNT_ID_LONG,
                          environment=prod_config.OANDA_ENVIRONMENT, label="RESEARCH-48H")
    frames = {}
    for inst in g20.TRADEABLE:
        df = client.fetch_warmup_bars(inst, count=FETCH_COUNT)
        frames[inst] = df.reset_index()  # columns: date, open, high, low, close
    return frames


def inject_cache(inst, df):
    df = df.set_index("date").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="first")]
    df = df[["open", "high", "low", "close"]]
    times = df.index.values
    lows = np.ascontiguousarray(df["low"].values, dtype=np.float64)
    highs = np.ascontiguousarray(df["high"].values, dtype=np.float64)
    g20.rs._price_cache[inst] = (times, lows, highs, df)


def main():
    frames = fetch_fresh()
    for inst, df in frames.items():
        inject_cache(inst, df)
        g20.rs._prep_zones(inst, g20.pg.MIN_RETRACEMENT)

    full = g20.build_speed_king_full()
    data_end = full.entry_time.max()
    window_start = data_end - pd.Timedelta(hours=WINDOW_HOURS)
    window = full[(full.entry_time >= window_start) & (full.entry_time <= data_end)].copy()

    window_start_pt = window_start.tz_localize("UTC").tz_convert("America/Los_Angeles").tz_localize(None)
    data_end_pt = data_end.tz_localize("UTC").tz_convert("America/Los_Angeles").tz_localize(None)
    print(f"\nWindow: {window_start_pt} -> {data_end_pt}  Pacific  "
          f"({window_start} -> {data_end} UTC)  ({len(window):,} signal candidates)")

    t, final_eq, max_dd, accepted, trace = g20.simulate_margin_constrained(
        window, start=g20.START, risk_pct=ROBUST_RISK_PCT, max_leverage=g20.MAX_LEVERAGE,
        cap=g20.CAP, track_detail=True)

    acc = t[t.accepted]
    print(f"Signals: {len(t):,}  Accepted: {len(acc):,} ({100*len(acc)/max(len(t),1):.1f}%)  "
          f"Rejected: {(~t.accepted).sum():,}")
    print(f"Max concurrent: {t.concurrent_at_entry.max() if len(t) else 0}")
    if len(acc):
        wr = 100 * acc.is_win.mean()
        w = acc.loc[acc.net_r > 0, "net_r"].sum(); l = -acc.loc[acc.net_r <= 0, "net_r"].sum()
        pf = w / l if l > 0 else float("inf")
        print(f"Accepted WR={wr:.2f}% PF={pf:.3f}")
    print(f"Start ${g20.START:.2f} -> End ${final_eq:.2f} ({100*(final_eq/g20.START-1):+.1f}%) "
          f"maxDD={max_dd:.1f}%\n")

    if len(trace):
        # Display in US/Pacific (America/Los_Angeles handles the PDT/PST switch correctly;
        # early Sept is PDT, UTC-7). close_time from simulate_margin_constrained is naive UTC.
        close_pt = trace.close_time.dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
        trace["close_time_pt"] = close_pt.dt.tz_localize(None)

        trace["hour"] = trace["close_time_pt"].dt.floor("h")
        hourly = trace.groupby("hour").agg(
            trades=("net_r", "size"), wr=("net_r", lambda x: 100 * (x > 0).mean()),
            total_r=("net_r", "sum"), equity_end=("equity", "last"))
        print("Hour-by-hour (Pacific time):")
        print(hourly.to_string())

        trace["day"] = trace["close_time_pt"].dt.date
        daily = trace.groupby("day").agg(
            trades=("net_r", "size"), wr=("net_r", lambda x: 100 * (x > 0).mean()),
            total_r=("net_r", "sum"), equity_end=("equity", "last"))
        print("\nDay-by-day (Pacific time):")
        print(daily.to_string())
    else:
        print("No accepted trades in this window.")

    # Re-express every timestamp column as tz-aware Pacific (America/Los_Angeles -- correctly
    # applies PDT/PST by date) before saving, per request. Source columns are naive UTC.
    # Kept tz-aware (not stripped back to naive) so the CSV's own offset suffix
    # (e.g. "-07:00") makes the timezone unambiguous on its own, without relying on a filename
    # or column-name convention.
    def to_pt(series):
        return series.dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")

    window["entry_time"] = to_pt(window["entry_time"])
    window["exit_time"] = to_pt(window["exit_time"])
    t["entry_time"] = to_pt(t["entry_time"])
    t["exit_time"] = to_pt(t["exit_time"])
    if len(trace):
        trace["close_time"] = to_pt(trace["close_time"])
        trace = trace.drop(columns=["close_time_pt", "hour", "day"])  # superseded by close_time itself now being PT

    window.to_csv(os.path.join(DATA_DIR, "goal27_speed_king_last_48h_trades.csv"), index=False)
    t.to_csv(os.path.join(DATA_DIR, "goal27_speed_king_last_48h_margin.csv"), index=False)
    trace.to_csv(os.path.join(DATA_DIR, "goal27_speed_king_last_48h_trace.csv"), index=False)
    print("\nSaved (entry_time/exit_time/close_time now Pacific, tz-aware): "
          "goal27_speed_king_last_48h_{trades,margin,trace}.csv")


if __name__ == "__main__":
    main()
