"""Direct user request (2026-09-08): "take this strategy and back test it with higher
time frame alignment. this means that you look at maybe 4 hours zone or 15 minute zones
and only in their zones will you then take trades from the 1 minute zone that align (15
min buy and 1 min buy) or (4h buy, 15 min buy, 1 min buy) etc. then compare."

Builds DTFX-every-swing zones independently at THREE timeframes -- 1-minute (unchanged),
15-minute, and 4-hour -- using the exact same mechanism at each (close-based symmetric
zigzag + every consecutive swing pair is a zone, goal95). A higher-timeframe zone's
directional bias is considered "alive" from the moment it's created until it either
invalidates (price breaches its own stop side) or expires -- the same lifecycle rule
dtfx.md already defines for 1-minute zones (MAX_WINDOW = 1440 bars OF THAT OWN TIMEFRAME:
15 days for a 15m zone, ~240 days for a 4h zone -- a deliberate, explicit choice to reuse
the spec's own "1440 own-bars" convention rather than invent a new one; not otherwise
specified anywhere, since HTF alignment is new experimentation, per dtfx.md's own framing).

Three segments compared, nested (strictest last):
  ALL       -- no HTF filter (today's actual deployed behavior)
  15M       -- 1-minute entry only counted if a SAME-SIDE 15-minute zone is alive at that instant
  15M+4H    -- also requires a SAME-SIDE 4-hour zone alive at that instant

Per-symbol MIN_ZONE_PIPS matches the currently-deployed config exactly (config.py's
get_min_zone_pips: 10 pips USDJPY, 5 pips others) -- same floor reused at every timeframe
for a clean, single-variable comparison (HTF zones are naturally far taller anyway, so
this floor rarely binds there).
"""
import os
import sys
import time
import heapq

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PRODLIVE_DIR = "/Users/brunoinzunza/Documents/GitHub/PRODLIVETRADING"
sys.path.insert(0, PRODLIVE_DIR)

import goal95_every_swing_zones as g95
import goal99_dtfx_base_backtest as g99
import goal79_two_account_portfolio as g79
import config as prod_config  # the ACTUAL currently-deployed config (per-symbol min zone pips)

INSTRUMENTS = {
    "EURUSD": ("lse_eurusd_1min_full.csv", 1e-4),
    "GBPUSD": ("lse_gbpusd_1min_full.csv", 1e-4),
    "EURGBP": ("lse_eurgbp_1min_full.csv", 1e-4),
    "USDJPY": ("lse_usdjpy_1min_2018on.csv", 1e-2),
}
SPREAD_PIPS = 1.6
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440         # own-bars, at whatever timeframe -- dtfx.md's stated convention
RISK_PCT = 0.03
PRE_END = pd.Timestamp("2021-01-01")
POST_START = pd.Timestamp("2026-04-01")


def build_zones_raw(high, low, close):
    confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high, low, close)
    return g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)


def trend_timeline(high, low, close):
    """Reconstructs the SAME trend state build_zones_every_swing tracks internally
    (dtfx.md #4's definition: a new high above the last confirmed high -> uptrend, a new
    low below the last confirmed low -> downtrend), but exposes trend-over-time as a step
    function instead of zone records. First attempt at HTF "alignment" used zone-alive
    windows (created -> invalidated/expired) instead -- scrapped after finding it was
    almost never restrictive (94%+ of 1-minute trades "aligned"): a zone can stay alive
    for up to 1440 own-bars (15 days at 15m, ~240 days at 4h) without invalidating, so
    BOTH a buy zone and a sell zone are very often alive simultaneously, making the filter
    nearly a no-op. Trend is a single, mutually-exclusive state -- exactly what "is the
    higher timeframe currently biased the same direction" actually means."""
    confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high, low, close)
    m = len(confirm_idx)
    out_idx = np.empty(m, dtype=np.int64)
    out_trend = np.empty(m, dtype=np.int8)
    has_high = has_low = False
    last_high_price = last_low_price = 0.0
    trend = 0
    n_out = 0
    for k in range(m):
        ci = confirm_idx[k]; p = price[k]
        if is_high[k]:
            if has_high:
                trend = 1 if p > last_high_price else -1
            last_high_price = p; has_high = True
        else:
            if has_low:
                trend = -1 if p < last_low_price else 1
            last_low_price = p; has_low = True
        out_idx[n_out] = ci; out_trend[n_out] = trend; n_out += 1
    return out_idx[:n_out], out_trend[:n_out]


def trend_at(pivot_ns, trend_vals, query_ns):
    """Step-function lookup: the trend value most recently confirmed at or before each
    query timestamp (0 if queried before the very first pivot confirms)."""
    idx = np.searchsorted(pivot_ns, query_ns, side="right") - 1
    out = np.zeros(len(query_ns), dtype=np.int8)
    valid = idx >= 0
    out[valid] = trend_vals[idx[valid]]
    return out


def resample_ohlc(df, rule):
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    out = pd.concat([o, h, l, c], axis=1, keys=["open", "high", "low", "close"]).dropna()
    return out


def main():
    t0 = time.time()
    rows = []
    all_candidates = {}   # instrument -> DataFrame of 1m candidates with alignment flags
    for inst, (fname, pip) in INSTRUMENTS.items():
        t1 = time.time()
        min_zone_pips = prod_config.get_min_zone_pips(inst)
        path = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        high1 = df["high"].values.astype(np.float64)
        low1 = df["low"].values.astype(np.float64)
        close1 = df["close"].values.astype(np.float64)
        times1_ns = df.index.values.astype("datetime64[ns]").astype(np.int64)

        # --- 1-minute zones + candidates (current deployed mechanism, unchanged) ---
        z_created, z_top, z_bottom, *_ , z_is_buy, _ = build_zones_raw(high1, low1, close1)
        min_zone_height = min_zone_pips * pip
        entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
            z_created, z_top, z_bottom, z_is_buy, high1, low1,
            np.full(len(high1), np.nan), np.full(len(high1), np.nan),
            SPREAD_PIPS * pip, MIN_STOP_PIPS * pip, MAX_WINDOW, False, min_zone_height, True)
        resolved = outcome > 0
        ei = entry_idx[resolved]; xi = exit_idx[resolved]
        stop_risk = np.abs(entry_price[resolved] - np.where(z_is_buy[resolved], z_bottom[resolved], z_top[resolved]))
        cand = pd.DataFrame({
            "entry_time": df.index.values[ei], "entry_ns": times1_ns[ei],
            "exit_time": df.index.values[xi],
            "entry_price": entry_price[resolved], "stop_risk": stop_risk,
            "is_buy": z_is_buy[resolved], "net_r": net_r[resolved], "is_win": outcome[resolved] == 1,
            "eff_risk_pips": stop_risk / pip,
        })

        # --- 15-minute and 4-hour trend state, built fresh on resampled OHLC ---
        for tf_name, rule in (("15m", "15min"), ("4h", "4h")):
            res = resample_ohlc(df[["open", "high", "low", "close"]], rule)
            h_, l_, c_ = res["high"].values, res["low"].values, res["close"].values
            times_tf_ns = res.index.values.astype("datetime64[ns]").astype(np.int64)
            pivot_ns_idx, trend_vals = trend_timeline(h_, l_, c_)
            pivot_ns = times_tf_ns[pivot_ns_idx]
            tf_trend = trend_at(pivot_ns, trend_vals, cand["entry_ns"].values)
            aligned = np.where(cand["is_buy"].values, tf_trend == 1, tf_trend == -1)
            cand[f"aligned_{tf_name}"] = aligned
            up_pct = 100.0 * (tf_trend == 1).mean()
            print(f"  {inst} {tf_name}: {len(pivot_ns):,} pivots, trend up {up_pct:.1f}% of the time  "
                  f"-> {aligned.sum():,}/{len(cand):,} 1m candidates aligned")

        cand["instrument"] = inst
        all_candidates[inst] = cand
        print(f"  {inst}: {len(cand):,} total 1m candidates (min_zone={min_zone_pips:.0f}p)  ({time.time()-t1:.1f}s)")

    combined = pd.concat(all_candidates.values(), ignore_index=True).sort_values("entry_time").reset_index(drop=True)

    def stats(df):
        n = len(df)
        if n == 0:
            return dict(n=0, wr=float("nan"), pf=float("nan"), avg_r=float("nan"), trades_per_day=float("nan"))
        wr = 100.0 * df["is_win"].mean()
        gw, gl = df.loc[df["is_win"], "net_r"].sum(), -df.loc[~df["is_win"], "net_r"].sum()
        pf = gw / gl if gl > 0 else float("inf")
        span_days = (df["entry_time"].max() - df["entry_time"].min()).total_seconds() / 86400
        return dict(n=n, wr=wr, pf=pf, avg_r=df["net_r"].mean(), trades_per_day=n / span_days if span_days else float("nan"))

    print(f"\n{'='*90}\nDETECTED CANDIDATES (pre-concurrency), by HTF alignment segment\n{'='*90}")
    segments = {
        "ALL (no HTF filter, today's actual behavior)": lambda d: d,
        "15M-aligned": lambda d: d[d["aligned_15m"]],
        "15M+4H-aligned": lambda d: d[d["aligned_15m"] & d["aligned_4h"]],
    }
    for regime_name, lo, hi in [("PRE-2021", None, PRE_END), ("POST-2026-04", POST_START, None)]:
        print(f"\n--- {regime_name} ---")
        for seg_name, seg_fn in segments.items():
            d = combined
            if lo is not None:
                d = d[d["entry_time"] >= lo]
            if hi is not None:
                d = d[d["entry_time"] < hi]
            s = stats(seg_fn(d))
            print(f"  {seg_name:<45} n={s['n']:>8,}  WR={s['wr']:6.2f}%  PF={s['pf']:6.3f}  "
                  f"avgR={s['avg_r']:+7.4f}  trades/day={s['trades_per_day']:6.2f}")

    combined.to_csv(os.path.join(DATA_DIR, "goal120_htf_candidates.csv"), index=False)
    print(f"\nSaved -> data/goal120_htf_candidates.csv")
    print(f"Detection phase runtime: {time.time()-t0:.1f}s")

    # ---- Two-account $ simulation per segment, same concurrency model used throughout
    # this project's other two-account comparisons (goal79's simulate_two_account: one
    # heap-based accept/reject slot per side, shared across all 4 symbols). Full-history
    # run so each segment gets a real, comparable long-run picture. ----
    print(f"\n{'='*90}\nTWO-ACCOUNT $200 SIMULATION (full history, {RISK_PCT:.0%} risk), by HTF alignment segment\n{'='*90}")
    for seg_name, seg_fn in segments.items():
        seg_trades = seg_fn(combined)[["instrument", "entry_time", "exit_time", "entry_price",
                                        "stop_risk", "net_r", "is_buy"]].sort_values("entry_time").reset_index(drop=True)
        result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
            seg_trades, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
            max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=False)
        acc = result[result["accepted"]]
        n = len(acc)
        wr = 100.0 * (acc["net_r"] > 0).mean() if n else float("nan")
        gw, gl = acc.loc[acc["net_r"] > 0, "net_r"].sum(), -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
        pf = gw / gl if gl > 0 else float("inf")
        span_days = (acc["entry_time"].max() - acc["entry_time"].min()).total_seconds() / 86400 if n else float("nan")
        final = eq_buy + eq_sell
        print(f"\n  {seg_name}")
        print(f"    accepted trades: {n:,}  ({n/span_days:.2f}/day over {span_days:.0f} days)")
        print(f"    WR={wr:.2f}%  PF={pf:.3f}  final=${final:,.2f}  maxDD={max_dd:.2f}%")

    return combined


if __name__ == "__main__":
    main()
