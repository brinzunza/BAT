"""Direct user request (2026-09-08): "show the results of each of those in the past
month" -- the same 3 HTF-alignment segments from goal120 (ALL / 15m-aligned /
15m+4h-aligned), restricted to the last 30 real days (fresh OANDA data through today),
instead of full history. Per-symbol MIN_ZONE_PIPS confirmed already correct (10p USDJPY,
5p others, config.py's get_min_zone_pips) -- same as goal120, just reused here.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PRODLIVE_DIR = "/Users/brunoinzunza/Documents/GitHub/PRODLIVETRADING"
sys.path.insert(0, PRODLIVE_DIR)
FRESH_DIR = "/tmp/dtfx_pastmonth"

import goal95_every_swing_zones as g95
import goal99_dtfx_base_backtest as g99
import goal79_two_account_portfolio as g79
import config as prod_config
from goal120_htf_alignment_backtest import trend_timeline, trend_at, resample_ohlc

SYMBOLS = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}
SPREAD_PIPS = 1.6
MIN_STOP_PIPS = 3.0
MAX_WINDOW = 1440
RISK_PCT = 0.03
LOOKBACK_DAYS = 30


def main():
    t0 = time.time()
    all_candidates = {}
    for inst, pip in SYMBOLS.items():
        t1 = time.time()
        min_zone_pips = prod_config.get_min_zone_pips(inst)
        df = pd.read_csv(os.path.join(FRESH_DIR, f"{inst}.csv"), parse_dates=["date"])
        df["date"] = df["date"].dt.tz_localize(None)
        df = df.drop_duplicates(subset="date").set_index("date").sort_index()
        high1 = df["high"].values.astype(np.float64)
        low1 = df["low"].values.astype(np.float64)
        close1 = df["close"].values.astype(np.float64)
        times1_ns = df.index.values.astype("datetime64[ns]").astype(np.int64)

        confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high1, low1, close1)
        z_created, z_top, z_bottom, _, _, z_is_buy, _ = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)
        min_zone_height = min_zone_pips * pip
        entry_idx, exit_idx, outcome, net_r, entry_price = g99.resolve_base_strategy(
            z_created, z_top, z_bottom, z_is_buy, high1, low1,
            np.full(len(high1), np.nan), np.full(len(high1), np.nan),
            SPREAD_PIPS * pip, MIN_STOP_PIPS * pip, MAX_WINDOW, False, min_zone_height, True)
        resolved = outcome > 0
        ei = entry_idx[resolved]; xi = exit_idx[resolved]
        stop_risk = np.abs(entry_price[resolved] - np.where(z_is_buy[resolved], z_bottom[resolved], z_top[resolved]))
        cand = pd.DataFrame({
            "entry_time": df.index.values[ei], "entry_ns": times1_ns[ei], "exit_time": df.index.values[xi],
            "entry_price": entry_price[resolved], "stop_risk": stop_risk,
            "is_buy": z_is_buy[resolved], "net_r": net_r[resolved], "is_win": outcome[resolved] == 1,
            "pips": np.where(outcome[resolved] == 1, net_r[resolved] * stop_risk / pip, -stop_risk / pip),
        })

        for tf_name, rule in (("15m", "15min"), ("4h", "4h")):
            res = resample_ohlc(df[["open", "high", "low", "close"]], rule)
            h_, l_, c_ = res["high"].values, res["low"].values, res["close"].values
            times_tf_ns = res.index.values.astype("datetime64[ns]").astype(np.int64)
            pivot_ns_idx, trend_vals = trend_timeline(h_, l_, c_)
            pivot_ns = times_tf_ns[pivot_ns_idx]
            tf_trend = trend_at(pivot_ns, trend_vals, cand["entry_ns"].values)
            cand[f"aligned_{tf_name}"] = np.where(cand["is_buy"].values, tf_trend == 1, tf_trend == -1)

        cand["instrument"] = inst
        all_candidates[inst] = cand
        print(f"  {inst}: {len(cand):,} total candidates (min_zone={min_zone_pips:.0f}p)  ({time.time()-t1:.1f}s)")

    combined = pd.concat(all_candidates.values(), ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    now = combined["entry_time"].max()
    window_start = now - pd.Timedelta(days=LOOKBACK_DAYS)
    month = combined[combined["entry_time"] >= window_start].reset_index(drop=True)
    print(f"\nPast {LOOKBACK_DAYS} days: {window_start} -> {now}  ({len(month):,} total candidates before segmenting)")

    segments = {
        "ALL (no HTF filter, today's actual behavior)": lambda d: d,
        "15M-aligned": lambda d: d[d["aligned_15m"]],
        "15M+4H-aligned": lambda d: d[d["aligned_15m"] & d["aligned_4h"]],
    }

    def stats(df):
        n = len(df)
        if n == 0:
            return dict(n=0, wr=float("nan"), pf=float("nan"), avg_pips_win=float("nan"),
                        avg_pips_loss=float("nan"), trades_per_day=float("nan"))
        wr = 100.0 * df["is_win"].mean()
        gw, gl = df.loc[df["is_win"], "net_r"].sum(), -df.loc[~df["is_win"], "net_r"].sum()
        pf = gw / gl if gl > 0 else float("inf")
        wins, losses = df[df["is_win"]], df[~df["is_win"]]
        return dict(n=n, wr=wr, pf=pf,
                     avg_pips_win=wins["pips"].mean() if len(wins) else float("nan"),
                     avg_pips_loss=losses["pips"].mean() if len(losses) else float("nan"),
                     trades_per_day=n / LOOKBACK_DAYS)

    print(f"\n{'='*95}\nPAST {LOOKBACK_DAYS} DAYS -- detected candidates, by HTF alignment segment\n{'='*95}")
    for seg_name, seg_fn in segments.items():
        s = stats(seg_fn(month))
        print(f"  {seg_name:<45} n={s['n']:>6,}  trades/day={s['trades_per_day']:6.2f}  WR={s['wr']:6.2f}%  "
              f"PF={s['pf']:6.3f}  avgPips(win)={s['avg_pips_win']:+6.2f}  avgPips(loss)={s['avg_pips_loss']:+6.2f}")

    print(f"\n{'='*95}\nPAST {LOOKBACK_DAYS} DAYS -- $200 two-account simulation (fresh start), by segment\n{'='*95}")
    for seg_name, seg_fn in segments.items():
        seg_trades = seg_fn(month)[["instrument", "entry_time", "exit_time", "entry_price",
                                     "stop_risk", "net_r", "is_buy"]].sort_values("entry_time").reset_index(drop=True)
        if len(seg_trades) == 0:
            print(f"\n  {seg_name}: no trades")
            continue
        result, eq_buy, eq_sell, max_dd, trace = g79.simulate_two_account(
            seg_trades, risk_pct=RISK_PCT, start_per_side=g79.START_PER_SIDE,
            max_leverage=g79.MAX_LEVERAGE, cap=g79.ACCOUNT_CAP, track_detail=False)
        acc = result[result["accepted"]]
        n = len(acc)
        wr = 100.0 * (acc["net_r"] > 0).mean() if n else float("nan")
        gw, gl = acc.loc[acc["net_r"] > 0, "net_r"].sum(), -acc.loc[acc["net_r"] <= 0, "net_r"].sum()
        pf = gw / gl if gl > 0 else float("inf")
        final = eq_buy + eq_sell
        print(f"\n  {seg_name}")
        print(f"    accepted trades: {n:,}  ({n/LOOKBACK_DAYS:.2f}/day)   WR={wr:.2f}%  PF={pf:.3f}")
        print(f"    final balance: ${final:,.2f}  (buy=${eq_buy:,.2f} sell=${eq_sell:,.2f})   "
              f"return={(final/200-1)*100:+.2f}%   maxDD={max_dd:.2f}%")

    month.to_csv(os.path.join(DATA_DIR, "goal121_past_month_candidates.csv"), index=False)
    print(f"\nSaved -> data/goal121_past_month_candidates.csv")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
