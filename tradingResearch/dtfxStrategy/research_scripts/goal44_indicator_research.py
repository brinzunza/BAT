#!/usr/bin/env python3
"""Goal 44: Exploratory Research & Synthetic Indicator Engineering.

Synthesizes:
1. Multi-Timeframe Trend (EMA15, EMA60, EMA240, EMA1440, Trend Slope)
2. Synthetic Volatility & Range (ATR14, Candle Range Ratio, Wick Rejection Force)
3. Momentum & Oscillator Exhaustion (RSI14, MACD Histogram, Bollinger %B)
4. Zone Dynamics (Absorption Climax, Entry Momentum, Age Decay)
5. Time-Decay & Concurrency Slot-Release Triggers

Screens across EURUSD, GBPUSD, EURGBP, USDJPY on true holdout data under dual-subaccount 1-per-side constraints.
"""
import os
import sys
import time
import heapq
import numpy as np
import pandas as pd

os.environ["MPLCONFIGDIR"] = "/tmp"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal5_rr_sweep as g5
import goal20_speed_king_margin_report as g20

position_size_and_margin = g20.position_size_and_margin
BLOCK = g20.BLOCK
RUIN_FLOOR = g20.RUIN_FLOOR
YEAR_NS = 365.25 * 86400 * 1e9

SPLIT = pd.Timestamp("2018-03-09")
START_PER_SIDE = 100.0
MAX_LEVERAGE = 50.0
ACCOUNT_CAP = 1_000_000.0
SPREAD_PIPS = 1.6
N_MC_SIMS = 3000

# 3-Factor Weights: Speed 40%, Win Rate 35%, Frequency 25%
WEIGHTS = dict(speed=0.40, wr=0.35, freq=0.25)


def compute_indicators(df):
    """Vectorized calculation of synthetic indicators on 1-minute OHLC dataframe."""
    c = df["close"].values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values

    # 1. EMAs
    ema15 = pd.Series(c).ewm(span=15, adjust=False).mean().values
    ema60 = pd.Series(c).ewm(span=60, adjust=False).mean().values
    ema240 = pd.Series(c).ewm(span=240, adjust=False).mean().values
    ema1440 = pd.Series(c).ewm(span=1440, adjust=False).mean().values

    # 2. ATR14 & ATR60
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    atr60 = pd.Series(tr).rolling(60, min_periods=1).mean().values

    # 3. Candle Geometry & Wick Rejection
    candle_range = h - l
    range_ratio = candle_range / (atr14 + 1e-9)
    buy_wick_rejection = (c - l) / (candle_range + 1e-9) # High value = rejected low and closed near high
    sell_wick_rejection = (h - c) / (candle_range + 1e-9) # High value = rejected high and closed near low

    # 4. RSI14
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-9)
    rsi14 = 100.0 - (100.0 / (1.0 + rs))

    # 5. MACD (12, 26, 9)
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    macd_sig = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_hist = macd_line - macd_sig

    # 6. Bollinger Bands (20, 2.0)
    sma20 = pd.Series(c).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(c).rolling(20, min_periods=1).std(ddof=0).values
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_b = (c - bb_lower) / (bb_upper - bb_lower + 1e-9)
    bb_bandwidth = (bb_upper - bb_lower) / (sma20 + 1e-9)

    return {
        "ema15": ema15, "ema60": ema60, "ema240": ema240, "ema1440": ema1440,
        "atr14": atr14, "atr60": atr60, "range_ratio": range_ratio,
        "buy_wick_rejection": buy_wick_rejection, "sell_wick_rejection": sell_wick_rejection,
        "rsi14": rsi14, "macd_hist": macd_hist, "bb_b": bb_b, "bb_bandwidth": bb_bandwidth,
        "close": c
    }


def pre_cache_all():
    print("Loading data and computing synthetic indicators across all 4 pairs...", flush=True)
    t0 = time.time()
    cached = {}
    for inst in g5.TRADEABLE:
        z = g5.rs._prep_zones(inst, 2)
        times, lows, highs, df = g5.rs.load_instrument(inst)
        inds = compute_indicators(df)

        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")
        pip_size = g5.pg.PIP_SIZE[inst]

        cached[inst] = {
            "times": times, "times_ns": times_ns, "lows": lows, "highs": highs,
            "df": df, "inds": inds,
            "top": np.ascontiguousarray(z["top"], dtype=np.float64),
            "bottom": np.ascontiguousarray(z["bottom"], dtype=np.float64),
            "mid": np.ascontiguousarray(z["mid"], dtype=np.float64),
            "is_buy": np.ascontiguousarray(z["is_buy"], dtype=np.bool_),
            "start_idx": np.ascontiguousarray(z["start_idx"], dtype=np.int64),
            "c_ns": np.ascontiguousarray(created_ns, dtype=np.int64),
            "pip_size": pip_size,
            "spread_price": SPREAD_PIPS * pip_size
        }
    print(f"Cached all indicators in {time.time()-t0:.2f}s.\n", flush=True)
    return cached


def run_fast_mc(trades, risk_pct=0.05, n_sims=N_MC_SIMS):
    t = trades.sort_values("entry_time").reset_index(drop=True)
    n = len(t)
    entry_ns = t.entry_time.values.astype("datetime64[ns]").astype(np.int64)
    exit_ns = t.exit_time.values.astype("datetime64[ns]").astype(np.int64)
    entry_price = t.entry_price.values
    stop_risk = t.stop_risk.values
    net_r = t.net_r.values
    instrument = t.instrument.values
    is_buy = t.is_buy.values

    rng = np.random.default_rng(42)
    days_to_1m = []

    for sim in range(n_sims):
        equity = {True: START_PER_SIDE, False: START_PER_SIDE}
        margin_used = {True: 0.0, False: 0.0}
        open_count = {True: 0, False: 0}
        heap = []
        clock = 0
        hit_target = False
        t_hit_days = -1.0
        trades_count = 0

        while trades_count < 50_000 and not hit_target:
            start_idx = rng.integers(0, n)
            end_idx = min(start_idx + BLOCK, n)
            offset = clock - entry_ns[start_idx]
            block_exit_max = clock

            for idx in range(start_idx, end_idx):
                et = entry_ns[idx] + offset
                xt = exit_ns[idx] + offset
                if xt > block_exit_max:
                    block_exit_max = xt

                while heap and heap[0][0] <= et:
                    _, r_dollars, margin, r, side = heapq.heappop(heap)
                    equity[side] += r_dollars * r
                    margin_used[side] -= margin
                    open_count[side] -= 1

                if equity[True] <= 1.0 or equity[False] <= 1.0:
                    break

                side = bool(is_buy[idx])
                sizing_base = min(equity[True], equity[False])
                _, risk_dollars, margin = position_size_and_margin(
                    instrument[idx], sizing_base, entry_price[idx], stop_risk[idx], risk_pct,
                    MAX_LEVERAGE, ACCOUNT_CAP
                )
                avail = min(equity[side], ACCOUNT_CAP) - margin_used[side]

                if margin <= avail and open_count[side] < 1:
                    heapq.heappush(heap, (xt, risk_dollars, margin, net_r[idx], side))
                    margin_used[side] += margin
                    open_count[side] += 1
                    trades_count += 1

                comb = equity[True] + equity[False]
                if comb >= 1_000_000.0:
                    hit_target = True
                    t_hit_days = (et / YEAR_NS) * 365.25
                    break

            clock = block_exit_max

        if hit_target:
            days_to_1m.append(t_hit_days)

    return np.median(days_to_1m) if len(days_to_1m) else float("inf")


def main():
    cached = pre_cache_all()

    # Base params
    base_n = 4.5
    base_rr = 0.18
    base_age_h = 0.30
    base_touches = 8

    filter_experiments = [
        ("Baseline (Apex-King)", base_n, base_rr, base_age_h, base_touches, 0.50,
         lambda d, e_idx, is_b: np.ones(len(e_idx), dtype=bool)),
        ("Trend: 1-hr EMA Filter", base_n, base_rr, base_age_h, base_touches, 0.50,
         lambda d, e_idx, is_b: (d["inds"]["close"][e_idx] > d["inds"]["ema60"][e_idx]) if is_b else (d["inds"]["close"][e_idx] < d["inds"]["ema60"][e_idx])),
        ("Trend: 4-hr Trend Alignment (EMA60 > EMA240)", base_n, base_rr, base_age_h, base_touches, 0.50,
         lambda d, e_idx, is_b: (d["inds"]["ema60"][e_idx] > d["inds"]["ema240"][e_idx]) if is_b else (d["inds"]["ema60"][e_idx] < d["inds"]["ema240"][e_idx])),
        ("RSI Momentum Gate (RSI > 45 / RSI < 55)", base_n, base_rr, base_age_h, base_touches, 0.50,
         lambda d, e_idx, is_b: (d["inds"]["rsi14"][e_idx] > 45.0) if is_b else (d["inds"]["rsi14"][e_idx] < 55.0)),
        ("Rejection Force > 50%", base_n, base_rr, base_age_h, base_touches, 0.50,
         lambda d, e_idx, is_b: (d["inds"]["buy_wick_rejection"][e_idx] > 0.50) if is_b else (d["inds"]["sell_wick_rejection"][e_idx] > 0.50)),
        ("Rejection Force > 65%", base_n, base_rr, base_age_h, base_touches, 0.50,
         lambda d, e_idx, is_b: (d["inds"]["buy_wick_rejection"][e_idx] > 0.65) if is_b else (d["inds"]["sell_wick_rejection"][e_idx] > 0.65)),
        ("Trend + Rejection > 50%", base_n, base_rr, base_age_h, base_touches, 0.50,
         lambda d, e_idx, is_b: (((d["inds"]["close"][e_idx] > d["inds"]["ema60"][e_idx]) & (d["inds"]["buy_wick_rejection"][e_idx] > 0.50)) if is_b else ((d["inds"]["close"][e_idx] < d["inds"]["ema60"][e_idx]) & (d["inds"]["sell_wick_rejection"][e_idx] > 0.50)))),
        ("Adaptive Probe Depth (0.65)", base_n, base_rr, base_age_h, base_touches, 0.65,
         lambda d, e_idx, is_b: np.ones(len(e_idx), dtype=bool)),
        ("Adaptive Probe Depth (0.75)", base_n, base_rr, base_age_h, base_touches, 0.75,
         lambda d, e_idx, is_b: np.ones(len(e_idx), dtype=bool)),
        ("High-Speed Zone Engine (N=4.0, RR=0.16, age=0.15h, touches=12)", 4.0, 0.16, 0.15, 12, 0.50,
         lambda d, e_idx, is_b: np.ones(len(e_idx), dtype=bool)),
        ("High-Speed + Rejection > 50%", 4.0, 0.16, 0.15, 12, 0.50,
         lambda d, e_idx, is_b: (d["inds"]["buy_wick_rejection"][e_idx] > 0.50) if is_b else (d["inds"]["sell_wick_rejection"][e_idx] > 0.50)),
        ("Ultra-Apex: (N=4.0, RR=0.16, age=0.20h, touches=10, probe=0.60)", 4.0, 0.16, 0.20, 10, 0.60,
         lambda d, e_idx, is_b: np.ones(len(e_idx), dtype=bool)),
        ("Ultra-Apex + 1-hr Trend", 4.0, 0.16, 0.20, 10, 0.60,
         lambda d, e_idx, is_b: (d["inds"]["close"][e_idx] > d["inds"]["ema60"][e_idx]) if is_b else (d["inds"]["close"][e_idx] < d["inds"]["ema60"][e_idx])),
        ("Ultra-Apex + RSI Momentum", 4.0, 0.16, 0.20, 10, 0.60,
         lambda d, e_idx, is_b: (d["inds"]["rsi14"][e_idx] > 45.0) if is_b else (d["inds"]["rsi14"][e_idx] < 55.0)),
    ]

    print("=" * 125)
    print("  EVALUATING SYNTHETIC INDICATOR & ADVANCED MICROSTRUCTURE COMBINATIONS")
    print("=" * 125)

    results = []

    for name, n_val, rr_val, age_val, touches_val, probe_val, filter_fn in filter_experiments:
        min_fill_age_ns = int(age_val * 3600 * 1e9)
        target_mult = n_val * rr_val

        frames = []
        for inst, d in cached.items():
            for side_is_buy in (True, False):
                mask = d["is_buy"] == side_is_buy
                top = np.ascontiguousarray(d["top"][mask], dtype=np.float64)
                bottom = np.ascontiguousarray(d["bottom"][mask], dtype=np.float64)
                mid = np.ascontiguousarray(d["mid"][mask], dtype=np.float64)
                is_buy_arr = np.ascontiguousarray(d["is_buy"][mask], dtype=np.bool_)
                start_idx = np.ascontiguousarray(d["start_idx"][mask], dtype=np.int64)
                c_ns = np.ascontiguousarray(d["c_ns"][mask], dtype=np.int64)

                half_width = (top - bottom) / 2.0
                risk_pct_zone = (n_val * half_width) / mid
                ok = risk_pct_zone >= g5.pg.MIN_RISK_PCT
                if not np.any(ok):
                    continue

                touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
                    top[ok], bottom[ok], mid[ok], is_buy_arr[ok], start_idx[ok], c_ns[ok],
                    d["times_ns"], d["lows"], d["highs"], min_fill_age_ns, n_val, target_mult,
                    False, probe_val, 1.0, touches_val, g5.pg.MAX_WINDOW_BARS
                )

                if len(entry_idx) == 0:
                    continue

                keep = filter_fn(d, entry_idx, side_is_buy)
                if not np.any(keep):
                    continue

                r_gross = np.where(is_win[keep], rr_val, -1.0)
                cost_in_r = np.where(is_win[keep], 0.0, d["spread_price"] / stop_risk[keep])
                net_r = r_gross - cost_in_r

                frames.append(pd.DataFrame({
                    "instrument": inst,
                    "entry_time": pd.to_datetime(d["times"][entry_idx[keep]]),
                    "exit_time": pd.to_datetime(d["times"][exit_idx[keep]]),
                    "entry_price": entry_mid[keep],
                    "stop_risk": stop_risk[keep],
                    "net_r": net_r,
                    "is_win": is_win[keep],
                    "is_buy": side_is_buy,
                }))

        if not frames:
            continue

        trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
        holdout = trades[trades.entry_time >= SPLIT].reset_index(drop=True)
        if len(holdout) == 0:
            continue

        years = (holdout.entry_time.max() - holdout.entry_time.min()).days / 365.25
        raw_tpy = len(holdout) / years
        raw_wr = 100.0 * (holdout.net_r > 0).mean()
        w_sum = holdout.net_r[holdout.net_r > 0].sum()
        l_sum = -holdout.net_r[holdout.net_r < 0].sum()
        pf = w_sum / l_sum if l_sum > 0 else 0.0

        # Simulate two account
        _, _, _, accepted, _ = g20.simulate_margin_constrained(
            holdout, start=200.0, risk_pct=0.05, max_leverage=50.0, cap=1_000_000.0, track_detail=False
        )
        acc_count = np.sum(accepted)
        acc_tpy = acc_count / years
        margin_pct = 100.0 * acc_count / len(holdout)

        # Run MC
        median_days = run_fast_mc(holdout, risk_pct=0.05, n_sims=3000)

        results.append({
            "Experiment": name,
            "Raw WR (%)": round(raw_wr, 2),
            "Profit Factor": round(pf, 3),
            "Acc TPY": round(acc_tpy, 0),
            "Margin Acc (%)": round(margin_pct, 1),
            "Days to $1M": round(median_days, 1),
            "Years to $1M": round(median_days / 365.25, 3)
        })
        print(f"  Processed {name:45s} | WR: {raw_wr:5.2f}% | PF: {pf:5.3f} | Acc TPY: {acc_tpy:6.0f} | Days to $1M: {median_days:5.1f} d", flush=True)

    df_exp = pd.DataFrame(results)
    print("\n" + "=" * 115)
    print("  FINAL SYNTHETIC INDICATOR EXPERIMENT RESULTS")
    print("=" * 115)
    print(df_exp.to_string(index=False))


if __name__ == "__main__":
    main()
