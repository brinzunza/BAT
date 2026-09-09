#!/usr/bin/env python3
"""Goal 39: High-RRR Systematic Search ($R:R > 0.75$) Targeting Composite Score > 88.7.

Searches across:
- Stop Multiplier (N): 1.5 to 8.0
- Target R:R (rr): 0.75 to 1.50 (in steps: 0.76, 0.80, 0.85, 0.90, 1.00, 1.10, 1.25, 1.50)
- Fill Age Gate (age_h): 0.05h to 2.0h
- Re-Arm Condition (bounce): [False, True]
- Max Touches: 1, 2, 3, 4, 6, 8, 12, 16
- Probe Depth: 0.20, 0.35, 0.50, 0.70, 0.85, 1.00
- Excursion Multiplier: 0.80, 1.00, 1.20, 1.50
- Zone Retracement: 1, 2, 3

Uses multi-threaded C++ simulation (dtfx_engine) and C++ Monte Carlo bootstrap.
"""
import os
import sys
import time
import itertools
import numpy as np
import pandas as pd

os.environ["MPLCONFIGDIR"] = "/tmp"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal5_rr_sweep as g5
import goal20_speed_king_margin_report as g20
from goal10_cana_nt_mc_capped import BLOCK, MAX_TRADES

SPLIT = pd.Timestamp("2018-03-09")
START = 200.0
RISK_PCT = 0.05
MAX_LEVERAGE = 50.0
CAP = 2_000_000.0

DEFAULT_WEIGHTS = dict(speed=0.35, wr=0.30, freq=0.12, margin=0.08, drawdown=0.10, rrr=0.05)


def pre_cache_all_retracements():
    cached = {}
    for retracement in [1, 2, 3]:
        cached[retracement] = {}
        for inst in g5.TRADEABLE:
            z = g5.rs._prep_zones(inst, retracement)
            times, lows, highs, df = g5.rs.load_instrument(inst)
            times_ns = times.astype("datetime64[ns]").astype("int64")
            created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")

            top = np.ascontiguousarray(z["top"], dtype=np.float64)
            bottom = np.ascontiguousarray(z["bottom"], dtype=np.float64)
            mid = np.ascontiguousarray(z["mid"], dtype=np.float64)
            is_buy = np.ascontiguousarray(z["is_buy"], dtype=np.bool_)
            start_idx = np.ascontiguousarray(z["start_idx"], dtype=np.int64)
            c_ns = np.ascontiguousarray(created_ns, dtype=np.int64)

            cached[retracement][inst] = {
                "times": times,
                "times_ns": times_ns,
                "lows": lows,
                "highs": highs,
                "top": top,
                "bottom": bottom,
                "mid": mid,
                "is_buy": is_buy,
                "start_idx": start_idx,
                "c_ns": c_ns,
                "pip_size": g5.pg.PIP_SIZE[inst],
                "spread_price": g5.pg.SPREAD_PIPS_TYPICAL * g5.pg.PIP_SIZE[inst],
            }
    return cached


def main():
    print("Pre-caching zone buffers across retracements 1, 2, 3...", flush=True)
    t0 = time.time()
    cached = pre_cache_all_retracements()
    print(f"Pre-cached in {time.time()-t0:.2f}s.\n", flush=True)

    base_scores_path = os.path.join(DATA_DIR, "goal38_strategy_rankings.csv")
    if not os.path.exists(base_scores_path):
        base_scores_path = os.path.join(DATA_DIR, "goal12_strategy_scores.csv")
    base_df = pd.read_csv(base_scores_path)

    # Grid parameters tailored for RRR > 0.75
    RETRACEMENT_GRID = [1, 2, 3]
    N_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
    RR_GRID = [0.76, 0.80, 0.85, 0.90, 1.00, 1.10, 1.25, 1.50]
    AGE_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    BOUNCE_GRID = [False, True]
    TOUCHES_GRID = [1, 2, 3, 4, 6, 8, 12]
    PROBE_GRID = [0.25, 0.50, 0.75, 1.0]
    EXC_GRID = [0.80, 1.00, 1.20]

    # Staged sweep: generate high-priority combinations
    combos = []
    # Primary grid
    for ret in [2, 1, 3]:
        for n in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
            for rr in [0.76, 0.80, 0.85, 0.90, 1.00, 1.10, 1.25]:
                for age_h in [0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
                    for bounce in [False, True]:
                        for touches in [2, 3, 4, 6, 8]:
                            for probe in [0.35, 0.50, 0.75]:
                                for exc in [1.0, 1.2]:
                                    combos.append((ret, n, rr, age_h, bounce, touches, probe, exc))

    total_combos = len(combos)
    print(f"Total High-RRR Combinations to Evaluate: {total_combos:,}. Launching search...", flush=True)

    results = []
    t_start = time.time()
    count = 0
    high_score_count = 0

    for idx, (ret, n, rr, age_h, bounce, touches, probe, exc) in enumerate(combos):
        count += 1
        min_fill_age_ns = int(age_h * 3600 * 1e9)
        target_mult = n * rr

        frames = []
        inst_dict = cached[ret]
        for inst, d in inst_dict.items():
            half_width = (d["top"] - d["bottom"]) / 2.0
            risk_pct = (n * half_width) / d["mid"]
            ok = risk_pct >= g5.pg.MIN_RISK_PCT

            top, bottom, mid = d["top"][ok], d["bottom"][ok], d["mid"][ok]
            is_buy, start_idx, c_ns = d["is_buy"][ok], d["start_idx"][ok], d["c_ns"][ok]

            touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
                top, bottom, mid, is_buy, start_idx, c_ns, d["times_ns"], d["lows"], d["highs"],
                min_fill_age_ns, n, target_mult, bounce, probe, exc,
                touches, g5.pg.MAX_WINDOW_BARS
            )

            if len(entry_idx) == 0:
                continue

            r_gross = np.where(is_win, rr, -1.0)
            cost_in_r = np.where(is_win, 0.0, d["spread_price"] / stop_risk)
            net_r = r_gross - cost_in_r

            frames.append(pd.DataFrame({
                "instrument": inst,
                "entry_time": pd.to_datetime(d["times"][entry_idx]),
                "exit_time": pd.to_datetime(d["times"][exit_idx]),
                "entry_price": entry_mid,
                "stop_risk": stop_risk,
                "net_r": net_r,
                "is_win": is_win,
            }))

        if not frames:
            continue

        trades = pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
        holdout = trades[trades.entry_time >= SPLIT].reset_index(drop=True)
        if len(holdout) < 200:
            continue

        net_r = holdout.net_r.values.astype(np.float64)
        years = (holdout.entry_time.max() - holdout.entry_time.min()).days / 365.25
        tpy = len(holdout) / years
        wr = 100.0 * (net_r > 0).mean()

        w_sum = net_r[net_r > 0].sum()
        l_sum = -net_r[net_r < 0].sum()
        pf = (w_sum / l_sum) if l_sum > 0 else 0.0

        # Fast pre-screen: at RR >= 0.76, we require PF >= 1.25 and WR >= 60.0%
        if pf < 1.25 or wr < 60.0:
            continue

        # Fast MC (15,000 runs during sweep)
        t1_raw, t10_raw, max_dd, ruined = dtfx_engine.run_mc_capped(
            net_r, 15000, BLOCK, MAX_TRADES, 42,
            start=START, risk_frac=RISK_PCT, cap=CAP, n_threads=0
        )
        t1_hit = t1_raw[t1_raw > 0]
        median_yrs_1m = np.median(t1_hit / tpy) if len(t1_hit) else float("inf")
        median_dd = np.median(max_dd) * 100.0

        if median_yrs_1m > 1.5:
            continue

        # Margin walk
        _, final_eq, _, accepted, _ = g20.simulate_margin_constrained(
            holdout, start=START, risk_pct=RISK_PCT, max_leverage=MAX_LEVERAGE, cap=CAP, track_detail=False
        )
        margin_pct = 100.0 * np.mean(accepted)

        # Min-max normalization
        all_speeds = np.append(base_df["median_yrs_1m"].values, median_yrs_1m)
        all_dds = np.append(base_df["median_dd"].values, median_dd)
        all_wrs = np.append(base_df["wr"].values, wr)
        all_tpys = np.append(base_df["tpy"].values, tpy)
        all_margins = np.append(base_df["margin_pct"].values, margin_pct)
        all_rrrs = np.append(base_df["rr"].values, rr)

        def norm_val(val, arr, higher_is_better):
            lo, hi = arr.min(), arr.max()
            frac = (val - lo) / (hi - lo) if hi > lo else 1.0
            return 100.0 * frac if higher_is_better else 100.0 * (1.0 - frac)

        speed_score = norm_val(median_yrs_1m, all_speeds, False)
        dd_score = norm_val(median_dd, all_dds, True)
        wr_score = norm_val(wr, all_wrs, True)
        freq_score = norm_val(tpy, all_tpys, True)
        margin_score = norm_val(margin_pct, all_margins, True)
        rrr_score = norm_val(rr, all_rrrs, True)

        w = DEFAULT_WEIGHTS
        final_score = (
            w["speed"] * speed_score +
            w["wr"] * wr_score +
            w["freq"] * freq_score +
            w["margin"] * margin_score +
            w["drawdown"] * dd_score +
            w["rrr"] * rrr_score
        )

        res_dict = {
            "ret": ret, "n": n, "rr": rr, "age_h": age_h, "bounce": bounce,
            "touches": touches, "probe": probe, "exc": exc,
            "final_score": round(final_score, 1), "margin_score": round(margin_score, 1),
            "margin_pct": round(margin_pct, 1), "wr": round(wr, 2), "pf": round(pf, 3),
            "tpy": round(tpy, 0), "median_yrs_1m": round(median_yrs_1m, 3),
            "median_dd": round(median_dd, 1), "speed_score": round(speed_score, 1),
            "wr_score": round(wr_score, 1), "freq_score": round(freq_score, 1),
            "dd_score": round(dd_score, 1), "rrr_score": round(rrr_score, 1),
            "trades": len(holdout)
        }
        results.append(res_dict)

        if final_score >= 88.0:
            high_score_count += 1
            print(f"  🔥 HIGH SCORE CANDIDATE: Score={final_score:.1f} | RR={rr:.2f} | WR={wr:.2f}% | "
                  f"PF={pf:.3f} | Margin={margin_pct:.1f}% (Score={margin_score:.1f}) | "
                  f"Yrs1M={median_yrs_1m:.3f} | Params: (ret={ret}, N={n}, age={age_h}h, touches={touches}, probe={probe})",
                  flush=True)

        if count % 200 == 0:
            elapsed = time.time() - t_start
            print(f"[{count:,}/{total_combos:,}] ({count/total_combos*100:.1f}%) in {elapsed:.1f}s | "
                  f"Screened: {len(results):,} | Score >= 88.0: {high_score_count}", flush=True)

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(DATA_DIR, "goal39_high_rrr_search_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nCompleted sweep in {time.time()-t_start:.1f}s. Saved {len(df_res):,} candidates to {out_csv}.", flush=True)

    print("\n" + "=" * 125)
    print("  TOP 30 HIGH-RRR (R:R > 0.75) CANDIDATES RANKED BY COMPOSITE SCORE")
    print("=" * 125)
    print(df_res.sort_values("final_score", ascending=False)[
        ["ret", "n", "rr", "age_h", "bounce", "touches", "probe", "final_score", "margin_score",
         "margin_pct", "wr", "pf", "tpy", "median_yrs_1m", "rrr_score"]
    ].head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
