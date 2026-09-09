#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import pandas as pd

# Set matplotlib cache directory to suppress warnings
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


def pre_cache_instruments():
    cached = {}
    for inst in g5.TRADEABLE:
        z = g5.rs._prep_zones(inst, g5.pg.MIN_RETRACEMENT)
        times, lows, highs, df = g5.rs.load_instrument(inst)
        times_ns = times.astype("datetime64[ns]").astype("int64")
        created_ns = z["created_at"].astype("datetime64[ns]").astype("int64")

        top = np.ascontiguousarray(z["top"], dtype=np.float64)
        bottom = np.ascontiguousarray(z["bottom"], dtype=np.float64)
        mid = np.ascontiguousarray(z["mid"], dtype=np.float64)
        is_buy = np.ascontiguousarray(z["is_buy"], dtype=np.bool_)
        start_idx = np.ascontiguousarray(z["start_idx"], dtype=np.int64)
        c_ns = np.ascontiguousarray(created_ns, dtype=np.int64)

        cached[inst] = {
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
    print("Pre-caching instruments in memory...", flush=True)
    t0 = time.time()
    cached = pre_cache_instruments()
    print(f"Pre-cached in {time.time()-t0:.2f}s.\n", flush=True)

    base_df = pd.read_csv(os.path.join(DATA_DIR, "goal12_strategy_scores.csv"))

    fine_grid = []
    # Test N in [4.0, 4.5, 4.8, 5.0, 5.2, 5.5, 6.0]
    # Test RR in [0.18, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]
    # Test age_h in [0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40]
    # Test bounce in [False, True]
    # Test touches in [3, 4, 6, 8]
    for n in [4.0, 4.5, 4.8, 5.0, 5.2, 5.5, 6.0]:
        for rr in [0.18, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]:
            for age_h in [0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40]:
                for bounce in [False, True]:
                    for touches in [3, 4, 6, 8]:
                        fine_grid.append((n, rr, age_h, bounce, touches))

    print(f"Total fine-grid combinations: {len(fine_grid):,}. Running search...", flush=True)
    results = []
    t_start = time.time()

    for idx, (n, rr, age_h, bounce, touches) in enumerate(fine_grid):
        min_fill_age_ns = int(age_h * 3600 * 1e9)
        target_mult = n * rr

        frames = []
        for inst, d in cached.items():
            half_width = (d["top"] - d["bottom"]) / 2.0
            risk_pct = (n * half_width) / d["mid"]
            ok = risk_pct >= g5.pg.MIN_RISK_PCT

            top, bottom, mid = d["top"][ok], d["bottom"][ok], d["mid"][ok]
            is_buy, start_idx, c_ns = d["is_buy"][ok], d["start_idx"][ok], d["c_ns"][ok]

            touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
                top, bottom, mid, is_buy, start_idx, c_ns, d["times_ns"], d["lows"], d["highs"],
                min_fill_age_ns, n, target_mult, bounce, 0.50, 1.0,
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
        if len(holdout) < 500:
            continue

        net_r = holdout.net_r.values.astype(np.float64)
        years = (holdout.entry_time.max() - holdout.entry_time.min()).days / 365.25
        tpy = len(holdout) / years
        wr = 100.0 * (net_r > 0).mean()

        # Fast MC
        t1_raw, t10_raw, max_dd, ruined = dtfx_engine.run_mc_capped(
            net_r, 25000, BLOCK, MAX_TRADES, 42,
            start=START, risk_frac=RISK_PCT, cap=CAP, n_threads=0
        )
        t1_hit = t1_raw[t1_raw > 0]
        median_yrs_1m = np.median(t1_hit / tpy) if len(t1_hit) else float("inf")
        median_dd = np.median(max_dd) * 100.0

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

        results.append({
            "n": n, "rr": rr, "age_h": age_h, "bounce": bounce, "touches": touches,
            "final_score": round(final_score, 1), "margin_score": round(margin_score, 1),
            "margin_pct": round(margin_pct, 1), "wr": round(wr, 2), "tpy": round(tpy, 0),
            "median_yrs_1m": round(median_yrs_1m, 3), "median_dd": round(median_dd, 1),
            "speed_score": round(speed_score, 1), "wr_score": round(wr_score, 1),
            "freq_score": round(freq_score, 1), "dd_score": round(dd_score, 1),
            "rrr_score": round(rrr_score, 1), "trades": len(holdout)
        })

        if (idx + 1) % 100 == 0 or idx == len(fine_grid) - 1:
            elapsed = time.time() - t_start
            print(f"[{idx+1:,}/{len(fine_grid):,}] ({idx/len(fine_grid)*100:.1f}%) in {elapsed:.1f}s | "
                  f"Evaluated: {len(results):,}", flush=True)

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(DATA_DIR, "goal37_fine_sweep_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_res):,} results to {out_csv}.", flush=True)

    print("\n" + "=" * 115)
    print("  TOP 25 CANDIDATES BY COMPOSITE SCORE")
    print("=" * 115)
    print(df_res.sort_values("final_score", ascending=False)[
        ["n", "rr", "age_h", "bounce", "touches", "final_score", "margin_score", "margin_pct",
         "wr", "tpy", "median_yrs_1m", "speed_score", "wr_score", "freq_score", "rrr_score"]
    ].head(25).to_string(index=False), flush=True)

    print("\n" + "=" * 115)
    print("  STRICT BEAT SPEED-KING (Final Score >= 87.6 AND margin_pct > 35.3%)")
    print("=" * 115)
    winners = df_res[(df_res["final_score"] >= 87.6) & (df_res["margin_pct"] > 35.3)].sort_values("final_score", ascending=False)
    if len(winners) > 0:
        print(winners[
            ["n", "rr", "age_h", "bounce", "touches", "final_score", "margin_score", "margin_pct",
             "wr", "tpy", "median_yrs_1m", "speed_score", "wr_score", "freq_score", "rrr_score"]
        ].to_string(index=False), flush=True)
    else:
        print("No candidates strictly >= 87.6 with margin > 35.3%, checking best overall:", flush=True)


if __name__ == "__main__":
    main()
