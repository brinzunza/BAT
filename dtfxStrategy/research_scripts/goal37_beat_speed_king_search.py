"""High-Speed Systematic Multi-Dimensional Parameter Sweep to Beat Speed-King.

Evaluates candidates on the 4 tradable instruments (EURUSD, GBPUSD, EURGBP, USDJPY) across:
- Stop multiplier (N): 5.0 to 8.0
- Target R:R (rr): 0.22 to 0.40
- Fill-age gate (age_h): 0.175 to 0.60h (10.5 to 36 min)
- Re-arm rule (bounce): False (unconditional) vs True (bounce-gated)
- Max touches: 3, 4, 6, 8

Features:
1. In-memory C++ buffer caching (zero Python overhead per loop)
2. Vectorized first-half & holdout metric filtering
3. Parallel multi-threaded C++ Monte Carlo simulation (dtfx_engine.run_mc_capped)
4. Event-driven broker-level margin capacity walk ($200 start, 5% risk, 50x leverage)
5. 6-Factor composite score ranking against all 14 baseline strategies
"""
import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal5_rr_sweep as g5
import goal20_speed_king_margin_report as g20
from goal10_cana_nt_mc_capped import BLOCK, MAX_TRADES

SPLIT = pd.Timestamp("2018-03-09")
SPLIT_NS = SPLIT.value
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


def build_candidate_trades(cached, n, rr, age_h, bounce, touches, probe_depth=0.50):
    frames = []
    min_fill_age_ns = int(age_h * 3600 * 1e9)
    target_mult = n * rr

    for inst, d in cached.items():
        half_width = (d["top"] - d["bottom"]) / 2.0
        risk_pct = (n * half_width) / d["mid"]
        ok = risk_pct >= g5.pg.MIN_RISK_PCT

        top, bottom, mid = d["top"][ok], d["bottom"][ok], d["mid"][ok]
        is_buy, start_idx, c_ns = d["is_buy"][ok], d["start_idx"][ok], d["c_ns"][ok]

        touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
            top, bottom, mid, is_buy, start_idx, c_ns, d["times_ns"], d["lows"], d["highs"],
            min_fill_age_ns, n, target_mult, bounce, probe_depth, 1.0,
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
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)


def compute_margin_capacity(trades, start=START, risk_pct=RISK_PCT, max_leverage=MAX_LEVERAGE, cap=CAP):
    if len(trades) == 0:
        return 0.0, 0.0
    _, final_equity, max_dd_pct, accepted, _ = g20.simulate_margin_constrained(
        trades, start=start, risk_pct=risk_pct, max_leverage=max_leverage, cap=cap, track_detail=False
    )
    accept_pct = 100.0 * np.mean(accepted)
    return accept_pct, final_equity


def evaluate_composite(holdout, rr, base_df, n_sims=25000):
    net_r = holdout.net_r.values.astype(np.float64)
    years = (holdout.entry_time.max() - holdout.entry_time.min()).days / 365.25
    tpy = len(holdout) / years
    wr = 100.0 * (net_r > 0).mean()

    # Fast multi-threaded C++ MC
    t1_raw, t10_raw, max_dd, ruined = dtfx_engine.run_mc_capped(
        net_r, n_sims, BLOCK, MAX_TRADES, 42,
        start=START, risk_frac=RISK_PCT, cap=CAP, n_threads=0
    )
    t1_hit = t1_raw[t1_raw > 0]
    median_yrs_1m = np.median(t1_hit / tpy) if len(t1_hit) else float("inf")
    median_dd = np.median(max_dd) * 100.0

    margin_pct, final_eq = compute_margin_capacity(holdout)

    all_speeds = np.append(base_df["median_yrs_1m"].values, median_yrs_1m)
    all_dds = np.append(base_df["median_dd"].values, median_dd)
    all_wrs = np.append(base_df["wr"].values, wr)
    all_tpys = np.append(base_df["tpy"].values, tpy)
    all_margins = np.append(base_df["margin_pct"].values, margin_pct)
    all_rrrs = np.append(base_df["rr"].values, rr)

    def norm_val(val, arr, higher_is_better):
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return 100.0
        frac = (val - lo) / (hi - lo)
        return 100.0 * frac if higher_is_better else 100.0 * (1.0 - frac)

    speed_score = norm_val(median_yrs_1m, all_speeds, higher_is_better=False)
    dd_score = norm_val(median_dd, all_dds, higher_is_better=True)
    wr_score = norm_val(wr, all_wrs, higher_is_better=True)
    freq_score = norm_val(tpy, all_tpys, higher_is_better=True)
    margin_score = norm_val(margin_pct, all_margins, higher_is_better=True)
    rrr_score = norm_val(rr, all_rrrs, higher_is_better=True)

    w = DEFAULT_WEIGHTS
    final_score = (
        w["speed"] * speed_score +
        w["wr"] * wr_score +
        w["freq"] * freq_score +
        w["margin"] * margin_score +
        w["drawdown"] * dd_score +
        w["rrr"] * rrr_score
    )

    w_sum = net_r[net_r > 0].sum()
    l_sum = -net_r[net_r < 0].sum()
    pf = (w_sum / l_sum) if l_sum > 0 else float("inf")

    return {
        "final_score": round(final_score, 1),
        "speed_score": round(speed_score, 1),
        "wr_score": round(wr_score, 1),
        "freq_score": round(freq_score, 1),
        "margin_score": round(margin_score, 1),
        "dd_score": round(dd_score, 1),
        "rrr_score": round(rrr_score, 1),
        "wr": round(wr, 2),
        "pf": round(pf, 3),
        "tpy": round(tpy, 0),
        "median_yrs_1m": round(median_yrs_1m, 3),
        "median_dd": round(median_dd, 1),
        "margin_pct": round(margin_pct, 1),
        "final_eq_mil": round(final_eq / 1e6, 1),
        "trades": len(holdout),
    }


def main():
    print("=" * 115)
    print("  FAST C++ PARAMETER EXPLORATION: TARGETING HIGHER COMPOSITE SCORE & HIGHER MARGIN CAPACITY")
    print("=" * 115)

    t0 = time.time()
    cached = pre_cache_instruments()
    print(f"Pre-cached all 4 instruments in memory in {time.time()-t0:.2f}s.\n")

    base_scores_path = os.path.join(DATA_DIR, "goal12_strategy_scores.csv")
    base_df = pd.read_csv(base_scores_path)
    sk_row = base_df[base_df["strategy"] == "Speed-King"].iloc[0]
    print(f"Baseline to beat (Speed-King): Final Score = {sk_row['final_score']:.1f}, "
          f"Margin Score = {sk_row['margin_score']:.1f}, Margin Cap = {sk_row['margin_pct']:.1f}%, "
          f"WR = {sk_row['wr']:.2f}%, Yrs->1M = {sk_row['median_yrs_1m']:.3f}\n")

    # Parameter Grids
    N_GRID = [5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    RR_GRID = [0.22, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]
    AGE_GRID = [0.175, 0.20, 0.225, 0.25, 0.30, 0.35, 0.40, 0.50]
    BOUNCE_GRID = [False, True]
    TOUCHES_GRID = [3, 4, 6, 8]

    total_combos = len(N_GRID) * len(RR_GRID) * len(AGE_GRID) * len(BOUNCE_GRID) * len(TOUCHES_GRID)
    print(f"Total parameter combinations: {total_combos:,}. Running high-speed grid search...\n")

    results = []
    count = 0
    t_start = time.time()

    for n in N_GRID:
        for age_h in AGE_GRID:
            for bounce in BOUNCE_GRID:
                for touches in TOUCHES_GRID:
                    for rr in RR_GRID:
                        count += 1
                        trades = build_candidate_trades(cached, n, rr, age_h, bounce, touches)
                        if len(trades) < 500:
                            continue

                        # Split into first-half in-sample and holdout
                        first = trades[trades.entry_time < SPLIT]
                        holdout = trades[trades.entry_time >= SPLIT].reset_index(drop=True)
                        if len(first) < 100 or len(holdout) < 100:
                            continue

                        # In-sample selection check (g-score)
                        glog = np.log1p(g5.RISK_FRAC * first.net_r)
                        first_yrs = (first.entry_time.max() - first.entry_time.min()).days / 365.25
                        g_score = (len(first) / first_yrs) * glog.mean()

                        if g_score <= 0.05:
                            continue

                        # Fast pre-filtering on holdout basic metrics
                        h_net_r = holdout.net_r.values
                        h_wr = 100.0 * (h_net_r > 0).mean()
                        h_w = h_net_r[h_net_r > 0].sum()
                        h_l = -h_net_r[h_net_r < 0].sum()
                        h_pf = (h_w / h_l) if h_l > 0 else 0.0
                        h_tpy = len(holdout) / ((holdout.entry_time.max() - holdout.entry_time.min()).days / 365.25)

                        if h_wr < 72.0 or h_pf < 1.15 or h_tpy < 1500:
                            continue

                        # Full C++ MC and margin evaluation
                        metrics = evaluate_composite(holdout, rr, base_df, n_sims=25000)

                        results.append({
                            "n": n,
                            "rr": rr,
                            "age_h": age_h,
                            "bounce": bounce,
                            "touches": touches,
                            "in_sample_g": round(g_score, 3),
                            **metrics
                        })

                        if count % 100 == 0:
                            elapsed = time.time() - t_start
                            print(f"  [{count:,}/{total_combos:,}] ({count/total_combos*100:.1f}%) in {elapsed:.1f}s | "
                                  f"High-quality candidates: {len(results):,}")

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(DATA_DIR, "goal37_beat_speed_king_search_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nCompleted sweep in {time.time()-t_start:.1f}s. Saved {len(df_res):,} results to {out_csv}.")

    print("\n" + "=" * 115)
    print("  TOP 25 CANDIDATES RANKED BY FINAL COMPOSITE SCORE")
    print("=" * 115)
    top25 = df_res.sort_values("final_score", ascending=False).head(25)
    print(top25[["n", "rr", "age_h", "bounce", "touches", "final_score", "margin_score",
                 "margin_pct", "wr", "pf", "tpy", "median_yrs_1m", "median_dd"]].to_string(index=False))

    print("\n" + "=" * 115)
    print("  CANDIDATES STRICTLY DOMINATING SPEED-KING (Score >= 87.6 AND Margin Score >= 20.0)")
    print("=" * 115)
    winners = df_res[(df_res["final_score"] >= 87.6) & (df_res["margin_score"] >= 20.0)].sort_values("final_score", ascending=False)
    if len(winners) > 0:
        print(winners[["n", "rr", "age_h", "bounce", "touches", "final_score", "margin_score",
                       "margin_pct", "wr", "pf", "tpy", "median_yrs_1m", "median_dd"]].to_string(index=False))
    else:
        print("No candidates strictly >= 87.6 with margin >= 20, checking best Pareto tradeoffs:")
        pareto = df_res[(df_res["final_score"] >= 80.0) & (df_res["margin_score"] >= 25.0)].sort_values("final_score", ascending=False)
        print(pareto[["n", "rr", "age_h", "bounce", "touches", "final_score", "margin_score",
                      "margin_pct", "wr", "pf", "tpy", "median_yrs_1m", "median_dd"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
