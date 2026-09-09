#!/usr/bin/env python3
"""Goal 40: High-Resolution Tight-N Sweep for High RRR (> 0.75)."""
import os
import sys
import time
import numpy as np
import pandas as pd

os.environ["MPLCONFIGDIR"] = "/tmp"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import dtfx_engine
import goal5_rr_sweep as g5
import goal20_speed_king_margin_report as g20

SPLIT = pd.Timestamp("2018-03-09")
base_df = pd.read_csv(os.path.join(DATA_DIR, "goal38_strategy_rankings.csv"))

cached = {}
for ret in [1, 2]:
    cached[ret] = {}
    for inst in g5.TRADEABLE:
        z = g5.rs._prep_zones(inst, ret)
        times, lows, highs, df = g5.rs.load_instrument(inst)
        cached[ret][inst] = {
            "times": times, "times_ns": times.astype("datetime64[ns]").astype("int64"),
            "lows": lows, "highs": highs, "top": np.ascontiguousarray(z["top"], dtype=np.float64),
            "bottom": np.ascontiguousarray(z["bottom"], dtype=np.float64),
            "mid": np.ascontiguousarray(z["mid"], dtype=np.float64),
            "is_buy": np.ascontiguousarray(z["is_buy"], dtype=np.bool_),
            "start_idx": np.ascontiguousarray(z["start_idx"], dtype=np.int64),
            "c_ns": np.ascontiguousarray(z["created_at"].astype("datetime64[ns]").astype("int64"), dtype=np.int64),
            "spread_price": g5.pg.SPREAD_PIPS_TYPICAL * g5.pg.PIP_SIZE[inst]
        }

DEFAULT_WEIGHTS = dict(speed=0.35, wr=0.30, freq=0.12, margin=0.08, drawdown=0.10, rrr=0.05)


def rank_cand(c_dict):
    all_raw = {}
    for _, row in base_df.iterrows():
        all_raw[row["strategy"]] = {
            "trades": int(row["trades"]), "tpy": float(row["tpy"]), "wr": float(row["wr"]),
            "rr": float(row["rr"]), "median_yrs_1m": float(row["median_yrs_1m"]),
            "median_dd": float(row["median_dd"]), "pct_ruined": float(row["pct_ruined"]),
            "margin_pct": float(row["margin_pct"]),
        }
    all_raw["Candidate"] = c_dict

    def norm_map(metric_name, higher_is_better):
        vals = {k: v[metric_name] for k, v in all_raw.items()}
        v_arr = np.array(list(vals.values()))
        lo, hi = v_arr.min(), v_arr.max()
        out = {}
        for k, val in vals.items():
            frac = (val - lo) / (hi - lo) if hi > lo else 1.0
            out[k] = 100.0 * frac if higher_is_better else 100.0 * (1.0 - frac)
        return out

    speed_n = norm_map("median_yrs_1m", False)
    dd_n = norm_map("median_dd", True)
    wr_n = norm_map("wr", True)
    freq_n = norm_map("tpy", True)
    margin_n = norm_map("margin_pct", True)
    rrr_n = norm_map("rr", True)

    w = DEFAULT_WEIGHTS
    scores = {}
    for k in all_raw:
        scores[k] = (w["speed"]*speed_n[k] + w["wr"]*wr_n[k] + w["freq"]*freq_n[k] +
                     w["margin"]*margin_n[k] + w["drawdown"]*dd_n[k] + w["rrr"]*rrr_n[k])
    return scores["Candidate"], scores["Apex-King"], speed_n["Candidate"], wr_n["Candidate"], freq_n["Candidate"], margin_n["Candidate"], dd_n["Candidate"], rrr_n["Candidate"]


def main():
    print("Sweeping tight N from 1.0 to 3.0 with RR in [0.76, 0.80, 0.85, 0.90, 1.00]...", flush=True)
    results = []
    t_start = time.time()
    count = 0

    for ret in [1, 2]:
        for n in [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]:
            for rr in [0.76, 0.80, 0.85, 0.90, 1.00]:
                for age_h in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
                    for touches in [2, 4, 6, 8]:
                        for probe in [0.25, 0.50, 0.75]:
                            for bounce in [False, True]:
                                count += 1
                                frames = []
                                target_mult = n * rr
                                for inst, d in cached[ret].items():
                                    half_width = (d["top"] - d["bottom"]) / 2.0
                                    risk_pct = (n * half_width) / d["mid"]
                                    ok = risk_pct >= g5.pg.MIN_RISK_PCT
                                    touch_num, entry_idx, exit_idx, is_win, entry_mid, stop_risk = dtfx_engine.simulate_generic_ex(
                                        d["top"][ok], d["bottom"][ok], d["mid"][ok], d["is_buy"][ok], d["start_idx"][ok], d["c_ns"][ok],
                                        d["times_ns"], d["lows"], d["highs"], int(age_h * 3600 * 1e9), n, target_mult, bounce, probe, 1.0,
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
                                        "is_win": is_win
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
                                pf = w_sum / l_sum if l_sum > 0 else 0.0

                                if pf < 1.10 or wr < 60.0:
                                    continue

                                t1_raw, t10_raw, max_dd, ruined = dtfx_engine.run_mc_capped(
                                    net_r, 10000, 100, 100000, 42,
                                    start=200.0, risk_frac=0.05, cap=2000000.0, n_threads=0
                                )
                                t1_hit = t1_raw[t1_raw > 0]
                                median_yrs_1m = np.median(t1_hit / tpy) if len(t1_hit) else float("inf")
                                median_dd = np.median(max_dd) * 100.0

                                if median_yrs_1m > 1.5:
                                    continue

                                _, final_eq, _, accepted, _ = g20.simulate_margin_constrained(
                                    holdout, start=200.0, risk_pct=0.05, max_leverage=50.0, cap=2000000.0, track_detail=False
                                )
                                margin_pct = 100.0 * np.mean(accepted)

                                c_dict = {
                                    "trades": len(holdout), "tpy": tpy, "wr": wr, "rr": rr,
                                    "median_yrs_1m": median_yrs_1m, "median_dd": median_dd,
                                    "pct_ruined": 0.0, "margin_pct": margin_pct
                                }
                                c_sc, a_sc, sp_s, wr_s, fr_s, mg_s, dd_s, rr_s = rank_cand(c_dict)

                                results.append({
                                    "ret": ret, "n": n, "rr": rr, "age_h": age_h, "bounce": bounce,
                                    "touches": touches, "probe": probe, "cand_score": round(c_sc, 1),
                                    "apex_score": round(a_sc, 1), "margin_pct": round(margin_pct, 1),
                                    "wr": round(wr, 2), "pf": round(pf, 3), "tpy": round(tpy, 0),
                                    "yrs_1m": round(median_yrs_1m, 3), "dd": round(median_dd, 1),
                                    "sp_s": round(sp_s, 1), "wr_s": round(wr_s, 1), "fr_s": round(fr_s, 1),
                                    "mg_s": round(mg_s, 1), "dd_s": round(dd_s, 1), "rr_s": round(rr_s, 1)
                                })

                                if c_sc >= 75.0:
                                    print(f"  🔥 Score={c_sc:.1f} (Apex={a_sc:.1f}) | RR={rr:.2f} | WR={wr:.2f}% | "
                                          f"PF={pf:.3f} | Margin={margin_pct:.1f}% | Yrs1M={median_yrs_1m:.3f} | "
                                          f"Params: (ret={ret}, N={n}, age={age_h}h, touches={touches}, probe={probe})",
                                          flush=True)

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(DATA_DIR, "goal40_tight_n_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nDone sweep in {time.time()-t_start:.1f}s. Saved {len(df_res)} candidates to {out_csv}.", flush=True)

    print("\n" + "=" * 125)
    print("  TOP 30 CANDIDATES RANKED BY CANDIDATE COMPOSITE SCORE")
    print("=" * 125)
    print(df_res.sort_values("cand_score", ascending=False)[
        ["ret", "n", "rr", "age_h", "bounce", "touches", "probe", "cand_score", "apex_score",
         "wr", "pf", "tpy", "yrs_1m", "margin_pct", "dd", "sp_s", "wr_s", "fr_s", "mg_s", "rr_s"]
    ].head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
