import os
import sys
import time
import heapq
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal79_two_account_portfolio as g79
import goal104_optimal_risk_two_account as g104

ZONE_HEIGHTS = [5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
RISK_GRID = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
N_SIMS = 1000
MAX_TRADES = 50_000
TARGET = 1_000_000.0
SEED = 42

def main():
    t0 = time.time()
    print("=" * 80)
    print("DTFX BASE STRATEGY: RISK % AND MIN RAW ZONE HEIGHT SWEEP TO $1M")
    print("Symmetric Half-Spread Execution Model: Entry Mid +/- spread/2, TP & SL +/- spread/2")
    print(f"Sub-accounts: 2 x $100 ($200 Total), 50x Leverage, Max Pos Cap: $2M")
    print(f"Monte Carlo: {N_SIMS} paths per grid point, Target: $1,000,000")
    print("=" * 80)
    
    print("\nLoading historical datasets and building zones across all 4 instruments...")
    cache = g104.load_and_build_zones()
    
    results = []
    
    for min_pips in ZONE_HEIGHTS:
        t_z = time.time()
        # candidate trades generated with symmetric_exit_spread=True
        trades = g104.build_candidate_trades(cache, min_pips, symmetric_exit_spread=True)
        n_trades = len(trades)
        wins = (trades["net_r"] > 0).sum()
        wr = 100.0 * wins / n_trades if n_trades > 0 else 0.0
        gross_win = trades[trades["net_r"] > 0]["net_r"].sum()
        gross_loss = -trades[trades["net_r"] <= 0]["net_r"].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else 0.0
        avg_r = trades["net_r"].mean()
        
        span_years = (trades["exit_time"].max() - trades["entry_time"].min()) / np.timedelta64(365, "D")
        tpy = n_trades / span_years if span_years > 0 else 0
        
        print(f"\n" + "-" * 80)
        print(f"MIN ZONE: {min_pips:>4.1f} pips | Pre-concurrency: {n_trades:,} trades ({tpy:,.0f}/yr) | WR: {wr:.2f}% | PF: {pf:.3f} | Avg R: {avg_r:+.4f}")
        print("-" * 80)
        
        for risk_pct in RISK_GRID:
            t_r = time.time()
            # 1. Deterministic full-history simulation
            t_acc, eq_buy, eq_sell, hist_max_dd, trace = g79.simulate_two_account(
                trades, risk_pct=risk_pct, start_per_side=100.0, max_leverage=50.0, cap=2_000_000.0, track_detail=False
            )
            final_eq = eq_buy + eq_sell
            hist_cagr = g104.cagr_from_run(span_years, final_eq, start=200.0)
            accepted_count = int(t_acc["accepted"].sum())
            
            # 2. Monte Carlo simulation
            g79.RISK_PCT = risk_pct
            days, trades_ct, max_dds, ruin_rate = g79.run_monte_carlo_two_account(
                trades, N_SIMS, TARGET, max_trades=MAX_TRADES, seed=SEED
            )
            
            pct_reached = 100.0 * len(days) / N_SIMS
            med_days = float(np.median(days)) if len(days) else float("nan")
            med_years = med_days / 365.25 if len(days) else float("nan")
            p5_years = float(np.percentile(days, 5) / 365.25) if len(days) else float("nan")
            p95_years = float(np.percentile(days, 95) / 365.25) if len(days) else float("nan")
            med_dd = float(np.median(max_dds))
            p95_dd = float(np.percentile(max_dds, 5))
            
            res = {
                "min_zone_pips": min_pips,
                "risk_pct": risk_pct,
                "n_trades_raw": n_trades,
                "accepted_trades": accepted_count,
                "raw_wr": wr,
                "raw_pf": pf,
                "raw_avg_r": avg_r,
                "hist_cagr": hist_cagr,
                "hist_max_dd": hist_max_dd,
                "pct_reached_1m": pct_reached,
                "med_years_to_1m": med_years,
                "p5_years_to_1m": p5_years,
                "p95_years_to_1m": p95_years,
                "med_max_dd": med_dd,
                "p95_worst_dd": p95_dd,
                "ruin_rate": 100.0 * ruin_rate,
            }
            results.append(res)
            
            med_yr_str = f"{med_years:5.2f}y" if not np.isnan(med_years) else "  N/A"
            p5_yr_str = f"{p5_years:5.2f}y" if not np.isnan(p5_years) else "  N/A"
            p95_yr_str = f"{p95_years:5.2f}y" if not np.isnan(p95_years) else "  N/A"
            
            print(f"  Risk={risk_pct:>5.1%} | Reached $1M: {pct_reached:>5.1f}% | Med Time: {med_yr_str} [{p5_yr_str} - {p95_yr_str}] | Med MaxDD: {med_dd:>6.1f}% | Worst DD: {p95_dd:>6.1f}% | Ruin: {100*ruin_rate:>5.2f}% ({time.time()-t_r:.1f}s)", flush=True)
            
    df_results = pd.DataFrame(results)
    out_csv = os.path.join(DATA_DIR, "dtfx_full_risk_zone_sweep_mc.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved all results to: {out_csv}", flush=True)
    
    # Print Top 15 fastest configurations with 0% ruin
    valid_zero_ruin = df_results[(df_results["ruin_rate"] == 0.0) & (df_results["pct_reached_1m"] == 100.0)]
    if len(valid_zero_ruin) > 0:
        top_fastest = valid_zero_ruin.sort_values("med_years_to_1m").head(15)
        print("\n" + "=" * 80)
        print("TOP 15 FASTEST CONFIGURATIONS TO $1M (0.00% RUIN PROBABILITY)")
        print("=" * 80)
        print(top_fastest[["min_zone_pips", "risk_pct", "med_years_to_1m", "p5_years_to_1m", "p95_years_to_1m", "med_max_dd", "p95_worst_dd", "raw_wr", "raw_pf"]].to_string(index=False))
        
    print(f"\nTotal sweep completed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
