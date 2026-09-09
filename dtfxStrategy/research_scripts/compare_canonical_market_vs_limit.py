import os
import sys
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

import goal79_two_account_portfolio as g79
import goal104_optimal_risk_two_account as g104

def run_comparison():
    t0 = time.time()
    print("=" * 88)
    print("CANONICAL BACKTEST: MARKET ORDER (SYMMETRIC SPREAD) vs. RESTING LIMIT ORDER (PENETRATION)")
    print("All 4 pairs (EURUSD, GBPUSD, EURGBP, USDJPY) across 17-year history (1-minute bars)")
    print("=" * 88)
    
    print("\nLoading datasets and building swing zones...")
    cache = g104.load_and_build_zones()
    
    zone_list = [5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    rows = []
    
    for z in zone_list:
        # Model 1: Market Order / Symmetric Half-Spread Model (Live Bot Convention)
        # Entry = mid +/- spread/2, TP = top/bottom -/+ spread/2, SL = bottom/top -/+ spread/2
        t_mkt = g104.build_candidate_trades(cache, z, symmetric_exit_spread=True)
        n_mkt = len(t_mkt)
        wr_mkt = 100.0 * (t_mkt["net_r"] > 0).sum() / n_mkt if n_mkt > 0 else 0.0
        gw_mkt = t_mkt[t_mkt["net_r"] > 0]["net_r"].sum()
        gl_mkt = -t_mkt[t_mkt["net_r"] <= 0]["net_r"].sum()
        pf_mkt = gw_mkt / gl_mkt if gl_mkt > 0 else 0.0
        avg_r_mkt = t_mkt["net_r"].mean()
        tot_r_mkt = t_mkt["net_r"].sum()
        
        # Model 2: Resting Limit Order with Strict Penetration Model (dtfx.md #8)
        # If penetration >= 1 spread beyond mid -> fills free at exact mid; shallow touch -> repriced as market fill
        t_lmt = g104.build_candidate_trades(cache, z, symmetric_exit_spread=False)
        n_lmt = len(t_lmt)
        wr_lmt = 100.0 * (t_lmt["net_r"] > 0).sum() / n_lmt if n_lmt > 0 else 0.0
        gw_lmt = t_lmt[t_lmt["net_r"] > 0]["net_r"].sum()
        gl_lmt = -t_lmt[t_lmt["net_r"] <= 0]["net_r"].sum()
        pf_lmt = gw_lmt / gl_lmt if gl_lmt > 0 else 0.0
        avg_r_lmt = t_lmt["net_r"].mean()
        tot_r_lmt = t_lmt["net_r"].sum()
        
        # Monte Carlo to $1M at 3% risk (500 paths)
        g79.RISK_PCT = 0.03
        days_mkt, _, dds_mkt, ruin_mkt = g79.run_monte_carlo_two_account(t_mkt, 500, 1_000_000.0, max_trades=30_000, seed=42)
        days_lmt, _, dds_lmt, ruin_lmt = g79.run_monte_carlo_two_account(t_lmt, 500, 1_000_000.0, max_trades=30_000, seed=42)
        
        med_yr_mkt = np.median(days_mkt) / 365.25 if len(days_mkt) else float("nan")
        med_yr_lmt = np.median(days_lmt) / 365.25 if len(days_lmt) else float("nan")
        med_dd_mkt = np.median(dds_mkt) if len(dds_mkt) else float("nan")
        med_dd_lmt = np.median(dds_lmt) if len(dds_lmt) else float("nan")
        
        rows.append({
            "min_zone": z,
            "mkt_trades": n_mkt,
            "lmt_trades": n_lmt,
            "mkt_wr": wr_mkt,
            "lmt_wr": wr_lmt,
            "mkt_pf": pf_mkt,
            "lmt_pf": pf_lmt,
            "mkt_avg_r": avg_r_mkt,
            "lmt_avg_r": avg_r_lmt,
            "mkt_tot_r": tot_r_mkt,
            "lmt_tot_r": tot_r_lmt,
            "mkt_med_yrs_1m": med_yr_mkt,
            "lmt_med_yrs_1m": med_yr_lmt,
            "mkt_med_dd": med_dd_mkt,
            "lmt_med_dd": med_dd_lmt,
        })
        
        print(f"\n--- Zone Floor: {z} pips ---")
        print(f"  Market Order (Symmetric Spread):  {n_mkt:>9,} trades | WR: {wr_mkt:>6.2f}% | PF: {pf_mkt:>5.3f} | Avg R: {avg_r_mkt:>+6.4f} | Med Time to $1M: {med_yr_mkt:>5.2f}y | MaxDD: {med_dd_mkt:>5.1f}%")
        print(f"  Resting Limit (Penetration Fill): {n_lmt:>9,} trades | WR: {wr_lmt:>6.2f}% | PF: {pf_lmt:>5.3f} | Avg R: {avg_r_lmt:>+6.4f} | Med Time to $1M: {med_yr_lmt:>5.2f}y | MaxDD: {med_dd_lmt:>5.1f}%", flush=True)
        
    df_res = pd.DataFrame(rows)
    out_csv = os.path.join(DATA_DIR, "canonical_market_vs_limit_comparison.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved full comparison to: {out_csv}")
    print(f"Completed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    run_comparison()
