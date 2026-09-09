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

YTD_START = pd.Timestamp("2026-01-01")
N_SIMS = 1000
TARGET = 1_000_000.0
MAX_TRADES = 50_000
SEED = 42

ZONE_FLOORS = [5.0, 7.0, 10.0, 15.0, 20.0]
RISK_LEVELS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

def main():
    t0 = time.time()
    print("=" * 85)
    print("2026 DATA MONTE CARLO RACE TO $1,000,000")
    print(f"Starting Capital: $200 (2 x $100 Sub-Accounts) | Leverage: 50:1 | Target: $1,000,000")
    print(f"Simulations per configuration: {N_SIMS:,} paths | Block-Bootstrap Resampling")
    print("=" * 85)

    print("\nLoading historical datasets and building zones...")
    cache = g104.load_and_build_zones()

    race_results = []

    for min_pips in ZONE_FLOORS:
        print(f"\n" + "-" * 85)
        print(f"RACE BRACKET: MIN ZONE = {min_pips:.1f} PIPS")
        print("-" * 85)
        
        # Build candidate trades and filter to 2026 period
        trades_full = g104.build_candidate_trades(cache, min_pips, symmetric_exit_spread=True)
        trades_2026 = trades_full[trades_full["entry_time"] >= YTD_START].reset_index(drop=True)
        n_candidates = len(trades_2026)
        wr_raw = 100.0 * (trades_2026["net_r"] > 0).sum() / n_candidates if n_candidates > 0 else 0
        gw = trades_2026[trades_2026["net_r"] > 0]["net_r"].sum()
        gl = -trades_2026[trades_2026["net_r"] <= 0]["net_r"].sum()
        pf_raw = gw / gl if gl > 0 else 0
        
        span_days = (trades_2026["exit_time"].max() - YTD_START) / np.timedelta64(1, "D")
        tpd = n_candidates / span_days if span_days > 0 else 0
        print(f"  2026 Trade Pool: {n_candidates:,} candidate trades ({tpd:.1f}/day) | Raw WR: {wr_raw:.2f}% | PF: {pf_raw:.3f}")

        for risk_pct in RISK_LEVELS:
            t_r = time.time()
            g79.RISK_PCT = risk_pct
            
            # Run 1,000-path Monte Carlo simulation on the 2026 trade pool
            days, trades_ct, max_dds, ruin_rate = g79.run_monte_carlo_two_account(
                trades_2026, N_SIMS, TARGET, max_trades=MAX_TRADES, seed=SEED
            )
            
            pct_reached = 100.0 * len(days) / N_SIMS
            med_days = float(np.median(days)) if len(days) else float("nan")
            p5_days = float(np.percentile(days, 5)) if len(days) else float("nan")
            p95_days = float(np.percentile(days, 95)) if len(days) else float("nan")
            
            med_months = med_days / 30.4375 if len(days) else float("nan")
            p5_months = p5_days / 30.4375 if len(days) else float("nan")
            p95_months = p95_days / 30.4375 if len(days) else float("nan")
            
            med_dd = float(np.median(max_dds))
            p95_dd = float(np.percentile(max_dds, 5))
            
            res = {
                "min_zone_pips": min_pips,
                "risk_pct": risk_pct,
                "pct_reached_1m": pct_reached,
                "med_days_to_1m": med_days,
                "med_months_to_1m": med_months,
                "p5_months_to_1m": p5_months,
                "p95_months_to_1m": p95_months,
                "med_max_dd": med_dd,
                "p95_worst_dd": p95_dd,
                "ruin_rate": 100.0 * ruin_rate,
                "raw_wr": wr_raw,
                "raw_pf": pf_raw,
            }
            race_results.append(res)
            
            med_m_str = f"{med_months:5.1f} mo ({med_days:4.0f}d)" if not np.isnan(med_months) else "     N/A"
            p5_m_str = f"{p5_months:4.1f}mo" if not np.isnan(p5_months) else " N/A"
            p95_m_str = f"{p95_months:4.1f}mo" if not np.isnan(p95_months) else " N/A"
            
            print(f"  Risk={risk_pct:>4.1%} | Reached $1M: {pct_reached:>5.1f}% | Median Time: {med_m_str} [{p5_m_str} - {p95_m_str}] | Median MaxDD: {med_dd:>6.1f}% | Worst DD: {p95_dd:>6.1f}% | Ruin: {100*ruin_rate:>4.2f}% ({time.time()-t_r:.1f}s)", flush=True)

    df_race = pd.DataFrame(race_results)
    out_csv = os.path.join(DATA_DIR, "mc_race_2026_to_1m.csv")
    df_race.to_csv(out_csv, index=False)
    print(f"\nSaved full race results to: {out_csv}")
    
    # Leaderboard
    valid_zero_ruin = df_race[(df_race["ruin_rate"] == 0.0) & (df_race["pct_reached_1m"] == 100.0)]
    if len(valid_zero_ruin) > 0:
        leaderboard = valid_zero_ruin.sort_values("med_days_to_1m").head(12)
        print("\n" + "=" * 85)
        print("MONTE CARLO RACE TO $1,000,000 LEADERBOARD (2026 DATA)")
        print("=" * 85)
        print(leaderboard[["min_zone_pips", "risk_pct", "med_days_to_1m", "med_months_to_1m", "p5_months_to_1m", "p95_months_to_1m", "med_max_dd", "p95_worst_dd", "raw_wr", "raw_pf"]].to_string(index=False))
        
    print(f"\nTotal race completed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
