import os
import sys
import numpy as np
import pandas as pd
import json
import time

def run_production_monte_carlo():
    # Load the audited full-history trades from our best confluence configuration
    from beid_real_engine import resample_ohlc_fast, compute_htf_trend
    from run_beid_ultimate_sweep import BEIDUltimateEngine
    
    tickers = {
        "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
        "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
        "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
        "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv"
    }
    
    print("Generating full-history trade pool for Production BEID (3M, 1H HTF, London/NY Session, 20p, 2 Touches)...")
    engine = BEIDUltimateEngine(
        entry_pct=0.50,
        rr=1.0,
        continuation_only=True,
        htf_filter=True,
        session_filter=True,
        allowed_hours=(7, 17),
        min_zone_pips=20.0,
        spread_pips=1.6,
        max_touches=2
    )
    
    all_trades = []
    for name, path in tickers.items():
        if os.path.exists(path):
            df_1m = pd.read_csv(path)
            df_ltf = resample_ohlc_fast(df_1m, 3)
            df_htf = resample_ohlc_fast(df_1m, 60)
            htf_trend = compute_htf_trend(df_htf)
            t_df = engine.backtest(df_ltf, name, df_htf_trend=htf_trend)
            all_trades.append(t_df)
            
    full_df = pd.concat(all_trades, ignore_index=True)
    full_df['exit_dt'] = pd.to_datetime(full_df['exit_time'])
    full_df = full_df.sort_values('exit_dt').reset_index(drop=True)
    
    n_trades = len(full_df)
    span_years = (full_df['exit_dt'].max() - full_df['exit_dt'].min()).total_seconds() / (365.25 * 86400)
    tpy = n_trades / span_years
    
    wins = sum(full_df['win'])
    wr = (wins / n_trades) * 100.0
    w_pips = full_df[full_df['net_pips'] > 0]['net_pips'].sum()
    l_pips = abs(full_df[full_df['net_pips'] <= 0]['net_pips'].sum())
    pf = w_pips / max(1e-6, l_pips)
    
    print(f"\nProduction BEID Trade Pool: {n_trades:,} trades ({tpy:.0f}/yr) | Win Rate: {wr:.2f}% | Profit Factor: {pf:.2f} | Net Pips: {full_df['net_pips'].sum():+,.1f}")
    
    # Save audited trades
    full_df.to_csv("dtfxStrategy/research_results/beid_production_audited_trades.csv", index=False)
    
    # Monte Carlo simulation across risk percentages
    # Initial capital = $200, Target = $1,000,000
    risk_grid = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
    n_sims = 2000
    target_equity = 1_000_000.0
    start_equity = 200.0
    max_years = 25.0
    max_trades_sim = min(int(max_years * tpy), 25000)
    
    net_r_arr = full_df['net_r'].values
    
    # Kelly Criterion calculation
    # p = win rate, b = avg win / avg loss
    avg_win_r = full_df[full_df['net_r'] > 0]['net_r'].mean()
    avg_loss_r = abs(full_df[full_df['net_r'] <= 0]['net_r'].mean())
    b_ratio = avg_win_r / max(1e-4, avg_loss_r)
    p_win = wins / n_trades
    kelly_f = (p_win * b_ratio - (1.0 - p_win)) / b_ratio
    half_kelly = kelly_f / 2.0
    quarter_kelly = kelly_f / 4.0
    
    print(f"\n--- Kelly Criterion Analysis ---")
    print(f"  Win Rate (p): {p_win*100:.2f}% | Win/Loss Payoff (b): {b_ratio:.3f}")
    print(f"  Full Kelly: {kelly_f*100:.2f}% | Half Kelly (Safe): {half_kelly*100:.2f}% | Quarter Kelly: {quarter_kelly*100:.2f}%")
    
    mc_results = []
    
    rng = np.random.default_rng(42)
    choices = rng.choice(net_r_arr, size=(n_sims, max_trades_sim), replace=True)
    
    for r_pct in risk_grid:
        mults = 1.0 + r_pct * choices
        mults = np.clip(mults, 1e-4, 10.0)
        
        cum_curves = start_equity * np.cumprod(mults, axis=1)
        
        target_hit_mask = cum_curves >= target_equity
        ruin_mask = cum_curves <= start_equity * 0.10 # 90% drawdown = ruin
        
        days_to_target = []
        ruin_count = 0
        
        running_peaks = np.maximum.accumulate(cum_curves, axis=1)
        dds = (running_peaks - cum_curves) / running_peaks
        max_dds_sim = np.max(dds, axis=1)
        
        for s in range(n_sims):
            hit_idx = np.where(target_hit_mask[s])[0]
            ruin_idx = np.where(ruin_mask[s])[0]
            
            first_hit = hit_idx[0] if len(hit_idx) > 0 else 999999
            first_ruin = ruin_idx[0] if len(ruin_idx) > 0 else 999999
            
            if first_hit < first_ruin and first_hit < 999999:
                days = (first_hit / tpy) * 365.25
                days_to_target.append(days)
            elif first_ruin < first_hit:
                ruin_count += 1
                
        prob_reach = (len(days_to_target) / n_sims) * 100.0
        med_days = float(np.median(days_to_target)) if len(days_to_target) > 0 else np.nan
        med_years = med_days / 365.25 if not np.isnan(med_days) else np.nan
        p5_years = float(np.percentile(days_to_target, 5) / 365.25) if len(days_to_target) > 0 else np.nan
        p95_years = float(np.percentile(days_to_target, 95) / 365.25) if len(days_to_target) > 0 else np.nan
        med_dd = float(np.median(max_dds_sim)) * 100.0
        p95_dd = float(np.percentile(max_dds_sim, 95)) * 100.0
        ruin_rate = (ruin_count / n_sims) * 100.0
        
        mc_results.append({
            'risk_pct': r_pct,
            'prob_reach_1m': round(prob_reach, 1),
            'med_years_to_1m': round(med_years, 2) if not np.isnan(med_years) else np.nan,
            'p5_years': round(p5_years, 2) if not np.isnan(p5_years) else np.nan,
            'p95_years': round(p95_years, 2) if not np.isnan(p95_years) else np.nan,
            'med_max_dd': round(med_dd, 1),
            'p95_worst_dd': round(p95_dd, 1),
            'ruin_rate': round(ruin_rate, 1)
        })
        
    df_mc = pd.DataFrame(mc_results)
    df_mc.to_csv("dtfxStrategy/research_results/beid_production_monte_carlo.csv", index=False)
    
    print("\n" + "=" * 105)
    print("PRODUCTION BEID MONTE CARLO SIMULATION RESULTS (2,000 PATHS, $200 START -> $1M TARGET):")
    print("=" * 105)
    print(f"{'Risk %':<8} | {'Prob $1M':<10} | {'Median Years':<14} | {'Best 5% (Yrs)':<14} | {'Worst 5% (Yrs)':<15} | {'Median Max DD':<14} | {'95% Worst DD':<14} | {'Ruin Rate':<10}")
    print("-" * 115)
    for _, r in df_mc.iterrows():
        med_str = f"{r['med_years_to_1m']:.2f} yrs" if not np.isnan(r['med_years_to_1m']) else "N/A"
        p5_str = f"{r['p5_years']:.2f} yrs" if not np.isnan(r['p5_years']) else "N/A"
        p95_str = f"{r['p95_years']:.2f} yrs" if not np.isnan(r['p95_years']) else "N/A"
        print(f"{r['risk_pct']*100:>5.1f}%  | {r['prob_reach_1m']:>8.1f}%  | {med_str:>12} | {p5_str:>12} | {p95_str:>13} | {r['med_max_dd']:>12.1f}% | {r['p95_worst_dd']:>12.1f}% | {r['ruin_rate']:>8.1f}%")

if __name__ == "__main__":
    run_production_monte_carlo()
