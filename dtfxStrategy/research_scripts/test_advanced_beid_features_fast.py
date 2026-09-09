import os
import sys
import numpy as np
import pandas as pd
import time
from beid_real_engine import resample_ohlc_fast, compute_htf_trend, get_pip_size
from test_advanced_beid_features import backtest_advanced_beid

def run_fast_feature_tests():
    tickers = {
        "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
        "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
        "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
        "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv"
    }
    
    print("Loading 4 datasets into memory once...")
    t0_load = time.time()
    raw_1m = {}
    for name, path in tickers.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Use last 2,000,000 bars (~8 years of multi-asset data) for ultra-fast, statistically massive evaluation
            raw_1m[name] = df.iloc[-2000000:].copy().reset_index(drop=True)
    print(f"Loaded in {time.time()-t0_load:.1f}s.")
    
    # Pre-resample all timeframes
    timeframes = [3, 5, 15, 60, 240]
    cached_resampled = {}
    for tf in timeframes:
        cached_resampled[tf] = {}
        for name, df_1m in raw_1m.items():
            cached_resampled[tf][name] = resample_ohlc_fast(df_1m, tf)
            
    # Precompute HTF trends
    htf_trends = {}
    for htf in [60, 240]:
        htf_trends[htf] = {}
        for name in raw_1m.keys():
            htf_trends[htf][name] = compute_htf_trend(cached_resampled[htf][name])
            
    test_grid = [
        # (tf, htf, depth, rr, be_trail, min_pip, touches)
        (3, 60, 0.50, 1.0, False, 15.0, 1),
        (3, 60, 0.50, 1.0, True, 15.0, 1),
        (3, 60, 0.50, 1.0, True, 20.0, 2),
        (3, 60, 0.618, 1.5, False, 20.0, 2),
        (3, 60, 0.618, 1.5, True, 20.0, 2),
        (5, 60, 0.50, 1.0, True, 20.0, 2),
        (5, 60, 0.618, 1.5, True, 20.0, 2),
        (5, 60, 0.618, 2.0, True, 20.0, 2),
        (15, 240, 0.50, 1.5, True, 25.0, 2),
        (15, 240, 0.618, 2.0, True, 25.0, 2)
    ]
    
    print("\n" + "=" * 115)
    print("RUNNING ADVANCED STRUCTURAL FEATURE TESTS (CONFLUENCE, BE TRAIL, DEEP DISCOUNT, TARGET RR)")
    print("=" * 115)
    
    results = []
    
    for tf, htf, depth, rr, be, min_p, touches in test_grid:
        t0 = time.time()
        trades = []
        for name in raw_1m.keys():
            df_ltf = cached_resampled[tf][name]
            trend_df = htf_trends[htf][name]
            t_df = backtest_advanced_beid(df_ltf, name, df_htf_trend=trend_df,
                                          entry_depth=depth, rr_target=rr,
                                          min_zone_pips=min_p, spread_pips=1.6,
                                          use_be_trail=be, max_touches=touches, session_only=True)
            trades.append(t_df)
            
        full = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
        if len(full) == 0: continue
        full['date'] = pd.to_datetime(full['exit_time'])
        
        pre = full[full['date'] < '2021-01-01']
        holdout = full[full['date'] >= '2026-04-01']
        
        tot_w = sum(full['win'])
        tot_wr = (tot_w / len(full)) * 100.0
        tot_pips = full['net_pips'].sum()
        tot_r = full['net_r'].sum()
        
        w_p = full[full['net_pips'] > 0]['net_pips'].sum()
        l_p = abs(full[full['net_pips'] <= 0]['net_pips'].sum())
        pf = w_p / max(1e-6, l_p)
        
        pre_pf = pre[pre['net_pips'] > 0]['net_pips'].sum() / max(1e-6, abs(pre[pre['net_pips'] <= 0]['net_pips'].sum())) if len(pre) > 0 else 0.0
        h_pf = holdout[holdout['net_pips'] > 0]['net_pips'].sum() / max(1e-6, abs(holdout[holdout['net_pips'] <= 0]['net_pips'].sum())) if len(holdout) > 0 else 0.0
        h_wr = (sum(holdout['win']) / len(holdout)) * 100.0 if len(holdout) > 0 else 0.0
        
        span_yrs = max(0.1, (full['date'].max() - full['date'].min()).total_seconds() / (365.25 * 86400))
        tpy = len(full) / span_yrs
        
        desc = f"TF:{tf:2d}M | HTF:{htf:3d}M | Entry:{depth*100:4.1f}% | RR:1:{rr:.1f} | BE:{str(be):<5} | MinP:{min_p:4.1f}p"
        print(f"{desc} | Trades: {len(full):<6,} ({tpy:>4.0f}/yr) | WR: {tot_wr:>4.1f}% | PF: {pf:>4.2f} | Pre PF: {pre_pf:>4.2f} | Holdout PF: {h_pf:>4.2f} | Net R: {tot_r:+,.1f}R ({time.time()-t0:.1f}s)")
        
        results.append({
            'tf': tf, 'htf': htf, 'entry_depth': depth, 'rr': rr, 'be_trail': be, 'min_p': min_p,
            'trades': len(full), 'tpy': round(tpy, 0), 'win_rate': round(tot_wr, 1),
            'pf': round(pf, 2), 'pre_pf': round(pre_pf, 2), 'holdout_pf': round(h_pf, 2),
            'net_pips': round(tot_pips, 1), 'net_r': round(tot_r, 1)
        })
        
    df_res = pd.DataFrame(results).sort_values('pf', ascending=False).reset_index(drop=True)
    df_res.to_csv("dtfxStrategy/research_results/beid_advanced_features_results.csv", index=False)
    print("\nSaved results to dtfxStrategy/research_results/beid_advanced_features_results.csv")

if __name__ == "__main__":
    run_fast_feature_tests()
