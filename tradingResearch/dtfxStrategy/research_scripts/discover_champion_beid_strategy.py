import os
import sys
import numpy as np
import pandas as pd
import time
from beid_real_engine import BEIDRealEngine, resample_ohlc_fast, compute_htf_trend, get_pip_size

def run_champion_search():
    tickers = {
        "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
        "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
        "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
        "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv"
    }
    
    print("=" * 95)
    print("SYSTEMATIC STRUCTURAL EXPLORATION: ENGINEERING THE HIGHEST-ALPHA BEID STRATEGY")
    print("Full Multi-Asset History (2000-2026) | True Bid/Ask Execution Model")
    print("=" * 95)
    
    raw_dfs_1m = {}
    for name, path in tickers.items():
        if os.path.exists(path):
            print(f"Loading {name}...")
            df = pd.read_csv(path)
            # Take last 1,500,000 bars (~6 years of continuous 1M data) for super fast, ultra-thorough sweep
            raw_dfs_1m[name] = df.iloc[-1500000:].copy().reset_index(drop=True)
            
    # Parameter Sweep Grid
    configs = []
    
    # 1. Baseline Test (Raw 50% entry, 1:1 RR, 3M, 15p zone)
    configs.append({
        'name': 'Baseline (3M, 50% Entry, 1:1 RR, 1-Touch, 15p)',
        'tf': 3, 'htf': None, 'session': False, 'displace': False, 'entry_pct': 0.50, 'rr': 1.0, 'touches': 1, 'min_pip': 15.0
    })
    
    # 2. HTF 15M Trend Filter
    configs.append({
        'name': 'HTF 15M Trend Alignment (3M, 15p)',
        'tf': 3, 'htf': 15, 'session': False, 'displace': False, 'entry_pct': 0.50, 'rr': 1.0, 'touches': 1, 'min_pip': 15.0
    })
    
    # 3. HTF 1H Trend Filter
    configs.append({
        'name': 'HTF 1H Macro Trend Alignment (3M, 15p)',
        'tf': 3, 'htf': 60, 'session': False, 'displace': False, 'entry_pct': 0.50, 'rr': 1.0, 'touches': 1, 'min_pip': 15.0
    })
    
    # 4. Session Liquidity Filter (London/NY Overlap 07:00-17:00 UTC)
    configs.append({
        'name': 'Session Filter: London/NY 07-17 UTC (3M, 15p)',
        'tf': 3, 'htf': None, 'session': True, 'displace': False, 'entry_pct': 0.50, 'rr': 1.0, 'touches': 1, 'min_pip': 15.0
    })
    
    # 5. Displacement Impulse Filter (Breakout Body Ratio >= 60%)
    configs.append({
        'name': 'Displacement Impulse >= 60% Body (3M, 15p)',
        'tf': 3, 'htf': None, 'session': False, 'displace': True, 'entry_pct': 0.50, 'rr': 1.0, 'touches': 1, 'min_pip': 15.0
    })
    
    # 6. Deep Discount Entry (61.8% Fibonacci Pullback)
    configs.append({
        'name': 'Deep Discount 61.8% Entry (3M, 15p, 1:1 RR)',
        'tf': 3, 'htf': None, 'session': False, 'displace': False, 'entry_pct': 0.618, 'rr': 1.0, 'touches': 1, 'min_pip': 15.0
    })
    
    # 7. Asymmetrical Risk-to-Reward (1:1.5 RR)
    configs.append({
        'name': 'Asymmetric 1:1.5 RR (3M, 15p, 50% Entry)',
        'tf': 3, 'htf': None, 'session': False, 'displace': False, 'entry_pct': 0.50, 'rr': 1.5, 'touches': 1, 'min_pip': 15.0
    })
    
    # 8. Asymmetrical Risk-to-Reward (1:2.0 RR)
    configs.append({
        'name': 'Asymmetric 1:2.0 RR (3M, 15p, 50% Entry)',
        'tf': 3, 'htf': None, 'session': False, 'displace': False, 'entry_pct': 0.50, 'rr': 2.0, 'touches': 1, 'min_pip': 15.0
    })
    
    # 9. Confluence Model: HTF 1H + Session (07-17 UTC) + 15p Zone
    configs.append({
        'name': 'Confluence: HTF 1H + London/NY Session (3M, 15p)',
        'tf': 3, 'htf': 60, 'session': True, 'displace': False, 'entry_pct': 0.50, 'rr': 1.0, 'touches': 1, 'min_pip': 15.0
    })
    
    # 10. Confluence Model: HTF 1H + Session + Displacement >= 55%
    configs.append({
        'name': 'Confluence: HTF 1H + Session + Displacement (3M, 15p)',
        'tf': 3, 'htf': 60, 'session': True, 'displace': True, 'entry_pct': 0.50, 'rr': 1.0, 'touches': 1, 'min_pip': 15.0
    })
    
    # 11. Confluence Model + Multi-Touch (2 Touches)
    configs.append({
        'name': 'Confluence (HTF 1H + Session) + 2 Touches (3M, 15p)',
        'tf': 3, 'htf': 60, 'session': True, 'displace': False, 'entry_pct': 0.50, 'rr': 1.0, 'touches': 2, 'min_pip': 15.0
    })
    
    # 12. Confluence Model + Multi-Touch (3 Touches)
    configs.append({
        'name': 'Confluence (HTF 1H + Session) + 3 Touches (3M, 15p)',
        'tf': 3, 'htf': 60, 'session': True, 'displace': False, 'entry_pct': 0.50, 'rr': 1.0, 'touches': 3, 'min_pip': 15.0
    })
    
    # 13. Confluence Model on 5M Timeframe
    configs.append({
        'name': 'Confluence (HTF 1H + Session) on 5M Bars (15p, 1:1 RR)',
        'tf': 5, 'htf': 60, 'session': True, 'displace': False, 'entry_pct': 0.50, 'rr': 1.0, 'touches': 2, 'min_pip': 15.0
    })
    
    # 14. Confluence Model on 15M Timeframe (with 4H HTF trend)
    configs.append({
        'name': 'Confluence (HTF 4H + Session) on 15M Bars (20p, 1:1.5 RR)',
        'tf': 15, 'htf': 240, 'session': True, 'displace': False, 'entry_pct': 0.50, 'rr': 1.5, 'touches': 2, 'min_pip': 20.0
    })
    
    # 15. Ultra-Selective Confluence: HTF 1H + Session + 20p Zone + 1:1.5 RR
    configs.append({
        'name': 'Ultra-Selective: HTF 1H + Session + 20p + 1:1.5 RR (3M)',
        'tf': 3, 'htf': 60, 'session': True, 'displace': False, 'entry_pct': 0.50, 'rr': 1.5, 'touches': 2, 'min_pip': 20.0
    })
    
    results = []
    
    for c_idx, cfg in enumerate(configs):
        t0 = time.time()
        print(f"\n[{c_idx+1}/{len(configs)}] Evaluating: {cfg['name']}...", flush=True)
        
        engine = BEIDRealEngine(
            entry_level_pct=cfg['entry_pct'],
            reward_risk_ratio=cfg['rr'],
            max_touches=cfg['touches'],
            min_zone_pips=cfg['min_pip'],
            spread_pips=1.6,
            htf_filter=(cfg['htf'] is not None),
            session_filter=cfg['session'],
            allowed_hours=(7, 17),
            displacement_filter=cfg['displace'],
            min_body_ratio=0.55
        )
        
        cfg_trades = []
        for name, df_1m in raw_dfs_1m.items():
            df_ltf = resample_ohlc_fast(df_1m, cfg['tf'])
            df_htf_trend = None
            if cfg['htf'] is not None:
                df_htf = resample_ohlc_fast(df_1m, cfg['htf'])
                df_htf_trend = compute_htf_trend(df_htf)
                
            t_df = engine.backtest(df_ltf, name, df_htf_trend=df_htf_trend)
            cfg_trades.append(t_df)
            
        full_trades = pd.concat(cfg_trades, ignore_index=True) if cfg_trades else pd.DataFrame()
        
        if len(full_trades) == 0:
            print(f"  --> 0 trades generated.")
            continue
            
        total_t = len(full_trades)
        wins = sum(full_trades['win'])
        wr = (wins / total_t) * 100.0
        tot_pips = full_trades['net_pips'].sum()
        tot_r = full_trades['net_r'].sum()
        
        win_pips = full_trades[full_trades['net_pips'] > 0]['net_pips'].sum()
        loss_pips = abs(full_trades[full_trades['net_pips'] <= 0]['net_pips'].sum())
        pf = win_pips / max(1e-6, loss_pips)
        avg_pip = tot_pips / total_t
        avg_r = tot_r / total_t
        
        first_d = pd.to_datetime(full_trades['exit_time'].iloc[0])
        last_d = pd.to_datetime(full_trades['exit_time'].iloc[-1])
        span_yrs = max(0.1, (last_d - first_d).total_seconds() / (365.25 * 86400))
        tpy = total_t / span_yrs
        
        # Split regime: Holdout check (Post 2026-04)
        full_trades['date_dt'] = pd.to_datetime(full_trades['exit_time'])
        holdout_trades = full_trades[full_trades['date_dt'] >= '2026-04-01']
        h_wr = (sum(holdout_trades['win']) / len(holdout_trades)) * 100.0 if len(holdout_trades) > 0 else 0.0
        h_win_p = holdout_trades[holdout_trades['net_pips'] > 0]['net_pips'].sum()
        h_loss_p = abs(holdout_trades[holdout_trades['net_pips'] <= 0]['net_pips'].sum())
        h_pf = h_win_p / max(1e-6, h_loss_p) if len(holdout_trades) > 0 else 0.0
        
        res = {
            'rank': 0,
            'name': cfg['name'],
            'trades': total_t,
            'trades_per_year': round(tpy, 0),
            'win_rate': round(wr, 1),
            'profit_factor': round(pf, 2),
            'net_pips': round(tot_pips, 1),
            'net_r': round(tot_r, 1),
            'avg_pip_trade': round(avg_pip, 2),
            'avg_r_trade': round(avg_r, 3),
            'holdout_trades': len(holdout_trades),
            'holdout_wr': round(h_wr, 1),
            'holdout_pf': round(h_pf, 2)
        }
        results.append(res)
        
        print(f"  --> {total_t:,} Trades ({tpy:,.0f}/yr) | WR: {wr:.1f}% | PF: {pf:.2f} | Net: {tot_pips:+,.1f} pips | Net R: {tot_r:+,.1f}R | Holdout PF: {h_pf:.2f} ({time.time()-t0:.2f}s)")
        
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('profit_factor', ascending=False).reset_index(drop=True)
    df_res['rank'] = df_res.index + 1
    
    out_csv = "dtfxStrategy/research_results/beid_champion_search_results.csv"
    df_res.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 115)
    print("RANKED STRUCTURAL CONFIGURATIONS (SORTED BY PROFIT FACTOR):")
    print("=" * 115)
    print(f"{'Rank':<5} | {'Model Name':<55} | {'Trades':<8} | {'WR%':<6} | {'PF':<6} | {'Net Pips':<12} | {'Net R':<10} | {'Holdout PF':<10}")
    print("-" * 125)
    for _, r in df_res.iterrows():
        print(f"#{r['rank']:<4} | {r['name']:<55} | {r['trades']:<8,} | {r['win_rate']:<6.1f} | {r['profit_factor']:<6.2f} | {r['net_pips']:<12,.1f} | {r['net_r']:<10.1f} | {r['holdout_pf']:<10.2f}")

if __name__ == "__main__":
    run_champion_search()
