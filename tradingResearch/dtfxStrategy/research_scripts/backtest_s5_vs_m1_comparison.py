import os
import sys
import numpy as np
import pandas as pd
import time
import json

def get_pip_size(ticker):
    if "JPY" in ticker:
        return 0.01
    return 0.0001

def backtest_beid_engine(df, ticker_name, spread_pips=1.6, timeframe_label="M1"):
    """
    Core BEID engine running on OHLC dataframe (works identically for S5, M1, etc.)
    - Entry at exact 50% midpoint
    - TP at exact boundary
    - SL at exact opposite boundary
    - Fills and exits on subsequent candles (i > fill_idx)
    """
    pip_size = get_pip_size(ticker_name)
    
    dates = df['time'].values if 'time' in df.columns else df['date'].values
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    
    n = len(opens)
    
    trend = 1
    pHH = opens[0]
    pHH_idx = 0
    cHH = np.nan
    cHH_idx = -1
    pHL = opens[0]
    pHL_idx = 0
    cHL = np.nan
    cHL_idx = -1
    last_cHL = np.nan
    
    pLL = opens[0]
    pLL_idx = 0
    cLL = np.nan
    cLL_idx = -1
    pLH = opens[0]
    pLH_idx = 0
    cLH = np.nan
    cLH_idx = -1
    last_cLH = np.nan
    
    in_pullback = False
    prev_green_open = np.nan
    prev_red_open = np.nan
    
    active_zones = []
    trades = []
    
    # Expiry bars: 2880 bars for M1 (48h), 2880*12 bars for S5 (48h)
    max_unfilled_bars = 2880 if timeframe_label == "M1" else 2880 * 12
    max_active = 30
    
    for i in range(n):
        o = opens[i]
        h = highs[i]
        l = lows[i]
        cl = closes[i]
        
        is_green = cl >= o
        is_red = cl < o
        
        # 1. Update Active Zones / Open Trades
        retained = []
        for z in active_zones:
            side = z['side']
            entry = z['entry']
            tp = z['tp']
            sl = z['sl']
            zone_pips = z['zone_pips']
            half_pips = z['half_pips']
            
            if not z['filled']:
                # Buy fill: price touches exact 50% midpoint
                if side == 1 and l <= entry:
                    z['filled'] = True
                    z['fill_idx'] = i
                    retained.append(z)
                # Sell fill: price touches exact 50% midpoint
                elif side == -1 and h >= entry:
                    z['filled'] = True
                    z['fill_idx'] = i
                    retained.append(z)
                else:
                    if i - z['created_idx'] > max_unfilled_bars:
                        pass
                    else:
                        retained.append(z)
            else:
                # Active trade: must resolve on subsequent candle (i > z['fill_idx'])
                if i > z['fill_idx']:
                    resolved = False
                    raw_result = ""
                    
                    if side == 1: # BUY Trade
                        hit_sl = (l <= sl) or (cl < sl)
                        hit_tp = (h >= tp) or (cl > tp)
                        
                        if hit_sl and hit_tp:
                            raw_result = "WIN" if cl >= entry else "LOSS"
                            resolved = True
                        elif hit_tp:
                            raw_result = "WIN"
                            resolved = True
                        elif hit_sl:
                            raw_result = "LOSS"
                            resolved = True
                    else: # SELL Trade
                        hit_sl = (h >= sl) or (cl > sl)
                        hit_tp = (l <= tp) or (cl < tp)
                        
                        if hit_sl and hit_tp:
                            raw_result = "WIN" if cl <= entry else "LOSS"
                            resolved = True
                        elif hit_tp:
                            raw_result = "WIN"
                            resolved = True
                        elif hit_sl:
                            raw_result = "LOSS"
                            resolved = True
                            
                    if resolved:
                        if raw_result == "WIN":
                            net_pips = half_pips - spread_pips
                            net_r = net_pips / max(1e-4, half_pips)
                        else:
                            net_pips = -(half_pips + spread_pips)
                            net_r = net_pips / max(1e-4, half_pips)
                            
                        trades.append({
                            'ticker': ticker_name,
                            'timeframe': timeframe_label,
                            'side': 'BUY' if side == 1 else 'SELL',
                            'type': z['type'],
                            'created_time': dates[z['created_idx']],
                            'fill_time': dates[z['fill_idx']],
                            'exit_time': dates[i],
                            'entry': entry,
                            'tp': tp,
                            'sl': sl,
                            'zone_pips': zone_pips,
                            'half_pips': half_pips,
                            'raw_result': raw_result,
                            'net_pips': net_pips,
                            'net_r': net_r,
                            'raw_r': 1.0 if raw_result == "WIN" else -1.0,
                            'duration_bars': i - z['fill_idx']
                        })
                    else:
                        retained.append(z)
                        
        active_zones = retained
        
        # 2. State Machine & Zone Creation
        if trend == 1: # UPTREND
            if h > pHH or np.isnan(pHH):
                pHH = h
                pHH_idx = i
                
            if not np.isnan(last_cHL) and cl < last_cHL:
                old_floor = last_cHL
                top_peak = pHH
                
                trend = -1
                cLH = top_peak
                cLH_idx = pHH_idx
                last_cLH = np.nan
                pLL = l
                pLL_idx = i
                cLL = np.nan
                in_pullback = False
                
                top = top_peak
                bottom = old_floor
                mid = (top + bottom) / 2.0
                zone_pips = (top - bottom) / pip_size
                half_pips = zone_pips / 2.0
                
                if top > bottom and len(active_zones) < max_active:
                    active_zones.append({
                        'id': len(trades) + len(active_zones) + 1,
                        'side': -1,
                        'type': 'SELL_FLIP',
                        'top': top,
                        'bottom': bottom,
                        'entry': mid,
                        'tp': bottom,
                        'sl': top,
                        'zone_pips': zone_pips,
                        'half_pips': half_pips,
                        'created_idx': i,
                        'filled': False,
                        'fill_idx': -1
                    })
            else:
                if not in_pullback and is_red and not np.isnan(prev_green_open) and cl < prev_green_open:
                    in_pullback = True
                    cHH = pHH
                    cHH_idx = pHH_idx
                    pHL = l
                    pHL_idx = i
                elif in_pullback:
                    if l < pHL:
                        pHL = l
                        pHL_idx = i
                    if cl > cHH:
                        in_pullback = False
                        cHL = pHL
                        cHL_idx = pHL_idx
                        last_cHL = cHL
                        
                        top = cHH
                        bottom = cHL
                        mid = (top + bottom) / 2.0
                        zone_pips = (top - bottom) / pip_size
                        half_pips = zone_pips / 2.0
                        
                        pHH = h
                        pHH_idx = i
                        
                        if top > bottom and len(active_zones) < max_active:
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': 1,
                                'type': 'BUY',
                                'top': top,
                                'bottom': bottom,
                                'entry': mid,
                                'tp': top,
                                'sl': bottom,
                                'zone_pips': zone_pips,
                                'half_pips': half_pips,
                                'created_idx': i,
                                'filled': False,
                                'fill_idx': -1
                            })
                            
        elif trend == -1: # DOWNTREND
            if l < pLL or np.isnan(pLL):
                pLL = l
                pLL_idx = i
                
            if not np.isnan(last_cLH) and cl > last_cLH:
                old_ceiling = last_cLH
                bottom_trough = pLL
                
                trend = 1
                cHL = bottom_trough
                cHL_idx = pLL_idx
                last_cLH = np.nan
                pHH = h
                pHH_idx = i
                cHH = np.nan
                in_pullback = False
                
                top = old_ceiling
                bottom = bottom_trough
                mid = (top + bottom) / 2.0
                zone_pips = (top - bottom) / pip_size
                half_pips = zone_pips / 2.0
                
                if top > bottom and len(active_zones) < max_active:
                    active_zones.append({
                        'id': len(trades) + len(active_zones) + 1,
                        'side': 1,
                        'type': 'BUY_FLIP',
                        'top': top,
                        'bottom': bottom,
                        'entry': mid,
                        'tp': top,
                        'sl': bottom,
                        'zone_pips': zone_pips,
                        'half_pips': half_pips,
                        'created_idx': i,
                        'filled': False,
                        'fill_idx': -1
                    })
            else:
                if not in_pullback and is_green and not np.isnan(prev_red_open) and cl > prev_red_open:
                    in_pullback = True
                    cLL = pLL
                    cLL_idx = pLL_idx
                    pLH = h
                    pLH_idx = i
                elif in_pullback:
                    if h > pLH:
                        pLH = h
                        pLH_idx = i
                    if cl < cLL:
                        in_pullback = False
                        cLH = pLH
                        cLH_idx = pLH_idx
                        last_cLH = cLH
                        
                        top = cLH
                        bottom = cLL
                        mid = (top + bottom) / 2.0
                        zone_pips = (top - bottom) / pip_size
                        half_pips = zone_pips / 2.0
                        
                        pLL = l
                        pLL_idx = i
                        
                        if top > bottom and len(active_zones) < max_active:
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': -1,
                                'type': 'SELL',
                                'top': top,
                                'bottom': bottom,
                                'entry': mid,
                                'tp': bottom,
                                'sl': top,
                                'zone_pips': zone_pips,
                                'half_pips': half_pips,
                                'created_idx': i,
                                'filled': False,
                                'fill_idx': -1
                            })
                            
        if is_green:
            prev_green_open = o
        elif is_red:
            prev_red_open = o
            
    return pd.DataFrame(trades)

def run_comparison():
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP']
    all_s5_trades = []
    all_m1_trades = []
    
    print("=" * 85)
    print("BEID STRATEGY TIMEFRAME COMPARISON: 1-MINUTE (M1) BARS vs 5-SECOND (S5) BARS")
    print("Exact Identical Time Period & Continuous Live Order Flow")
    print("=" * 85)
    
    for sym in symbols:
        cache_path = f"dtfxStrategy/data/s5_cache/{sym}_s5.parquet"
        if not os.path.exists(cache_path):
            print(f"Skipping {sym}: {cache_path} not found.")
            continue
            
        df_s5 = pd.read_parquet(cache_path)
        df_s5['time_dt'] = pd.to_datetime(df_s5['time'])
        df_s5 = df_s5.sort_values('time_dt').reset_index(drop=True)
        
        # Resample to M1 bars
        df_m1 = df_s5.set_index('time_dt').resample('1min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna().reset_index()
        df_m1.rename(columns={'time_dt': 'time'}, inplace=True)
        
        print(f"\n[{sym}] Loaded {len(df_s5):,} S5 bars and resampled into {len(df_m1):,} M1 bars")
        print(f"  Time Window: {df_s5['time'].iloc[0]} to {df_s5['time'].iloc[-1]}")
        
        # 1. Run on S5 bars
        t_s5 = backtest_beid_engine(df_s5, sym, spread_pips=1.6, timeframe_label="S5")
        all_s5_trades.append(t_s5)
        
        # 2. Run on M1 bars
        t_m1 = backtest_beid_engine(df_m1, sym, spread_pips=1.6, timeframe_label="M1")
        all_m1_trades.append(t_m1)
        
        # Print summary per ticker
        print(f"  --> S5: {len(t_s5):,} Trades | Target Hit: {(sum(t_s5['raw_result']=='WIN')/max(1,len(t_s5)))*100:.1f}% | Avg Zone: {t_s5['zone_pips'].mean():.1f}p | Net Pips: {t_s5['net_pips'].sum():+,.1f}p | Net R: {t_s5['net_r'].sum():+,.1f}R")
        print(f"  --> M1: {len(t_m1):,} Trades | Target Hit: {(sum(t_m1['raw_result']=='WIN')/max(1,len(t_m1)))*100:.1f}% | Avg Zone: {t_m1['zone_pips'].mean():.1f}p | Net Pips: {t_m1['net_pips'].sum():+,.1f}p | Net R: {t_m1['net_r'].sum():+,.1f}R")
        
    s5_full = pd.concat(all_s5_trades, ignore_index=True) if all_s5_trades else pd.DataFrame()
    m1_full = pd.concat(all_m1_trades, ignore_index=True) if all_m1_trades else pd.DataFrame()
    
    s5_full.to_csv("dtfxStrategy/research_results/beid_s5_trades.csv", index=False)
    m1_full.to_csv("dtfxStrategy/research_results/beid_m1_identical_period_trades.csv", index=False)
    
    print("\n" + "=" * 85)
    print("AGGREGATE PORTFOLIO COMPARISON SUMMARY (EXACT SAME TIME PERIOD):")
    print("=" * 85)
    
    for tf_name, df_trades in [("5-SECOND (S5)", s5_full), ("1-MINUTE (M1)", m1_full)]:
        if len(df_trades) == 0:
            continue
        total_t = len(df_trades)
        wins = sum(df_trades['raw_result'] == 'WIN')
        wr = (wins / total_t) * 100.0
        raw_r = df_trades['raw_r'].sum()
        net_pips = df_trades['net_pips'].sum()
        net_r = df_trades['net_r'].sum()
        avg_zone = df_trades['zone_pips'].mean()
        med_zone = df_trades['zone_pips'].median()
        
        win_pips = df_trades[df_trades['net_pips'] > 0]['net_pips'].sum()
        loss_pips = abs(df_trades[df_trades['net_pips'] <= 0]['net_pips'].sum())
        pf = win_pips / max(1e-6, loss_pips)
        
        # Trades with >= 5.0 pip zones
        sub_5 = df_trades[df_trades['zone_pips'] >= 5.0]
        pips_5 = sub_5['net_pips'].sum() if len(sub_5) > 0 else 0.0
        r_5 = sub_5['net_r'].sum() if len(sub_5) > 0 else 0.0
        
        print(f"\n--- {tf_name} PERFORMANCE ---")
        print(f"  Total Trades: {total_t:,}")
        print(f"  Raw Target Hit Rate: {wr:.2f}% ({wins:,} Wins / {total_t - wins:,} Losses)")
        print(f"  Zero-Friction Net Return: {raw_r:+,.1f} R")
        print(f"  Average Zone Size: {avg_zone:.2f} pips (Median: {med_zone:.2f} pips)")
        print(f"  Net Total Pips (-1.6 pips spread): {net_pips:+,.1f} Pips (Profit Factor: {pf:.2f})")
        print(f"  Net Total R (-1.6 pips spread): {net_r:+,.1f} R")
        print(f"  Filtered &ge; 5.0 Pip Zones ({len(sub_5):,} trades): {pips_5:+,.1f} Pips | {r_5:+,.1f} R")

if __name__ == "__main__":
    run_comparison()
