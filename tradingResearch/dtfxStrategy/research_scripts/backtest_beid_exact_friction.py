import numpy as np
import pandas as pd
import os
import time
import json

def get_pip_size(ticker):
    if "JPY" in ticker:
        return 0.01
    return 0.0001

def backtest_beid_exact_friction(df_ticker, ticker_name, spread_pips=1.6, max_active_zones=25):
    """
    Backtests BEID where entry is EXACT 50% midpoint, TP is EXACT zone boundary, SL is EXACT opposite boundary.
    Spread friction (1.6 pips) is applied directly to the final PnL:
    - WIN: + (half_height_pips - 1.6 pips)
    - LOSS: - (half_height_pips + 1.6 pips)
    """
    pip_size = get_pip_size(ticker_name)
    
    dates = df_ticker['date'].values if 'date' in df_ticker.columns else df_ticker['time'].values
    opens = df_ticker['open'].values.astype(np.float64)
    highs = df_ticker['high'].values.astype(np.float64)
    lows = df_ticker['low'].values.astype(np.float64)
    closes = df_ticker['close'].values.astype(np.float64)
    
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
                    if i - z['created_idx'] > 2880: # Expire after 48h (2880 1m candles)
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
                            net_r = net_pips / half_pips
                        else:
                            net_pips = -(half_pips + spread_pips)
                            net_r = net_pips / half_pips
                            
                        trades.append({
                            'ticker': ticker_name,
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
                            'duration_bars': i - z['fill_idx'],
                            'year': str(dates[i])[:4]
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
                
                if top > bottom and len(active_zones) < max_active_zones:
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
                        
                        if top > bottom and len(active_zones) < max_active_zones:
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
                
                if top > bottom and len(active_zones) < max_active_zones:
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
                        
                        if top > bottom and len(active_zones) < max_active_zones:
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

if __name__ == "__main__":
    tickers = {
        "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
        "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
        "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
        "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv"
    }
    
    all_trade_dfs = []
    
    print("=" * 80)
    print("BACKTEST: BEID EXACT MIDPOINT ENTRY & EXACT EXITS WITH -1.6 PIPS SPREAD DEDUCTION")
    print("=" * 80)
    
    for name, path in tickers.items():
        if os.path.exists(path):
            print(f"\nLoading {name} (1,000,000 recent 1-minute bars)...")
            df = pd.read_csv(path)
            df_sample = df.iloc[-1000000:].copy().reset_index(drop=True)
            trade_df = backtest_beid_exact_friction(df_sample, name, spread_pips=1.6)
            all_trade_dfs.append(trade_df)
            
            if len(trade_df) > 0:
                raw_wins = sum(trade_df['raw_result'] == 'WIN')
                raw_losses = sum(trade_df['raw_result'] == 'LOSS')
                raw_wr = (raw_wins / len(trade_df)) * 100.0
                raw_net_r = trade_df['raw_r'].sum()
                
                net_pips = trade_df['net_pips'].sum()
                net_r = trade_df['net_r'].sum()
                avg_zone = trade_df['zone_pips'].mean()
                
                pos_pnl_trades = sum(trade_df['net_pips'] > 0)
                net_wr = (pos_pnl_trades / len(trade_df)) * 100.0
                
                gross_win_pips = trade_df[trade_df['net_pips'] > 0]['net_pips'].sum()
                gross_loss_pips = abs(trade_df[trade_df['net_pips'] <= 0]['net_pips'].sum())
                pip_pf = gross_win_pips / max(1e-6, gross_loss_pips)
                
                print(f"--> {name} Results:")
                print(f"    Total Trades: {len(trade_df):,}")
                print(f"    Raw Target Hits: {raw_wins} Wins ({raw_wr:.1f}%) / {raw_losses} Losses")
                print(f"    Raw Net Return: {raw_net_r:+.1f} R (without friction)")
                print(f"    Net PnL (after 1.6 pips friction): {net_pips:+,.1f} Pips | {net_r:+,.1f} R")
                print(f"    Net Profit Factor (Pips): {pip_pf:.2f}")
                print(f"    Avg Zone Size: {avg_zone:.1f} pips | Median: {trade_df['zone_pips'].median():.1f} pips")
                
    if all_trade_dfs:
        full_df = pd.concat(all_trade_dfs, ignore_index=True)
        out_csv = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_exact_friction_trades.csv"
        full_df.to_csv(out_csv, index=False)
        
        print("\n" + "=" * 80)
        print("AGGREGATE 4-FOREX PORTFOLIO PERFORMANCE (EXACT MID ENTRIES + -1.6 PIPS PER TRADE):")
        print("=" * 80)
        total_trades = len(full_df)
        raw_w = sum(full_df['raw_result'] == 'WIN')
        raw_l = sum(full_df['raw_result'] == 'LOSS')
        raw_wr = (raw_w / total_trades) * 100.0
        raw_r = full_df['raw_r'].sum()
        
        tot_net_pips = full_df['net_pips'].sum()
        tot_net_r = full_df['net_r'].sum()
        
        win_pips = full_df[full_df['net_pips'] > 0]['net_pips'].sum()
        loss_pips = abs(full_df[full_df['net_pips'] <= 0]['net_pips'].sum())
        agg_pip_pf = win_pips / max(1e-6, loss_pips)
        
        pos_trades = sum(full_df['net_pips'] > 0)
        net_wr = (pos_trades / total_trades) * 100.0
        
        print(f"Total Portfolio Trades: {total_trades:,}")
        print(f"Raw Target Hit Rate: {raw_wr:.2f}% ({raw_w:,} Wins / {raw_l:,} Losses)")
        print(f"Raw Net R (Zero Friction): {raw_r:+,.1f} R")
        print(f"Net Total Pips (-1.6 pips per trade): {tot_net_pips:+,.1f} Pips")
        print(f"Net Total R (-1.6 pips per trade): {tot_net_r:+,.1f} R")
        print(f"Net Profit Factor: {agg_pip_pf:.2f}")
        print(f"Net Profitable Trade Rate: {net_wr:.2f}%")
        print("=" * 80)
