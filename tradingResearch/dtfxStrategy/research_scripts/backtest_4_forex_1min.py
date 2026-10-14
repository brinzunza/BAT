import numpy as np
import pandas as pd
import os
import time

def backtest_beid_fast(df_ticker, ticker_name, max_active_zones=20):
    """
    High-performance NumPy-based backtest of BEID strategy on 1-minute historical data.
    """
    print(f"\nProcessing {ticker_name} ({len(df_ticker):,} 1-minute candles)...")
    t0 = time.time()
    
    # Extract columns to numpy arrays for maximum execution speed
    dates = df_ticker['date'].values if 'date' in df_ticker.columns else df_ticker['time'].values
    opens = df_ticker['open'].values.astype(np.float64)
    highs = df_ticker['high'].values.astype(np.float64)
    lows = df_ticker['low'].values.astype(np.float64)
    closes = df_ticker['close'].values.astype(np.float64)
    
    n = len(opens)
    
    # State tracking
    trend = 1 # 1: UPTREND, -1: DOWNTREND
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
    
    # Active zones structure: list of dicts
    # Each zone: {id, side, top, bottom, entry, tp, sl, created_idx, filled, fill_idx, touched_50, touched_50_idx}
    active_zones = []
    
    # Trades recorded
    trades = []
    
    for i in range(n):
        o = opens[i]
        h = highs[i]
        l = lows[i]
        cl = closes[i]
        
        is_green = cl >= o
        is_red = cl < o
        
        # -------------------------------------------------------------
        # 1. Update Active Zones / Open Trades
        # -------------------------------------------------------------
        retained_zones = []
        for z in active_zones:
            side = z['side']
            entry = z['entry']
            tp = z['tp']
            sl = z['sl']
            
            if not z['filled']:
                # Check for 50% limit fill
                if side == 1 and l <= entry: # BUY fill
                    z['filled'] = True
                    z['fill_idx'] = i
                    z['touched_50'] = True
                    z['touched_50_idx'] = i
                    retained_zones.append(z)
                elif side == -1 and h >= entry: # SELL fill
                    z['filled'] = True
                    z['fill_idx'] = i
                    z['touched_50'] = True
                    z['touched_50_idx'] = i
                    retained_zones.append(z)
                else:
                    # Still waiting for fill. Check structural cancellation if trend flipped against it
                    if i - z['created_idx'] > 2880: # Expire if unfilled after 48h (2880 1m candles)
                        pass
                    else:
                        retained_zones.append(z)
            else:
                # Trade is active! Must resolve on subsequent candle (i > z['fill_idx'])
                if i > z['fill_idx']:
                    resolved = False
                    result = 0 # 1 for WIN, -1 for LOSS
                    
                    if side == 1: # BUY Trade
                        hit_sl = l <= sl or cl < sl
                        hit_tp = h >= tp or cl > tp
                        
                        if hit_sl and hit_tp:
                            result = 1 if cl >= entry else -1
                            resolved = True
                        elif hit_tp:
                            result = 1
                            resolved = True
                        elif hit_sl:
                            result = -1
                            resolved = True
                    else: # SELL Trade
                        hit_sl = h >= sl or cl > sl
                        hit_tp = l <= tp or cl < tp
                        
                        if hit_sl and hit_tp:
                            result = 1 if cl <= entry else -1
                            resolved = True
                        elif hit_tp:
                            result = 1
                            resolved = True
                        elif hit_sl:
                            result = -1
                            resolved = True
                            
                    if resolved:
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
                            'result': 'WIN' if result == 1 else 'LOSS',
                            'r_mult': float(result),
                            'duration_bars': i - z['fill_idx'],
                            'year': str(dates[i])[:4]
                        })
                        # Zone terminates upon resolution! Do not retain
                    else:
                        retained_zones.append(z)
                        
        active_zones = retained_zones
        
        # -------------------------------------------------------------
        # 2. State Machine Updates & Zone Generation
        # -------------------------------------------------------------
        if trend == 1: # UPTREND
            if h > pHH or np.isnan(pHH):
                pHH = h
                pHH_idx = i
                
            # Check Trend Flip: red body closes below last_cHL
            if not np.isnan(last_cHL) and cl < last_cHL:
                old_floor = last_cHL
                top_peak = pHH
                
                trend = -1 # FLIP TO DOWNTREND
                cLH = top_peak
                cLH_idx = pHH_idx
                last_cLH = np.nan
                pLL = l
                pLL_idx = i
                cLL = np.nan
                in_pullback = False
                
                # SELL FLIP ZONE
                top = top_peak
                bottom = old_floor
                mid = (top + bottom) / 2.0
                
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
                        'created_idx': i,
                        'filled': False,
                        'fill_idx': -1,
                        'touched_50': False,
                        'touched_50_idx': -1
                    })
            else:
                # Normal Uptrend
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
                        
                        pHH = h
                        pHH_idx = i
                        
                        # BUY ZONE
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
                                'created_idx': i,
                                'filled': False,
                                'fill_idx': -1,
                                'touched_50': False,
                                'touched_50_idx': -1
                            })
                            
        elif trend == -1: # DOWNTREND
            if l < pLL or np.isnan(pLL):
                pLL = l
                pLL_idx = i
                
            # Check Trend Flip: green body closes above last_cLH
            if not np.isnan(last_cLH) and cl > last_cLH:
                old_ceiling = last_cLH
                bottom_trough = pLL
                
                trend = 1 # FLIP TO UPTREND
                cHL = bottom_trough
                cHL_idx = pLL_idx
                last_cHL = np.nan
                pHH = h
                pHH_idx = i
                cHH = np.nan
                in_pullback = False
                
                # BUY FLIP ZONE
                top = old_ceiling
                bottom = bottom_trough
                mid = (top + bottom) / 2.0
                
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
                        'created_idx': i,
                        'filled': False,
                        'fill_idx': -1,
                        'touched_50': False,
                        'touched_50_idx': -1
                    })
            else:
                # Normal Downtrend
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
                        
                        pLL = l
                        pLL_idx = i
                        
                        # SELL ZONE
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
                                'created_idx': i,
                                'filled': False,
                                'fill_idx': -1,
                                'touched_50': False,
                                'touched_50_idx': -1
                            })
                            
        if is_green:
            prev_green_open = o
        elif is_red:
            prev_red_open = o
            
    elapsed = time.time() - t0
    print(f"Completed {ticker_name} in {elapsed:.2f}s | {len(trades):,} trades generated.")
    return pd.DataFrame(trades)

if __name__ == "__main__":
    tickers = {
        "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
        "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
        "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
        "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv"
    }
    
    all_trade_dfs = []
    
    for name, path in tickers.items():
        if os.path.exists(path):
            print(f"Loading {name} from {path}...")
            # For quick initial deep backtest, sample or load recent 1,500,000 candles (approx 4-5 years of 1M data) or full
            df = pd.read_csv(path)
            # Take the most recent 1,000,000 1-minute bars for comprehensive multi-year statistical significance
            df_sample = df.iloc[-1000000:].copy().reset_index(drop=True)
            trade_df = backtest_beid_fast(df_sample, name)
            all_trade_dfs.append(trade_df)
            
            # Print ticker summary
            if len(trade_df) > 0:
                wins = sum(trade_df['result'] == 'WIN')
                losses = sum(trade_df['result'] == 'LOSS')
                wr = (wins / len(trade_df)) * 100.0
                net_r = trade_df['r_mult'].sum()
                pf = wins / max(1, losses)
                print(f"--> {name} 1M Results: {len(trade_df):,} Trades | {wr:.1f}% Win Rate ({wins}W / {losses}L) | Net: {net_r:+.1f}R | Profit Factor: {pf:.2f}")
                
    if all_trade_dfs:
        full_df = pd.concat(all_trade_dfs, ignore_index=True)
        full_df.to_csv("/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_4forex_1min_trades.csv", index=False)
        print("\n" + "="*70)
        print(f"4 FOREX TICKERS 1-MINUTE AGGREGATE PERFORMANCE ({len(full_df):,} TOTAL TRADES):")
        total_w = sum(full_df['result'] == 'WIN')
        total_l = sum(full_df['result'] == 'LOSS')
        print(f"Overall Win Rate: {(total_w / len(full_df))*100.0:.2f}% ({total_w:,}W / {total_l:,}L)")
        print(f"Total Net Return: {full_df['r_mult'].sum():+,.1f} R")
        print(f"Overall Profit Factor: {(total_w / max(1, total_l)):.2f}")
        print("="*70)
