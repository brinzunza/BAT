import os
import sys
import numpy as np
import pandas as pd
import time

def get_pip_size(ticker):
    if "JPY" in ticker:
        return 0.01
    return 0.0001

def resample_ohlc(df_1m, timeframe_mins):
    if timeframe_mins == 1:
        df = df_1m.copy()
        if 'time' not in df.columns and 'date' in df.columns:
            df['time'] = df['date']
        return df
    
    df = df_1m.copy()
    time_col = 'date' if 'date' in df.columns else 'time'
    df['dt'] = pd.to_datetime(df[time_col])
    df = df.set_index('dt')
        
    rule = f"{timeframe_mins}min" if timeframe_mins < 60 else f"{timeframe_mins//60}h"
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna().reset_index()
    resampled.rename(columns={'dt': 'time'}, inplace=True)
    return resampled

def backtest_fixed_beid_engine(df, ticker_name, max_touches=1, spread_pips=1.6):
    pip_size = get_pip_size(ticker_name)
    time_col = 'date' if 'date' in df.columns else 'time'
    dates = df[time_col].values
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n = len(opens)
    
    trend = 1
    pHH = opens[0]; pHH_idx = 0
    cHH = np.nan; cHH_idx = -1
    pHL = opens[0]; pHL_idx = 0
    cHL = np.nan; cHL_idx = -1
    last_cHL = np.nan
    
    pLL = opens[0]; pLL_idx = 0
    cLL = np.nan; cLL_idx = -1
    pLH = opens[0]; pLH_idx = 0
    cLH = np.nan; cLH_idx = -1
    last_cLH = np.nan
    
    in_pullback = False
    pullback_start_idx = 0
    prev_green_open = np.nan
    prev_red_open = np.nan
    
    active_zones = []
    trades = []
    max_active = 25
    
    for i in range(n):
        o, h, l, cl = opens[i], highs[i], lows[i], closes[i]
        is_green = cl >= o
        is_red = cl < o
        
        # 1. Update Active Zones
        retained = []
        for z in active_zones:
            side, entry, tp, sl = z['side'], z['entry'], z['tp'], z['sl']
            half_pips = z['half_pips']
            
            if not z['filled']:
                if side == 1:
                    if l <= sl: continue # Invalidate
                    if l <= entry:
                        z['filled'] = True; z['fill_idx'] = i; z['touch_count'] += 1
                        retained.append(z)
                    elif i - z['created_idx'] <= 2880: retained.append(z)
                elif side == -1:
                    if h >= sl: continue # Invalidate
                    if h >= entry:
                        z['filled'] = True; z['fill_idx'] = i; z['touch_count'] += 1
                        retained.append(z)
                    elif i - z['created_idx'] <= 2880: retained.append(z)
            else:
                if i > z['fill_idx']:
                    resolved, win = False, False
                    if side == 1:
                        if (h >= tp) or (cl > tp): resolved, win = True, True
                        elif (l <= sl) or (cl < sl): resolved, win = True, False
                    else:
                        if (l <= tp) or (cl < tp): resolved, win = True, True
                        elif (h >= sl) or (cl > sl): resolved, win = True, False
                        
                    if resolved:
                        net_pips = (half_pips - spread_pips) if win else -(half_pips + spread_pips)
                        trades.append({
                            'ticker': ticker_name,
                            'fill_time': dates[z['fill_idx']],
                            'exit_time': dates[i],
                            'zone_pips': z['zone_pips'],
                            'touch_count': z['touch_count'],
                            'win': win,
                            'net_pips': net_pips,
                            'net_r': net_pips / max(1e-4, half_pips),
                            'raw_r': 1.0 if win else -1.0
                        })
                        if win and z['touch_count'] < max_touches:
                            z['filled'] = False; z['fill_idx'] = -1; z['created_idx'] = i
                            retained.append(z)
                    else:
                        retained.append(z)
        active_zones = retained
        
        # 2. Robust State Machine Updates (Lockup-Free)
        if trend == 1: # UPTREND
            if not in_pullback:
                if h > pHH:
                    pHH = h; pHH_idx = i
                # Check for trend flip on close below last_cHL
                if not np.isnan(last_cHL) and cl < last_cHL:
                    trend = -1
                    top = pHH; bottom = last_cHL; mid = (top + bottom)/2.0
                    zone_pips = (top - bottom)/pip_size
                    if top > bottom and len(active_zones) < max_active:
                        active_zones.append({
                            'id': len(trades)+len(active_zones)+1, 'side': -1, 'type': 'SELL_FLIP',
                            'top': top, 'bottom': bottom, 'entry': mid, 'tp': bottom, 'sl': top,
                            'zone_pips': zone_pips, 'half_pips': zone_pips/2.0,
                            'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                        })
                    pLL = l; pLL_idx = i; cLH = top; last_cLH = np.nan; in_pullback = False
                elif is_red and not np.isnan(prev_green_open) and cl < prev_green_open:
                    in_pullback = True
                    pullback_start_idx = i
                    cHH = pHH; cHH_idx = pHH_idx
                    pHL = l; pHL_idx = i
            else:
                if l < pHL:
                    pHL = l; pHL_idx = i
                
                # Check trend flip while in pullback
                floor_to_check = last_cHL if not np.isnan(last_cHL) else opens[0]
                if cl < floor_to_check:
                    trend = -1
                    top = cHH; bottom = floor_to_check; mid = (top + bottom)/2.0
                    zone_pips = (top - bottom)/pip_size
                    if top > bottom and len(active_zones) < max_active:
                        active_zones.append({
                            'id': len(trades)+len(active_zones)+1, 'side': -1, 'type': 'SELL_FLIP',
                            'top': top, 'bottom': bottom, 'entry': mid, 'tp': bottom, 'sl': top,
                            'zone_pips': zone_pips, 'half_pips': zone_pips/2.0,
                            'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                        })
                    pLL = l; pLL_idx = i; cLH = top; last_cLH = np.nan; in_pullback = False
                elif cl > cHH: # Breakout continuation
                    in_pullback = False
                    cHL = pHL; cHL_idx = pHL_idx; last_cHL = cHL
                    top = cHH; bottom = cHL; mid = (top + bottom)/2.0
                    zone_pips = (top - bottom)/pip_size
                    pHH = h; pHH_idx = i
                    if top > bottom and len(active_zones) < max_active:
                        active_zones.append({
                            'id': len(trades)+len(active_zones)+1, 'side': 1, 'type': 'BUY',
                            'top': top, 'bottom': bottom, 'entry': mid, 'tp': top, 'sl': bottom,
                            'zone_pips': zone_pips, 'half_pips': zone_pips/2.0,
                            'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                        })
                elif i - pullback_start_idx > 500:
                    in_pullback = False
                    pHH = h; pHH_idx = i
                    
        elif trend == -1: # DOWNTREND
            if not in_pullback:
                if l < pLL:
                    pLL = l; pLL_idx = i
                if not np.isnan(last_cLH) and cl > last_cLH:
                    trend = 1
                    top = last_cLH; bottom = pLL; mid = (top + bottom)/2.0
                    zone_pips = (top - bottom)/pip_size
                    if top > bottom and len(active_zones) < max_active:
                        active_zones.append({
                            'id': len(trades)+len(active_zones)+1, 'side': 1, 'type': 'BUY_FLIP',
                            'top': top, 'bottom': bottom, 'entry': mid, 'tp': top, 'sl': bottom,
                            'zone_pips': zone_pips, 'half_pips': zone_pips/2.0,
                            'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                        })
                    pHH = h; pHH_idx = i; cHL = bottom; last_cHL = np.nan; in_pullback = False
                elif is_green and not np.isnan(prev_red_open) and cl > prev_red_open:
                    in_pullback = True
                    pullback_start_idx = i
                    cLL = pLL; cLL_idx = pLL_idx
                    pLH = h; pLH_idx = i
            else:
                if h > pLH:
                    pLH = h; pLH_idx = i
                ceiling_to_check = last_cLH if not np.isnan(last_cLH) else opens[0]
                if cl > ceiling_to_check:
                    trend = 1
                    top = ceiling_to_check; bottom = cLL; mid = (top + bottom)/2.0
                    zone_pips = (top - bottom)/pip_size
                    if top > bottom and len(active_zones) < max_active:
                        active_zones.append({
                            'id': len(trades)+len(active_zones)+1, 'side': 1, 'type': 'BUY_FLIP',
                            'top': top, 'bottom': bottom, 'entry': mid, 'tp': top, 'sl': bottom,
                            'zone_pips': zone_pips, 'half_pips': zone_pips/2.0,
                            'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                        })
                    pHH = h; pHH_idx = i; cHL = bottom; last_cHL = np.nan; in_pullback = False
                elif cl < cLL: # Breakdown continuation
                    in_pullback = False
                    cLH = pLH; cLH_idx = pLH_idx; last_cLH = cLH
                    top = cLH; bottom = cLL; mid = (top + bottom)/2.0
                    zone_pips = (top - bottom)/pip_size
                    pLL = l; pLL_idx = i
                    if top > bottom and len(active_zones) < max_active:
                        active_zones.append({
                            'id': len(trades)+len(active_zones)+1, 'side': -1, 'type': 'SELL',
                            'top': top, 'bottom': bottom, 'entry': mid, 'tp': bottom, 'sl': top,
                            'zone_pips': zone_pips, 'half_pips': zone_pips/2.0,
                            'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                        })
                elif i - pullback_start_idx > 500:
                    in_pullback = False
                    pLL = l; pLL_idx = i
                    
        if is_green: prev_green_open = o
        elif is_red: prev_red_open = o
        
    return pd.DataFrame(trades)

def run_audit():
    tickers = {
        "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
        "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
        "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
        "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv"
    }
    
    print("=" * 95)
    print("HONEST FULL-HISTORY BEID AUDIT ACROSS REGIMES (FIXED LOCKUP-FREE ENGINE)")
    print("=" * 95)
    
    all_trades = []
    
    for name, path in tickers.items():
        if not os.path.exists(path):
            continue
        print(f"Loading {name} full dataset...")
        df_1m = pd.read_csv(path)
        # Resample to 3-minute bars (or 15-minute)
        df_3m = resample_ohlc(df_1m, 3)
        trades_3m = backtest_fixed_beid_engine(df_3m, name, max_touches=1, spread_pips=1.6)
        all_trades.append(trades_3m)
        print(f"  --> {name}: {len(trades_3m):,} trades on 3M across full history")
        
    full_df = pd.concat(all_trades, ignore_index=True)
    full_df['date'] = pd.to_datetime(full_df['exit_time'])
    
    # Analyze by min zone filter and regime
    for min_pip in [0.0, 5.0, 10.0, 15.0, 20.0]:
        sub = full_df[full_df['zone_pips'] >= min_pip].copy()
        print("\n" + "=" * 80)
        print(f"MIN ZONE FILTER: >= {min_pip:.1f} PIPS (Total Trades: {len(sub):,})")
        print("=" * 80)
        
        pre_2021 = sub[sub['date'] < '2021-01-01']
        contam = sub[(sub['date'] >= '2021-01-01') & (sub['date'] < '2026-04-01')]
        holdout = sub[sub['date'] >= '2026-04-01']
        
        for r_name, r_sub in [('PRE-2021 (Design Era)', pre_2021), ('2021-2026 (Broker Feed)', contam), ('POST-2026-04 (Holdout)', holdout), ('FULL HISTORY (2000-2026)', sub)]:
            if len(r_sub) == 0: continue
            w = sum(r_sub['win'])
            wr = (w / len(r_sub)) * 100.0
            tot_pips = r_sub['net_pips'].sum()
            tot_r = r_sub['net_r'].sum()
            win_pips = r_sub[r_sub['net_pips'] > 0]['net_pips'].sum()
            loss_pips = abs(r_sub[r_sub['net_pips'] <= 0]['net_pips'].sum())
            pf = win_pips / max(1e-6, loss_pips)
            print(f"  {r_name:<26} | Trades: {len(r_sub):<7,} | WR: {wr:<5.1f}% | Net Pips: {tot_pips:<10,.1f} | Net R: {tot_r:<8.1f} | PF: {pf:<5.2f}")

if __name__ == "__main__":
    run_audit()
