import os
import pandas as pd
import numpy as np
import json
import time

def get_pip_size(ticker):
    if "JPY" in ticker:
        return 0.01
    return 0.0001

def backtest_ticker_spread_beid(df_ticker, ticker_name, spread_pips=1.6, min_zone_pips=5.0, max_active_zones=25):
    pip_size = get_pip_size(ticker_name)
    spread_val = spread_pips * pip_size
    half_spread = spread_val / 2.0
    min_zone_val = min_zone_pips * pip_size
    
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
            risk_dist = z['risk']
            reward_dist = z['reward']
            
            if not z['filled']:
                # Buy Limit fill: price reaches entry (50% + spread/2)
                if side == 1 and l <= entry:
                    z['filled'] = True
                    z['fill_idx'] = i
                    retained.append(z)
                # Sell Limit fill: price reaches entry (50% - spread/2)
                elif side == -1 and h >= entry:
                    z['filled'] = True
                    z['fill_idx'] = i
                    retained.append(z)
                else:
                    if i - z['created_idx'] > 2880: # Expire after 48h
                        pass
                    else:
                        retained.append(z)
            else:
                # Active trade: must resolve on a subsequent different candle (i > z['fill_idx'])
                if i > z['fill_idx']:
                    resolved = False
                    result = ""
                    pnl_r = 0.0
                    
                    if side == 1: # BUY Trade
                        # SL hit: low reaches bottom + spread/2
                        hit_sl = l <= sl or cl < sl
                        # TP hit: high reaches top - spread/2
                        hit_tp = h >= tp or cl > tp
                        
                        if hit_sl and hit_tp:
                            if cl >= entry:
                                result = "WIN"
                                pnl_r = reward_dist / risk_dist
                                resolved = True
                            else:
                                result = "LOSS"
                                pnl_r = -1.0
                                resolved = True
                        elif hit_tp:
                            result = "WIN"
                            pnl_r = reward_dist / risk_dist
                            resolved = True
                        elif hit_sl:
                            result = "LOSS"
                            pnl_r = -1.0
                            resolved = True
                            
                    elif side == -1: # SELL Trade
                        # SL hit: high reaches top - spread/2
                        hit_sl = h >= sl or cl > sl
                        # TP hit: low reaches bottom + spread/2
                        hit_tp = l <= tp or cl < tp
                        
                        if hit_sl and hit_tp:
                            if cl <= entry:
                                result = "WIN"
                                pnl_r = reward_dist / risk_dist
                                resolved = True
                            else:
                                result = "LOSS"
                                pnl_r = -1.0
                                resolved = True
                        elif hit_tp:
                            result = "WIN"
                            pnl_r = reward_dist / risk_dist
                            resolved = True
                        elif hit_sl:
                            result = "LOSS"
                            pnl_r = -1.0
                            resolved = True
                            
                    if resolved:
                        trades.append({
                            'ticker': ticker_name,
                            'side': 'BUY' if side == 1 else 'SELL',
                            'type': z['type'],
                            'zone_height_pips': round(z['zone_height'] / pip_size, 2),
                            'created_time': str(dates[z['created_idx']]),
                            'fill_time': str(dates[z['fill_idx']]),
                            'exit_time': str(dates[i]),
                            'entry': entry,
                            'tp': tp,
                            'sl': sl,
                            'risk_pips': round(risk_dist / pip_size, 2),
                            'reward_pips': round(reward_dist / pip_size, 2),
                            'raw_rr': round(reward_dist / risk_dist, 3),
                            'result': result,
                            'r_mult': pnl_r,
                            'duration_bars': i - z['fill_idx']
                        })
                    else:
                        retained.append(z)
        active_zones = retained
        
        # 2. State Machine & Zone Creation
        if trend == 1:
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
                zone_height = top - bottom
                
                if zone_height >= min_zone_val:
                    mid = (top + bottom) / 2.0
                    # SELL Zone: Entry = 50% - spread/2, TP = bottom + spread/2, SL = top - spread/2
                    entry = mid - half_spread
                    tp = bottom + half_spread
                    sl = top - half_spread
                    risk = sl - entry
                    reward = entry - tp
                    
                    if reward > 0 and risk > 0 and len(active_zones) < max_active_zones:
                        active_zones.append({
                            'side': -1,
                            'type': 'SELL_FLIP',
                            'top': top,
                            'bottom': bottom,
                            'zone_height': zone_height,
                            'entry': entry,
                            'tp': tp,
                            'sl': sl,
                            'risk': risk,
                            'reward': reward,
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
                        zone_height = top - bottom
                        
                        pHH = h
                        pHH_idx = i
                        
                        if zone_height >= min_zone_val:
                            mid = (top + bottom) / 2.0
                            # BUY Zone: Entry = 50% + spread/2, TP = top - spread/2, SL = bottom + spread/2
                            entry = mid + half_spread
                            tp = top - half_spread
                            sl = bottom + half_spread
                            risk = entry - sl
                            reward = tp - entry
                            
                            if reward > 0 and risk > 0 and len(active_zones) < max_active_zones:
                                active_zones.append({
                                    'side': 1,
                                    'type': 'BUY',
                                    'top': top,
                                    'bottom': bottom,
                                    'zone_height': zone_height,
                                    'entry': entry,
                                    'tp': tp,
                                    'sl': sl,
                                    'risk': risk,
                                    'reward': reward,
                                    'created_idx': i,
                                    'filled': False,
                                    'fill_idx': -1
                                })
        elif trend == -1:
            if l < pLL or np.isnan(pLL):
                pLL = l
                pLL_idx = i
                
            if not np.isnan(last_cLH) and cl > last_cLH:
                old_ceiling = last_cLH
                bottom_trough = pLL
                
                trend = 1
                cHL = bottom_trough
                cHL_idx = pLL_idx
                last_cHL = np.nan
                pHH = h
                pHH_idx = i
                cHH = np.nan
                in_pullback = False
                
                top = old_ceiling
                bottom = bottom_trough
                zone_height = top - bottom
                
                if zone_height >= min_zone_val:
                    mid = (top + bottom) / 2.0
                    # BUY Flip Zone: Entry = 50% + spread/2, TP = top - spread/2, SL = bottom + spread/2
                    entry = mid + half_spread
                    tp = top - half_spread
                    sl = bottom + half_spread
                    risk = entry - sl
                    reward = tp - entry
                    
                    if reward > 0 and risk > 0 and len(active_zones) < max_active_zones:
                        active_zones.append({
                            'side': 1,
                            'type': 'BUY_FLIP',
                            'top': top,
                            'bottom': bottom,
                            'zone_height': zone_height,
                            'entry': entry,
                            'tp': tp,
                            'sl': sl,
                            'risk': risk,
                            'reward': reward,
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
                        zone_height = top - bottom
                        
                        pLL = l
                        pLL_idx = i
                        
                        if zone_height >= min_zone_val:
                            mid = (top + bottom) / 2.0
                            # SELL Zone: Entry = 50% - spread/2, TP = bottom + spread/2, SL = top - spread/2
                            entry = mid - half_spread
                            tp = bottom + half_spread
                            sl = top - half_spread
                            risk = sl - entry
                            reward = entry - tp
                            
                            if reward > 0 and risk > 0 and len(active_zones) < max_active_zones:
                                active_zones.append({
                                    'side': -1,
                                    'type': 'SELL',
                                    'top': top,
                                    'bottom': bottom,
                                    'zone_height': zone_height,
                                    'entry': entry,
                                    'tp': tp,
                                    'sl': sl,
                                    'risk': risk,
                                    'reward': reward,
                                    'created_idx': i,
                                    'filled': False,
                                    'fill_idx': -1
                                })
        if is_green:
            prev_green_open = o
        elif is_red:
            prev_red_open = o
            
    return pd.DataFrame(trades)

def run_portfolio_backtest():
    tickers = {
        "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
        "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
        "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
        "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv"
    }
    
    # We will test on recent 1,000,000 candles for each ticker (~3-4 years continuous 1M tick data)
    # Testing min zone sizes: 4.0, 5.0, 6.0, 7.0, 8.0 pips
    min_zones = [4.0, 5.0, 6.0, 8.0]
    
    print("=== BEID 4-FOREX PORTFOLIO BACKTEST (1.6 PIPS SPREAD) ===")
    print("Rules:")
    print("  BUY : Entry = 50% + 0.8 pips | TP = Top - 0.8 pips | SL = Bottom + 0.8 pips")
    print("  SELL: Entry = 50% - 0.8 pips | TP = Bottom + 0.8 pips | SL = Top - 0.8 pips")
    print("-" * 80)
    
    portfolio_dfs = []
    
    for mz in min_zones:
        ticker_results = []
        all_trades = []
        for name, path in tickers.items():
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df_sub = df.iloc[-1000000:].copy().reset_index(drop=True)
            tdf = backtest_ticker_spread_beid(df_sub, name, spread_pips=1.6, min_zone_pips=mz)
            if len(tdf) > 0:
                wins = sum(tdf['result'] == 'WIN')
                losses = sum(tdf['result'] == 'LOSS')
                wr = (wins / len(tdf)) * 100.0
                net_r = tdf['r_mult'].sum()
                gross_w = sum(tdf[tdf['result'] == 'WIN']['r_mult'])
                gross_l = abs(sum(tdf[tdf['result'] == 'LOSS']['r_mult']))
                pf = gross_w / max(0.001, gross_l)
                ticker_results.append({
                    "min_zone": mz,
                    "ticker": name,
                    "trades": len(tdf),
                    "wins": wins,
                    "losses": losses,
                    "win_rate": wr,
                    "net_r": net_r,
                    "pf": pf
                })
                all_trades.append(tdf)
                
        comb = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        tot_t = len(comb)
        tot_w = sum(comb['result'] == 'WIN')
        tot_l = sum(comb['result'] == 'LOSS')
        comb_wr = (tot_w / tot_t) * 100.0 if tot_t > 0 else 0
        comb_net_r = comb['r_mult'].sum() if tot_t > 0 else 0
        comb_gw = sum(comb[comb['result'] == 'WIN']['r_mult']) if tot_t > 0 else 0
        comb_gl = abs(sum(comb[comb['result'] == 'LOSS']['r_mult'])) if tot_t > 0 else 1
        comb_pf = comb_gw / max(0.001, comb_gl)
        exp = comb_net_r / tot_t if tot_t > 0 else 0
        
        print(f"\n[MinZone: {mz:3.1f} pips] COMBINED PORTFOLIO: {tot_t:,} Trades | {comb_wr:.1f}% WR | Net: {comb_net_r:+,.1f}R | PF: {comb_pf:.2f} | Exp: {exp:+.3f}R")
        for tr in ticker_results:
            print(f"   {tr['ticker']:6s}: {tr['trades']:4d} Trades | {tr['win_rate']:5.1f}% WR | Net: {tr['net_r']:+6.1f}R | PF: {tr['pf']:.2f}")

    # For the optimal min_zone (5.0 pips), export master trade history and generate dashboard
    best_mz = 5.0
    master_trades = []
    for name, path in tickers.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df_sub = df.iloc[-1000000:].copy().reset_index(drop=True)
            tdf = backtest_ticker_spread_beid(df_sub, name, spread_pips=1.6, min_zone_pips=best_mz)
            master_trades.append(tdf)
            
    full_portfolio = pd.concat(master_trades, ignore_index=True)
    full_portfolio['fill_time_dt'] = pd.to_datetime(full_portfolio['fill_time'])
    full_portfolio = full_portfolio.sort_values('fill_time_dt').reset_index(drop=True)
    full_portfolio['cum_r'] = full_portfolio['r_mult'].cumsum()
    
    out_csv = "/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/beid_4forex_portfolio_spread_trades.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    full_portfolio.to_csv(out_csv, index=False)
    print(f"\nSaved {len(full_portfolio):,} portfolio trades to {out_csv}.")
    
    return full_portfolio

if __name__ == "__main__":
    run_portfolio_backtest()
