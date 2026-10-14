import os
import pandas as pd
import numpy as np
import json
import time

def load_eurusd_master():
    csv_path = "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv"
    print(f"Loading EURUSD master dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    date_col = 'date' if 'date' in df.columns else 'time'
    df['date_dt'] = pd.to_datetime(df[date_col])
    return df, date_col

def run_beid_spread_backtest(df_subset, date_col, spread_pips=1.6, min_zone_pips=5.0, pip_size=0.0001):
    spread_val = spread_pips * pip_size
    half_spread = spread_val / 2.0
    min_zone_val = min_zone_pips * pip_size
    
    dates = df_subset[date_col].values
    opens = df_subset['open'].values.astype(np.float64)
    highs = df_subset['high'].values.astype(np.float64)
    lows = df_subset['low'].values.astype(np.float64)
    closes = df_subset['close'].values.astype(np.float64)
    
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
                # Buy Limit fill: price must reach entry (mid + half_spread)
                if side == 1 and l <= entry:
                    z['filled'] = True
                    z['fill_idx'] = i
                    retained.append(z)
                # Sell Limit fill: price must reach entry (mid - half_spread)
                elif side == -1 and h >= entry:
                    z['filled'] = True
                    z['fill_idx'] = i
                    retained.append(z)
                else:
                    if i - z['created_idx'] > 2880: # expire after 48h
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
                        # SL hit if low reaches SL (bottom)
                        hit_sl = l <= sl or cl < sl
                        # TP hit if high reaches TP (top - half_spread)
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
                        # SL hit if high reaches SL (top)
                        hit_sl = h >= sl or cl > sl
                        # TP hit if low reaches TP (bottom + half_spread)
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
                
                # Check min zone size filter
                if zone_height >= min_zone_val:
                    mid = (top + bottom) / 2.0
                    entry = mid - half_spread
                    tp = bottom + half_spread
                    sl = top
                    risk = sl - entry
                    reward = entry - tp
                    
                    if reward > 0 and len(active_zones) < 20:
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
                            entry = mid + half_spread
                            tp = top - half_spread
                            sl = bottom
                            risk = entry - sl
                            reward = tp - entry
                            
                            if reward > 0 and len(active_zones) < 20:
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
                    entry = mid + half_spread
                    tp = top - half_spread
                    sl = bottom
                    risk = entry - sl
                    reward = tp - entry
                    
                    if reward > 0 and len(active_zones) < 20:
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
                            entry = mid - half_spread
                            tp = bottom + half_spread
                            sl = top
                            risk = sl - entry
                            reward = entry - tp
                            
                            if reward > 0 and len(active_zones) < 20:
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

def sweep_min_zone_sizes():
    df, date_col = load_eurusd_master()
    # 2026 data
    df_2026 = df[df['date_dt'].dt.year == 2026].copy().reset_index(drop=True)
    # Also 2024-2026 deep dataset
    df_3yr = df[df['date_dt'].dt.year >= 2023].copy().reset_index(drop=True)
    
    print(f"Loaded 2026 ({len(df_2026):,} bars) and 2023-2026 ({len(df_3yr):,} bars).")
    
    min_zone_candidates = [3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0]
    
    print("\n" + "="*85)
    print("MINIMUM ZONE SIZE SWEEP (SPREAD = 1.6 PIPS, EUR/USD 2026):")
    print("="*85)
    
    sweep_results = []
    
    for mz in min_zone_candidates:
        tdf = run_beid_spread_backtest(df_2026, date_col, spread_pips=1.6, min_zone_pips=mz)
        if len(tdf) == 0:
            continue
        n_t = len(tdf)
        wins = sum(tdf['result'] == 'WIN')
        losses = sum(tdf['result'] == 'LOSS')
        wr = (wins / n_t) * 100.0
        net_r = tdf['r_mult'].sum()
        gross_win = sum(tdf[tdf['result'] == 'WIN']['r_mult'])
        gross_loss = abs(sum(tdf[tdf['result'] == 'LOSS']['r_mult']))
        pf = gross_win / max(0.001, gross_loss)
        avg_rr = tdf['raw_rr'].mean()
        exp = net_r / n_t
        
        sweep_results.append({
            "min_zone_pips": mz,
            "trades": n_t,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wr, 1),
            "avg_rr": round(avg_rr, 2),
            "net_r": round(net_r, 1),
            "profit_factor": round(pf, 2),
            "expectancy_r": round(exp, 3)
        })
        
        print(f"MinZone: {mz:4.1f} pips | Trades: {n_t:4d} | WR: {wr:5.1f}% | Avg RR: {avg_rr:.2f} | Net R: {net_r:+6.1f}R | PF: {pf:4.2f} | Exp: {exp:+.3f}R")
        
    sweep_df = pd.DataFrame(sweep_results)
    
    # Save sweep results
    sweep_df.to_csv("/Users/brunoinzunza/Documents/GitHub/tradingResearch/dtfxStrategy/research_results/min_zone_sweep_results.csv", index=False)
    
    # Run optimal min zone size (e.g. 7.0 - 10.0 pips) across 2026 and compute Kelly & compounding
    best_mz = sweep_df.loc[sweep_df['net_r'].idxmax()]['min_zone_pips']
    print(f"\nOptimal Minimum Zone Size for 1.6 pips spread is: {best_mz:.1f} pips")
    
    # Simulate $200 account on optimal min zone size
    opt_trades = run_beid_spread_backtest(df_2026, date_col, spread_pips=1.6, min_zone_pips=best_mz)
    
    # Calculate Kelly Criterion
    p = sum(opt_trades['result'] == 'WIN') / len(opt_trades)
    q = 1.0 - p
    b = opt_trades[opt_trades['result'] == 'WIN']['r_mult'].mean() # average win payout per 1R risk
    kelly = p - (q / b)
    print(f"\n--- KELLY CRITERION ANALYSIS ---")
    print(f"Win Probability (p): {p*100:.1f}%")
    print(f"Win/Loss Ratio (b): {b:.2f}")
    print(f"Full Kelly Fraction: {kelly*100:.2f}%")
    print(f"Half Kelly (Recommended): {kelly*50:.2f}%")
    print(f"Quarter Kelly (Conservative): {kelly*25:.2f}%")
    
    # Compounding simulation on $200 account
    risk_tiers = [0.01, 0.02, 0.03, round(kelly*0.25, 3), round(kelly*0.5, 3)]
    sim_out = {}
    for r in [0.01, 0.02, 0.03, 0.05]:
        bal = 200.0
        peak = 200.0
        mdd = 0.0
        eq_curve = [bal]
        for _, tr in opt_trades.iterrows():
            pos_risk = bal * r
            pnl = pos_risk * tr['r_mult']
            bal += pnl
            if bal > peak:
                peak = bal
            dd = (peak - bal) / peak * 100.0
            if dd > mdd:
                mdd = dd
            eq_curve.append(round(bal, 2))
        sim_out[f"{int(r*100)}% Risk"] = {
            "risk_pct": f"{int(r*100)}%",
            "final_bal": round(bal, 2),
            "return_pct": round(((bal - 200.0) / 200.0)*100.0, 1),
            "max_dd": round(mdd, 1),
            "curve": eq_curve
        }
        print(f"[{int(r*100)}% Risk] Final: ${bal:,.2f} (+{((bal-200)/200)*100:.1f}%) | Max DD: {mdd:.1f}%")

    return sweep_df, opt_trades, sim_out, best_mz, kelly

if __name__ == "__main__":
    sweep_min_zone_sizes()
