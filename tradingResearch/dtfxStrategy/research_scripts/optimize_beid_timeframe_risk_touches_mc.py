import os
import sys
import time
import numpy as np
import pandas as pd
import json

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

def backtest_beid_multitouch(df, ticker_name, max_touches=1, spread_pips=1.6):
    """
    Backtests BEID with multi-touch zone rearming:
    - If a trade hits TP, the zone is rearmed for a subsequent touch (up to max_touches).
    - If a trade hits SL, the zone is permanently invalidated.
    - If price breaks through the opposite boundary before filling, zone is invalidated.
    - Entry at exact 50%, TP at exact boundary, SL at exact opposite boundary.
    - Subsequent candle rule strictly enforced (exit must be after fill_idx).
    """
    pip_size = get_pip_size(ticker_name)
    time_col = 'time' if 'time' in df.columns else 'date'
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
    prev_green_open = np.nan
    prev_red_open = np.nan
    
    active_zones = []
    trades = []
    max_active = 25
    
    for i in range(n):
        o = opens[i]; h = highs[i]; l = lows[i]; cl = closes[i]
        is_green = cl >= o; is_red = cl < o
        
        # 1. Update Active Zones
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
                if side == 1:
                    if l <= sl:
                        continue # Invalidate if price dropped below SL floor before 50% fill
                    if l <= entry:
                        z['filled'] = True
                        z['fill_idx'] = i
                        z['touch_count'] += 1
                        retained.append(z)
                    else:
                        if i - z['created_idx'] > 2880: # Expire after 48h
                            pass
                        else:
                            retained.append(z)
                elif side == -1: # Sell fill
                    if h >= sl:
                        continue # Invalidate if price rose above SL ceiling before 50% fill
                    if h >= entry:
                        z['filled'] = True
                        z['fill_idx'] = i
                        z['touch_count'] += 1
                        retained.append(z)
                    else:
                        if i - z['created_idx'] > 2880:
                            pass
                        else:
                            retained.append(z)
            else:
                # Active trade: must resolve on subsequent candle (i > z['fill_idx'])
                if i > z['fill_idx']:
                    resolved = False
                    raw_result = ""
                    
                    if side == 1: # BUY
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
                    else: # SELL
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
                            'touch_num': z['touch_count'],
                            'raw_result': raw_result,
                            'net_pips': net_pips,
                            'net_r': net_r,
                            'raw_r': 1.0 if raw_result == "WIN" else -1.0,
                            'duration_bars': i - z['fill_idx']
                        })
                        
                        # MULTI-TOUCH REARMING RULE:
                        # If trade won and touches < max_touches, re-arm zone!
                        if raw_result == "WIN" and z['touch_count'] < max_touches:
                            z['filled'] = False
                            z['fill_idx'] = -1
                            z['created_idx'] = i
                            retained.append(z)
                    else:
                        retained.append(z)
                        
        active_zones = retained
        
        # 2. State Machine & Zone Generation
        if trend == 1: # UPTREND
            if h > pHH or np.isnan(pHH):
                pHH = h; pHH_idx = i
                
            if not np.isnan(last_cHL) and cl < last_cHL:
                old_floor = last_cHL
                top_peak = pHH
                trend = -1
                cLH = top_peak; cLH_idx = pHH_idx
                last_cLH = np.nan
                pLL = l; pLL_idx = i
                cLL = np.nan; in_pullback = False
                
                top = top_peak; bottom = old_floor
                mid = (top + bottom) / 2.0
                zone_pips = (top - bottom) / pip_size
                half_pips = zone_pips / 2.0
                
                if top > bottom and len(active_zones) < max_active:
                    active_zones.append({
                        'id': len(trades) + len(active_zones) + 1,
                        'side': -1, 'type': 'SELL_FLIP',
                        'top': top, 'bottom': bottom, 'entry': mid, 'tp': bottom, 'sl': top,
                        'zone_pips': zone_pips, 'half_pips': half_pips,
                        'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                    })
            else:
                if not in_pullback and is_red and not np.isnan(prev_green_open) and cl < prev_green_open:
                    in_pullback = True
                    cHH = pHH; cHH_idx = pHH_idx
                    pHL = l; pHL_idx = i
                elif in_pullback:
                    if l < pHL: pHL = l; pHL_idx = i
                    if cl > cHH:
                        in_pullback = False
                        cHL = pHL; cHL_idx = pHL_idx; last_cHL = cHL
                        top = cHH; bottom = cHL
                        mid = (top + bottom) / 2.0
                        zone_pips = (top - bottom) / pip_size
                        half_pips = zone_pips / 2.0
                        pHH = h; pHH_idx = i
                        
                        if top > bottom and len(active_zones) < max_active:
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': 1, 'type': 'BUY',
                                'top': top, 'bottom': bottom, 'entry': mid, 'tp': top, 'sl': bottom,
                                'zone_pips': zone_pips, 'half_pips': half_pips,
                                'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                            })
                            
        elif trend == -1: # DOWNTREND
            if l < pLL or np.isnan(pLL):
                pLL = l; pLL_idx = i
                
            if not np.isnan(last_cLH) and cl > last_cLH:
                old_ceiling = last_cLH
                bottom_trough = pLL
                trend = 1
                cHL = bottom_trough; cHL_idx = pLL_idx
                last_cLH = np.nan
                pHH = h; pHH_idx = i
                cHH = np.nan; in_pullback = False
                
                top = old_ceiling; bottom = bottom_trough
                mid = (top + bottom) / 2.0
                zone_pips = (top - bottom) / pip_size
                half_pips = zone_pips / 2.0
                
                if top > bottom and len(active_zones) < max_active:
                    active_zones.append({
                        'id': len(trades) + len(active_zones) + 1,
                        'side': 1, 'type': 'BUY_FLIP',
                        'top': top, 'bottom': bottom, 'entry': mid, 'tp': top, 'sl': bottom,
                        'zone_pips': zone_pips, 'half_pips': half_pips,
                        'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                    })
            else:
                if not in_pullback and is_green and not np.isnan(prev_red_open) and cl > prev_red_open:
                    in_pullback = True
                    cLL = pLL; cLL_idx = pLL_idx
                    pLH = h; pLH_idx = i
                elif in_pullback:
                    if h > pLH: pLH = h; pLH_idx = i
                    if cl < cLL:
                        in_pullback = False
                        cLH = pLH; cLH_idx = pLH_idx; last_cLH = cLH
                        top = cLH; bottom = cLL
                        mid = (top + bottom) / 2.0
                        zone_pips = (top - bottom) / pip_size
                        half_pips = zone_pips / 2.0
                        pLL = l; pLL_idx = i
                        
                        if top > bottom and len(active_zones) < max_active:
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': -1, 'type': 'SELL',
                                'top': top, 'bottom': bottom, 'entry': mid, 'tp': bottom, 'sl': top,
                                'zone_pips': zone_pips, 'half_pips': half_pips,
                                'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                            })
                            
        if is_green: prev_green_open = o
        elif is_red: prev_red_open = o
            
    return pd.DataFrame(trades)

def fast_monte_carlo(net_r_arr, trades_per_year, risk_pct=0.03, start_equity=200.0, target=1_000_000.0, n_sims=1000, max_years=10.0):
    """
    Ultra-fast vectorized NumPy Monte Carlo simulation to reach $1,000,000 from $200.
    """
    if len(net_r_arr) < 20 or trades_per_year < 5:
        return np.nan, 0.0, 100.0, 100.0
        
    n_trades = len(net_r_arr)
    max_trades = min(int(max_years * trades_per_year), 10000)
    if max_trades < 50:
        return np.nan, 0.0, 100.0, 100.0
        
    rng = np.random.default_rng(42)
    choices = rng.choice(net_r_arr, size=(n_sims, max_trades), replace=True)
    
    mults = 1.0 + risk_pct * choices
    mults = np.clip(mults, 1e-4, 10.0) # Safety clip
    
    # Cumulative compound curves
    cum_curves = start_equity * np.cumprod(mults, axis=1) # Shape: (n_sims, max_trades)
    
    # Identify hitting target
    target_hit_mask = cum_curves >= target
    ruin_mask = cum_curves <= start_equity * 0.10
    
    days_to_target = []
    ruin_count = 0
    max_dds = []
    
    # Running peaks for drawdown
    running_peaks = np.maximum.accumulate(cum_curves, axis=1)
    dds = (running_peaks - cum_curves) / running_peaks
    max_dds_sim = np.max(dds, axis=1)
    
    for s in range(n_sims):
        hit_indices = np.where(target_hit_mask[s])[0]
        ruin_indices = np.where(ruin_mask[s])[0]
        
        first_hit = hit_indices[0] if len(hit_indices) > 0 else 999999
        first_ruin = ruin_indices[0] if len(ruin_indices) > 0 else 999999
        
        if first_hit < first_ruin and first_hit < 999999:
            days = (first_hit / trades_per_year) * 365.25
            days_to_target.append(days)
        elif first_ruin < first_hit:
            ruin_count += 1
            
    prob_reach = (len(days_to_target) / n_sims) * 100.0
    med_days = float(np.median(days_to_target)) if len(days_to_target) > 0 else np.nan
    med_max_dd = float(np.median(max_dds_sim)) * 100.0
    ruin_rate = (ruin_count / n_sims) * 100.0
    
    return med_days, prob_reach, med_max_dd, ruin_rate

def run_full_grid_search():
    tickers = {
        "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
        "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
        "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
        "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv"
    }
    
    timeframes = [1, 2, 3, 5, 10, 15, 30] # in minutes
    touch_limits = [1, 2, 3, 5] # max touches per zone
    min_pip_filters = [0.0, 4.0, 6.0, 8.0, 10.0, 15.0]
    risk_levels = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
    
    print("=" * 85)
    print("BEID MULTI-DIMENSIONAL GRID SEARCH OPTIMIZATION TO $1M (MONTE CARLO)")
    print("Variables: X min bars, X risk %, X min pip range, X max zone touches")
    print("=" * 85)
    
    raw_data = {}
    for name, path in tickers.items():
        if os.path.exists(path):
            print(f"Loading {name}...")
            df = pd.read_csv(path)
            raw_data[name] = df.iloc[-500000:].copy().reset_index(drop=True)
            
    grid_results = []
    total_combos = len(timeframes) * len(touch_limits)
    curr_combo = 0
    t0_all = time.time()
    
    for tf in timeframes:
        resampled_data = {}
        for name, df_1m in raw_data.items():
            resampled_data[name] = resample_ohlc(df_1m, tf)
            
        sample_df = list(resampled_data.values())[0]
        time_col = 'time' if 'time' in sample_df.columns else 'date'
        first_t = pd.to_datetime(sample_df[time_col].iloc[0])
        last_t = pd.to_datetime(sample_df[time_col].iloc[-1])
        span_years = max(0.1, (last_t - first_t).total_seconds() / (365.25 * 86400))
        
        for touches in touch_limits:
            curr_combo += 1
            t_sub0 = time.time()
            
            tf_trades = []
            for name, df_tf in resampled_data.items():
                t_df = backtest_beid_multitouch(df_tf, name, max_touches=touches, spread_pips=1.6)
                tf_trades.append(t_df)
                
            combined_trades = pd.concat(tf_trades, ignore_index=True) if tf_trades else pd.DataFrame()
            
            if len(combined_trades) == 0:
                continue
                
            for min_pip in min_pip_filters:
                sub = combined_trades[combined_trades['zone_pips'] >= min_pip]
                if len(sub) < 30:
                    continue
                    
                n_t = len(sub)
                trades_per_yr = n_t / span_years
                wins = sum(sub['raw_result'] == 'WIN')
                wr = (wins / n_t) * 100.0
                tot_pips = sub['net_pips'].sum()
                net_r_arr = sub['net_r'].values
                
                win_pips = sub[sub['net_pips'] > 0]['net_pips'].sum()
                loss_pips = abs(sub[sub['net_pips'] <= 0]['net_pips'].sum())
                pf = win_pips / max(1e-6, loss_pips)
                
                for r_pct in risk_levels:
                    med_days, prob_1m, med_dd, ruin_pct = fast_monte_carlo(
                        net_r_arr, trades_per_yr, risk_pct=r_pct, start_equity=200.0, target=1_000_000.0, n_sims=1000
                    )
                    
                    grid_results.append({
                        'timeframe_min': tf,
                        'max_touches': touches,
                        'min_pip_range': min_pip,
                        'risk_pct': r_pct,
                        'total_trades': n_t,
                        'trades_per_year': round(trades_per_yr, 1),
                        'raw_win_rate': round(wr, 1),
                        'profit_factor': round(pf, 2),
                        'net_pips': round(tot_pips, 1),
                        'med_days_to_1m': round(med_days, 1) if not np.isnan(med_days) else np.nan,
                        'med_months_to_1m': round(med_days / 30.42, 2) if not np.isnan(med_days) else np.nan,
                        'prob_reach_1m': round(prob_1m, 1),
                        'med_max_dd': round(med_dd, 1),
                        'ruin_rate': round(ruin_pct, 1)
                    })
                    
            print(f"[{curr_combo:02d}/{total_combos}] TF: {tf:2d}M | Touches: {touches} | Trades: {len(combined_trades):,} ({time.time()-t_sub0:.2f}s)", flush=True)
                    
    df_grid = pd.DataFrame(grid_results)
    out_csv = "dtfxStrategy/research_results/optimal_beid_mc_grid_results.csv"
    df_grid.to_csv(out_csv, index=False)
    print(f"\nEvaluated {len(df_grid):,} parameter configurations in {time.time()-t0_all:.1f}s. Saved to {out_csv}")
    
    # Filter winning configurations: prob_reach >= 70%, ruin_rate <= 15%
    valid = df_grid[(df_grid['prob_reach_1m'] >= 70.0) & (df_grid['ruin_rate'] <= 15.0)].copy()
    valid = valid.sort_values('med_days_to_1m').reset_index(drop=True)
    
    print("\n" + "=" * 105)
    print("TOP 15 FASTEST CONFIGURATIONS TO MEDIAN $1,000,000 IN MONTE CARLO ($200 START):")
    print("=" * 105)
    print(f"{'Rank':<5} | {'TF (min)':<8} | {'Touches':<8} | {'Min Pip':<8} | {'Risk %':<8} | {'Win Rate':<9} | {'PF':<6} | {'Trades/Yr':<10} | {'Med Months to $1M':<18} | {'Med Max DD':<10} | {'Prob $1M':<8}")
    print("-" * 125)
    for idx, row in valid.head(20).iterrows():
        print(f"#{idx+1:<4} | {row['timeframe_min']:<8} | {row['max_touches']:<8} | {row['min_pip_range']:<8.1f} | {row['risk_pct']*100:<7.1f}% | {row['raw_win_rate']:<8.1f}% | {row['profit_factor']:<6.2f} | {row['trades_per_year']:<10.0f} | {row['med_months_to_1m']:<18.1f} | {row['med_max_dd']:<9.1f}% | {row['prob_reach_1m']:<7.1f}%")

if __name__ == "__main__":
    run_full_grid_search()
