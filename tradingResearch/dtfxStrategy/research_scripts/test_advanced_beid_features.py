import os
import sys
import numpy as np
import pandas as pd
import time
from beid_real_engine import resample_ohlc_fast, compute_htf_trend, get_pip_size

def backtest_advanced_beid(df_ltf, ticker_name, df_htf_trend=None,
                           entry_depth=0.618, # 61.8% deep discount entry
                           rr_target=1.5,     # 1:1.5 RR target
                           min_zone_pips=20.0,
                           spread_pips=1.6,
                           use_be_trail=True, # Move SL to BE at +0.5R
                           max_touches=2,
                           session_only=True):
    
    pip_size = get_pip_size(ticker_name)
    time_col = 'date' if 'date' in df_ltf.columns else 'time'
    
    df = df_ltf.copy()
    df['time_dt'] = pd.to_datetime(df[time_col])
    
    if df_htf_trend is not None:
        df = pd.merge_asof(df.sort_values('time_dt'), df_htf_trend.sort_values('time_dt'), on='time_dt', direction='backward')
        htf_arr = df['htf_trend'].fillna(0).values.astype(np.int8)
    else:
        htf_arr = np.zeros(len(df), dtype=np.int8)
        
    dates = df['time_dt'].values
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    n = len(opens)
    
    hours = pd.to_datetime(dates).hour.values
    in_session = (hours >= 7) & (hours <= 17) if session_only else np.ones(n, dtype=bool)
    
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
        is_green = cl >= o; is_red = cl < o
        
        # 1. Update Active Zones & Open Trades
        retained = []
        for z in active_zones:
            side = z['side']
            entry = z['entry']
            tp = z['tp']
            sl = z['sl']
            risk_pips = z['risk_pips']
            reward_pips = z['reward_pips']
            
            if not z['filled']:
                if side == 1:
                    if l <= sl: continue
                    if l <= entry:
                        z['filled'] = True; z['fill_idx'] = i; z['touch_count'] += 1
                        z['be_moved'] = False
                        retained.append(z)
                    elif i - z['created_idx'] <= 2880: retained.append(z)
                elif side == -1:
                    if h >= sl: continue
                    if h >= entry:
                        z['filled'] = True; z['fill_idx'] = i; z['touch_count'] += 1
                        z['be_moved'] = False
                        retained.append(z)
                    elif i - z['created_idx'] <= 2880: retained.append(z)
            else:
                if i > z['fill_idx']:
                    resolved = False
                    win = False
                    be_hit = False
                    
                    # Trailing BE check: if price pushed 50% toward TP, move SL to Entry
                    if use_be_trail and not z['be_moved']:
                        if side == 1 and h >= (entry + (tp - entry) * 0.5):
                            z['sl'] = entry # Protected to breakeven!
                            z['be_moved'] = True
                        elif side == -1 and l <= (entry - (entry - tp) * 0.5):
                            z['sl'] = entry
                            z['be_moved'] = True
                            
                    curr_sl = z['sl']
                    
                    if side == 1: # BUY
                        hit_tp = (h >= tp) or (cl > tp)
                        hit_sl = (l <= curr_sl) or (cl < curr_sl)
                        if hit_tp and hit_sl:
                            resolved = True; win = (cl >= entry)
                        elif hit_tp:
                            resolved = True; win = True
                        elif hit_sl:
                            resolved = True
                            win = False
                            if z['be_moved'] and curr_sl == entry:
                                be_hit = True
                    else: # SELL
                        hit_tp = (l <= tp) or (cl < tp)
                        hit_sl = (h >= curr_sl) or (cl > curr_sl)
                        if hit_tp and hit_sl:
                            resolved = True; win = (cl <= entry)
                        elif hit_tp:
                            resolved = True; win = True
                        elif hit_sl:
                            resolved = True
                            win = False
                            if z['be_moved'] and curr_sl == entry:
                                be_hit = True
                                
                    if resolved:
                        if win:
                            net_pips = reward_pips - spread_pips
                            net_r = net_pips / max(1e-4, risk_pips)
                        elif be_hit:
                            net_pips = -spread_pips # Breakeven exit (just spread cost)
                            net_r = -spread_pips / max(1e-4, risk_pips)
                        else:
                            net_pips = -(risk_pips + spread_pips)
                            net_r = net_pips / max(1e-4, risk_pips)
                            
                        trades.append({
                            'ticker': ticker_name,
                            'side': 'BUY' if side == 1 else 'SELL',
                            'fill_time': dates[z['fill_idx']],
                            'exit_time': dates[i],
                            'zone_pips': z['zone_pips'],
                            'win': win,
                            'be_hit': be_hit,
                            'net_pips': net_pips,
                            'net_r': net_r,
                            'raw_r': rr_target if win else (0.0 if be_hit else -1.0)
                        })
                        if win and z['touch_count'] < max_touches:
                            z['filled'] = False; z['fill_idx'] = -1; z['created_idx'] = i
                            retained.append(z)
                    else:
                        retained.append(z)
        active_zones = retained
        
        # 2. State Machine Zone Creation (Continuation Preferred)
        if trend == 1: # UPTREND
            if not in_pullback:
                if h > pHH: pHH = h; pHH_idx = i
                if not np.isnan(last_cHL) and cl < last_cHL:
                    trend = -1
                    pLL = l; pLL_idx = i; cLH = pHH; last_cLH = np.nan; in_pullback = False
                elif is_red and not np.isnan(prev_green_open) and cl < prev_green_open:
                    in_pullback = True
                    pullback_start_idx = i
                    cHH = pHH; cHH_idx = pHH_idx
                    pHL = l; pHL_idx = i
            else:
                if l < pHL: pHL = l; pHL_idx = i
                floor_to_check = last_cHL if not np.isnan(last_cHL) else opens[0]
                if cl < floor_to_check:
                    trend = -1
                    pLL = l; pLL_idx = i; cLH = cHH; last_cLH = np.nan; in_pullback = False
                elif cl > cHH: # Continuation BUY
                    in_pullback = False
                    cHL = pHL; cHL_idx = pHL_idx; last_cHL = cHL
                    top = cHH; bottom = cHL
                    zone_pips = (top - bottom) / pip_size
                    
                    pass_filters = True
                    if zone_pips < min_zone_pips: pass_filters = False
                    if df_htf_trend is not None and htf_arr[i] != 1: pass_filters = False
                    if session_only and not in_session[i]: pass_filters = False
                    
                    if pass_filters and len(active_zones) < max_active:
                        entry = bottom + (top - bottom) * (1.0 - entry_depth) # e.g. 61.8% deep discount from top = entry at 38.2% from bottom
                        sl = bottom
                        risk_pips = (entry - sl) / pip_size
                        tp = entry + (risk_pips * pip_size * rr_target)
                        reward_pips = (tp - entry) / pip_size
                        
                        active_zones.append({
                            'id': len(trades)+len(active_zones)+1, 'side': 1, 'type': 'BUY',
                            'top': top, 'bottom': bottom, 'entry': entry, 'tp': tp, 'sl': sl,
                            'zone_pips': zone_pips, 'risk_pips': risk_pips, 'reward_pips': reward_pips,
                            'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                        })
                    pHH = h; pHH_idx = i
                elif i - pullback_start_idx > 500:
                    in_pullback = False; pHH = h; pHH_idx = i
                    
        elif trend == -1: # DOWNTREND
            if not in_pullback:
                if l < pLL: pLL = l; pLL_idx = i
                if not np.isnan(last_cLH) and cl > last_cLH:
                    trend = 1
                    pHH = h; pHH_idx = i; cHL = pLL; last_cHL = np.nan; in_pullback = False
                elif is_green and not np.isnan(prev_red_open) and cl > prev_red_open:
                    in_pullback = True
                    pullback_start_idx = i
                    cLL = pLL; cLL_idx = pLL_idx
                    pLH = h; pLH_idx = i
            else:
                if h > pLH: pLH = h; pLH_idx = i
                ceiling_to_check = last_cLH if not np.isnan(last_cLH) else opens[0]
                if cl > ceiling_to_check:
                    trend = 1
                    pHH = h; pHH_idx = i; cHL = cLL; last_cHL = np.nan; in_pullback = False
                elif cl < cLL: # Continuation SELL
                    in_pullback = False
                    cLH = pLH; cLH_idx = pLH_idx; last_cLH = cLH
                    top = cLH; bottom = cLL
                    zone_pips = (top - bottom) / pip_size
                    
                    pass_filters = True
                    if zone_pips < min_zone_pips: pass_filters = False
                    if df_htf_trend is not None and htf_arr[i] != -1: pass_filters = False
                    if session_only and not in_session[i]: pass_filters = False
                    
                    if pass_filters and len(active_zones) < max_active:
                        entry = top - (top - bottom) * (1.0 - entry_depth)
                        sl = top
                        risk_pips = (sl - entry) / pip_size
                        tp = entry - (risk_pips * pip_size * rr_target)
                        reward_pips = (entry - tp) / pip_size
                        
                        active_zones.append({
                            'id': len(trades)+len(active_zones)+1, 'side': -1, 'type': 'SELL',
                            'top': top, 'bottom': bottom, 'entry': entry, 'tp': tp, 'sl': sl,
                            'zone_pips': zone_pips, 'risk_pips': risk_pips, 'reward_pips': reward_pips,
                            'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                        })
                    pLL = l; pLL_idx = i
                elif i - pullback_start_idx > 500:
                    in_pullback = False; pLL = l; pLL_idx = i
                    
        if is_green: prev_green_open = o
        elif is_red: prev_red_open = o
        
    return pd.DataFrame(trades)

if __name__ == "__main__":
    tickers = {
        "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
        "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
        "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
        "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv"
    }
    
    print("=" * 105)
    print("TESTING ADVANCED CONFLUENCE + TRAILING BE + DEEP DISCOUNT ENTRIES (2000-2026)")
    print("=" * 105)
    
    for tf, htf, depth, rr, be in [
        (3, 60, 0.50, 1.0, False),
        (3, 60, 0.50, 1.0, True),
        (3, 60, 0.618, 1.5, False),
        (3, 60, 0.618, 1.5, True),
        (5, 60, 0.50, 1.0, True),
        (5, 60, 0.618, 1.5, True),
        (15, 240, 0.50, 1.5, True),
        (15, 240, 0.618, 2.0, True)
    ]:
        all_t = []
        for name, path in tickers.items():
            if not os.path.exists(path): continue
            df_1m = pd.read_csv(path)
            df_ltf = resample_ohlc_fast(df_1m, tf)
            df_htf = resample_ohlc_fast(df_1m, htf)
            htf_trend = compute_htf_trend(df_htf)
            t_df = backtest_advanced_beid(df_ltf, name, df_htf_trend=htf_trend,
                                          entry_depth=depth, rr_target=rr,
                                          min_zone_pips=20.0, spread_pips=1.6,
                                          use_be_trail=be, max_touches=2, session_only=True)
            all_t.append(t_df)
            
        full = pd.concat(all_t, ignore_index=True) if all_t else pd.DataFrame()
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
        
        pre_pf = pre[pre['net_pips'] > 0]['net_pips'].sum() / max(1e-6, abs(pre[pre['net_pips'] <= 0]['net_pips'].sum()))
        h_pf = holdout[holdout['net_pips'] > 0]['net_pips'].sum() / max(1e-6, abs(holdout[holdout['net_pips'] <= 0]['net_pips'].sum())) if len(holdout) > 0 else 0.0
        
        span_yrs = max(0.1, (full['date'].max() - full['date'].min()).total_seconds() / (365.25 * 86400))
        tpy = len(full) / span_yrs
        
        desc = f"TF:{tf:2d}M | HTF:{htf:3d}M | Entry:{depth*100:4.1f}% | RR:1:{rr:.1f} | BE_Trail:{str(be):<5}"
        print(f"{desc} | Trades: {len(full):<6,} ({tpy:>4.0f}/yr) | WR: {tot_wr:>4.1f}% | PF: {pf:>4.2f} | Pre PF: {pre_pf:>4.2f} | Holdout PF: {h_pf:>4.2f} | Net R: {tot_r:+,.1f}R")
