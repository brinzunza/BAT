import os
import sys
import numpy as np
import pandas as pd
import time

def get_pip_size(ticker):
    if "JPY" in ticker:
        return 0.01
    return 0.0001

def resample_ohlc_fast(df_1m, timeframe_mins):
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

def compute_htf_trend(df_htf):
    opens = df_htf['open'].values.astype(np.float64)
    highs = df_htf['high'].values.astype(np.float64)
    lows = df_htf['low'].values.astype(np.float64)
    closes = df_htf['close'].values.astype(np.float64)
    n = len(opens)
    
    trend = np.zeros(n, dtype=np.int8)
    curr_trend = 1
    pHH = opens[0]; pLL = opens[0]
    last_cHL = opens[0]; last_cLH = opens[0]
    
    for i in range(n):
        cl = closes[i]; h = highs[i]; l = lows[i]
        if curr_trend == 1:
            if h > pHH: pHH = h
            if cl < last_cHL:
                curr_trend = -1; pLL = l; last_cLH = pHH
        else:
            if l < pLL: pLL = l
            if cl > last_cLH:
                curr_trend = 1; pHH = h; last_cHL = pLL
        trend[i] = curr_trend
        
    df_res = df_htf.copy()
    df_res['htf_trend'] = trend
    time_col = 'time' if 'time' in df_res.columns else 'date'
    df_res['time_dt'] = pd.to_datetime(df_res[time_col])
    return df_res[['time_dt', 'htf_trend']]

class BEIDUltimateEngine:
    def __init__(self,
                 entry_pct=0.50,
                 rr=1.0,
                 continuation_only=False, # If True, only trade BUY in uptrend, SELL in downtrend (ignore Flips)
                 htf_filter=False,
                 session_filter=False,
                 allowed_hours=(7, 17),
                 min_zone_pips=10.0,
                 spread_pips=1.6,
                 max_touches=1,
                 max_unfilled_bars=2880,
                 max_active_zones=25):
        
        self.entry_pct = entry_pct
        self.rr = rr
        self.continuation_only = continuation_only
        self.htf_filter = htf_filter
        self.session_filter = session_filter
        self.allowed_hours = allowed_hours
        self.min_zone_pips = min_zone_pips
        self.spread_pips = spread_pips
        self.max_touches = max_touches
        self.max_unfilled_bars = max_unfilled_bars
        self.max_active_zones = max_active_zones

    def backtest(self, df_ltf, ticker_name, df_htf_trend=None):
        pip_size = get_pip_size(ticker_name)
        time_col = 'date' if 'date' in df_ltf.columns else 'time'
        
        df = df_ltf.copy()
        df['time_dt'] = pd.to_datetime(df[time_col])
        
        if self.htf_filter and df_htf_trend is not None:
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
        
        if self.session_filter:
            hours = pd.to_datetime(dates).hour.values
            in_session = (hours >= self.allowed_hours[0]) & (hours <= self.allowed_hours[1])
        else:
            in_session = np.ones(n, dtype=bool)
            
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
        
        for i in range(n):
            o, h, l, cl = opens[i], highs[i], lows[i], closes[i]
            is_green = cl >= o; is_red = cl < o
            
            # 1. Update Active Zones
            retained = []
            for z in active_zones:
                side, entry, tp, sl = z['side'], z['entry'], z['tp'], z['sl']
                risk_pips = z['risk_pips']
                reward_pips = z['reward_pips']
                
                if not z['filled']:
                    if side == 1:
                        if l <= sl: continue # Invalidation
                        if l <= entry:
                            z['filled'] = True; z['fill_idx'] = i; z['touch_count'] += 1
                            retained.append(z)
                        elif i - z['created_idx'] <= self.max_unfilled_bars:
                            retained.append(z)
                    elif side == -1:
                        if h >= sl: continue # Invalidation
                        if h >= entry:
                            z['filled'] = True; z['fill_idx'] = i; z['touch_count'] += 1
                            retained.append(z)
                        elif i - z['created_idx'] <= self.max_unfilled_bars:
                            retained.append(z)
                else:
                    if i > z['fill_idx']:
                        resolved, win = False, False
                        if side == 1:
                            hit_tp = (h >= tp) or (cl > tp)
                            hit_sl = (l <= sl) or (cl < sl)
                            if hit_tp and hit_sl:
                                resolved = True; win = (cl >= entry)
                            elif hit_tp:
                                resolved = True; win = True
                            elif hit_sl:
                                resolved = True; win = False
                        else:
                            hit_tp = (l <= tp) or (cl < tp)
                            hit_sl = (h >= sl) or (cl > sl)
                            if hit_tp and hit_sl:
                                resolved = True; win = (cl <= entry)
                            elif hit_tp:
                                resolved = True; win = True
                            elif hit_sl:
                                resolved = True; win = False
                                
                        if resolved:
                            if win:
                                net_pips = reward_pips - self.spread_pips
                                net_r = net_pips / max(1e-4, risk_pips)
                            else:
                                net_pips = -(risk_pips + self.spread_pips)
                                net_r = net_pips / max(1e-4, risk_pips)
                                
                            trades.append({
                                'ticker': ticker_name,
                                'side': 'BUY' if side == 1 else 'SELL',
                                'type': z['type'],
                                'created_time': dates[z['created_idx']],
                                'fill_time': dates[z['fill_idx']],
                                'exit_time': dates[i],
                                'zone_pips': z['zone_pips'],
                                'risk_pips': risk_pips,
                                'reward_pips': reward_pips,
                                'touch_count': z['touch_count'],
                                'win': win,
                                'net_pips': net_pips,
                                'net_r': net_r,
                                'raw_r': self.rr if win else -1.0,
                                'duration_bars': i - z['fill_idx']
                            })
                            
                            if win and z['touch_count'] < self.max_touches:
                                z['filled'] = False; z['fill_idx'] = -1; z['created_idx'] = i
                                retained.append(z)
                        else:
                            retained.append(z)
            active_zones = retained
            
            # 2. State Machine Zone Creation
            if trend == 1: # UPTREND
                if not in_pullback:
                    if h > pHH: pHH = h; pHH_idx = i
                    if not np.isnan(last_cHL) and cl < last_cHL:
                        trend = -1
                        top = pHH; bottom = last_cHL
                        zone_pips = (top - bottom) / pip_size
                        
                        pass_filters = (not self.continuation_only)
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != -1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = top - (top - bottom) * self.entry_pct
                            sl = top
                            risk_pips = (sl - entry) / pip_size
                            tp = entry - (risk_pips * pip_size * self.rr)
                            reward_pips = (entry - tp) / pip_size
                            
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': -1, 'type': 'SELL_FLIP',
                                'top': top, 'bottom': bottom, 'entry': entry, 'tp': tp, 'sl': sl,
                                'zone_pips': zone_pips, 'risk_pips': risk_pips, 'reward_pips': reward_pips,
                                'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                            })
                        pLL = l; pLL_idx = i; cLH = top; last_cLH = np.nan; in_pullback = False
                    elif is_red and not np.isnan(prev_green_open) and cl < prev_green_open:
                        in_pullback = True
                        pullback_start_idx = i
                        cHH = pHH; cHH_idx = pHH_idx
                        pHL = l; pHL_idx = i
                else: # Inside Pullback
                    if l < pHL: pHL = l; pHL_idx = i
                    floor_to_check = last_cHL if not np.isnan(last_cHL) else opens[0]
                    if cl < floor_to_check:
                        trend = -1
                        top = cHH; bottom = floor_to_check
                        zone_pips = (top - bottom) / pip_size
                        
                        pass_filters = (not self.continuation_only)
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != -1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = top - (top - bottom) * self.entry_pct
                            sl = top
                            risk_pips = (sl - entry) / pip_size
                            tp = entry - (risk_pips * pip_size * self.rr)
                            reward_pips = (entry - tp) / pip_size
                            
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': -1, 'type': 'SELL_FLIP',
                                'top': top, 'bottom': bottom, 'entry': entry, 'tp': tp, 'sl': sl,
                                'zone_pips': zone_pips, 'risk_pips': risk_pips, 'reward_pips': reward_pips,
                                'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                            })
                        pLL = l; pLL_idx = i; cLH = top; last_cLH = np.nan; in_pullback = False
                    elif cl > cHH: # Continuation BUY ZONE
                        in_pullback = False
                        cHL = pHL; cHL_idx = pHL_idx; last_cHL = cHL
                        top = cHH; bottom = cHL
                        zone_pips = (top - bottom) / pip_size
                        
                        pass_filters = True
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != 1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = bottom + (top - bottom) * self.entry_pct
                            sl = bottom
                            risk_pips = (entry - sl) / pip_size
                            tp = entry + (risk_pips * pip_size * self.rr)
                            reward_pips = (tp - entry) / pip_size
                            
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': 1, 'type': 'BUY',
                                'top': top, 'bottom': bottom, 'entry': entry, 'tp': tp, 'sl': sl,
                                'zone_pips': zone_pips, 'risk_pips': risk_pips, 'reward_pips': reward_pips,
                                'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                            })
                        pHH = h; pHH_idx = i
                    elif i - pullback_start_idx > 500:
                        in_pullback = False
                        pHH = h; pHH_idx = i
                        
            elif trend == -1: # DOWNTREND
                if not in_pullback:
                    if l < pLL: pLL = l; pLL_idx = i
                    if not np.isnan(last_cLH) and cl > last_cLH:
                        trend = 1
                        top = last_cLH; bottom = pLL
                        zone_pips = (top - bottom) / pip_size
                        
                        pass_filters = (not self.continuation_only)
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != 1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = bottom + (top - bottom) * self.entry_pct
                            sl = bottom
                            risk_pips = (entry - sl) / pip_size
                            tp = entry + (risk_pips * pip_size * self.rr)
                            reward_pips = (tp - entry) / pip_size
                            
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': 1, 'type': 'BUY_FLIP',
                                'top': top, 'bottom': bottom, 'entry': entry, 'tp': tp, 'sl': sl,
                                'zone_pips': zone_pips, 'risk_pips': risk_pips, 'reward_pips': reward_pips,
                                'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                            })
                        pHH = h; pHH_idx = i; cHL = bottom; last_cHL = np.nan; in_pullback = False
                    elif is_green and not np.isnan(prev_red_open) and cl > prev_red_open:
                        in_pullback = True
                        pullback_start_idx = i
                        cLL = pLL; cLL_idx = pLL_idx
                        pLH = h; pLH_idx = i
                else: # Inside Pullback
                    if h > pLH: pLH = h; pLH_idx = i
                    ceiling_to_check = last_cLH if not np.isnan(last_cLH) else opens[0]
                    if cl > ceiling_to_check:
                        trend = 1
                        top = ceiling_to_check; bottom = cLL
                        zone_pips = (top - bottom) / pip_size
                        
                        pass_filters = (not self.continuation_only)
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != 1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = bottom + (top - bottom) * self.entry_pct
                            sl = bottom
                            risk_pips = (entry - sl) / pip_size
                            tp = entry + (risk_pips * pip_size * self.rr)
                            reward_pips = (tp - entry) / pip_size
                            
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': 1, 'type': 'BUY_FLIP',
                                'top': top, 'bottom': bottom, 'entry': entry, 'tp': tp, 'sl': sl,
                                'zone_pips': zone_pips, 'risk_pips': risk_pips, 'reward_pips': reward_pips,
                                'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                            })
                        pHH = h; pHH_idx = i; cHL = bottom; last_cHL = np.nan; in_pullback = False
                    elif cl < cLL: # Continuation SELL ZONE
                        in_pullback = False
                        cLH = pLH; cLH_idx = pLH_idx; last_cLH = cLH
                        top = cLH; bottom = cLL
                        zone_pips = (top - bottom) / pip_size
                        
                        pass_filters = True
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != -1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = top - (top - bottom) * self.entry_pct
                            sl = top
                            risk_pips = (sl - entry) / pip_size
                            tp = entry - (risk_pips * pip_size * self.rr)
                            reward_pips = (entry - tp) / pip_size
                            
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': -1, 'type': 'SELL',
                                'top': top, 'bottom': bottom, 'entry': entry, 'tp': tp, 'sl': sl,
                                'zone_pips': zone_pips, 'risk_pips': risk_pips, 'reward_pips': reward_pips,
                                'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                            })
                        pLL = l; pLL_idx = i
                    elif i - pullback_start_idx > 500:
                        in_pullback = False
                        pLL = l; pLL_idx = i
                        
            if is_green: prev_green_open = o
            elif is_red: prev_red_open = o
            
        return pd.DataFrame(trades)

def run_sweep():
    tickers = {
        "EURUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/eurusd_1min_2000_2026_master.csv",
        "GBPUSD": "/Users/brunoinzunza/Documents/GitHub/trading-data/gbpusd_1min_2000_2026_master.csv",
        "USDJPY": "/Users/brunoinzunza/Documents/GitHub/trading-data/usdjpy_1min_2000_2026_master.csv",
        "EURGBP": "/Users/brunoinzunza/Documents/GitHub/trading-data/lse_eurgbp_1min_full.csv"
    }
    
    print("Loading data across 4 forex tickers...")
    raw_dfs = {}
    for k, v in tickers.items():
        if os.path.exists(v):
            raw_dfs[k] = pd.read_csv(v)
            
    # Test combinations:
    # 1. Base 3M, 15p, 1:1 RR
    # 2. Continuation Only (no flips)
    # 3. Continuation Only + 1H HTF Alignment
    # 4. Continuation Only + London/NY Session (07-17)
    # 5. Continuation Only + 1H HTF + London/NY Session
    # 6. Continuation Only + 1H HTF + London/NY Session + 20p zone
    # 7. Continuation Only + 1H HTF + London/NY Session + 20p zone + 1:1.5 RR
    # 8. Same on 5M bars
    # 9. Same on 15M bars
    
    test_suite = [
        {'name': '1. Raw 3M (All Zones, 15p)', 'tf': 3, 'cont_only': False, 'htf': None, 'sess': False, 'min_p': 15.0, 'rr': 1.0, 'touch': 1},
        {'name': '2. Trend Continuation Only (3M, 15p)', 'tf': 3, 'cont_only': True, 'htf': None, 'sess': False, 'min_p': 15.0, 'rr': 1.0, 'touch': 1},
        {'name': '3. Continuation + 1H Trend Alignment (3M, 15p)', 'tf': 3, 'cont_only': True, 'htf': 60, 'sess': False, 'min_p': 15.0, 'rr': 1.0, 'touch': 1},
        {'name': '4. Continuation + London/NY Session (3M, 15p)', 'tf': 3, 'cont_only': True, 'htf': None, 'sess': True, 'min_p': 15.0, 'rr': 1.0, 'touch': 1},
        {'name': '5. Continuation + 1H Trend + London/NY (3M, 15p)', 'tf': 3, 'cont_only': True, 'htf': 60, 'sess': True, 'min_p': 15.0, 'rr': 1.0, 'touch': 1},
        {'name': '6. Full Confluence 3M (1H Trend + Session + 20p)', 'tf': 3, 'cont_only': True, 'htf': 60, 'sess': True, 'min_p': 20.0, 'rr': 1.0, 'touch': 1},
        {'name': '7. Full Confluence 3M + 2 Touches (20p)', 'tf': 3, 'cont_only': True, 'htf': 60, 'sess': True, 'min_p': 20.0, 'rr': 1.0, 'touch': 2},
        {'name': '8. Full Confluence 5M Bars (1H Trend + Session + 20p)', 'tf': 5, 'cont_only': True, 'htf': 60, 'sess': True, 'min_p': 20.0, 'rr': 1.0, 'touch': 1},
        {'name': '9. Full Confluence 15M Bars (4H Trend + Session + 25p)', 'tf': 15, 'cont_only': True, 'htf': 240, 'sess': True, 'min_p': 25.0, 'rr': 1.5, 'touch': 1},
        {'name': '10. Alpha King: 5M Bars + 1H Trend + Session + 20p + 1:1.5 RR', 'tf': 5, 'cont_only': True, 'htf': 60, 'sess': True, 'min_p': 20.0, 'rr': 1.5, 'touch': 2},
    ]
    
    results = []
    print("\n" + "=" * 105)
    print("RUNNING SYSTEMATIC FULL-HISTORY ALPHA SEARCH (2000-2026)...")
    print("=" * 105)
    
    for item in test_suite:
        t0 = time.time()
        engine = BEIDUltimateEngine(
            entry_pct=0.50,
            rr=item['rr'],
            continuation_only=item['cont_only'],
            htf_filter=(item['htf'] is not None),
            session_filter=item['sess'],
            allowed_hours=(7, 17),
            min_zone_pips=item['min_p'],
            spread_pips=1.6,
            max_touches=item['touch']
        )
        
        trades = []
        for name, df_1m in raw_dfs.items():
            df_ltf = resample_ohlc_fast(df_1m, item['tf'])
            df_htf_trend = None
            if item['htf'] is not None:
                df_htf = resample_ohlc_fast(df_1m, item['htf'])
                df_htf_trend = compute_htf_trend(df_htf)
            t_df = engine.backtest(df_ltf, name, df_htf_trend=df_htf_trend)
            trades.append(t_df)
            
        full = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
        if len(full) == 0:
            print(f"{item['name']:<55} | 0 trades")
            continue
            
        full['date'] = pd.to_datetime(full['exit_time'])
        
        # Performance splits
        pre_2021 = full[full['date'] < '2021-01-01']
        holdout = full[full['date'] >= '2026-04-01']
        
        tot_w = sum(full['win'])
        tot_wr = (tot_w / len(full)) * 100.0
        tot_pips = full['net_pips'].sum()
        tot_r = full['net_r'].sum()
        
        w_p = full[full['net_pips'] > 0]['net_pips'].sum()
        l_p = abs(full[full['net_pips'] <= 0]['net_pips'].sum())
        pf = w_p / max(1e-6, l_p)
        
        # Pre-2021 PF
        pre_w_p = pre_2021[pre_2021['net_pips'] > 0]['net_pips'].sum()
        pre_l_p = abs(pre_2021[pre_2021['net_pips'] <= 0]['net_pips'].sum())
        pre_pf = pre_w_p / max(1e-6, pre_l_p) if len(pre_2021) > 0 else 0.0
        
        # Holdout PF
        h_w_p = holdout[holdout['net_pips'] > 0]['net_pips'].sum()
        h_l_p = abs(holdout[holdout['net_pips'] <= 0]['net_pips'].sum())
        h_pf = h_w_p / max(1e-6, h_l_p) if len(holdout) > 0 else 0.0
        h_wr = (sum(holdout['win'])/len(holdout))*100.0 if len(holdout) > 0 else 0.0
        
        span_yrs = max(0.1, (full['date'].max() - full['date'].min()).total_seconds() / (365.25 * 86400))
        tpy = len(full) / span_yrs
        
        results.append({
            'name': item['name'],
            'trades': len(full),
            'tpy': round(tpy, 0),
            'win_rate': round(tot_wr, 1),
            'pf': round(pf, 2),
            'net_pips': round(tot_pips, 1),
            'net_r': round(tot_r, 1),
            'pre_pf': round(pre_pf, 2),
            'holdout_trades': len(holdout),
            'holdout_wr': round(h_wr, 1),
            'holdout_pf': round(h_pf, 2)
        })
        
        print(f"{item['name']:<55} | Trades: {len(full):<6,} ({tpy:>4.0f}/yr) | WR: {tot_wr:>4.1f}% | PF: {pf:>4.2f} | Pre PF: {pre_pf:>4.2f} | Holdout PF: {h_pf:>4.2f} ({time.time()-t0:.1f}s)")
        
    df_out = pd.DataFrame(results).sort_values('pf', ascending=False).reset_index(drop=True)
    df_out.to_csv("dtfxStrategy/research_results/beid_ultimate_alpha_sweep.csv", index=False)
    print("\nSaved sweep results to dtfxStrategy/research_results/beid_ultimate_alpha_sweep.csv")

if __name__ == "__main__":
    run_sweep()
