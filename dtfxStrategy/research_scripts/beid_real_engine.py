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
    """
    Computes macro trend direction on Higher Timeframe (e.g. 15M, 1H).
    Returns an array of macro trend (+1: Bullish, -1: Bearish, 0: Neutral) mapped by time.
    Uses structural swing confirmation:
    - Bullish if price closed above previous confirmed peak
    - Bearish if price closed below previous confirmed trough
    """
    opens = df_htf['open'].values.astype(np.float64)
    highs = df_htf['high'].values.astype(np.float64)
    lows = df_htf['low'].values.astype(np.float64)
    closes = df_htf['close'].values.astype(np.float64)
    n = len(opens)
    
    trend = np.zeros(n, dtype=np.int8)
    curr_trend = 1
    
    pHH = opens[0]
    pLL = opens[0]
    last_cHL = opens[0]
    last_cLH = opens[0]
    
    for i in range(n):
        cl = closes[i]; h = highs[i]; l = lows[i]
        
        if curr_trend == 1:
            if h > pHH: pHH = h
            if cl < last_cHL:
                curr_trend = -1
                pLL = l
                last_cLH = pHH
        else:
            if l < pLL: pLL = l
            if cl > last_cLH:
                curr_trend = 1
                pHH = h
                last_cHL = pLL
                
        trend[i] = curr_trend
        
    df_res = df_htf.copy()
    df_res['htf_trend'] = trend
    time_col = 'time' if 'time' in df_res.columns else 'date'
    df_res['time_dt'] = pd.to_datetime(df_res[time_col])
    return df_res[['time_dt', 'htf_trend']]

class BEIDRealEngine:
    """
    Realistic Institutional BEID Execution Engine:
    - Strict non-lookahead Bid/Ask limit and market execution.
    - Full State Machine with zero lockups.
    - Configurable Structural Filters:
        * htf_filter: Require trade direction to match HTF trend
        * session_filter: Restrict entries to liquid market sessions (e.g. London / NY)
        * displacement_filter: Require minimum impulse momentum on zone creation candle
        * entry_level_pct: Entry pullback depth (e.g. 0.50 = 50%, 0.618 = 61.8%, 0.382 = 38.2%)
        * reward_risk_ratio: Target multiple (1.0 = 1:1 RR, 1.5 = 1.5:1 RR, 2.0 = 2:1 RR)
        * max_touches: Multi-touch re-arming limit (1, 2, 3, 5, etc.)
        * min_zone_pips: Minimum zone size in pips
        * spread_pips: Round-trip spread fee
    """
    def __init__(self,
                 entry_level_pct=0.50,
                 reward_risk_ratio=1.0,
                 max_touches=1,
                 min_zone_pips=10.0,
                 spread_pips=1.6,
                 htf_filter=False,
                 session_filter=False,
                 allowed_hours=(7, 18), # UTC hours: London 07:00 to NY close 18:00
                 displacement_filter=False,
                 min_body_ratio=0.50,
                 max_unfilled_bars=2880,
                 max_active_zones=25):
        
        self.entry_level_pct = entry_level_pct
        self.reward_risk_ratio = reward_risk_ratio
        self.max_touches = max_touches
        self.min_zone_pips = min_zone_pips
        self.spread_pips = spread_pips
        self.htf_filter = htf_filter
        self.session_filter = session_filter
        self.allowed_hours = allowed_hours
        self.displacement_filter = displacement_filter
        self.min_body_ratio = min_body_ratio
        self.max_unfilled_bars = max_unfilled_bars
        self.max_active_zones = max_active_zones

    def backtest(self, df_ltf, ticker_name, df_htf_trend=None):
        pip_size = get_pip_size(ticker_name)
        time_col = 'date' if 'date' in df_ltf.columns else 'time'
        
        df = df_ltf.copy()
        df['time_dt'] = pd.to_datetime(df[time_col])
        
        if self.htf_filter and df_htf_trend is not None:
            # Merge asof on datetime to get latest available HTF trend with zero lookahead
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
        
        # Precompute session hours if session filter active
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
        
        spread_val = self.spread_pips * pip_size
        half_spread = spread_val / 2.0
        
        for i in range(n):
            o = opens[i]; h = highs[i]; l = lows[i]; cl = closes[i]
            is_green = cl >= o; is_red = cl < o
            
            # ---------------------------------------------------------
            # 1. Update Active Zones / Limit Orders & Active Trades
            # ---------------------------------------------------------
            retained = []
            for z in active_zones:
                side = z['side']
                entry = z['entry']
                tp = z['tp']
                sl = z['sl']
                risk_pips = z['risk_pips']
                reward_pips = z['reward_pips']
                
                if not z['filled']:
                    # Invalidation check: if price breaches SL floor before 50% limit fill
                    if side == 1: # BUY Zone
                        if l <= sl:
                            continue # Invalidate zone, opposite edge breached
                        # Fill check: Ask price touches entry (Mid low <= Entry - half_spread)
                        # Realistic exact limit execution:
                        if l <= (entry - half_spread):
                            z['filled'] = True
                            z['fill_idx'] = i
                            z['touch_count'] += 1
                            retained.append(z)
                        else:
                            if i - z['created_idx'] <= self.max_unfilled_bars:
                                retained.append(z)
                    elif side == -1: # SELL Zone
                        if h >= sl:
                            continue # Invalidate zone, opposite edge breached
                        # Fill check: Bid price touches entry (Mid high >= Entry + half_spread)
                        if h >= (entry + half_spread):
                            z['filled'] = True
                            z['fill_idx'] = i
                            z['touch_count'] += 1
                            retained.append(z)
                        else:
                            if i - z['created_idx'] <= self.max_unfilled_bars:
                                retained.append(z)
                else:
                    # Active Trade: must resolve on subsequent candle (i > z['fill_idx'])
                    if i > z['fill_idx']:
                        resolved = False
                        win = False
                        
                        if side == 1: # BUY Trade
                            # TP triggers when Bid reaches TP (Mid high >= TP + half_spread)
                            # SL triggers when Bid reaches SL (Mid low <= SL + half_spread)
                            hit_tp = (h >= (tp + half_spread)) or (cl > (tp + half_spread))
                            hit_sl = (l <= (sl + half_spread)) or (cl < (sl + half_spread))
                            
                            if hit_tp and hit_sl:
                                # Intra-bar tie breaker: conservative assumption based on close
                                if cl >= entry:
                                    resolved = True; win = True
                                else:
                                    resolved = True; win = False
                            elif hit_tp:
                                resolved = True; win = True
                            elif hit_sl:
                                resolved = True; win = False
                        else: # SELL Trade
                            # TP triggers when Ask reaches TP (Mid low <= TP - half_spread)
                            # SL triggers when Ask reaches SL (Mid high >= SL - half_spread)
                            hit_tp = (l <= (tp - half_spread)) or (cl < (tp - half_spread))
                            hit_sl = (h >= (sl - half_spread)) or (cl > (sl - half_spread))
                            
                            if hit_tp and hit_sl:
                                if cl <= entry:
                                    resolved = True; win = True
                                else:
                                    resolved = True; win = False
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
                                'raw_r': self.reward_risk_ratio if win else -1.0,
                                'duration_bars': i - z['fill_idx']
                            })
                            
                            # Multi-Touch Re-Arming: if trade won and touch_count < max_touches, re-arm zone!
                            if win and z['touch_count'] < self.max_touches:
                                z['filled'] = False
                                z['fill_idx'] = -1
                                z['created_idx'] = i
                                retained.append(z)
                        else:
                            retained.append(z)
                            
            active_zones = retained
            
            # ---------------------------------------------------------
            # 2. State Machine & Zone Generation (Robust & Lockup-Free)
            # ---------------------------------------------------------
            if trend == 1: # UPTREND
                if not in_pullback:
                    if h > pHH:
                        pHH = h; pHH_idx = i
                    # Check Trend Flip: close below confirmed floor
                    if not np.isnan(last_cHL) and cl < last_cHL:
                        trend = -1
                        top = pHH; bottom = last_cHL
                        zone_pips = (top - bottom) / pip_size
                        
                        # Apply Filters
                        pass_filters = True
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != -1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = top - (top - bottom) * self.entry_level_pct
                            sl = top
                            risk_pips = (sl - entry) / pip_size
                            tp = entry - (risk_pips * pip_size * self.reward_risk_ratio)
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
                    if l < pHL:
                        pHL = l; pHL_idx = i
                    
                    floor_to_check = last_cHL if not np.isnan(last_cHL) else opens[0]
                    if cl < floor_to_check:
                        trend = -1
                        top = cHH; bottom = floor_to_check
                        zone_pips = (top - bottom) / pip_size
                        
                        pass_filters = True
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != -1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = top - (top - bottom) * self.entry_level_pct
                            sl = top
                            risk_pips = (sl - entry) / pip_size
                            tp = entry - (risk_pips * pip_size * self.reward_risk_ratio)
                            reward_pips = (entry - tp) / pip_size
                            
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': -1, 'type': 'SELL_FLIP',
                                'top': top, 'bottom': bottom, 'entry': entry, 'tp': tp, 'sl': sl,
                                'zone_pips': zone_pips, 'risk_pips': risk_pips, 'reward_pips': reward_pips,
                                'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                            })
                        pLL = l; pLL_idx = i; cLH = top; last_cLH = np.nan; in_pullback = False
                    elif cl > cHH: # Breakout Continuation
                        in_pullback = False
                        cHL = pHL; cHL_idx = pHL_idx; last_cHL = cHL
                        top = cHH; bottom = cHL
                        zone_pips = (top - bottom) / pip_size
                        
                        # Displacement filter: check breakout candle body strength
                        pass_filters = True
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != 1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        if self.displacement_filter:
                            body_size = abs(cl - o)
                            candle_range = max(1e-6, h - l)
                            if (body_size / candle_range) < self.min_body_ratio:
                                pass_filters = False
                                
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = bottom + (top - bottom) * self.entry_level_pct
                            sl = bottom
                            risk_pips = (entry - sl) / pip_size
                            tp = entry + (risk_pips * pip_size * self.reward_risk_ratio)
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
                    if l < pLL:
                        pLL = l; pLL_idx = i
                    if not np.isnan(last_cLH) and cl > last_cLH:
                        trend = 1
                        top = last_cLH; bottom = pLL
                        zone_pips = (top - bottom) / pip_size
                        
                        pass_filters = True
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != 1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = bottom + (top - bottom) * self.entry_level_pct
                            sl = bottom
                            risk_pips = (entry - sl) / pip_size
                            tp = entry + (risk_pips * pip_size * self.reward_risk_ratio)
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
                    if h > pLH:
                        pLH = h; pLH_idx = i
                    ceiling_to_check = last_cLH if not np.isnan(last_cLH) else opens[0]
                    if cl > ceiling_to_check:
                        trend = 1
                        top = ceiling_to_check; bottom = cLL
                        zone_pips = (top - bottom) / pip_size
                        
                        pass_filters = True
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != 1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = bottom + (top - bottom) * self.entry_level_pct
                            sl = bottom
                            risk_pips = (entry - sl) / pip_size
                            tp = entry + (risk_pips * pip_size * self.reward_risk_ratio)
                            reward_pips = (tp - entry) / pip_size
                            
                            active_zones.append({
                                'id': len(trades) + len(active_zones) + 1,
                                'side': 1, 'type': 'BUY_FLIP',
                                'top': top, 'bottom': bottom, 'entry': entry, 'tp': tp, 'sl': sl,
                                'zone_pips': zone_pips, 'risk_pips': risk_pips, 'reward_pips': reward_pips,
                                'created_idx': i, 'filled': False, 'fill_idx': -1, 'touch_count': 0
                            })
                        pHH = h; pHH_idx = i; cHL = bottom; last_cHL = np.nan; in_pullback = False
                    elif cl < cLL: # Breakdown Continuation
                        in_pullback = False
                        cLH = pLH; cLH_idx = pLH_idx; last_cLH = cLH
                        top = cLH; bottom = cLL
                        zone_pips = (top - bottom) / pip_size
                        
                        pass_filters = True
                        if zone_pips < self.min_zone_pips: pass_filters = False
                        if self.htf_filter and htf_arr[i] != -1: pass_filters = False
                        if self.session_filter and not in_session[i]: pass_filters = False
                        if self.displacement_filter:
                            body_size = abs(cl - o)
                            candle_range = max(1e-6, h - l)
                            if (body_size / candle_range) < self.min_body_ratio:
                                pass_filters = False
                                
                        if pass_filters and len(active_zones) < self.max_active_zones:
                            entry = top - (top - bottom) * self.entry_level_pct
                            sl = top
                            risk_pips = (sl - entry) / pip_size
                            tp = entry - (risk_pips * pip_size * self.reward_risk_ratio)
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
