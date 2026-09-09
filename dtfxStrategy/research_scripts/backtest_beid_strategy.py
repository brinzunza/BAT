import os
import glob
import pandas as pd
import numpy as np
import json
from datetime import datetime

def parse_candle_data(csv_path):
    df = pd.read_csv(csv_path)
    # Check column names
    df.columns = [c.lower() for c in df.columns]
    
    candles = []
    for idx, row in df.iterrows():
        t = str(row['time'])
        o = float(row['open'])
        h = float(row['high'])
        l = float(row['low'])
        c = float(row['close'])
        candles.append({
            "idx": idx,
            "t": t,
            "o": o,
            "h": h,
            "l": l,
            "c": c
        })
    return candles

def run_beid_state_machine_and_backtest(candles, spread_pips=0.0):
    state = {
        "trend": "UPTREND",
        "pHH": None,
        "pHH_idx": None,
        "cHH": None,
        "cHH_idx": None,
        "pHL": None,
        "pHL_idx": None,
        "cHL": None,
        "cHL_idx": None,
        "last_cHL": None,
        "last_cHL_idx": None,
        "pLL": None,
        "pLL_idx": None,
        "cLL": None,
        "cLL_idx": None,
        "pLH": None,
        "pLH_idx": None,
        "cLH": None,
        "cLH_idx": None,
        "last_cLH": None,
        "last_cLH_idx": None,
        "in_pullback": False
    }
    
    zones = []
    prev_green_open = None
    prev_red_open = None

    # Step 1: Detect all structural zones via BEID/DTFX State Machine
    for i, c in enumerate(candles):
        t, o, h, l, cl = c["t"], c["o"], c["h"], c["l"], c["c"]
        is_green = cl >= o
        is_red = cl < o

        if state["trend"] == "UPTREND":
            if state["pHH"] is None or h > state["pHH"]:
                state["pHH"] = h
                state["pHH_idx"] = i
                
            # Check Trend Flip breakdown (red candle body closes below last_cHL floor)
            if state["last_cHL"] is not None and cl < state["last_cHL"]:
                old_floor = state["last_cHL"]
                old_floor_idx = state["last_cHL_idx"]
                top_peak = state["pHH"]
                top_peak_idx = state["pHH_idx"]
                
                left_idx = min(old_floor_idx if old_floor_idx is not None else i, top_peak_idx if top_peak_idx is not None else i)
                
                state["trend"] = "DOWNTREND"
                state["cLH"] = top_peak
                state["cLH_idx"] = top_peak_idx
                state["last_cLH"] = None
                state["pLL"] = l
                state["pLL_idx"] = i
                state["cLL"] = None
                state["in_pullback"] = False
                
                # BEID Sell Flip Zone: Top = Peak, Bottom = Broken Floor, Entry = 50% Mid, TP = Bottom (Floor), SL = Top (Peak)
                top = top_peak
                bottom = old_floor
                mid = round((top + bottom) / 2.0, 6)
                
                zones.append({
                    "id": f"Z_{len(zones)+1}",
                    "name": "SELL FLIP ZONE",
                    "type": "SELL_FLIP",
                    "side": "SELL",
                    "left_idx": left_idx,
                    "left_time": candles[left_idx]["t"],
                    "created_idx": i,
                    "created_time": t,
                    "top": top,
                    "bottom": bottom,
                    "entry": mid,
                    "tp": bottom,  # TP at bottom of zone for sell
                    "sl": top,     # SL at top of zone for sell
                    "risk": top - mid,
                    "reward": mid - bottom
                })
            else:
                # Normal Uptrend
                if not state["in_pullback"] and is_red and prev_green_open is not None and cl < prev_green_open:
                    state["in_pullback"] = True
                    state["cHH"] = state["pHH"]
                    state["cHH_idx"] = state["pHH_idx"]
                    state["pHL"] = l
                    state["pHL_idx"] = i
                elif state["in_pullback"]:
                    if l < state["pHL"]:
                        state["pHL"] = l
                        state["pHL_idx"] = i
                    if cl > state["cHH"]:
                        state["in_pullback"] = False
                        state["cHL"] = state["pHL"]
                        state["cHL_idx"] = state["pHL_idx"]
                        state["last_cHL"] = state["cHL"]
                        state["last_cHL_idx"] = state["cHL_idx"]
                        
                        left_idx = min(state["cHH_idx"] if state["cHH_idx"] is not None else i, state["cHL_idx"] if state["cHL_idx"] is not None else i)
                        
                        top = state["cHH"]
                        bottom = state["cHL"]
                        mid = round((top + bottom) / 2.0, 6)
                        
                        state["pHH"] = h
                        state["pHH_idx"] = i
                        
                        # BEID Buy Zone: Top = cHH, Bottom = cHL, Entry = 50% Mid, TP = Top (cHH), SL = Bottom (cHL)
                        zones.append({
                            "id": f"Z_{len(zones)+1}",
                            "name": f"BUY ZONE {len(zones)+1}",
                            "type": "BUY",
                            "side": "BUY",
                            "left_idx": left_idx,
                            "left_time": candles[left_idx]["t"],
                            "created_idx": i,
                            "created_time": t,
                            "top": top,
                            "bottom": bottom,
                            "entry": mid,
                            "tp": top,     # TP at top of zone for buy
                            "sl": bottom,  # SL at bottom of zone for buy
                            "risk": mid - bottom,
                            "reward": top - mid
                        })

        elif state["trend"] == "DOWNTREND":
            if state["pLL"] is None or l < state["pLL"]:
                state["pLL"] = l
                state["pLL_idx"] = i
                
            # Check Trend Flip breakout (green candle body closes above last_cLH ceiling)
            if state["last_cLH"] is not None and cl > state["last_cLH"]:
                old_ceiling = state["last_cLH"]
                old_ceiling_idx = state["last_cLH_idx"]
                bottom_trough = state["pLL"]
                bottom_trough_idx = state["pLL_idx"]
                
                left_idx = min(old_ceiling_idx if old_ceiling_idx is not None else i, bottom_trough_idx if bottom_trough_idx is not None else i)
                
                state["trend"] = "UPTREND"
                state["cHL"] = bottom_trough
                state["cHL_idx"] = bottom_trough_idx
                state["last_cHL"] = None
                state["pHH"] = h
                state["pHH_idx"] = i
                state["cHH"] = None
                state["in_pullback"] = False
                
                # BEID Buy Flip Zone: Top = Broken Ceiling, Bottom = Trough, Entry = 50% Mid, TP = Top (Ceiling), SL = Bottom (Trough)
                top = old_ceiling
                bottom = bottom_trough
                mid = round((top + bottom) / 2.0, 6)
                
                zones.append({
                    "id": f"Z_{len(zones)+1}",
                    "name": "BUY FLIP ZONE",
                    "type": "BUY_FLIP",
                    "side": "BUY",
                    "left_idx": left_idx,
                    "left_time": candles[left_idx]["t"],
                    "created_idx": i,
                    "created_time": t,
                    "top": top,
                    "bottom": bottom,
                    "entry": mid,
                    "tp": top,     # TP at top of zone for buy
                    "sl": bottom,  # SL at bottom of zone for buy
                    "risk": mid - bottom,
                    "reward": top - mid
                })
            else:
                # Normal Downtrend
                if not state["in_pullback"] and is_green and prev_red_open is not None and cl > prev_red_open:
                    state["in_pullback"] = True
                    state["cLL"] = state["pLL"]
                    state["cLL_idx"] = state["pLL_idx"]
                    state["pLH"] = h
                    state["pLH_idx"] = i
                elif state["in_pullback"]:
                    if h > state["pLH"]:
                        state["pLH"] = h
                        state["pLH_idx"] = i
                    if cl < state["cLL"]:
                        state["in_pullback"] = False
                        state["cLH"] = state["pLH"]
                        state["cLH_idx"] = state["pLH_idx"]
                        state["last_cLH"] = state["cLH"]
                        state["last_cLH_idx"] = state["cLH_idx"]
                        
                        left_idx = min(state["cLL_idx"] if state["cLL_idx"] is not None else i, state["cLH_idx"] if state["cLH_idx"] is not None else i)
                        
                        top = state["cLH"]
                        bottom = state["cLL"]
                        mid = round((top + bottom) / 2.0, 6)
                        
                        state["pLL"] = l
                        state["pLL_idx"] = i
                        
                        # BEID Sell Zone: Top = cLH, Bottom = cLL, Entry = 50% Mid, TP = Bottom (cLL), SL = Top (cLH)
                        zones.append({
                            "id": f"Z_{len(zones)+1}",
                            "name": f"SELL ZONE {len(zones)+1}",
                            "type": "SELL",
                            "side": "SELL",
                            "left_idx": left_idx,
                            "left_time": candles[left_idx]["t"],
                            "created_idx": i,
                            "created_time": t,
                            "top": top,
                            "bottom": bottom,
                            "entry": mid,
                            "tp": bottom,  # TP at bottom of zone for sell
                            "sl": top,     # SL at top of zone for sell
                            "risk": top - mid,
                            "reward": mid - bottom
                        })

        if is_green:
            prev_green_open = o
        elif is_red:
            prev_red_open = o

    # Step 2: Simulate Trade Execution for each BEID Zone
    trades = []
    
    for z in zones:
        side = z["side"]
        c_idx = z["created_idx"]
        entry_price = z["entry"]
        tp_price = z["tp"]
        sl_price = z["sl"]
        risk_dist = z["risk"]
        
        if risk_dist <= 0:
            continue
            
        filled = False
        fill_idx = None
        fill_time = None
        exit_idx = None
        exit_time = None
        exit_price = None
        result = None
        r_mult = 0.0
        
        for j in range(c_idx + 1, len(candles)):
            cj = candles[j]
            h, l, cl = cj["h"], cj["l"], cj["c"]
            
            if not filled:
                # Check for Limit Order Fill at 50% Midpoint
                if side == "BUY" and l <= entry_price:
                    filled = True
                    fill_idx = j
                    fill_time = cj["t"]
                elif side == "SELL" and h >= entry_price:
                    filled = True
                    fill_idx = j
                    fill_time = cj["t"]
            else:
                # Trade is active! Must check resolution on SUBSEQUENT candles (j > fill_idx)
                if j > fill_idx:
                    if side == "BUY":
                        # Check SL (hits bottom of zone)
                        hit_sl = l <= sl_price or cl < sl_price
                        # Check TP (hits top of zone)
                        hit_tp = h >= tp_price or cl > tp_price
                        
                        if hit_sl and hit_tp:
                            # Intra-candle conflict: determine by close
                            if cl >= entry_price:
                                result = "WIN"
                                exit_price = tp_price
                                r_mult = 1.0
                            else:
                                result = "LOSS"
                                exit_price = sl_price
                                r_mult = -1.0
                            exit_idx = j
                            exit_time = cj["t"]
                            break
                        elif hit_tp:
                            result = "WIN"
                            exit_price = tp_price
                            r_mult = 1.0
                            exit_idx = j
                            exit_time = cj["t"]
                            break
                        elif hit_sl:
                            result = "LOSS"
                            exit_price = sl_price
                            r_mult = -1.0
                            exit_idx = j
                            exit_time = cj["t"]
                            break
                            
                    elif side == "SELL":
                        # Check SL (hits top of zone)
                        hit_sl = h >= sl_price or cl > sl_price
                        # Check TP (hits bottom of zone)
                        hit_tp = l <= tp_price or cl < tp_price
                        
                        if hit_sl and hit_tp:
                            if cl <= entry_price:
                                result = "WIN"
                                exit_price = tp_price
                                r_mult = 1.0
                            else:
                                result = "LOSS"
                                exit_price = sl_price
                                r_mult = -1.0
                            exit_idx = j
                            exit_time = cj["t"]
                            break
                        elif hit_tp:
                            result = "WIN"
                            exit_price = tp_price
                            r_mult = 1.0
                            exit_idx = j
                            exit_time = cj["t"]
                            break
                        elif hit_sl:
                            result = "LOSS"
                            exit_price = sl_price
                            r_mult = -1.0
                            exit_idx = j
                            exit_time = cj["t"]
                            break
                            
        if filled and result is not None:
            trades.append({
                "zone_id": z["id"],
                "zone_name": z["name"],
                "zone_type": z["type"],
                "side": side,
                "created_time": z["created_time"],
                "fill_time": fill_time,
                "exit_time": exit_time,
                "fill_idx": fill_idx,
                "exit_idx": exit_idx,
                "entry": entry_price,
                "tp": tp_price,
                "sl": sl_price,
                "top": z["top"],
                "bottom": z["bottom"],
                "result": result,
                "r_mult": r_mult,
                "duration_candles": exit_idx - fill_idx
            })
            
    return zones, trades

if __name__ == "__main__":
    test_files = [
        "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data/EURUSD_15m.csv",
        "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data/EURUSD_1h.csv",
        "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data/GBPUSD_15m.csv",
        "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data/GBPUSD_1h.csv",
        "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data_ltf_deep/EURUSD_5m.csv"
    ]
    
    print("=== BEID STRATEGY MULTI-ASSET BACKTEST ===")
    print("Rules: Entry @ 50% Mid | TP @ Top/Bottom | SL @ Bottom/Top | RR = 1:1 (1.0R)")
    print("-" * 75)
    
    for path in test_files:
        if not os.path.exists(path):
            continue
        candles = parse_candle_data(path)
        zones, trades = run_beid_state_machine_and_backtest(candles)
        
        n_trades = len(trades)
        if n_trades == 0:
            print(f"[{path.split('/')[-1]}] {len(candles)} candles | 0 trades resolved")
            continue
            
        wins = sum(1 for t in trades if t["result"] == "WIN")
        losses = sum(1 for t in trades if t["result"] == "LOSS")
        win_rate = (wins / n_trades) * 100.0
        net_r = sum(t["r_mult"] for t in trades)
        pf = (wins * 1.0) / max(1, losses * 1.0)
        
        print(f"[{path.split('/')[-1]}] {len(candles)} candles | {len(zones)} zones | {n_trades} trades")
        print(f"   Win Rate: {win_rate:.1f}% ({wins}W / {losses}L) | Net Return: {net_r:+.1f}R | Profit Factor: {pf:.2f}")
        for t in trades[:3]:
            print(f"      Trade {t['zone_id']} ({t['side']}): Entry {t['entry']:.5f} -> Exit {t['exit_time']} | Result: {t['result']} ({t['r_mult']:+.1f}R)")
