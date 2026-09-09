import json
import pandas as pd
from test_real_dtfx_data import process_real_market_data, run_real_dtfx_state_machine

def track_strict_50pct_zone_lifecycles(candles, zones):
    for z in zones:
        side = z["side"]
        c_idx = z["created_idx"]
        top = z["top"]
        bottom = z["bottom"]
        mid = z["mid"]
        
        touched_50 = False
        touched_50_time = None
        touched_50_idx = None
        end_idx = len(candles) - 1
        end_reason = "Active / Unmitigated (50% Not Yet Tested & Left)"
        
        # Check subsequent candles starting after creation
        for j in range(c_idx + 1, len(candles)):
            cj = candles[j]
            h, l, cl = cj["h"], cj["l"], cj["c"]
            
            if side == "BUY":
                # Check 50% Midpoint touch
                if l <= mid:
                    if not touched_50:
                        touched_50 = True
                        touched_50_time = cj["t"]
                        touched_50_idx = j
                
                # Check structural invalidation: candle body closes below bottom floor
                if cl < bottom:
                    end_idx = j
                    end_reason = f"Invalidated (Closed below bottom floor {bottom:.5f})"
                    break
                
                # Zone ends ONLY if 50% was touched on an earlier candle AND a subsequent different candle leaves zone back up
                if touched_50 and j > touched_50_idx and (cl > top or h > top):
                    end_idx = j
                    end_reason = f"Trade Completed (Touched 50% on {touched_50_time}, Exited Up on {cj['t']})"
                    break
                    
            elif side == "SELL":
                # Check 50% Midpoint touch
                if h >= mid:
                    if not touched_50:
                        touched_50 = True
                        touched_50_time = cj["t"]
                        touched_50_idx = j
                
                # Check structural invalidation: candle body closes above top ceiling
                if cl > top:
                    end_idx = j
                    end_reason = f"Invalidated (Closed above top ceiling {top:.5f})"
                    break
                
                # Zone ends ONLY if 50% was touched on an earlier candle AND a subsequent different candle leaves zone back down
                if touched_50 and j > touched_50_idx and (cl < bottom or l < bottom):
                    end_idx = j
                    end_reason = f"Trade Completed (Touched 50% on {touched_50_time}, Exited Down on {cj['t']})"
                    break
                    
        z["touched_50"] = touched_50
        z["touched_50_time"] = touched_50_time
        z["touched_50_idx"] = touched_50_idx
        z["end_idx"] = end_idx
        z["end_time"] = candles[end_idx]["t"]
        z["end_reason"] = end_reason
        
    return zones

if __name__ == "__main__":
    csv_file = "/Users/brunoinzunza/.gemini/antigravity/scratch/smc_strategy_backtest/data/EURUSD_15m.csv"
    candles = process_real_market_data(csv_file, max_candles=600)
    zones, pivots, events = run_real_dtfx_state_machine(candles)
    zones = track_strict_50pct_zone_lifecycles(candles, zones)
    
    print("--- Strict 50% Touch (Separate Candle) & Subsequent Exit Lifecycles ---")
    for z in zones:
        t50 = f"Touched 50% at {z['touched_50_time']}" if z['touched_50'] else "50% NEVER Touched"
        print(f"[{z['id']}] {z['name']} ({z['side']}): Left={z['left_time']} | Created={z['created_time']} | End={z['end_time']} | {t50} | {z['end_reason']}")
