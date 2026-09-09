import os
import sys
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PROD_DIR = "/Users/brunoinzunza/Documents/github/PRODLIVETRADING"
sys.path.insert(0, PROD_DIR)

import config
from broker.oanda_client import OANDAClient, SYMBOL_MAP
import oandapyV20.endpoints.instruments as instruments
import goal95_every_swing_zones as g95
import goal99_dtfx_base_backtest as g99

SYMBOLS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}
MIN_ZONE_PIPS = 5.0
MIN_STOP_PIPS = 3.0

def fetch_s5_candles(client, symbol, total_count=30000):
    """Fetch consecutive 5-second True Bid/Ask candles backwards using OANDA API."""
    instrument = SYMBOL_MAP[symbol]
    all_rows = []
    to_time = None
    print(f"  [{symbol}] Fetching {total_count:,} S5 (5-sec) Bid/Ask candles...", end=" ", flush=True)
    t0 = time.time()
    
    while len(all_rows) < total_count:
        fetch_size = min(total_count - len(all_rows), 5000)
        params = {"count": fetch_size, "granularity": "S5", "price": "BA"}
        if to_time is not None:
            params["to"] = to_time
            
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.api.request(r)
        raw_candles = r.response.get("candles", [])
        complete = [c for c in raw_candles if c.get("complete")]
        if not complete:
            break
            
        for c in complete:
            t_str = c["time"]
            t_dt = pd.Timestamp(t_str).tz_localize(None)
            bid = c["bid"]
            ask = c["ask"]
            all_rows.append({
                "date": t_dt,
                "bid_o": float(bid["o"]), "bid_h": float(bid["h"]), "bid_l": float(bid["l"]), "bid_c": float(bid["c"]),
                "ask_o": float(ask["o"]), "ask_h": float(ask["h"]), "ask_l": float(ask["l"]), "ask_c": float(ask["c"]),
                "mid_o": (float(bid["o"]) + float(ask["o"])) / 2.0,
                "mid_h": (float(bid["h"]) + float(ask["h"])) / 2.0,
                "mid_l": (float(bid["l"]) + float(ask["l"])) / 2.0,
                "mid_c": (float(bid["c"]) + float(ask["c"])) / 2.0,
            })
            
        next_time = complete[0]["time"]
        if to_time is not None and pd.Timestamp(next_time) >= pd.Timestamp(to_time):
            break
        to_time = next_time
        if len(raw_candles) < fetch_size:
            break
            
    df = pd.DataFrame(all_rows).drop_duplicates(subset="date").set_index("date").sort_index()
    print(f"Done! {len(df):,} S5 slices loaded ({df.index[0]} to {df.index[-1]}) in {time.time()-t0:.1f}s")
    return df

def run_tick_backtest():
    print("=" * 85)
    print("2026 HIGH-RESOLUTION TICK & S5 BID/ASK TRUE SPREAD BACKTEST")
    print("Testing resting limit execution against separate Bid and Ask order books")
    print("=" * 85)
    
    client = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG, environment=config.OANDA_ENVIRONMENT)
    
    results = []
    
    for sym in SYMBOLS:
        print(f"\n--- Processing {sym} ---")
        # Fetch S5 high-resolution Bid/Ask dataset
        df_s5 = fetch_s5_candles(client, sym, total_count=35000)
        pip = PIP[sym]
        
        # 1. Resample to 1-Minute M1 bars to identify macro swing zones identically
        df_m1 = df_s5.resample("1min").agg({
            "mid_o": "first", "mid_h": "max", "mid_l": "min", "mid_c": "last"
        }).dropna()
        df_m1.rename(columns={"mid_o": "open", "mid_h": "high", "mid_l": "low", "mid_c": "close"}, inplace=True)
        
        high_m1 = df_m1["high"].values.astype(np.float64)
        low_m1 = df_m1["low"].values.astype(np.float64)
        close_m1 = df_m1["close"].values.astype(np.float64)
        m1_times = df_m1.index
        
        confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high_m1, low_m1, close_m1)
        (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
         ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)
        
        print(f"  {sym}: Identified {len(z_created):,} swing zones from M1 structure.")
        
        # 2. Step through S5 ticks to simulate exact Bid/Ask limit execution
        m = len(z_created)
        trades_m1_model = []
        trades_s5_tick_model = []
        
        for zi in range(m):
            top = z_top[zi]; bottom = z_bottom[zi]; is_buy = z_is_buy[zi]
            height = top - bottom
            if height < MIN_ZONE_PIPS * pip:
                continue
            mid = (top + bottom) / 2.0
            eff_risk = (mid - bottom) if is_buy else (top - mid)
            if eff_risk < MIN_STOP_PIPS * pip:
                continue
                
            c_m1_idx = z_created[zi]
            c_time = m1_times[c_m1_idx]
            
            # Slice S5 ticks forward from zone creation time
            s5_forward = df_s5[df_s5.index >= c_time]
            if len(s5_forward) < 10:
                continue
                
            # Stage 1: Overshoot check
            overshoot_time = None
            for t_idx, row in s5_forward.iterrows():
                if is_buy and row["mid_h"] > top:
                    overshoot_time = t_idx
                    break
                elif (not is_buy) and row["mid_l"] < bottom:
                    overshoot_time = t_idx
                    break
            if overshoot_time is None:
                continue
                
            s5_after_overshoot = s5_forward[s5_forward.index >= overshoot_time]
            
            # Stage 2: Limit Fill Check on True Bid/Ask
            # For Buy Limit: Fills when ASK drops to <= mid
            # For Sell Limit: Fills when BID rises to >= mid
            fill_time = None
            fill_row = None
            for t_idx, row in s5_after_overshoot.iterrows():
                # Invalidation check: if price breaches opposite edge before filling
                if is_buy and row["bid_l"] <= bottom:
                    break
                if (not is_buy) and row["ask_h"] >= top:
                    break
                    
                if is_buy and row["ask_l"] <= mid:
                    fill_time = t_idx
                    fill_row = row
                    break
                elif (not is_buy) and row["bid_h"] >= mid:
                    fill_time = t_idx
                    fill_row = row
                    break
            if fill_time is None:
                continue
                
            # Stage 3: Exit Resolution on True Bid/Ask
            # For Buy: Stop triggers when BID <= bottom; TP triggers when BID >= top
            # For Sell: Stop triggers when ASK >= top; TP triggers when ASK <= bottom
            s5_after_fill = s5_after_overshoot[s5_after_overshoot.index >= fill_time]
            outcome = None
            exit_time = None
            
            for t_idx, row in s5_after_fill.iterrows():
                if is_buy:
                    if row["bid_l"] <= bottom:
                        outcome = "loss"
                        exit_time = t_idx
                        break
                    elif row["bid_h"] >= top:
                        outcome = "win"
                        exit_time = t_idx
                        break
                else:
                    if row["ask_h"] >= top:
                        outcome = "loss"
                        exit_time = t_idx
                        break
                    elif row["ask_l"] <= bottom:
                        outcome = "win"
                        exit_time = t_idx
                        break
                        
            if outcome is not None:
                net_r = (top - mid) / eff_risk if outcome == "win" else -1.0
                trades_s5_tick_model.append({
                    "symbol": sym,
                    "zone_created": c_time,
                    "fill_time": fill_time,
                    "exit_time": exit_time,
                    "outcome": outcome,
                    "net_r": net_r,
                    "side": "buy" if is_buy else "sell"
                })
                
        # Summarize S5 results for this symbol
        n_t = len(trades_s5_tick_model)
        if n_t > 0:
            df_sym_res = pd.DataFrame(trades_s5_tick_model)
            wins = (df_sym_res["outcome"] == "win").sum()
            wr = 100.0 * wins / n_t
            gw = df_sym_res[df_sym_res["net_r"] > 0]["net_r"].sum()
            gl = -df_sym_res[df_sym_res["net_r"] <= 0]["net_r"].sum()
            pf = gw / gl if gl > 0 else float("inf")
            avg_r = df_sym_res["net_r"].mean()
            print(f"  >>> {sym} S5 TRUE BID/ASK RESULTS: {n_t:,} trades | Win Rate: {wr:.2f}% | Profit Factor: {pf:.3f} | Avg R: {avg_r:+.4f}")
            results.append({
                "symbol": sym,
                "trades": n_t,
                "win_rate": wr,
                "profit_factor": pf,
                "avg_r": avg_r,
                "total_r": df_sym_res["net_r"].sum()
            })
            
    df_all = pd.DataFrame(results)
    out_csv = os.path.join(DATA_DIR, "s5_tick_backtest_results.csv")
    df_all.to_csv(out_csv, index=False)
    print("\n" + "=" * 85)
    print("SUMMARY ACROSS ALL TESTED INSTRUMENTS ON 5-SECOND TRUE BID/ASK DATA")
    print("=" * 85)
    print(df_all.to_string(index=False))
    print(f"\nWrote full tick backtest results to: {out_csv}")

if __name__ == "__main__":
    run_tick_backtest()
