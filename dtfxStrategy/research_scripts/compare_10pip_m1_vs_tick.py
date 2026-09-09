import os
import sys
import time
import math
import numpy as np
import pandas as pd

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
MIN_ZONE_PIPS = 10.0
MIN_STOP_PIPS = 3.0

def fetch_s5_data(client, symbol, count=40000):
    """Fetch consecutive 5-second True Bid/Ask candles backwards using OANDA API."""
    instrument = SYMBOL_MAP[symbol]
    all_rows = []
    to_time = None
    print(f"  [{symbol}] Fetching {count:,} S5 True Bid/Ask candles...", end=" ", flush=True)
    t0 = time.time()
    
    while len(all_rows) < count:
        fetch_size = min(count - len(all_rows), 5000)
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
    print(f"Done! {len(df):,} S5 slices ({df.index[0]} to {df.index[-1]}) in {time.time()-t0:.1f}s")
    return df

def main():
    print("=" * 90)
    print("HEAD-TO-HEAD BACKTEST: 10-PIP ZONE FLOOR")
    print("1-MINUTE (M1) CANDLE DATA  vs.  TRUE SUB-MINUTE TICK (S5 BID/ASK) DATA")
    print("=" * 90)

    client = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG, environment=config.OANDA_ENVIRONMENT)
    
    all_summary = []
    
    for sym in SYMBOLS:
        print(f"\n==================== {sym} (10-PIP FLOOR) ====================")
        pip = PIP[sym]
        min_zone_height = MIN_ZONE_PIPS * pip
        min_stop_price = MIN_STOP_PIPS * pip
        
        # 1. Fetch S5 data (~40k bars = approx 3 days of high-frequency data)
        df_s5 = fetch_s5_data(client, sym, count=45000)
        
        # 2. Resample to M1 bars
        df_m1 = df_s5.resample("1min").agg({
            "mid_o": "first", "mid_h": "max", "mid_l": "min", "mid_c": "last"
        }).dropna()
        df_m1.rename(columns={"mid_o": "open", "mid_h": "high", "mid_l": "low", "mid_c": "close"}, inplace=True)
        
        high_m1 = df_m1["high"].values.astype(np.float64)
        low_m1 = df_m1["low"].values.astype(np.float64)
        close_m1 = df_m1["close"].values.astype(np.float64)
        m1_times = df_m1.index
        
        # Build 10-pip zones on M1 closes
        confirm_idx, pivot_idx, is_high, price = g95.close_based_zigzag_symmetric(high_m1, low_m1, close_m1)
        (z_created, z_top, z_bottom, z_top_idx, z_bottom_idx, z_is_buy, z_is_flip
         ) = g95.build_zones_every_swing(confirm_idx, pivot_idx, is_high, price)
        
        m = len(z_created)
        print(f"  Total 10-pip zones identified: {m}")
        
        m1_trades = []
        tick_trades = []
        
        for zi in range(m):
            top = z_top[zi]; bottom = z_bottom[zi]; is_buy = z_is_buy[zi]
            height = top - bottom
            if height < min_zone_height:
                continue
            mid = (top + bottom) / 2.0
            eff_risk = (mid - bottom) if is_buy else (top - mid)
            if eff_risk < min_stop_price:
                continue
                
            c_idx = z_created[zi]
            c_time = m1_times[c_idx]
            
            # --- MODEL A: 1-MINUTE (M1) RESOLUTION ---
            # Overshoot check on M1
            m1_forward = df_m1[df_m1.index >= c_time]
            m1_overshoot_ts = None
            for ts, row in m1_forward.iterrows():
                if is_buy and row["high"] > top:
                    m1_overshoot_ts = ts; break
                elif (not is_buy) and row["low"] < bottom:
                    m1_overshoot_ts = ts; break
            if m1_overshoot_ts is None:
                continue
                
            m1_after_ov = m1_forward[m1_forward.index >= m1_overshoot_ts]
            
            # M1 Tag (Touch 50% Mid)
            m1_tag_ts = None
            for ts, row in m1_after_ov.iterrows():
                # Invalidation
                if is_buy and row["low"] <= bottom: break
                if (not is_buy) and row["high"] >= top: break
                if is_buy and row["low"] <= mid:
                    m1_tag_ts = ts; break
                elif (not is_buy) and row["high"] >= mid:
                    m1_tag_ts = ts; break
            if m1_tag_ts is None:
                continue
                
            # M1 Exit Resolution
            m1_after_tag = m1_after_ov[m1_after_ov.index >= m1_tag_ts]
            m1_outcome = None
            for ts, row in m1_after_tag.iterrows():
                if is_buy:
                    if row["low"] <= bottom:
                        m1_outcome = "loss"; break
                    elif row["high"] >= top:
                        m1_outcome = "win"; break
                else:
                    if row["high"] >= top:
                        m1_outcome = "loss"; break
                    elif row["low"] <= bottom:
                        m1_outcome = "win"; break
            if m1_outcome is not None:
                m1_trades.append({"zone_idx": zi, "outcome": m1_outcome, "time": m1_tag_ts})
                
            # --- MODEL B: TRUE TICK (S5 BID/ASK) RESOLUTION ---
            s5_forward = df_s5[df_s5.index >= c_time]
            s5_ov_ts = None
            for ts, row in s5_forward.iterrows():
                if is_buy and row["mid_h"] > top:
                    s5_ov_ts = ts; break
                elif (not is_buy) and row["mid_l"] < bottom:
                    s5_ov_ts = ts; break
            if s5_ov_ts is None:
                continue
                
            s5_after_ov = s5_forward[s5_forward.index >= s5_ov_ts]
            
            # S5 True Limit Fill: Buy limit fills when Ask <= Mid; Sell limit fills when Bid >= Mid
            s5_fill_ts = None
            for ts, row in s5_after_ov.iterrows():
                if is_buy and row["bid_l"] <= bottom: break
                if (not is_buy) and row["ask_h"] >= top: break
                if is_buy and row["ask_l"] <= mid:
                    s5_fill_ts = ts; break
                elif (not is_buy) and row["bid_h"] >= mid:
                    s5_fill_ts = ts; break
            if s5_fill_ts is None:
                continue
                
            # S5 Exit Resolution on True Bid/Ask starting from next 5-second slice
            s5_after_fill = s5_after_ov[s5_after_ov.index > s5_fill_ts]
            tick_outcome = None
            for ts, row in s5_after_fill.iterrows():
                if is_buy:
                    if row["bid_l"] <= bottom:
                        tick_outcome = "loss"; break
                    elif row["bid_h"] >= top:
                        tick_outcome = "win"; break
                else:
                    if row["ask_h"] >= top:
                        tick_outcome = "loss"; break
                    elif row["ask_l"] <= bottom:
                        tick_outcome = "win"; break
            if tick_outcome is not None:
                tick_trades.append({"zone_idx": zi, "outcome": tick_outcome, "time": s5_fill_ts})

        # Calculate statistics
        n_m1 = len(m1_trades)
        n_tk = len(tick_trades)
        
        if n_m1 > 0:
            df_m1_t = pd.DataFrame(m1_trades)
            wr_m1 = (df_m1_t["outcome"] == "win").mean() * 100.0
            n_losses_m1 = (df_m1_t["outcome"] == "loss").sum()
            pf_m1 = (df_m1_t["outcome"] == "win").sum() / n_losses_m1 if n_losses_m1 > 0 else float("inf")
        else:
            df_m1_t = pd.DataFrame(columns=["zone_idx", "outcome", "time"])
            wr_m1 = float("nan")
            pf_m1 = float("nan")
            
        if n_tk > 0:
            df_tk_t = pd.DataFrame(tick_trades)
            wr_tk = (df_tk_t["outcome"] == "win").mean() * 100.0
            n_losses_tk = (df_tk_t["outcome"] == "loss").sum()
            pf_tk = (df_tk_t["outcome"] == "win").sum() / n_losses_tk if n_losses_tk > 0 else float("inf")
        else:
            df_tk_t = pd.DataFrame(columns=["zone_idx", "outcome", "time"])
            wr_tk = float("nan")
            pf_tk = float("nan")
        
        # Check agreement rate on identical zones
        common_zones = set(df_m1_t["zone_idx"]).intersection(set(df_tk_t["zone_idx"]))
        matches = 0
        for zid in common_zones:
            out_m1 = df_m1_t[df_m1_t["zone_idx"] == zid]["outcome"].values[0]
            out_tk = df_tk_t[df_tk_t["zone_idx"] == zid]["outcome"].values[0]
            if out_m1 == out_tk:
                matches += 1
        agreement_pct = (matches / len(common_zones) * 100.0) if len(common_zones) > 0 else float("nan")
        
        print(f"  1-Minute (M1) Model:    {n_m1:>4} trades | Win Rate: {wr_m1:>6.2f}% | Profit Factor: {pf_m1:>5.3f}")
        print(f"  True Tick (S5) Model:   {n_tk:>4} trades | Win Rate: {wr_tk:>6.2f}% | Profit Factor: {pf_tk:>5.3f}")
        print(f"  Outcome Match Accuracy: {agreement_pct:>6.2f}% ({matches}/{len(common_zones)} identical win/loss calls)")
        
        all_summary.append({
            "symbol": sym,
            "m1_trades": n_m1, "m1_wr": wr_m1, "m1_pf": pf_m1,
            "tick_trades": n_tk, "tick_wr": wr_tk, "tick_pf": pf_tk,
            "agreement_pct": agreement_pct
        })
        
    df_summary = pd.DataFrame(all_summary)
    out_csv = os.path.join(DATA_DIR, "compare_10pip_m1_vs_tick.csv")
    df_summary.to_csv(out_csv, index=False)
    
    print("\n" + "=" * 90)
    print("FINAL SUMMARY: 10-PIP ZONE FLOOR (M1 BARS vs. TRUE SUB-MINUTE S5 TICKS)")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print(f"\nSaved detailed comparison to: {out_csv}")

if __name__ == "__main__":
    main()
