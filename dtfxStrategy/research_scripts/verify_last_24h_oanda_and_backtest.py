import os
import sys
import time
import json
import math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
PROD_DIR = "/Users/brunoinzunza/Documents/github/PRODLIVETRADING"
sys.path.insert(0, PROD_DIR)

import config
from broker.oanda_client import OANDAClient, SYMBOL_MAP
REVERSE_MAP = {v: k for k, v in SYMBOL_MAP.items()}
from strategy.engine import SymbolEngine, _zone_id
import goal95_every_swing_zones as g95
import goal99_dtfx_base_backtest as g99

SYMBOLS = ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"]
PIP = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "EURGBP": 1e-4, "USDJPY": 1e-2}

def main():
    print("=" * 85)
    print("OANDA LAST 24-HOUR DATA PULL, BACKTEST & REAL TRADE RECONCILIATION")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()} UTC")
    print(f"Environment: {config.ENV_LABEL}")
    print("=" * 85)

    client_long = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG,
                              environment=config.OANDA_ENVIRONMENT, label="LONG-SUBACCOUNT")
    client_short = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_SHORT,
                               environment=config.OANDA_ENVIRONMENT, label="SHORT-SUBACCOUNT")

    print("\n--- 1. Fetching OANDA Account Balances & Recent Broker Trades ---")
    try:
        bal_long = client_long.account_balance()
        bal_short = client_short.account_balance()
        eq_long = client_long.account_equity()
        eq_short = client_short.account_equity()
        print(f"  Long Account ({config.OANDA_ACCOUNT_ID_LONG}):  Balance=${bal_long:,.2f} | Equity=${eq_long:,.2f}")
        print(f"  Short Account ({config.OANDA_ACCOUNT_ID_SHORT}): Balance=${bal_short:,.2f} | Equity=${eq_short:,.2f}")
    except Exception as e:
        print(f"  Error fetching account summaries: {e}")

    # Fetch closed trades from both accounts
    now_utc = datetime.now(timezone.utc)
    cutoff_24h = now_utc - timedelta(hours=24)
    print(f"\n  Querying trades since 24h cutoff ({cutoff_24h.strftime('%Y-%m-%d %H:%M:%S')} UTC)...")

    closed_trades = []
    for client, side_label in [(client_long, "buy"), (client_short, "sell")]:
        try:
            trades = client.list_recent_closed_trades(count=100)
            for t in trades:
                # parse close time
                c_time_str = t.get("closeTime") or t.get("openTime")
                if c_time_str:
                    c_dt = pd.to_datetime(c_time_str).to_pydatetime()
                    if c_dt.tzinfo is None:
                        c_dt = c_dt.replace(tzinfo=timezone.utc)
                    if c_dt >= cutoff_24h:
                        t["subaccount_side"] = side_label
                        closed_trades.append(t)
        except Exception as e:
            print(f"  Error pulling trades from {side_label} account: {e}")

    print(f"  Total closed trades in last 24h across both accounts: {len(closed_trades)}")
    for t in closed_trades:
        sym = REVERSE_MAP.get(t.get("instrument"), t.get("instrument"))
        pnl = float(t.get("realizedPL", 0.0))
        u = float(t.get("initialUnits", 0.0))
        op = float(t.get("price", 0.0))
        cp = float(t.get("averageClosePrice", 0.0))
        ot = t.get("openTime")
        ct = t.get("closeTime")
        print(f"    Broker Trade: {sym} {t.get('subaccount_side').upper()} {u:,.0f} units | Open: {ot} @ {op:.5f} | Close: {ct} @ {cp:.5f} | PnL: ${pnl:+,.2f}")

    # Also check open trades
    open_trades = []
    for client, side_label in [(client_long, "buy"), (client_short, "sell")]:
        try:
            ot = client.open_trades_detailed()
            for t in ot:
                t["subaccount_side"] = side_label
                open_trades.append(t)
        except Exception as e:
            print(f"  Error pulling open trades from {side_label} account: {e}")
    if open_trades:
        print(f"\n  Currently Open Positions ({len(open_trades)}):")
        for t in open_trades:
            sym = REVERSE_MAP.get(t.get("instrument"), t.get("instrument"))
            u = float(t.get("currentUnits", 0.0))
            op = float(t.get("price", 0.0))
            upl = float(t.get("unrealizedPL", 0.0))
            print(f"    Open Position: {sym} {t.get('subaccount_side').upper()} {u:,.0f} units @ {op:.5f} | Unrealized PnL: ${upl:+,.2f}")
    else:
        print("  No positions currently open.")

    print("\n--- 2. Fetching Complete 1-Minute M1 Candle Data from OANDA (Warmup + Last 24h) ---")
    data_by_sym = {}
    for sym in SYMBOLS:
        print(f"  Fetching {sym} candles...", end=" ", flush=True)
        # Fetch 5,000 bars (approx 3.5 days) to ensure solid warmup
        df = client_long.fetch_warmup_bars(sym, count=5000)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data_by_sym[sym] = df
        print(f"{len(df):,} bars loaded (from {df.index[0]} to {df.index[-1]})")

    print("\n--- 3. Running Live DTFX Base Engine Replay over Last 24 Hours ---")
    # Feed candles sequentially through SymbolEngine to simulate exact live bot behavior
    engines = {sym: SymbolEngine(sym) for sym in SYMBOLS}
    
    # Pre-seed engines with historical warmup bars before the 24h cutoff
    # Let cutoff be exactly 24 hours ago
    cutoff_ts = pd.to_datetime(cutoff_24h.strftime('%Y-%m-%d %H:%M:%S'))
    
    for sym in SYMBOLS:
        df = data_by_sym[sym]
        warmup_df = df[df.index < cutoff_ts]
        engines[sym].seed_zones(warmup_df)
        print(f"  {sym}: Pre-seeded with {len(warmup_df):,} warmup bars. Active zones at 24h mark: {len(engines[sym].active_zones)}")

    # Now step bar-by-bar across the synchronized 24h timeline
    # Combine all bar timestamps in the 24h window
    eval_dfs = {sym: data_by_sym[sym][data_by_sym[sym].index >= cutoff_ts] for sym in SYMBOLS}
    all_timestamps = sorted(list(set().union(*[df.index.tolist() for df in eval_dfs.values()])))
    print(f"\nStepping through {len(all_timestamps)} one-minute intervals in the last 24 hours...")

    sim_signals = []
    
    for ts in all_timestamps:
        for sym in SYMBOLS:
            df = eval_dfs[sym]
            if ts in df.index:
                row = df.loc[ts]
                o, h, l, c = row["open"], row["high"], row["low"], row["close"]
                signals, invalidated = engines[sym].on_new_bar(ts, o, h, l, c, log_new=False)
                for sig in signals:
                    sim_signals.append({
                        "symbol": sym,
                        "time": ts,
                        "side": sig["side"],
                        "top": sig["top"],
                        "bottom": sig["bottom"],
                        "mid": sig["mid"],
                        "stop_price": sig["stop_price"],
                        "target_price": sig["target_price"],
                        "created_at": sig["created_at"],
                    })

    print(f"\n  Total DTFX signals generated in last 24h: {len(sim_signals)}")
    for sig in sim_signals:
        print(f"    Signal: {sig['symbol']} {sig['side'].upper()} @ {sig['time']} | 50% Mid: {sig['mid']:.5f} | SL: {sig['stop_price']:.5f} | TP: {sig['target_price']:.5f} | Zone Created: {sig['created_at']}")

    print("\n--- 4. Concurrency & Order Resolution Simulation (1 per side) ---")
    # Simulate trade outcomes for signals under resting limit execution
    executed_trades = []
    # Track open slot per side
    open_slot = {"buy": None, "sell": None}
    
    for sig in sim_signals:
        sym = sig["symbol"]
        side = sig["side"]
        is_buy = (side == "buy")
        df = data_by_sym[sym]
        sig_time = sig["time"]
        
        # Check if slot is available at signal time
        current_open = open_slot[side]
        if current_open is not None and current_open["exit_time"] > sig_time:
            # Slot occupied by another trade
            print(f"    [Concurrency Gate]: {sym} {side.upper()} at {sig_time} rejected (slot occupied by {current_open['symbol']} until {current_open['exit_time']})")
            continue
            
        # Resolve trade forward in time from sig_time
        bars_after = df[df.index >= sig_time]
        entry_price = sig["mid"]
        stop_price = sig["stop_price"]
        target_price = sig["target_price"]
        
        outcome = None
        exit_time = None
        exit_price = None
        
        for bar_ts, row in bars_after.iterrows():
            h, l = row["high"], row["low"]
            if is_buy:
                if l <= stop_price:
                    outcome = "loss"
                    exit_time = bar_ts
                    exit_price = stop_price
                    break
                elif h >= target_price:
                    outcome = "win"
                    exit_time = bar_ts
                    exit_price = target_price
                    break
            else:
                if h >= stop_price:
                    outcome = "loss"
                    exit_time = bar_ts
                    exit_price = stop_price
                    break
                elif l <= target_price:
                    outcome = "win"
                    exit_time = bar_ts
                    exit_price = target_price
                    break
                    
        trade_res = {
            "symbol": sym,
            "side": side,
            "entry_time": sig_time,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "exit_time": exit_time if exit_time else "STILL_OPEN",
            "exit_price": exit_price,
            "outcome": outcome if outcome else "OPEN",
        }
        executed_trades.append(trade_res)
        open_slot[side] = trade_res
        print(f"    -> Accepted Trade: {sym} {side.upper()} | Entry: {sig_time} @ {entry_price:.5f} | Exit: {exit_time} @ {exit_price} | Outcome: {outcome}")

    print("\n--- 5. Comparison: Backtest vs OANDA Broker Trades ---")
    print(f"  Backtest Accepted Trades: {len(executed_trades)}")
    print(f"  OANDA Broker Closed/Open Trades: {len(closed_trades) + len(open_trades)}")

    df_comp = pd.DataFrame(executed_trades)
    out_csv = os.path.join(DATA_DIR, "last_24h_reconciliation.csv")
    df_comp.to_csv(out_csv, index=False)
    print(f"\nSaved reconciliation output to: {out_csv}")

if __name__ == "__main__":
    main()
