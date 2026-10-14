import os
import sys
import time
import sqlite3
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
    print("=" * 90)
    print("OVERNIGHT 24-HOUR PROD RECONCILIATION & LIVE BACKTEST REPORT")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()} UTC")
    print(f"Environment: {config.ENV_LABEL}")
    print("=" * 90)

    client_long = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG,
                              environment=config.OANDA_ENVIRONMENT, label="LONG-SUBACCOUNT")
    client_short = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_SHORT,
                               environment=config.OANDA_ENVIRONMENT, label="SHORT-SUBACCOUNT")

    print("\n--- 1. Current Account Status ---")
    try:
        bal_long = client_long.account_balance()
        bal_short = client_short.account_balance()
        eq_long = client_long.account_equity()
        eq_short = client_short.account_equity()
        tot_bal = bal_long + bal_short
        tot_eq = eq_long + eq_short
        print(f"  Long Account ({config.OANDA_ACCOUNT_ID_LONG}):  Balance=${bal_long:,.2f} | Equity=${eq_long:,.2f}")
        print(f"  Short Account ({config.OANDA_ACCOUNT_ID_SHORT}): Balance=${bal_short:,.2f} | Equity=${eq_short:,.2f}")
        print(f"  Combined Total: Balance=${tot_bal:,.2f} | Equity=${tot_eq:,.2f}")
    except Exception as e:
        print(f"  Error fetching account summaries: {e}")

    # 2. Fetch all closed trades from OANDA for last 24h
    now_utc = datetime.now(timezone.utc)
    cutoff_24h = now_utc - timedelta(hours=24)
    print(f"\n--- 2. Fetching Closed Trades from OANDA API (Past 24h since {cutoff_24h.strftime('%Y-%m-%d %H:%M:%S')} UTC) ---")

    closed_trades = []
    for client, side_label in [(client_long, "buy"), (client_short, "sell")]:
        try:
            trades = client.list_recent_closed_trades(count=150)
            for t in trades:
                c_time_str = t.get("closeTime") or t.get("openTime")
                if c_time_str:
                    c_dt = pd.to_datetime(c_time_str).to_pydatetime()
                    if c_dt.tzinfo is None:
                        c_dt = c_dt.replace(tzinfo=timezone.utc)
                    if c_dt >= cutoff_24h:
                        t["subaccount_side"] = side_label
                        t["close_dt"] = c_dt
                        t["open_dt"] = pd.to_datetime(t.get("openTime")).to_pydatetime()
                        closed_trades.append(t)
        except Exception as e:
            print(f"  Error pulling trades from {side_label} account: {e}")

    closed_trades.sort(key=lambda x: x["open_dt"])
    n_real = len(closed_trades)
    print(f"  Total Closed Trades Executed by OANDA: {n_real}")

    real_wins = 0
    real_gross_win = 0.0
    real_gross_loss = 0.0
    real_pnl = 0.0

    trade_records = []
    for t in closed_trades:
        sym = REVERSE_MAP.get(t.get("instrument"), t.get("instrument"))
        pnl = float(t.get("realizedPL", 0.0))
        u = float(t.get("initialUnits", 0.0))
        op = float(t.get("price", 0.0))
        cp = float(t.get("averageClosePrice", 0.0))
        ot = t["open_dt"].strftime("%Y-%m-%d %H:%M:%S")
        ct = t["close_dt"].strftime("%Y-%m-%d %H:%M:%S")
        side = t.get("subaccount_side").upper()
        is_win = pnl > 0
        if is_win:
            real_wins += 1
            real_gross_win += pnl
        else:
            real_gross_loss += abs(pnl)
        real_pnl += pnl
        
        trade_records.append({
            "symbol": sym, "side": side, "units": u, "open_time": ot, "close_time": ct,
            "open_price": op, "close_price": cp, "pnl": pnl, "win": is_win
        })
        print(f"    [{side:4s}] {sym} {u:>6.0f}u | Open: {ot} @ {op:.5f} | Close: {ct} @ {cp:.5f} | PnL: ${pnl:+6.2f} ({'WIN' if is_win else 'LOSS'})")

    real_wr = (real_wins / n_real * 100.0) if n_real > 0 else 0.0
    real_pf = (real_gross_win / real_gross_loss) if real_gross_loss > 0 else float("inf")
    print(f"\n  Real Execution Summary: {n_real} Trades | Win Rate: {real_wr:.2f}% ({real_wins}W / {n_real-real_wins}L) | Net PnL: ${real_pnl:+,.2f} | PF: {real_pf:.3f}")

    # Check open positions
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
            print(f"    Open: {sym} {t.get('subaccount_side').upper()} {u:,.0f}u @ {op:.5f} | Unrealized PnL: ${upl:+,.2f}")
    else:
        print("\n  No positions currently open.")

    # 3. Pull fresh 1-minute M1 candles from OANDA (warmup + last 24h)
    print("\n--- 3. Fetching Full M1 Historical Candle Feeds from OANDA ---")
    data_by_sym = {}
    for sym in SYMBOLS:
        print(f"  Fetching {sym} M1 candles...", end=" ", flush=True)
        df = client_long.fetch_warmup_bars(sym, count=6000)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data_by_sym[sym] = df
        print(f"{len(df):,} bars loaded ({df.index[0]} to {df.index[-1]})")

    # 4. Run Live DTFX Strategy Engine Bar-by-Bar Replay
    print("\n--- 4. Running DTFX Base Strategy Replay with Concurrency Simulation ---")
    cutoff_ts = pd.to_datetime(cutoff_24h.strftime('%Y-%m-%d %H:%M:%S'))
    engines = {sym: SymbolEngine(sym) for sym in SYMBOLS}
    
    # Warmup
    for sym in SYMBOLS:
        df = data_by_sym[sym]
        warmup_df = df[df.index < cutoff_ts]
        engines[sym].seed_zones(warmup_df)
        
    eval_dfs = {sym: data_by_sym[sym][data_by_sym[sym].index >= cutoff_ts] for sym in SYMBOLS}
    all_timestamps = sorted(list(set().union(*[df.index.tolist() for df in eval_dfs.values()])))
    print(f"  Stepping through {len(all_timestamps):,} bars across last 24h timeline...")

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

    # Simulate concurrency (max 1 position per side)
    sim_executed = []
    open_slot = {"buy": None, "sell": None}
    
    for sig in sim_signals:
        sym = sig["symbol"]
        side = sig["side"]
        is_buy = (side == "buy")
        df = data_by_sym[sym]
        sig_time = sig["time"]
        
        current_open = open_slot[side]
        if current_open is not None and current_open["exit_time"] > sig_time:
            continue
            
        bars_after = df[df.index >= sig_time]
        entry_price = sig["mid"]
        stop_price = sig["stop_price"]
        target_price = sig["target_price"]
        eff_risk = abs(entry_price - stop_price)
        
        outcome = None
        exit_time = None
        exit_price = None
        
        for bar_ts, row in bars_after.iterrows():
            h, l = row["high"], row["low"]
            if is_buy:
                if l <= stop_price:
                    outcome = "loss"; exit_time = bar_ts; exit_price = stop_price; break
                elif h >= target_price:
                    outcome = "win"; exit_time = bar_ts; exit_price = target_price; break
            else:
                if h >= stop_price:
                    outcome = "loss"; exit_time = bar_ts; exit_price = stop_price; break
                elif l <= target_price:
                    outcome = "win"; exit_time = bar_ts; exit_price = target_price; break
                    
        trade_res = {
            "symbol": sym,
            "side": side,
            "entry_time": sig_time,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "exit_time": exit_time if exit_time else pd.Timestamp.max,
            "exit_price": exit_price,
            "outcome": outcome if outcome else "OPEN",
            "net_r": (abs(target_price - entry_price) / eff_risk) if outcome == "win" else (-1.0 if outcome == "loss" else 0.0)
        }
        sim_executed.append(trade_res)
        open_slot[side] = trade_res

    df_sim = pd.DataFrame(sim_executed)
    n_sim = len(df_sim)
    sim_wins = (df_sim["outcome"] == "win").sum()
    sim_wr = (sim_wins / n_sim * 100.0) if n_sim > 0 else 0.0
    sim_gw = df_sim[df_sim["net_r"] > 0]["net_r"].sum()
    sim_gl = -df_sim[df_sim["net_r"] <= 0]["net_r"].sum()
    sim_pf = sim_gw / sim_gl if sim_gl > 0 else float("inf")
    sim_tot_r = df_sim["net_r"].sum()

    print(f"\n--- 5. Comparison Summary: Real OANDA PROD vs Backtest Replay ---")
    print(f"{'Metric':<30} | {'Real OANDA Execution':<25} | {'Backtest Replay':<25}")
    print("-" * 88)
    print(f"{'Total Trades':<30} | {n_real:<25} | {n_sim:<25}")
    print(f"{'Win Rate':<30} | {real_wr:>6.2f}% ({real_wins}W/{n_real-real_wins}L){'':<10} | {sim_wr:>6.2f}% ({sim_wins}W/{n_sim-sim_wins}L)")
    print(f"{'Profit Factor':<30} | {real_pf:>6.3f}{'':<19} | {sim_pf:>6.3f}")
    print(f"{'Net Realized PnL':<30} | ${real_pnl:>+8.2f}{'':<17} | {sim_tot_r:>+6.2f} R")

    out_file = os.path.join(DATA_DIR, "overnight_24h_reconciliation.csv")
    df_sim.to_csv(out_file, index=False)
    print(f"\nSaved overnight reconciliation report to: {out_file}")

if __name__ == "__main__":
    main()
