"""Query SQLite Database and OANDA API for last 24h of trades, positions, orders, and transactions.
"""
import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime, timezone, timedelta

ROOT = '/Users/brunoinzunza/Documents/github/PRODLIVETRADING'
sys.path.insert(0, ROOT)
import config
from broker.oanda_client import OANDAClient
import oandapyV20.endpoints.transactions as trans
import oandapyV20.endpoints.trades as o_trades
import oandapyV20.endpoints.orders as o_orders

print("=" * 90)
print("  CHECKING LOCAL / VM SQLITE DATABASE")
print("=" * 90)

for env in ['practice', 'live']:
    db_path = os.path.join(ROOT, 'logs', env, 'trades.db')
    print(f"\n--- Database Environment: {env.upper()} ({db_path}) ---")
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        try:
            trades = pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC LIMIT 20", conn)
            print(f"Closed Trades in DB ({len(trades)}):")
            if len(trades) > 0:
                print(trades.to_string())
            else:
                print("  (No closed trades recorded yet)")
        except Exception as e:
            print(f"Error reading trades table: {e}")

        try:
            active = pd.read_sql_query("SELECT * FROM active_positions", conn)
            print(f"\nActive Open Positions in DB ({len(active)}):")
            if len(active) > 0:
                print(active.to_string())
            else:
                print("  (No active positions open)")
        except Exception as e:
            print(f"Error reading active_positions: {e}")

        try:
            pending = pd.read_sql_query("SELECT * FROM pending_orders", conn)
            print(f"\nPending Orders in DB ({len(pending)}):")
            if len(pending) > 0:
                print(pending.to_string())
            else:
                print("  (No pending resting orders)")
        except Exception as e:
            print(f"Error reading pending_orders: {e}")
        conn.close()
    else:
        print(f"  (DB file not found at {db_path})")

print("\n" + "=" * 90)
print("  CHECKING LIVE OANDA REST API FOR LAST 24 HOURS")
print("=" * 90)

c_long = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_LONG, environment=config.OANDA_ENVIRONMENT, label="LONG-002")
c_short = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID_SHORT, environment=config.OANDA_ENVIRONMENT, label="SHORT-003")

since_time = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%SZ')
print(f"Querying events since: {since_time} UTC (Last 24 Hours)\n")

for name, client in [('LONG SUB-ACCOUNT (002)', c_long), ('SHORT SUB-ACCOUNT (003)', c_short)]:
    bal = client.account_balance()
    print(f"🔹 {name}: Balance = ${bal:,.2f}")
    
    # 1. Open trades
    r_tr = o_trades.OpenTrades(accountID=client.account_id)
    tr_resp = client.api.request(r_tr)
    open_tr = tr_resp.get('trades', [])
    print(f"   • Open Positions on OANDA ({len(open_tr)}):")
    if open_tr:
        for ot in open_tr:
            print(f"     ID: {ot.get('id')} | {ot.get('instrument')} | Units: {ot.get('currentUnits')} | Entry: {ot.get('price')} | Unrealized PnL: ${float(ot.get('unrealizedPL', 0)):+,.2f} | Open Time: {ot.get('openTime')}")
    else:
        print("     (None)")

    # 2. Pending resting orders
    r_ord = o_orders.OrdersPending(accountID=client.account_id)
    ord_resp = client.api.request(r_ord)
    pend_ord = ord_resp.get('orders', [])
    print(f"   • Resting Pending Orders on OANDA ({len(pend_ord)}):")
    if pend_ord:
        for po in pend_ord:
            print(f"     ID: {po.get('id')} | Type: {po.get('type')} | {po.get('instrument')} | Units: {po.get('units')} | Price: {po.get('price')} | Created: {po.get('createTime')}")
    else:
        print("     (None)")

    # 3. Transaction history in last 24h
    try:
        r_hist = trans.TransactionsSinceID(accountID=client.account_id, params={'id': 1})
        tx_resp = client.api.request(r_hist)
        all_tx = tx_resp.get('transactions', [])
        recent_tx = [t for t in all_tx if t.get('time', '') >= since_time]
        print(f"   • OANDA Transactions in Last 24 Hours ({len(recent_tx)}):")
        if recent_tx:
            for t in recent_tx:
                t_type = t.get('type')
                t_time = t.get('time')
                t_id = t.get('id')
                t_inst = t.get('instrument', '')
                t_units = t.get('units', '')
                t_pl = t.get('pl', '')
                t_reason = t.get('reason', '')
                print(f"     [{t_time}] {t_type:20s} ID:{t_id:5s} {t_inst:7s} {t_units:6s} PnL:{t_pl:6s} {t_reason}")
        else:
            print("     (No transactions occurred in the last 24 hours)")
    except Exception as e:
        print(f"     Error querying transactions: {e}")
    print()

print("=" * 90)
