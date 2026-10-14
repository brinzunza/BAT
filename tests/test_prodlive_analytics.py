import sqlite3, numpy as np, pandas as pd, pytest
from src.analytics.prodlive import queries as Q, sync as S
SCHEMA = """CREATE TABLE trades(id INTEGER PRIMARY KEY AUTOINCREMENT, account_id TEXT, side TEXT, oanda_trade_id TEXT, symbol TEXT, entry_time TEXT, entry_price REAL, stop_price REAL, backstop_price REAL, target_price REAL, qty REAL, exit_time TEXT, exit_price REAL, realized_pnl REAL, financing REAL, commission REAL, entry_slippage_cost REAL, exit_slippage_cost REAL, outcome TEXT, r_multiple REAL, pnl_pct_equity REAL, zone_top REAL, zone_bottom REAL, zone_mid REAL, zone_created_at TEXT, zone_age_minutes REAL, entry_hour_utc INTEGER, combined_balance_at_entry REAL, risk_pct_configured REAL, holding_duration_seconds REAL, created_at TEXT, updated_at TEXT);
CREATE TABLE active_positions(id INTEGER PRIMARY KEY, symbol TEXT, opened_at TEXT, entry_time TEXT); CREATE TABLE pending_orders(id INTEGER PRIMARY KEY, status TEXT);
CREATE TABLE account_snapshots(id INTEGER PRIMARY KEY, timestamp TEXT, combined_balance REAL, combined_nav REAL, open_positions_count INTEGER); CREATE TABLE system_events(id INTEGER PRIMARY KEY, timestamp TEXT, event_type TEXT, symbol TEXT, details_json TEXT);"""
@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"; c = sqlite3.connect(p); c.executescript(SCHEMA)
    rows = [("A", "buy", "1", "EUR_USD", "2026-09-01T08:00:00+00:00", 1.1, "2026-09-01T10:00:00+00:00", 20.0, -1.0, 1.0, 0.1, 7200, "TP", 2.0),
            ("A", "sell", "2", "EUR_USD", "2026-09-02T08:00:00+00:00", 1.1, "2026-09-02T09:00:00+00:00", -10.0, 0.0, 0.0, 0.1, 3600, "SL", -1.0),
            ("A", "buy", "3", "GBP_USD", "2026-09-03T08:00:00+00:00", 1.3, "2026-09-03T12:00:00+00:00", 10.0, 0.0, 0.0, 0.1, 14400, "TP", 1.0),
            ("A", "buy", "4", "GBP_USD", "2026-09-04T08:00:00+00:00", 1.3, None, None, None, None, None, None, None, None)]
    for a, s, i, sym, et, ep, xt, pnl, fin, com, sl, hold, out, r in rows:
        c.execute("insert into trades(account_id,side,oanda_trade_id,symbol,entry_time,entry_price,qty,exit_time,realized_pnl,financing,commission,entry_slippage_cost,exit_slippage_cost,outcome,r_multiple,holding_duration_seconds,created_at,updated_at,entry_hour_utc) values(?,?,?,?,?,?,1,?,?,?,?,?,0,?,?,?,'x','x',8)", (a, s, i, sym, et, ep, xt, pnl, fin, com, sl, out, r, hold))
    for k, (t, b) in enumerate([("2026-09-01T00:00:00+00:00", 200), ("2026-09-02T00:00:00+00:00", 220), ("2026-09-03T00:00:00+00:00", 198), ("2026-09-04T00:00:00+00:00", 210)]): c.execute("insert into account_snapshots values(?,?,?,?,0)", (k, t, b, b))
    c.execute("insert into system_events values(1,'2026-09-01T00:00:00+00:00','START',null,null)"); c.commit(); c.close(); return str(p)
def test_trades_excludes_open_and_nets_costs(db):
    d = Q.trades(db); assert len(d) == 3 and d.net_pnl.round(6).tolist() == [18.0, -10.0, 10.0] and d.win.tolist() == [True, False, True]
def test_summary(db):
    s = Q.summary(db); assert s["n"] == 3 and abs(s["win_rate"] - 2 / 3) < 1e-9 and s["net_pnl"] == 18 and s["profit_factor"] == 2.8 and abs(s["avg_r"] - 2 / 3) < 1e-9
    assert s["max_drawdown"] == pytest.approx(198 / 220 - 1) and s["latest_nav"] == 210
def test_breakdown_and_filters(db):
    b = Q.breakdown("symbol", db); assert b.loc["EUR_USD", "n"] == 2 and b.loc["GBP_USD", "net_pnl"] == 10
    assert len(Q.trades(db, symbol="GBP_USD")) == 1 and len(Q.trades(db, since="2026-09-02")) == 2
def test_sql_is_read_only(db):
    with pytest.raises(Exception): Q.sql("delete from trades", db)
def test_sync_local_snapshot(db, tmp_path, monkeypatch):
    monkeypatch.setattr(S, "DATA", tmp_path / "data"); monkeypatch.setattr(S, "LATEST", tmp_path / "data" / "latest.db")
    out = S.sync(src=db); assert out.exists() and len(Q.trades(str(out))) == 3
def test_sync_rejects_non_bot_db(tmp_path, monkeypatch):
    bad = tmp_path / "bad.db"; sqlite3.connect(bad).execute("create table x(a)").connection.commit()
    monkeypatch.setattr(S, "DATA", tmp_path / "data"); monkeypatch.setattr(S, "LATEST", tmp_path / "data" / "latest.db")
    with pytest.raises(SystemExit): S.sync(src=str(bad))
