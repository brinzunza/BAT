"""Local trade-analytics dashboard. Reads logs/trades.csv (written by bot.py on every
completed trade) and serves a single-page live view.

Run with: python dashboard.py
Then open: http://localhost:8420

Does NOT require the bot itself to be running -- it only reads the CSV. If OANDA
credentials are available in .env, it also shows a live "currently open position"
banner by querying the account directly; if that call fails for any reason (bot not
running, no credentials, network hiccup), the banner just says so and the rest of the
dashboard works normally from the CSV alone.
"""
import json
import logging
import os
import threading
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

import config

logging.getLogger("oandapyV20").setLevel(logging.WARNING)

PORT = 8420
TRADE_LOG_PATH = os.path.join(config.LOG_DIR, "trades.csv")
RISK_PCT_FOR_SHARPE = 0.01  # matches every backtest Sharpe quoted throughout this project's
                             # research -- NOT config.RISK_PCT, so the number here is directly
                             # comparable to those, not a statement about real position sizing


DB_PATH = os.path.join(config.LOG_DIR, "trades.db")


def load_trades():
    if os.path.exists(DB_PATH):
        try:
            import sqlite3
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY exit_time ASC", conn)
            conn.close()
            if len(df):
                df["entry_time"] = pd.to_datetime(df["entry_time"])
                df["exit_time"] = pd.to_datetime(df["exit_time"])
                df = df.dropna(subset=["exit_time"]).sort_values("exit_time").reset_index(drop=True)
                for col in ["entry_price", "stop_price", "target_price", "exit_price", "realized_pnl", "r_multiple", "qty", "pnl_pct_equity"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
        except Exception:
            pass

    if not os.path.exists(TRADE_LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(TRADE_LOG_PATH, parse_dates=["entry_time", "exit_time"])
    df = df.dropna(subset=["exit_time"]).sort_values("exit_time").reset_index(drop=True)
    for col in ["entry_price", "stop_price", "target_price", "exit_price", "realized_pnl", "r_multiple", "qty"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def sharpe_from_r_multiples(df):
    """Same methodology as every backtest number in this project: build an equity curve
    compounding at a fixed % risk per trade on the R-multiple, resample to daily, annualize
    by sqrt(252). Needs several distinct days of trades to mean anything -- returns None
    otherwise rather than reporting a meaningless number off 2-3 data points."""
    if len(df) < 5:
        return None
    equity = 100.0
    curve = []
    for _, row in df.sort_values("exit_time").iterrows():
        r = row["r_multiple"]
        if pd.isna(r):
            continue
        equity = max(equity * (1 + RISK_PCT_FOR_SHARPE * r), 0.0)
        curve.append((row["exit_time"], equity))
    if len(curve) < 5:
        return None
    s = pd.Series([c[1] for c in curve], index=[c[0] for c in curve])
    daily = s.resample("D").last().ffill()
    ret = daily.pct_change().dropna()
    if len(ret) < 3 or ret.std() == 0:
        return None
    return float(ret.mean() / ret.std() * np.sqrt(252))


def compute_stats(df):
    if len(df) == 0:
        return dict(trades=0)
    wins = df[df["realized_pnl"] > 0]
    losses = df[df["realized_pnl"] <= 0]
    return dict(
        trades=len(df),
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=100 * len(wins) / len(df),
        avg_win=float(wins["realized_pnl"].mean()) if len(wins) else None,
        avg_loss=float(losses["realized_pnl"].mean()) if len(losses) else None,
        largest_win=float(df["realized_pnl"].max()),
        largest_loss=float(df["realized_pnl"].min()),
        sum_pnl=float(df["realized_pnl"].sum()),
        sum_r=float(df["r_multiple"].sum(skipna=True)),
        avg_r=float(df["r_multiple"].mean(skipna=True)),
        sharpe=sharpe_from_r_multiples(df),
    )


def build_payload():
    df = load_trades()
    now = datetime.now(timezone.utc)
    exit_time_utc = df["exit_time"].dt.tz_convert("UTC") if len(df) and df["exit_time"].dt.tz is not None \
        else (df["exit_time"].dt.tz_localize("UTC") if len(df) else df.get("exit_time"))

    windows = {
        "today": now.replace(hour=0, minute=0, second=0, microsecond=0),
        "past_week": now - timedelta(days=7),
        "past_month": now - timedelta(days=30),
        "past_year": now - timedelta(days=365),
        "total": None,
    }
    periods = {}
    for label, since in windows.items():
        sub = df if since is None else df[exit_time_utc >= since]
        periods[label] = compute_stats(sub)

    live_status = get_live_status()

    trades_out = []
    for _, row in df.sort_values("exit_time", ascending=False).iterrows():
        trades_out.append(dict(
            symbol=row["symbol"], side=row["side"],
            entry_time=str(row["entry_time"]), exit_time=str(row["exit_time"]),
            entry_price=row["entry_price"], exit_price=row["exit_price"],
            stop_price=row["stop_price"], target_price=row["target_price"],
            realized_pnl=row["realized_pnl"], r_multiple=row["r_multiple"],
            outcome=row.get("outcome"),
        ))

    return dict(generated_at=now.isoformat(), live_status=live_status,
                periods=periods, trades=trades_out)


def get_live_status():
    try:
        from broker.oanda_client import OANDAClient, SYMBOL_MAP
        if not config.OANDA_API_TOKEN or not config.OANDA_ACCOUNT_ID:
            return dict(available=False, reason="no OANDA credentials in .env")
        client = OANDAClient(config.OANDA_API_TOKEN, config.OANDA_ACCOUNT_ID, config.OANDA_ENVIRONMENT)
        import oandapyV20.endpoints.trades as trades_ep
        r = trades_ep.OpenTrades(accountID=config.OANDA_ACCOUNT_ID)
        client.api.request(r)
        open_trades = r.response.get("trades", [])
        if not open_trades:
            return dict(available=True, open=False)
        t = open_trades[0]
        return dict(available=True, open=True, symbol=t.get("instrument"),
                    side="buy" if float(t.get("currentUnits", 0)) > 0 else "sell",
                    entry_price=float(t.get("price", 0)),
                    unrealized_pnl=float(t.get("unrealizedPL", 0)))
    except Exception as e:
        return dict(available=False, reason=str(e))


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>DTFX Live Trade Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>
:root {
  --bg: #0b0f14; --surface: #121922; --surface-2: #1a232e; --border: #232f3b;
  --text: #e7eef4; --text-muted: #7d8ea0; --accent: #4fd1c5;
  --win: #3fb950; --win-soft: rgba(63,185,80,0.14);
  --loss: #f85149; --loss-soft: rgba(248,81,73,0.14);
  --font-mono: 'SF Mono', 'IBM Plex Mono', ui-monospace, monospace;
  --font-display: -apple-system, 'Segoe UI', system-ui, sans-serif;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: var(--font-mono); margin: 0; padding: 2rem 1.5rem 4rem; }
.page { max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 1.5rem; }
h1 { font-family: var(--font-display); font-weight: 800; font-size: 1.6rem; margin: 0; }
.subtitle { color: var(--text-muted); font-size: 0.85rem; }
.banner { display: flex; align-items: center; gap: 1rem; border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.3rem; background: var(--surface); }
.banner.open { border-left: 3px solid var(--accent); }
.banner.flat { border-left: 3px solid var(--text-muted); }
.banner.unavailable { border-left: 3px solid var(--border); color: var(--text-muted); }
.tabs { display: flex; gap: 0.4rem; flex-wrap: wrap; }
.tab { font-family: var(--font-display); font-size: 0.82rem; padding: 0.5rem 1rem; border-radius: 999px; border: 1px solid var(--border); background: var(--surface); cursor: pointer; color: var(--text-muted); }
.tab.active { background: var(--accent); color: #06231f; border-color: var(--accent); font-weight: 600; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1px; background: var(--border); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.stat { background: var(--surface); padding: 1rem 1.1rem; display: flex; flex-direction: column; gap: 0.3rem; }
.stat-label { font-size: 0.65rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-muted); }
.stat-value { font-family: var(--font-display); font-weight: 700; font-size: 1.3rem; }
.stat-value.win { color: var(--win); } .stat-value.loss { color: var(--loss); }
table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
th { text-align: left; font-size: 0.65rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-muted); padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--surface); }
td { padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--border); }
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.2rem; }
.panel-title { font-family: var(--font-display); font-weight: 700; margin-bottom: 0.8rem; }
.r-pos { color: var(--win); } .r-neg { color: var(--loss); }
.side-buy { color: var(--win); } .side-sell { color: var(--loss); }
.empty { color: var(--text-muted); padding: 2rem; text-align: center; }
#tableWrap { max-height: 480px; overflow: auto; }
.footer { color: var(--text-muted); font-size: 0.72rem; }
</style>
</head>
<body>
<div class="page">
  <div>
    <h1>DTFX Live Trade Dashboard</h1>
    <div class="subtitle" id="genTime">loading...</div>
  </div>
  <div id="banner" class="banner">checking live status...</div>
  <div class="tabs" id="tabs"></div>
  <div class="stat-grid" id="statGrid"></div>
  <div class="panel">
    <div class="panel-title">Trade log</div>
    <div id="tableWrap"><table id="tradeTable"><thead><tr>
      <th>Exit</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit price</th>
      <th>Outcome</th><th>P&amp;L</th><th>R</th>
    </tr></thead><tbody></tbody></table></div>
  </div>
  <div class="footer">Auto-refreshes every 20s. Sharpe uses a 1%-risk-per-trade R-multiple equity curve, matching this project's backtest methodology -- not a statement about real position sizing.</div>
</div>
<script>
const PERIOD_LABELS = {today: "Today", past_week: "Past 7 Days", past_month: "Past 30 Days", past_year: "Past Year", total: "All Time"};
let currentTab = "total";
let latestData = null;

function fmtMoney(x) { return x == null ? "&mdash;" : (x>=0?"+$":"-$") + Math.abs(x).toFixed(2); }
function fmtR(x) { return x == null || isNaN(x) ? "&mdash;" : (x>=0?"+":"") + x.toFixed(2); }
function cls(x) { return x == null ? "" : (x >= 0 ? "r-pos" : "r-neg"); }

function renderBanner(status) {
  const el = document.getElementById("banner");
  if (!status || !status.available) {
    el.className = "banner unavailable";
    el.innerHTML = `Live position status unavailable${status && status.reason ? " (" + status.reason + ")" : ""} &mdash; showing closed-trade history only.`;
  } else if (!status.open) {
    el.className = "banner flat";
    el.innerHTML = `No open position right now.`;
  } else {
    el.className = "banner open";
    el.innerHTML = `<b>Open position:</b> ${status.side.toUpperCase()} ${status.symbol} @ ${status.entry_price.toFixed(5)} &nbsp; unrealized P&amp;L: <span class="${cls(status.unrealized_pnl)}">${fmtMoney(status.unrealized_pnl)}</span>`;
  }
}

function renderTabs() {
  const tabs = document.getElementById("tabs");
  tabs.innerHTML = Object.entries(PERIOD_LABELS).map(([key, label]) =>
    `<div class="tab ${key===currentTab?'active':''}" data-key="${key}">${label}</div>`).join("");
  tabs.querySelectorAll(".tab").forEach(t => t.onclick = () => { currentTab = t.dataset.key; render(); });
}

function renderStats(stats) {
  const grid = document.getElementById("statGrid");
  if (!stats || stats.trades === 0) {
    grid.innerHTML = `<div class="stat" style="grid-column:1/-1"><div class="empty">No completed trades in this period yet.</div></div>`;
    return;
  }
  const wr = stats.win_rate_pct;
  const cards = [
    ["Trades", stats.trades, ""],
    ["Win rate", wr.toFixed(1) + "%", wr>=50?"win":"loss"],
    ["Won / Lost", `${stats.wins} / ${stats.losses}`, ""],
    ["Avg win", fmtMoney(stats.avg_win), "win"],
    ["Avg loss", fmtMoney(stats.avg_loss), "loss"],
    ["Largest win", fmtMoney(stats.largest_win), "win"],
    ["Largest loss", fmtMoney(stats.largest_loss), "loss"],
    ["Sum P&amp;L", fmtMoney(stats.sum_pnl), cls(stats.sum_pnl)],
    ["Sum R", fmtR(stats.sum_r), cls(stats.sum_r)],
    ["Avg R", fmtR(stats.avg_r), cls(stats.avg_r)],
    ["Sharpe", stats.sharpe == null ? "n/a (too few days)" : stats.sharpe.toFixed(2), ""],
  ];
  grid.innerHTML = cards.map(([label, value, c]) =>
    `<div class="stat"><div class="stat-label">${label}</div><div class="stat-value ${c}">${value}</div></div>`).join("");
}

function renderTable(trades) {
  const tbody = document.querySelector("#tradeTable tbody");
  if (!trades.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty">No completed trades yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = trades.map(t => `
    <tr>
      <td>${t.exit_time.replace("T"," ").slice(0,16)}</td>
      <td>${t.symbol}</td>
      <td class="side-${t.side}">${t.side.toUpperCase()}</td>
      <td>${Number(t.entry_price).toFixed(5)}</td>
      <td>${Number(t.exit_price).toFixed(5)}</td>
      <td>${t.outcome || ""}</td>
      <td class="${cls(t.realized_pnl)}">${fmtMoney(t.realized_pnl)}</td>
      <td class="${cls(t.r_multiple)}">${fmtR(t.r_multiple)}</td>
    </tr>`).join("");
}

function render() {
  if (!latestData) return;
  document.getElementById("genTime").textContent = "Last updated: " + new Date(latestData.generated_at).toLocaleString();
  renderBanner(latestData.live_status);
  renderTabs();
  renderStats(latestData.periods[currentTab]);
  renderTable(latestData.trades);
}

async function poll() {
  try {
    const res = await fetch("/api/data");
    latestData = await res.json();
    render();
  } catch (e) { console.error(e); }
}
poll();
setInterval(poll, 20000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep console quiet; this is a local dev tool

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/data":
            try:
                payload = json.dumps(build_payload(), default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    server = ThreadingHTTPServer(("localhost", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Dashboard running at {url} (Ctrl+C to stop)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard")
        server.shutdown()


if __name__ == "__main__":
    main()
