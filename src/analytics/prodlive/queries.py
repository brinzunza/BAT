"""Analytics over a PRODLIVETRADING trades.db snapshot. Pure functions: path -> pandas objects. Read-only."""
import json, sqlite3
import numpy as np, pandas as pd
from pathlib import Path
from .sync import LATEST
def connect(path=None):
    p = Path(path or LATEST)
    if not p.exists(): raise FileNotFoundError(f"{p} not found - run `python -m analytics.prodlive sync` first")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)
def _q(sql, path=None, params=()):
    c = connect(path)
    try: return pd.read_sql_query(sql, c, params=params)
    finally: c.close()
def _times(df, cols):
    for c in cols:
        if c in df: df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df
def trades(path=None, account=None, symbol=None, since=None):
    d = _times(_q("select * from trades order by exit_time", path), ["entry_time", "exit_time", "zone_created_at"])
    d = d[d.exit_time.notna()]
    if account: d = d[d.account_id == account]
    if symbol: d = d[d.symbol == symbol]
    if since: d = d[d.exit_time >= pd.Timestamp(since, tz="UTC")]
    d = d.copy(); d["net_pnl"] = d.realized_pnl.fillna(0) + d.financing.fillna(0) - d.commission.fillna(0).abs()
    d["win"] = d.net_pnl > 0; return d.reset_index(drop=True)
def snapshots(path=None): return _times(_q("select * from account_snapshots order by timestamp", path), ["timestamp"])
def open_positions(path=None): return _times(_q("select * from active_positions", path), ["opened_at", "entry_time"])
def pending_orders(path=None): return _q("select * from pending_orders where status='PENDING'", path)
def events(path=None, n=200, kinds=None):
    d = _times(_q("select * from system_events order by timestamp desc limit ?", path, (n,)), ["timestamp"])
    return d[d.event_type.isin(kinds)] if kinds else d
def equity_curve(path=None):
    s = snapshots(path)
    if s.empty: return s
    s["peak"] = s.combined_nav.fillna(s.combined_balance).cummax(); eq = s.combined_nav.fillna(s.combined_balance); s["drawdown"] = eq / s.peak - 1; s["equity"] = eq; return s
def stats_of(d, snaps=None):
    """Trade stats. R-multiple fields use the bot's own r_multiple column (NaN-safe)."""
    if d.empty: return dict(n=0)
    r = d.r_multiple.dropna(); w = d[d.win]; l = d[~d.win]; gp, gl = w.net_pnl.sum(), -l.net_pnl.sum()
    out = dict(n=len(d), win_rate=d.win.mean(), net_pnl=d.net_pnl.sum(), avg_win=w.net_pnl.mean() if len(w) else np.nan, avg_loss=l.net_pnl.mean() if len(l) else np.nan,
               profit_factor=gp / gl if gl > 0 else np.inf, avg_r=r.mean() if len(r) else np.nan, median_r=r.median() if len(r) else np.nan, total_r=r.sum(),
               avg_hold_h=d.holding_duration_seconds.mean() / 3600 if "holding_duration_seconds" in d else np.nan,
               first=d.entry_time.min(), last=d.exit_time.max(), days=max((d.exit_time.max() - d.entry_time.min()).days, 1))
    out["trades_per_week"] = out["n"] / (out["days"] / 7)
    out["outcomes"] = d.outcome.value_counts().to_dict() if "outcome" in d else {}
    sl = (d.entry_slippage_cost.fillna(0) + d.exit_slippage_cost.fillna(0)); out["slippage_total"] = sl.sum(); out["slippage_per_trade"] = sl.mean()
    if snaps is not None and len(snaps) > 2:
        eq = snaps.set_index("timestamp").equity.resample("1D").last().dropna(); ret = eq.pct_change().dropna()
        out["max_drawdown"] = snaps.drawdown.min(); out["return_pct"] = eq.iloc[-1] / eq.iloc[0] - 1
        out["daily_sharpe"] = ret.mean() / ret.std() * np.sqrt(252) if len(ret) > 5 and ret.std() > 0 else np.nan
    return out
def summary(path=None, **flt):
    d = trades(path, **flt); s = equity_curve(path); out = stats_of(d, s if len(s) else None)
    out["open_positions"] = len(open_positions(path)); out["pending_orders"] = len(pending_orders(path))
    if len(s): out["latest_balance"] = s.combined_balance.iloc[-1]; out["latest_nav"] = s.equity.iloc[-1]; out["snapshot_time"] = s.timestamp.iloc[-1]
    return out
def breakdown(by, path=None, **flt):
    """by in: symbol, side, entry_hour_utc, outcome, weekday, month, account_id"""
    d = trades(path, **flt)
    if d.empty: return d
    d["weekday"] = d.exit_time.dt.day_name(); d["month"] = d.exit_time.dt.strftime("%Y-%m")
    g = d.groupby(by).agg(n=("win", "size"), win_rate=("win", "mean"), net_pnl=("net_pnl", "sum"), avg_r=("r_multiple", "mean"), total_r=("r_multiple", "sum"), avg_hold_h=("holding_duration_seconds", lambda x: x.mean() / 3600))
    return g.sort_values("net_pnl", ascending=False)
def period_returns(freq="W", path=None, **flt):
    d = trades(path, **flt)
    if d.empty: return d
    g = d.set_index("exit_time").net_pnl.resample(freq).agg(["sum", "count"]); g.columns = ["net_pnl", "trades"]; return g[g.trades > 0]
def vs_expectation(path=None, exp_win_rate=None, exp_avg_r=None, **flt):
    """Compare live results with the backtest expectation (win rate / avg R) with a binomial z-score for the win rate and a t-stat for avg R."""
    d = trades(path, **flt); out = {}; n = len(d)
    if n and exp_win_rate is not None: p = exp_win_rate; out["win_rate"] = (d.win.mean(), p, (d.win.mean() - p) / np.sqrt(p * (1 - p) / n))
    r = d.r_multiple.dropna()
    if len(r) > 2 and exp_avg_r is not None: out["avg_r"] = (r.mean(), exp_avg_r, (r.mean() - exp_avg_r) / (r.std(ddof=1) / np.sqrt(len(r))))
    out["n"] = n; return out
def sql(query, path=None):
    """Ad-hoc read-only SQL (connection is opened mode=ro so writes fail)."""
    return _q(query, path)
