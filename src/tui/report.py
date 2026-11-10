"""Backtest analytics for the TUI.

``BacktestEngine`` returns one row per *execution* (open/close). This module
turns that into things a trader wants to look at: a mark-to-market equity curve
per bar, drawdown, round trips and risk-adjusted metrics.

Annualisation: periods per year are inferred from the sample itself
(``periods / years``), so crypto (24/7) and equities (weekday sessions) both
come out consistent without hard-coding 252 or 365.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

OPEN_ACTIONS = {"BUY", "SELL_SHORT"}
CLOSE_ACTIONS = {"CLOSE", "CLOSE_SHORT", "CLOSE_LONG"}


@dataclass
class BacktestReport:
    summary: Dict[str, object]
    equity: pd.Series                 # mark-to-market, one value per bar
    drawdown: pd.Series               # fraction <= 0
    round_trips: pd.DataFrame
    executions: pd.DataFrame
    bars: pd.DataFrame
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    open_trade: Optional[Dict[str, object]] = None
    initial_cash: float = 0.0
    markers: List[tuple] = field(default_factory=list)  # (bar_index, 'buy'|'sell'|'exit')


def equity_curve(trades: pd.DataFrame, bars: pd.DataFrame, initial_cash: float) -> pd.Series:
    """Replay executions over the bars: equity = cash + signed_qty * close."""
    n = len(bars)
    close = bars["Close"].to_numpy(float)
    idx = bars["timestamp"] if "timestamp" in bars else pd.RangeIndex(n)
    if trades is None or len(trades) == 0:
        return pd.Series(np.full(n, initial_cash), index=idx, name="equity")
    ev_idx = trades["Index"].to_numpy(int)
    ev_cash = trades["Balance"].to_numpy(float)
    ev_qty = (trades["Position"].to_numpy(float) * trades["Shares"].to_numpy(float))
    ev_qty = np.where(trades["Position"].to_numpy() == 0, 0.0, ev_qty)
    pos = np.searchsorted(ev_idx, np.arange(n), side="right") - 1
    cash = np.where(pos >= 0, ev_cash[np.clip(pos, 0, None)], initial_cash)
    qty = np.where(pos >= 0, ev_qty[np.clip(pos, 0, None)], 0.0)
    return pd.Series(cash + qty * close, index=idx, name="equity")


def pair_round_trips(trades: pd.DataFrame, bars: Optional[pd.DataFrame] = None):
    """Match opening and closing executions. Returns (round_trips, open_trade)."""
    rows, current = [], None
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=_RT_COLS), None
    for _, t in trades.iterrows():
        action = str(t["Action"]).upper()
        if action in OPEN_ACTIONS:
            current = t
        elif action in CLOSE_ACTIONS and current is not None:
            side = "LONG" if str(current["Action"]).upper() == "BUY" else "SHORT"
            entry, exit_ = float(current["Price"]), float(t["Price"])
            shares = float(current["Shares"])
            pnl = float(t["Profit"]) if pd.notna(t.get("Profit")) else (
                (exit_ - entry) * shares if side == "LONG" else (entry - exit_) * shares)
            cost = entry * shares if entry * shares else np.nan
            rows.append(dict(side=side, entry_time=current["Time"], exit_time=t["Time"], entry_price=entry,
                             exit_price=exit_, shares=shares, pnl=pnl, ret_pct=pnl / cost * 100 if cost else np.nan,
                             bars=int(t["Index"]) - int(current["Index"]),
                             entry_bar=int(current["Index"]), exit_bar=int(t["Index"])))
            current = None
    rt = pd.DataFrame(rows, columns=_RT_COLS)
    if len(rt):
        rt["duration"] = pd.to_datetime(rt["exit_time"]) - pd.to_datetime(rt["entry_time"])
    open_trade = None
    if current is not None:
        open_trade = dict(side="LONG" if str(current["Action"]).upper() == "BUY" else "SHORT",
                          entry_time=current["Time"], entry_price=float(current["Price"]),
                          shares=float(current["Shares"]), entry_bar=int(current["Index"]))
        if bars is not None and len(bars):
            last = float(bars["Close"].iloc[-1])
            e, s = open_trade["entry_price"], open_trade["shares"]
            open_trade["unrealized"] = (last - e) * s if open_trade["side"] == "LONG" else (e - last) * s
    return rt, open_trade


_RT_COLS = ["side", "entry_time", "exit_time", "entry_price", "exit_price", "shares", "pnl", "ret_pct", "bars",
            "entry_bar", "exit_bar"]


def _max_streak(flags: np.ndarray, value: bool) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f == value else 0
        best = max(best, cur)
    return best


def _periodic_returns(equity: pd.Series):
    """Resample equity to a sensible frequency; return (returns, periods_per_year, label)."""
    if not isinstance(equity.index, pd.DatetimeIndex) or len(equity) < 3:
        r = equity.pct_change().dropna()
        return r, np.nan, "bar"
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    if days >= 20:
        e, label = equity.resample("1D").last().dropna(), "daily"
    elif days >= 2:
        e, label = equity.resample("1h").last().dropna(), "hourly"
    else:
        e, label = equity, "bar"
    r = e.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    years = max(days / 365.25, 1e-9)
    return r, len(r) / years, label


def build_report(trades: pd.DataFrame, bars: pd.DataFrame, initial_cash: float) -> BacktestReport:
    bars = bars.reset_index(drop=True)
    if "timestamp" in bars:
        bars = bars.copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    equity = equity_curve(trades, bars, initial_cash)
    peak = equity.cummax()
    dd = (equity / peak - 1.0).fillna(0.0)
    rt, open_trade = pair_round_trips(trades, bars)

    final = float(equity.iloc[-1]) if len(equity) else initial_cash
    total_ret = final / initial_cash - 1
    if isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 1:
        span = equity.index[-1] - equity.index[0]
        days = span.total_seconds() / 86400
    else:
        days = float("nan")
    cagr = (1 + total_ret) ** (365.25 / days) - 1 if days and days >= 30 and 1 + total_ret > 0 else np.nan

    rets, ppy, freq_label = _periodic_returns(equity)
    vol = rets.std() * np.sqrt(ppy) if len(rets) > 2 and np.isfinite(ppy) else np.nan
    sharpe = rets.mean() / rets.std() * np.sqrt(ppy) if len(rets) > 2 and rets.std() > 0 and np.isfinite(ppy) else np.nan
    downside = np.sqrt(np.mean(np.minimum(rets, 0) ** 2)) if len(rets) > 2 else np.nan
    sortino = rets.mean() / downside * np.sqrt(ppy) if downside and downside > 0 and np.isfinite(ppy) else np.nan
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = (cagr if np.isfinite(cagr) else total_ret) / abs(max_dd) if max_dd < 0 else np.nan

    # longest time under water
    under = (dd < 0).to_numpy()
    longest, run = 0, 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)

    wins = rt[rt["pnl"] > 0]["pnl"]
    losses = rt[rt["pnl"] <= 0]["pnl"]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    n_trades = len(rt)
    win_rate = len(wins) / n_trades if n_trades else np.nan
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    payoff = avg_win / abs(avg_loss) if len(wins) and len(losses) and avg_loss != 0 else np.nan
    pf = gross_win / gross_loss if gross_loss > 0 else (np.inf if gross_win > 0 else np.nan)
    expectancy = rt["pnl"].mean() if n_trades else np.nan

    qty_held = np.zeros(len(bars))
    if len(trades):
        ev_idx = trades["Index"].to_numpy(int)
        pos_after = np.where(trades["Position"].to_numpy() == 0, 0.0, trades["Position"].to_numpy(float))
        p = np.searchsorted(ev_idx, np.arange(len(bars)), side="right") - 1
        qty_held = np.where(p >= 0, pos_after[np.clip(p, 0, None)], 0.0)
    exposure = float((qty_held != 0).mean()) if len(bars) else 0.0

    close = bars["Close"].to_numpy(float)
    bh = close[-1] / close[0] - 1 if len(close) > 1 and close[0] else np.nan

    side_stats = {}
    for side in ("LONG", "SHORT"):
        sub = rt[rt["side"] == side]
        side_stats[side] = dict(n=len(sub), pnl=float(sub["pnl"].sum()) if len(sub) else 0.0,
                                win_rate=float((sub["pnl"] > 0).mean()) if len(sub) else np.nan)

    summary: Dict[str, object] = dict(
        initial=initial_cash, final=final, net_pnl=final - initial_cash, total_return=total_ret, cagr=cagr,
        sharpe=sharpe, sortino=sortino, volatility=vol, max_drawdown=max_dd, calmar=calmar,
        max_dd_bars=longest, trades=n_trades, win_rate=win_rate, profit_factor=pf, expectancy=expectancy,
        avg_win=avg_win, avg_loss=avg_loss, payoff=payoff,
        best_trade=float(rt["pnl"].max()) if n_trades else np.nan,
        worst_trade=float(rt["pnl"].min()) if n_trades else np.nan,
        max_consec_wins=_max_streak((rt["pnl"] > 0).to_numpy(), True) if n_trades else 0,
        max_consec_losses=_max_streak((rt["pnl"] > 0).to_numpy(), False) if n_trades else 0,
        avg_bars=float(rt["bars"].mean()) if n_trades else np.nan,
        avg_duration=rt["duration"].mean() if n_trades else pd.NaT,
        exposure=exposure, buy_hold=bh, alpha=total_ret - bh if np.isfinite(bh) else np.nan,
        days=days, bars=len(bars), freq=freq_label, long=side_stats["LONG"], short=side_stats["SHORT"],
        open_unrealized=(open_trade or {}).get("unrealized", 0.0),
    )

    monthly = pd.DataFrame()
    if isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 2:
        try:
            m = equity.resample("ME").last().dropna()
        except ValueError:  # pandas < 2.2 spells month-end "M"
            m = equity.resample("M").last().dropna()
        base = pd.concat([pd.Series([initial_cash]), m.reset_index(drop=True)], ignore_index=True)
        mr = base.pct_change().iloc[1:].to_numpy() * 100
        if len(m):
            tmp = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": mr})
            monthly = tmp.pivot(index="year", columns="month", values="ret").reindex(columns=range(1, 13))

    markers = []
    for _, t in (trades.iterrows() if len(trades) else []):
        a = str(t["Action"]).upper()
        kind = "buy" if a == "BUY" else "sell" if a == "SELL_SHORT" else "exit"
        markers.append((int(t["Index"]), kind))

    return BacktestReport(summary=summary, equity=equity, drawdown=dd, round_trips=rt, executions=trades, bars=bars,
                          monthly=monthly, open_trade=open_trade, initial_cash=initial_cash, markers=markers)
