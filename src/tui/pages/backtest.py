"""Backtest page: configure, run, analyse and compare strategy runs."""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, ProgressBar, Select, Static, TabbedContent, TabPane

from src.core.adapters.legacy import LegacyDataProviderAdapter
from src.core.application.backtest import BacktestService
from src.core.domain.models import BacktestConfig
from src.tui import charts as C
from src.tui.datasets import DatasetInfo, scan_datasets, short_name
from src.tui.diagnostics import (Finding, check_config, check_data, check_result, check_signals,
                                 explain_exception)
from src.tui.report import BacktestReport, build_report
from src.tui.state import BacktestRecord
from src.tui.strategies import STRATEGIES, strategy_options
from src.tui.widgets.common import (CardGrid, EventLog, LabeledInput, LabeledSelect, ParamForm, Plot, Section, fmt_td,
                                    pnl_text, select_value)
from src.tui.widgets.price import PriceChart

ROOT = Path(__file__).resolve().parents[3]
CARDS = {
    "pnl": "Net P&L", "ret": "Return", "cagr": "CAGR", "final": "Final equity",
    "sharpe": "Sharpe", "sortino": "Sortino", "dd": "Max drawdown", "calmar": "Calmar",
    "trades": "Round trips", "win": "Win rate", "pf": "Profit factor", "exp": "Expectancy / trade",
    "payoff": "Payoff ratio", "expo": "Time in market", "bh": "Buy & hold", "alpha": "Alpha vs B&H",
}
TRIP_COLS = [("#", "n"), ("Side", "side"), ("Entry", "entry_time"), ("Exit", "exit_time"), ("Entry px", "entry_price"),
             ("Exit px", "exit_price"), ("Qty", "shares"), ("P&L", "pnl"), ("Ret %", "ret_pct"), ("Held", "duration")]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
RESAMPLE = [("Native", "none"), ("5 min", "5min"), ("15 min", "15min"), ("30 min", "30min"), ("1 hour", "1h"),
            ("4 hours", "4h"), ("1 day", "1D")]


def symbol_from_filename(name: str) -> str:
    """``X_BTCUSD_minute_…csv`` -> ``X:BTCUSD`` (Polygon naming, used for forex spread logic)."""
    m = re.match(r"^([A-Z])_([A-Z0-9]+)_", name)
    return f"{m.group(1)}:{m.group(2)}" if m else name.split("_")[0].split(".")[0]


def fmt_params(vals: Dict[str, Any]) -> str:
    return ",".join(f"{v:g}" if isinstance(v, float) else str(v) for v in vals.values())


class _SignalCache:
    """Hands the already-computed signal frame to ``BacktestService`` so indicators are not computed twice."""

    def __init__(self, strategy: Any, signals: pd.DataFrame):
        self._strategy, self._signals = strategy, signals

    def __getattr__(self, name: str) -> Any:
        return getattr(self._strategy, name)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._signals.copy()


class BacktestPage(Widget):
    BINDINGS = [Binding("ctrl+r", "run", "Run backtest")]

    def __init__(self) -> None:
        super().__init__()
        self._bt_busy = False
        self._bt_skip = threading.Event()          # 'Skip' during the replay
        self._bt_axis: Dict[str, Any] = {}
        self._bt_frames = 0
        self._bt_ds: List[DatasetInfo] = []
        self._bt_rec: Optional[BacktestRecord] = None
        self._bt_trips: Optional[pd.DataFrame] = None
        self._bt_sort: Dict[str, Any] = {"col": None, "rev": False}

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        first = next(iter(STRATEGIES.values()))
        with Horizontal():
            with VerticalScroll(classes="config"):
                yield Section("Data")
                yield LabeledSelect("Source", [("Local dataset", "local")], select_id="bt-source", value="local")
                with Vertical(id="bt-local-box"):
                    yield LabeledSelect("Dataset", [("(scanning…)", "")], select_id="bt-dataset")
                    with Horizontal(classes="row"):
                        yield LabeledInput("From (optional)", "", placeholder="YYYY-MM-DD", input_id="bt-lfrom")
                        yield LabeledInput("To (optional)", "", placeholder="YYYY-MM-DD", input_id="bt-lto")
                    yield LabeledSelect("Resample to", RESAMPLE, select_id="bt-resample", value="none")
                with Vertical(id="bt-provider-box", classes="hidden"):
                    yield LabeledSelect("Timespan", [("Minute", "minute"), ("Hour", "hour"), ("Day", "day")],
                                        select_id="bt-timespan", value="minute")
                    with Horizontal(classes="row"):
                        yield LabeledInput("From", (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                                           input_id="bt-from")
                        yield LabeledInput("To", (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                                           input_id="bt-to")
                    yield LabeledInput("Max bars", "50000", input_id="bt-limit")
                yield LabeledInput("Symbol / ticker", "X:BTCUSD", input_id="bt-ticker",
                                   placeholder="AAPL · X:BTCUSD · C:EURUSD")
                yield Section("Strategy")
                yield LabeledSelect("Strategy", strategy_options(), select_id="bt-strategy", value=first.key)
                yield Static(first.description, id="bt-desc", classes="muted")
                yield ParamForm(first, prefix="bt-p", id="bt-params")
                yield Static("", id="bt-warn")
                yield Section("Execution")
                yield LabeledSelect("Trading mode", [("Long only (buy / close)", "long_only"),
                                                     ("Long / short (flip)", "long_short")],
                                    select_id="bt-mode", value="long_only")
                with Horizontal(classes="row"):
                    yield LabeledInput("Initial balance", "10000", input_id="bt-balance")
                    yield LabeledInput("Per trade %", "100", input_id="bt-pos")
                yield LabeledInput("Spread (pips, forex)", "1.0", input_id="bt-spread")
                yield LabeledSelect("Replay speed", [("Instant (no animation)", "0"), ("Fast · ~2 s", "2"),
                                                     ("Normal · ~6 s", "6"), ("Slow · ~15 s", "15")],
                                    select_id="bt-replay", value="2")
                with Horizontal(classes="btn-row actions"):
                    yield Button("Run ▶", id="bt-run", variant="primary")
                    yield Button("Defaults", id="bt-reset")
            with Vertical(classes="results"):
                yield Static("Configure a run, then press [b]Run ▶[/b] (or ctrl+r).", id="bt-status", classes="status-line")
                with Vertical(id="bt-runpanel"):
                    with Horizontal(id="bt-runbar"):
                        yield ProgressBar(total=100, show_eta=False, id="bt-progress")
                        yield Button("Skip ⏭", id="bt-skip", compact=True, disabled=True)
                    yield EventLog(id="bt-runlog")
                with TabbedContent(id="bt-tabs"):
                    with TabPane("Overview", id="tab-overview"):
                        with VerticalScroll():
                            yield CardGrid(CARDS, columns=4, id="bt-cards")
                            yield Static("", id="bt-note")
                            yield Plot(id="bt-equity")
                            yield Plot(id="bt-dd")
                    with TabPane("Price & signals", id="tab-price"):
                        yield Static("", id="bt-view", classes="muted")
                        yield PriceChart(id="bt-price")
                    with TabPane("Trades", id="tab-trades"):
                        with Horizontal(classes="btn-row"):
                            yield Select([("All trades", "all"), ("Long", "long"), ("Short", "short"),
                                          ("Winners", "win"), ("Losers", "loss")], value="all", allow_blank=False,
                                         id="bt-trip-filter", compact=True)
                            yield Static("", id="bt-trip-info")
                        yield DataTable(id="bt-trips", cursor_type="row", zebra_stripes=True)
                    with TabPane("Analysis", id="tab-analysis"):
                        with VerticalScroll():
                            yield Plot(id="bt-hist")
                            yield Plot(id="bt-hour")
                            yield Plot(id="bt-weekday")
                            yield Static("", id="bt-monthly")
                            yield Static("", id="bt-risk")
                    with TabPane("History", id="tab-history"):
                        with Horizontal(classes="btn-row"):
                            yield Button("Open", id="bt-open")
                            yield Button("Export…", id="bt-export")
                            yield Button("Load settings", id="bt-load")
                            yield Button("Clear", id="bt-clear")
                        yield DataTable(id="bt-history", cursor_type="row", zebra_stripes=True)
                        yield Plot(id="bt-compare")

    def on_mount(self) -> None:
        trips = self.query_one("#bt-trips", DataTable)
        for label, key in TRIP_COLS:
            trips.add_column(label, key=key)
        hist = self.query_one("#bt-history", DataTable)
        hist.add_columns("#", "Time", "Strategy", "Symbol", "Mode", "Return", "Sharpe", "Max DD", "Trades", "Win %")
        self._scan()
        self._refresh_sources()
        self._validate_form()
        log = self.query_one("#bt-runlog", EventLog)
        log.add("info", "Run log: every step of a backtest — data checks, signals, simulation, warnings — shows up here.")

    def on_show(self) -> None:
        self._refresh_sources()

    # -------------------------------------------------------------- data sources
    @work(thread=True, exclusive=True, group="bt-scan")
    def _scan(self) -> None:
        infos = scan_datasets()
        self.app.call_from_thread(self._set_datasets, infos)

    def _set_datasets(self, infos: List[DatasetInfo]) -> None:
        self._bt_ds = infos
        sel = self.query_one("#bt-dataset", Select)
        if infos:
            opts = [(f"{short_name(i.name, 30)} · {i.rows:,} × {i.timeframe}", str(i.path)) for i in infos]
        else:
            opts = [("No datasets — download one in Research", "")]
        sel.set_options(opts)
        sel.value = opts[0][1]
        self._sync_symbol()

    def _refresh_sources(self) -> None:
        sel = self.query_one("#bt-source", Select)
        opts = [("Local dataset", "local")] + self.app.bat.connections.data_sources()
        cur = select_value(sel, "local")
        sel.set_options(opts)
        sel.value = cur if cur in [v for _, v in opts] else "local"

    def _sync_symbol(self) -> None:
        path = select_value(self.query_one("#bt-dataset", Select), "")
        if path:
            self.query_one("#bt-ticker", Input).value = symbol_from_filename(Path(path).name)

    @on(Select.Changed, "#bt-source")
    def _source_changed(self, ev: Select.Changed) -> None:
        local = select_value(ev.select, "local") == "local"
        self.query_one("#bt-local-box").display = local
        self.query_one("#bt-provider-box").display = not local

    @on(Select.Changed, "#bt-dataset")
    def _dataset_changed(self, ev: Select.Changed) -> None:
        self._sync_symbol()

    # ------------------------------------------------------------------ strategy
    @on(Select.Changed, "#bt-strategy")
    async def _strategy_changed(self, ev: Select.Changed) -> None:
        spec = STRATEGIES[select_value(ev.select)]
        form = self.query_one("#bt-params", ParamForm)
        self.query_one("#bt-desc", Static).update(spec.description)
        if form.spec is not spec:
            await form.set_spec(spec)
        self._validate_form()

    @on(ParamForm.Changed)
    def _params_changed(self) -> None:
        self._validate_form()

    def _validate_form(self) -> Optional[Dict[str, Any]]:
        spec = STRATEGIES[select_value(self.query_one("#bt-strategy", Select))]
        warn = self.query_one("#bt-warn", Static)
        try:
            vals = spec.parse(self.query_one("#bt-params", ParamForm).raw_values())
        except ValueError as e:
            warn.update(Text(f"✗ {e}", style=C.DOWN))
            return None
        msgs = spec.warnings(vals)
        warn.update(Text("\n".join("⚠ " + m for m in msgs), style=C.AMBER) if msgs else "")
        return vals

    @on(Button.Pressed, "#bt-reset")
    async def _reset(self) -> None:
        spec = STRATEGIES[select_value(self.query_one("#bt-strategy", Select))]
        await self.query_one("#bt-params", ParamForm).set_spec(spec)
        for wid, v in (("#bt-balance", "10000"), ("#bt-pos", "100"), ("#bt-spread", "1.0")):
            self.query_one(wid, Input).value = v
        self._validate_form()

    # --------------------------------------------------------------------- run
    def _status(self, msg: str, kind: str = "") -> None:
        s = self.query_one("#bt-status", Static)
        s.update(msg)
        s.set_classes(f"status-line {kind}".strip())

    def _num(self, wid: str, label: str, lo: float, hi: Optional[float] = None, integer: bool = False) -> float:
        raw = self.query_one(wid, Input).value.strip()
        try:
            v = float(raw)
        except ValueError:
            raise ValueError(f"{label}: '{raw}' is not a number")
        if v < lo or (hi is not None and v > hi):
            raise ValueError(f"{label}: must be between {lo:g} and {hi:g}" if hi is not None else f"{label}: must be ≥ {lo:g}")
        return int(v) if integer else v

    # ------------------------------------------------------------- run log helpers
    def _log(self, level: str, text: str) -> None:
        self.query_one("#bt-runlog", EventLog).add(level, text)
        if level == "error":
            self.app.log_event("error", "backtest", text)

    def _log_findings(self, findings: List[Finding]) -> None:
        for f in findings:
            self._log({"ok": "ok", "info": "info", "warn": "warn", "error": "error"}[f.level], self._mark(f) + f.text)

    @staticmethod
    def _mark(f: Finding) -> str:
        return {"ok": "✓ ", "info": "ℹ ", "warn": "⚠ ", "error": "✗ "}[f.level]

    # --------------------------------------------------------------------- run
    @on(Button.Pressed, "#bt-run")
    def _run_clicked(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#bt-skip")
    def _skip_clicked(self) -> None:
        self._bt_skip.set()

    def _reject(self, msg: str) -> None:
        """A run that cannot even start: say so loudly in every place the user might look."""
        self._status(f"✗ {msg}", "error")
        self._log("error", msg)
        self.app.notify_err(msg)

    def action_run(self) -> None:
        if self._bt_busy:
            self.app.notify("A backtest is already running", severity="warning")
            return
        key = select_value(self.query_one("#bt-strategy", Select))
        spec = STRATEGIES[key]
        try:
            params = spec.parse(self.query_one("#bt-params", ParamForm).raw_values())
        except ValueError as e:
            self._reject(f"Fix the strategy parameters first: {e}")
            return
        try:
            cfg: Dict[str, Any] = dict(
                key=key, params=params,
                source=select_value(self.query_one("#bt-source", Select), "local"),
                dataset=select_value(self.query_one("#bt-dataset", Select), ""),
                resample=select_value(self.query_one("#bt-resample", Select), "none"),
                lfrom=self.query_one("#bt-lfrom", Input).value.strip() or None,
                lto=self.query_one("#bt-lto", Input).value.strip() or None,
                timespan=select_value(self.query_one("#bt-timespan", Select), "minute"),
                pfrom=self.query_one("#bt-from", Input).value.strip(), pto=self.query_one("#bt-to", Input).value.strip(),
                limit=self._num("#bt-limit", "Max bars", 100, 1_000_000, integer=True),
                symbol=self.query_one("#bt-ticker", Input).value.strip() or "X:BTCUSD",
                mode=select_value(self.query_one("#bt-mode", Select), "long_only"),
                balance=self._num("#bt-balance", "Initial balance", 1),
                pos=self._num("#bt-pos", "Per trade %", 1, 100),
                spread=self._num("#bt-spread", "Spread", 0, 10000),
                replay=float(select_value(self.query_one("#bt-replay", Select), "2")),
            )
            if cfg["source"] == "local" and not cfg["dataset"]:
                raise ValueError("No dataset selected — download one in Research, or pick a provider as the source")
            if cfg["source"] != "local" and cfg["source"] not in self.app.bat.connections.providers:
                raise ValueError(f"Provider '{cfg['source']}' is not connected — open Connections (key 7) first")
            for tag in ("lfrom", "lto"):
                if cfg[tag]:
                    pd.Timestamp(cfg[tag])
        except ValueError as e:
            self._reject(str(e))
            return
        self._bt_busy = True
        self._bt_skip.clear()
        self._bt_frames = 0
        self.query_one("#bt-run", Button).disabled = True
        self.query_one("#bt-skip", Button).disabled = True
        self.query_one("#bt-progress", ProgressBar).update(total=100, progress=0)
        self.query_one("#bt-tabs", TabbedContent).active = "tab-overview"
        self.query_one("#bt-cards", CardGrid).reset()
        log = self.query_one("#bt-runlog", EventLog)
        log.wipe()
        self._status("Starting…", "busy")
        self._log("info", f"▶ {spec.label}({fmt_params(params)}) · {cfg['symbol']} · {cfg['mode']} · "
                          f"${cfg['balance']:,.0f} · {cfg['pos']:g}% per trade")
        self._log_findings(check_config(cfg["symbol"], cfg["spread"], cfg["mode"], cfg["pos"]))
        self._run(cfg)

    @work(thread=True, exclusive=True, group="backtest")
    def _run(self, cfg: Dict[str, Any]) -> None:
        ui = self.app.call_from_thread
        step = "loading data"
        try:
            spec = STRATEGIES[cfg["key"]]
            # ---- 1. data ------------------------------------------------------------------
            t0 = time.time()
            ui(self._status, "Loading data…", "busy")
            if cfg["source"] == "local":
                ui(self._log, "info", f"Reading {Path(cfg['dataset']).name} …")
                kw = dict(start=cfg["lfrom"], end=cfg["lto"],
                          resample=None if cfg["resample"] == "none" else cfg["resample"])
                df = self.app.bat.datasets.get(cfg["dataset"], **kw)
                src = Path(cfg["dataset"]).name
            else:
                ui(self._log, "info", f"Requesting {cfg['symbol']} from {cfg['source']} …")
                provider = self.app.bat.connections.providers[cfg["source"]]
                df = LegacyDataProviderAdapter(provider).historical_bars(
                    cfg["symbol"], timespan=cfg["timespan"], from_date=cfg["pfrom"], to_date=cfg["pto"],
                    limit=cfg["limit"])
                src = cfg["source"]
            if df is not None and cfg["limit"] and len(df) > cfg["limit"] and cfg["source"] == "local":
                df = df.tail(cfg["limit"]).reset_index(drop=True)
            t_load = time.time() - t0
            strat = spec.build(cfg["params"])
            lookback = int(strat.get_required_lookback()) if hasattr(strat, "get_required_lookback") else 0
            findings = check_data(df, lookback)
            ui(self._log_findings, findings)
            if any(f.blocking for f in findings):
                raise RuntimeError("stopped: fix the data problem above")

            # ---- 2. signals ----------------------------------------------------------------
            step = "computing signals"
            ui(self._status, f"Computing {spec.label} signals on {len(df):,} bars…", "busy")
            t1 = time.time()
            sig = strat.generate_signals(df.copy())
            t_sig = time.time() - t1
            names = strat.get_signal_names()
            ui(self._log, "ok", f"Indicators computed in {t_sig:.2f}s")
            ui(self._log_findings, check_signals(sig, names, strat.get_indicators(), cfg["mode"], lookback))
            n_buy, n_sell = int(sig[names["buy"]].astype(bool).sum()), int(sig[names["sell"]].astype(bool).sum())

            # ---- 3. simulation ---------------------------------------------------------------
            step = "simulating trades"
            ui(self._status, "Simulating trades…", "busy")
            bt_cfg = BacktestConfig(initial_cash=cfg["balance"], trading_mode=cfg["mode"], symbol=cfg["symbol"],
                                    position_fraction=cfg["pos"] / 100.0, spread_pips=cfg["spread"])
            t2 = time.time()
            result = BacktestService().run(df, _SignalCache(strat, sig), bt_cfg)    # reuse the signals just computed
            t_run = time.time() - t2
            trades = result.trades
            ui(self._log, "ok", f"Simulated {len(trades):,} executions in {t_run:.2f}s "
                                f"({len(df) / max(t_run, 1e-9):,.0f} bars/s)")

            # ---- 4. analysis -------------------------------------------------------------------
            step = "analysing results"
            report = build_report(trades, result.signal_data, cfg["balance"])
            ui(self._log_findings, check_result(report.summary, n_buy, n_sell, cfg["mode"], report.open_trade))
            rec = BacktestRecord(label=f"{spec.label}({fmt_params(cfg['params'])})", strategy_key=cfg["key"],
                                 params=cfg["params"], symbol=cfg["symbol"], mode=cfg["mode"], source=src,
                                 report=report, config={**cfg, "indicators": strat.get_indicators(),
                                                        "signals": n_buy + n_sell, "t_load": t_load, "t_run": t_run})

            # ---- 5. replay (results are final; this just plays them back bar by bar) -------------
            step = "replaying"
            if cfg["replay"] > 0 and len(trades) > 0:
                self._replay(result.signal_data, trades, cfg, rec, ui)
            ui(self._done, rec)
        except Exception as e:  # noqa: BLE001
            ui(self._failed, step, e)

    def _replay(self, bars: pd.DataFrame, trades: pd.DataFrame, cfg: Dict[str, Any], rec: BacktestRecord, ui) -> None:
        n, target = len(bars), float(cfg["replay"])
        close = bars["Close"].to_numpy(float)
        ts = pd.to_datetime(bars["timestamp"]) if "timestamp" in bars else pd.Series(range(n))
        ui(self._begin_replay, C.time_labels(ts, 5), close / close[0] * cfg["balance"], n, rec)
        frames = int(np.clip(target * 12, 12, 90))
        cuts = np.unique(np.linspace(max(n // 40, 30), n, frames).astype(int))[:-1]       # last frame = final result
        delay, prev = target / (len(cuts) + 1), 0
        for cut in cuts:
            if self._bt_skip.is_set():
                break
            t0 = time.time()
            part = trades[trades["Index"] < cut].reset_index(drop=True)
            rep = build_report(part, bars.iloc[:cut], cfg["balance"])
            new = trades[(trades["Index"] >= prev) & (trades["Index"] < cut)]
            ui(self._frame, rep, int(cut), n, new, ts.iloc[int(cut) - 1])
            prev = int(cut)
            time.sleep(max(delay - (time.time() - t0), 0.0))

    def _begin_replay(self, labels, bh_full, n: int, rec: BacktestRecord) -> None:
        self._bt_axis = {"labels": labels, "bh": bh_full, "n": n, "init": rec.report.initial_cash,
                         "indicators": rec.config["indicators"], "symbol": rec.symbol, "first": True}
        self.query_one("#bt-skip", Button).disabled = False
        self._log("info", f"▶ Replaying the simulation bar by bar ({n:,} bars) — the numbers below are final, this is "
                          f"just the playback. Press Skip to jump to the end.")

    def _frame(self, rep, cut: int, n: int, new: pd.DataFrame, now) -> None:
        self._bt_frames += 1
        s, ax = rep.summary, self._bt_axis
        self.query_one("#bt-progress", ProgressBar).update(total=n, progress=cut)
        self._status(f"▶ Replaying · bar {cut:,}/{n:,} · {pd.Timestamp(now):%Y-%m-%d %H:%M} · "
                     f"equity {C.fmt_money(s['final'])} ({C.fmt_pct(s['total_return'] * 100)}) · "
                     f"{s['trades']} round trips", "busy")
        self._fill_cards_summary(s, note=False)
        self._fill_live_plots(rep, cut, n)
        shown = 0
        for _, t in new.iterrows():
            if shown >= 4:
                self._log("info", f"   … and {len(new) - shown} more executions in this step")
                break
            a, px, tm = str(t["Action"]), float(t["Price"]), pd.Timestamp(t["Time"])
            if pd.notna(t.get("Profit")):
                pnl = float(t["Profit"])
                self._log("ok" if pnl > 0 else "loss", f"  ◆ {a:<11} @ {px:,.5g}  {C.fmt_money(pnl, signed=True)}   "
                                                        f"{tm:%m-%d %H:%M}")
            else:
                self._log("trade", f"  {'▲' if a == 'BUY' else '▼'} {a:<11} @ {px:,.5g}  qty {float(t['Shares']):.5g}   "
                                   f"{tm:%m-%d %H:%M}")
            shown += 1

    def _fill_live_plots(self, rep, cut: int, n: int) -> None:
        ax = self._bt_axis
        init = ax["init"]
        eq = np.full(n, np.nan)
        eq[:cut] = rep.equity.to_numpy()
        bh = ax["bh"].copy()
        bh[cut:] = np.nan
        dd = np.full(n, np.nan)
        dd[:cut] = rep.drawdown.to_numpy() * 100
        labels, money = ax["labels"], (lambda v: "$" + C.fmt_num(v))
        self.query_one("#bt-equity", Plot).set_builder(lambda w, h: C.line_chart(
            {"Strategy": eq, "Buy & hold": bh}, w, h, colors={"Strategy": C.BLUE, "Buy & hold": C.MUTED},
            hline=init, x_labels=labels, y_fmt=money, title="Equity (mark-to-market) — replaying"))
        lo = min(float(np.nanmin(dd)) * 1.15, -0.5)
        self.query_one("#bt-dd", Plot).set_builder(lambda w, h: C.line_chart(
            {"Drawdown %": dd}, w, h, colors={"Drawdown %": C.DOWN}, fill_to=0.0, y_range=(lo, 0.0),
            x_labels=labels, y_fmt=lambda v: f"{v:.1f}%", title="Drawdown"))
        chart = self.query_one("#bt-price", PriceChart)
        chart.set_data(rep.bars, markers=rep.markers, indicators=ax["indicators"], title=ax["symbol"], follow=True,
                       view_len=120, reset_view=ax["first"])
        ax["first"] = False

    def _failed(self, step: str, e: Exception) -> None:
        self._bt_busy = False
        self.query_one("#bt-run", Button).disabled = False
        self.query_one("#bt-skip", Button).disabled = True
        hint = explain_exception(e)
        stopped = isinstance(e, RuntimeError) and str(e).startswith("stopped")
        if stopped:
            self._status("✗ Backtest stopped — see the run log for the data problem", "error")
            self._log("error", "Backtest stopped before running.")
        else:
            self._status(f"✗ Backtest failed while {step}: {hint}", "error")
            self._log("error", f"Failed while {step}: {hint}")
            self.app.log_event("error", "backtest", f"{step}: {type(e).__name__}: {e}")
        self.app.notify_err(f"Backtest {'stopped' if stopped else 'failed'}: see the run log")

    def _done(self, rec: BacktestRecord) -> None:
        self._bt_busy = False
        self.query_one("#bt-run", Button).disabled = False
        self.query_one("#bt-skip", Button).disabled = True
        bar = self.query_one("#bt-progress", ProgressBar)
        bar.update(total=100, progress=100)
        self.app.bat.backtests.append(rec)
        s = rec.report.summary
        self._log("ok", f"✓ Done · {C.fmt_pct(s['total_return'] * 100)} · {s['trades']} round trips · "
                        f"Sharpe {s['sharpe']:.2f}" if np.isfinite(s["sharpe"]) else
                  f"✓ Done · {C.fmt_pct(s['total_return'] * 100)} · {s['trades']} round trips")
        self.app.log_event("ok", "backtest", f"{rec.label} on {rec.symbol}: {C.fmt_pct(s['total_return'] * 100)} · "
                                             f"{s['trades']} trades · {rec.config['t_run']:.2f}s")
        self.show(rec)

    # ------------------------------------------------------------------ display
    def show(self, rec: BacktestRecord) -> None:
        self._bt_rec = rec
        rep, s, cfg = rec.report, rec.report.summary, rec.config
        warn = ""
        if cfg["signals"] == 0:
            warn = "  ⚠ the strategy produced no signals on this data"
        self._status(f"{rec.label} · {rec.symbol} · {rec.mode} · {s['bars']:,} bars in {cfg['t_run']:.2f}s "
                     f"(data {cfg['t_load']:.2f}s){warn}", "error" if warn else "ok")
        self._fill_cards(rec)
        self._fill_plots(rec)
        self._fill_trades(rec)
        self._fill_analysis(rec)
        self._fill_history()

    def _fill_cards(self, rec: BacktestRecord) -> None:
        self._fill_cards_summary(rec.report.summary, note=True)

    def _fill_cards_summary(self, s: Dict[str, Any], note: bool = True) -> None:
        g = self.query_one("#bt-cards", CardGrid)
        pc = C.pnl_color
        g.set("pnl", C.fmt_money(s["net_pnl"], signed=True), f"from {C.fmt_money(s['initial'])}", pc(s["net_pnl"]))
        g.set("ret", C.fmt_pct(s["total_return"] * 100), f"{s['days']:.1f} days · {s['bars']:,} bars", pc(s["total_return"]))
        g.set("cagr", C.fmt_pct(s["cagr"] * 100) if np.isfinite(s["cagr"]) else "n/a", "annualised" if np.isfinite(s["cagr"]) else "needs ≥ 30 days", pc(s["cagr"]) if np.isfinite(s["cagr"]) else None)
        g.set("final", C.fmt_money(s["final"]), f"open P&L {C.fmt_money(s['open_unrealized'], signed=True)}")
        g.set("sharpe", f"{s['sharpe']:.2f}" if np.isfinite(s["sharpe"]) else "–", f"{s['freq']} returns", pc(s["sharpe"]) if np.isfinite(s["sharpe"]) else None)
        g.set("sortino", f"{s['sortino']:.2f}" if np.isfinite(s["sortino"]) else "–", "downside risk only")
        g.set("dd", C.fmt_pct(s["max_drawdown"] * 100, signed=False), f"longest under water {s['max_dd_bars']:,} bars", C.DOWN if s["max_drawdown"] < -0.1 else None)
        g.set("calmar", f"{s['calmar']:.2f}" if np.isfinite(s["calmar"]) else "–", "return ÷ drawdown")
        g.set("trades", f"{s['trades']:,}", f"{s['long']['n']} long · {s['short']['n']} short")
        wr = s["win_rate"]
        g.set("win", f"{wr * 100:.1f}%" if np.isfinite(wr) else "–", f"best streak {s['max_consec_wins']} · worst {s['max_consec_losses']}", C.UP if np.isfinite(wr) and wr >= 0.5 else None)
        pf = s["profit_factor"]
        g.set("pf", "∞" if pf == np.inf else (f"{pf:.2f}" if np.isfinite(pf) else "–"), "gross win ÷ gross loss", C.UP if np.isfinite(pf) and pf > 1 else C.DOWN if np.isfinite(pf) else None)
        g.set("exp", C.fmt_money(s["expectancy"], signed=True) if np.isfinite(s["expectancy"]) else "–", f"best {C.fmt_money(s['best_trade'], decimals=2)} · worst {C.fmt_money(s['worst_trade'], decimals=2)}", pc(s["expectancy"]))
        g.set("payoff", f"{s['payoff']:.2f}" if np.isfinite(s["payoff"]) else "–", f"avg win {C.fmt_money(s['avg_win'], decimals=2)} · loss {C.fmt_money(s['avg_loss'], decimals=2)}")
        g.set("expo", f"{s['exposure'] * 100:.0f}%", f"avg hold {fmt_td(s['avg_duration']) if pd.notna(s['avg_duration']) else '–'}")
        g.set("bh", C.fmt_pct(s["buy_hold"] * 100), "benchmark", pc(s["buy_hold"]))
        g.set("alpha", C.fmt_pct(s["alpha"] * 100), "strategy − benchmark", pc(s["alpha"]))
        if not note:
            return
        note = self.query_one("#bt-note", Static)
        if s["days"] < 60:
            note.update(Text(f"⚠ Short sample ({s['days']:.0f} days): CAGR, Sharpe, Sortino and Calmar are annualised "
                             f"from a small window and can be misleading.", style=C.AMBER))
        else:
            note.update("")

    def _fill_plots(self, rec: BacktestRecord) -> None:
        rep, s = rec.report, rec.report.summary
        init, eq, bars = rep.initial_cash, rep.equity, rep.bars
        bh = bars["Close"].to_numpy(float) / bars["Close"].iloc[0] * init
        labels = C.time_labels(eq.index, 5)
        money = lambda v: "$" + C.fmt_num(v)
        self.query_one("#bt-equity", Plot).set_builder(lambda w, h: C.line_chart(
            {"Strategy": eq.to_numpy(), "Buy & hold": bh}, w, h, colors={"Strategy": C.BLUE, "Buy & hold": C.MUTED},
            hline=init, x_labels=labels, y_fmt=money, title="Equity (mark-to-market)"))
        dd = rep.drawdown.to_numpy() * 100
        lo = min(float(dd.min()) * 1.15, -0.5)
        self.query_one("#bt-dd", Plot).set_builder(lambda w, h: C.line_chart(
            {"Drawdown %": dd}, w, h, colors={"Drawdown %": C.DOWN}, fill_to=0.0, y_range=(lo, 0.0),
            x_labels=labels, y_fmt=lambda v: f"{v:.1f}%", title="Drawdown"))
        self.query_one("#bt-dd", Plot).styles.height = 9
        chart = self.query_one("#bt-price", PriceChart)
        chart.set_data(bars, markers=rep.markers, indicators=rec.config["indicators"], title=rec.symbol,
                       view_len=min(len(bars), 120))
        first = rep.markers[0][0] if rep.markers else 0
        chart.focus_bar(first, 120)
        self._update_view_label(chart)

    @on(PriceChart.ViewChanged)
    def _view_changed(self, ev: PriceChart.ViewChanged) -> None:
        self._update_view_label(self.query_one("#bt-price", PriceChart))

    def _update_view_label(self, chart: PriceChart) -> None:
        self.query_one("#bt-view", Static).update(
            "▲ long entry  ▼ short entry  ◆ exit   ·   ←/→ pan · +/− or wheel zoom · 0 show all · hover for OHLC")

    # -- trades table -----------------------------------------------------------
    def _trip_rows(self) -> pd.DataFrame:
        rt = self._bt_rec.report.round_trips.copy() if self._bt_rec else pd.DataFrame()
        if rt.empty:
            return rt
        rt.insert(0, "n", range(1, len(rt) + 1))
        f = select_value(self.query_one("#bt-trip-filter", Select), "all")
        if f == "long":
            rt = rt[rt["side"] == "LONG"]
        elif f == "short":
            rt = rt[rt["side"] == "SHORT"]
        elif f == "win":
            rt = rt[rt["pnl"] > 0]
        elif f == "loss":
            rt = rt[rt["pnl"] <= 0]
        if self._bt_sort["col"] in rt.columns:
            rt = rt.sort_values(self._bt_sort["col"], ascending=not self._bt_sort["rev"], kind="stable")
        return rt

    def _fill_trades(self, rec: Optional[BacktestRecord] = None) -> None:
        t = self.query_one("#bt-trips", DataTable)
        t.clear()
        rt = self._trip_rows()
        for _, r in rt.iterrows():
            t.add_row(str(int(r["n"])),
                      Text(r["side"], style=C.UP if r["side"] == "LONG" else C.DOWN),
                      f"{pd.Timestamp(r['entry_time']):%m-%d %H:%M}", f"{pd.Timestamp(r['exit_time']):%m-%d %H:%M}",
                      f"{r['entry_price']:,.4f}".rstrip("0").rstrip("."), f"{r['exit_price']:,.4f}".rstrip("0").rstrip("."),
                      f"{r['shares']:.6g}", pnl_text(r["pnl"]),
                      Text(C.fmt_pct(r["ret_pct"]), style=C.pnl_color(r["pnl"]), justify="right"),
                      fmt_td(r["duration"]), key=str(int(r["entry_bar"])))
        info = self.query_one("#bt-trip-info", Static)
        info.update(f" {len(rt)} trades · net {C.fmt_money(float(rt['pnl'].sum()) if len(rt) else 0.0, signed=True)}"
                    f"   [dim]click a header to sort · Enter shows the trade on the chart[/]")

    @on(Select.Changed, "#bt-trip-filter")
    def _filter_changed(self) -> None:
        self._fill_trades()

    @on(DataTable.HeaderSelected, "#bt-trips")
    def _sort_trips(self, ev: DataTable.HeaderSelected) -> None:
        col = str(ev.column_key.value)
        self._bt_sort["rev"] = (not self._bt_sort["rev"]) if self._bt_sort["col"] == col else True
        self._bt_sort["col"] = col
        self._fill_trades()

    @on(DataTable.RowSelected, "#bt-trips")
    def _jump_to_trade(self, ev: DataTable.RowSelected) -> None:
        try:
            bar = int(ev.row_key.value)
        except (TypeError, ValueError):
            return
        self.query_one("#bt-tabs", TabbedContent).active = "tab-price"
        chart = self.query_one("#bt-price", PriceChart)
        chart.focus_bar(bar, 120)
        chart.focus()

    # -- analysis ---------------------------------------------------------------
    def _fill_analysis(self, rec: BacktestRecord) -> None:
        rep, s = rec.report, rec.report.summary
        rt = rep.round_trips
        self.query_one("#bt-hist", Plot).set_builder(
            lambda w, h: C.histogram(rt["pnl"].to_numpy(), w, h, title="Trade P&L distribution ($ per round trip)"))
        by_hour = by_day = None
        if len(rt):
            ent = pd.to_datetime(rt["entry_time"])
            by_hour = rt.groupby(ent.dt.hour)["pnl"].sum().reindex(range(24), fill_value=0.0)
            by_day = rt.groupby(ent.dt.dayofweek)["pnl"].sum().reindex(range(7), fill_value=0.0)
        self.query_one("#bt-hour", Plot).set_builder(
            (lambda w, h: C.bar_chart(by_hour.to_numpy(), w, h, labels=[f"{i:02d}" for i in range(24)],
                                      title="P&L by entry hour")) if by_hour is not None else None)
        self.query_one("#bt-weekday", Plot).set_builder(
            (lambda w, h: C.bar_chart(by_day.to_numpy(), w, h, labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                      title="P&L by entry weekday")) if by_day is not None else None)
        for pid in ("#bt-hist", "#bt-hour", "#bt-weekday"):
            self.query_one(pid, Plot).styles.height = 11
        mon = self.query_one("#bt-monthly", Static)
        if len(rep.monthly):
            hm = C.heatmap(rep.monthly.to_numpy(), [str(y) for y in rep.monthly.index], MONTHS, cell_width=7,
                           fmt=lambda v: f"{v:+.1f}%")
            mon.update(Text.assemble(("Monthly returns\n", "bold"), hm))
        else:
            mon.update("")
        tbl = Table(title="Risk & trade detail", title_justify="left", show_header=False, box=None, padding=(0, 2))
        rows = [("Volatility (ann.)", C.fmt_pct(s["volatility"] * 100, signed=False) if np.isfinite(s["volatility"]) else "–"),
                ("Long trades", f"{s['long']['n']}  ·  {C.fmt_money(s['long']['pnl'], signed=True)}"
                                f"  ·  win {s['long']['win_rate'] * 100:.0f}%" if s["long"]["n"] else "0"),
                ("Short trades", f"{s['short']['n']}  ·  {C.fmt_money(s['short']['pnl'], signed=True)}"
                                 f"  ·  win {s['short']['win_rate'] * 100:.0f}%" if s["short"]["n"] else "0"),
                ("Avg bars in trade", f"{s['avg_bars']:.1f}" if np.isfinite(s["avg_bars"]) else "–"),
                ("Strategy signals", f"{rec.config['signals']:,}")]
        for k, v in rows:
            tbl.add_row(Text(k, style="dim"), v)
        self.query_one("#bt-risk", Static).update(tbl)

    # -- history ----------------------------------------------------------------
    def _fill_history(self) -> None:
        h = self.query_one("#bt-history", DataTable)
        h.clear()
        recs = self.app.bat.backtests
        for r in reversed(recs):
            s = r.report.summary
            h.add_row(str(r.id), f"{r.created:%H:%M:%S}", r.label, r.symbol, "L/S" if r.mode == "long_short" else "Long",
                      Text(C.fmt_pct(s["total_return"] * 100), style=C.pnl_color(s["total_return"])),
                      f"{s['sharpe']:.2f}" if np.isfinite(s["sharpe"]) else "–", C.fmt_pct(s["max_drawdown"] * 100, signed=False),
                      str(s["trades"]), f"{s['win_rate'] * 100:.0f}" if np.isfinite(s["win_rate"]) else "–", key=str(r.id))
        series = {f"#{r.id} {r.label}"[:28]: (r.report.equity.to_numpy() / r.report.initial_cash - 1) * 100
                  for r in recs[-6:]}
        cmp = self.query_one("#bt-compare", Plot)
        cmp.styles.height = 14
        cmp.set_builder((lambda w, h: C.line_chart(series, w, h, y_fmt=lambda v: f"{v:.1f}%", hline=0.0,
                                                   title="Return comparison (last 6 runs)")) if series else None)

    def _selected_record(self) -> Optional[BacktestRecord]:
        h = self.query_one("#bt-history", DataTable)
        if h.row_count == 0:
            return None
        try:
            key = h.coordinate_to_cell_key(h.cursor_coordinate).row_key.value
        except Exception:
            return None
        return next((r for r in self.app.bat.backtests if str(r.id) == key), None)

    @on(Button.Pressed, "#bt-open")
    @on(DataTable.RowSelected, "#bt-history")
    def _open(self) -> None:
        rec = self._selected_record()
        if rec:
            self.show(rec)
            self.query_one("#bt-tabs", TabbedContent).active = "tab-overview"

    @on(Button.Pressed, "#bt-load")
    async def _load_settings(self) -> None:
        rec = self._selected_record()
        if rec:
            await self.prefill(strategy_key=rec.strategy_key, params=rec.params, symbol=rec.symbol)
            self.app.notify_ok(f"Loaded settings of #{rec.id}")

    @on(Button.Pressed, "#bt-clear")
    def _clear(self) -> None:
        self.app.bat.backtests.clear()
        self._fill_history()

    @on(Button.Pressed, "#bt-export")
    def _export(self) -> None:
        rec = self._selected_record() or self._bt_rec
        if not rec:
            self.app.notify("Run a backtest first", severity="warning")
            return
        out = ROOT / "runs" / "backtests"
        out.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{rec.created:%Y%m%d_%H%M%S}_{rec.label}_{rec.symbol}")
        rec.report.executions.to_csv(out / f"{stem}_executions.csv", index=False)
        rec.report.round_trips.to_csv(out / f"{stem}_round_trips.csv", index=False)
        summary = {k: (v if isinstance(v, (int, float, str, dict, type(None))) else str(v))
                   for k, v in rec.report.summary.items()}
        (out / f"{stem}_summary.json").write_text(json.dumps(
            {"label": rec.label, "symbol": rec.symbol, "mode": rec.mode, "params": rec.params, "summary": summary},
            indent=2, default=str))
        self.app.notify_ok(f"Exported 3 files → runs/backtests/{stem}_*")

    # ---------------------------------------------------------------- prefill API
    async def prefill(self, strategy_key: Optional[str] = None, params: Optional[Dict[str, Any]] = None,
                      dataset: Optional[str] = None, symbol: Optional[str] = None) -> None:
        if strategy_key and strategy_key in STRATEGIES:
            spec = STRATEGIES[strategy_key]
            # build the form first: the Select.Changed handler skips when the spec already matches
            await self.query_one("#bt-params", ParamForm).set_spec(spec, params)
            self.query_one("#bt-desc", Static).update(spec.description)
            self.query_one("#bt-strategy", Select).value = strategy_key
        if dataset:
            self.query_one("#bt-source", Select).value = "local"
            try:
                self.query_one("#bt-dataset", Select).value = str(dataset)
            except Exception:
                pass
        if symbol:
            self.query_one("#bt-ticker", Input).value = symbol
        self._validate_form()
