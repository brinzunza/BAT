"""Live trading page: configure a session, then watch it trade."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, ContentSwitcher, DataTable, Input, Select, Static

from src.tui import charts as C
from src.tui.services.demo import DemoFeed
from src.tui.services.live import LiveConfig, LiveEvent, LiveSession, LiveSnapshot, make_bar_source
from src.tui.strategies import STRATEGIES, strategy_options
from src.tui.widgets.common import (EventLog, LabeledInput, LabeledSelect, ParamForm, Plot, Section, pnl_text,
                                    select_value)
from src.tui.widgets.modals import ChoiceModal, ConfirmModal
from src.tui.widgets.price import PriceChart

ROOT = Path(__file__).resolve().parents[3]
LIVE_SOURCES = ("alpaca", "oanda", "synth")
DEFAULT_SYMBOL = {"demo": "BTC/USD", "alpaca": "BTC/USD", "oanda": "EURUSD", "synth": "SYNTH"}
START_PRICE = {"BTC": 60_000.0, "ETH": 3_000.0, "SOL": 150.0, "EUR": 1.08, "GBP": 1.27, "AAPL": 190.0}
LEVELS = [("Trades & errors", "trade"), ("Signals", "signal"), ("Info", "info"), ("Everything (verbose)", "detail")]


def demo_price(symbol: str) -> float:
    for k, v in START_PRICE.items():
        if symbol.upper().startswith(k):
            return v
    return 100.0


class LivePage(Widget):
    BINDINGS = [Binding("x", "close_position", "Close position")]

    def __init__(self) -> None:
        super().__init__()
        self._lv_trades_n = -1
        self._lv_first = True
        self._lv_last_source = "demo"

    # ----------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        first = STRATEGIES["rsi"]
        with ContentSwitcher(initial="live-config", id="live-switch"):
            with Horizontal(id="live-config"):
                with VerticalScroll(classes="config"):
                    yield Section("Market")
                    yield LabeledSelect("Data source", [("Demo market (offline)", "demo")], select_id="lv-source",
                                        value="demo")
                    yield LabeledInput("Symbol", "BTC/USD", input_id="lv-symbol",
                                       placeholder="BTC/USD · AAPL · EURUSD")
                    with Horizontal(classes="row"):
                        yield LabeledInput("Tick every (s)", "1", input_id="lv-interval")
                        yield LabeledInput("History bars", "300", input_id="lv-lookback")
                    yield Section("Strategy")
                    yield LabeledSelect("Strategy", strategy_options(), select_id="lv-strategy", value=first.key)
                    yield Static(first.description, id="lv-desc", classes="muted")
                    yield ParamForm(first, prefix="lv-p", id="lv-params")
                    yield Section("Execution")
                    yield LabeledSelect("Trading mode", [("Long only (buy / close)", "long_only"),
                                                         ("Long / short (flip)", "long_short")],
                                        select_id="lv-mode", value="long_only")
                    with Horizontal(classes="row"):
                        yield LabeledSelect("Sizing", [("% of account", "pct"), ("Fixed quantity", "qty")],
                                            select_id="lv-sizing", value="pct")
                        yield LabeledInput("Size", "20", input_id="lv-size")
                    yield LabeledSelect("Broker", [("Simulated broker", "simulated")], select_id="lv-broker",
                                        value="simulated")
                    yield LabeledInput("Starting balance (simulated)", "10000", input_id="lv-balance")
                    with Horizontal(classes="btn-row actions"):
                        yield Button("Start session ▶", id="lv-start", variant="primary")
                yield VerticalScroll(Static(id="lv-preview"), classes="results")
            with Vertical(id="live-dash"):
                with Horizontal(id="live-bar", classes="btn-row"):
                    yield Static("IDLE", id="lv-badge", classes="live-badge stopped")
                    yield Static("", id="lv-title")
                    yield Button("Stop", id="lv-stop", variant="error")
                    yield Button("Close position", id="lv-close", variant="warning")
                    yield Button("Tick now", id="lv-tick")
                    yield Button("Export", id="lv-export")
                    yield Button("◀ Setup", id="lv-back")
                with Horizontal(id="live-dash-top"):
                    with Vertical(id="live-main"):
                        yield PriceChart(id="lv-price")
                        yield Plot(id="lv-equity")
                    with VerticalScroll(id="live-side"):
                        yield Static(id="lv-pos", classes="panel")
                        yield Static(id="lv-acct", classes="panel")
                        yield Static(id="lv-perf", classes="panel")
                with Horizontal(id="live-bottom"):
                    yield DataTable(id="live-trades", cursor_type="row", zebra_stripes=True)
                    with Vertical(id="live-events"):
                        yield Select(LEVELS, value="trade", allow_blank=False, id="lv-level", compact=True)
                        yield EventLog(id="lv-log")

    def on_mount(self) -> None:
        t = self.query_one("#live-trades", DataTable)
        t.add_columns("Time", "Action", "Price", "Qty", "P&L", "Result")
        self.query_one("#lv-equity", Plot).styles.height = 9
        self.query_one("#lv-log", EventLog).min_level = "trade"
        self.set_interval(0.5, self._tick_clock)
        self._refresh_sources()
        self._refresh_preview()

    def on_show(self) -> None:
        self._refresh_sources()
        live = self.app.bat.live
        if live and (live.running or live.last_snapshot):
            self.query_one("#live-switch", ContentSwitcher).current = "live-dash"

    # --------------------------------------------------------- form plumbing
    def _refresh_sources(self) -> None:
        cm = self.app.bat.connections
        src = self.query_one("#lv-source", Select)
        opts = [("Demo market (offline)", "demo")] + [(l, k) for l, k in cm.data_sources() if k in LIVE_SOURCES]
        cur = select_value(src, "demo")
        src.set_options(opts)
        src.value = cur if cur in [v for _, v in opts] else "demo"
        br = self.query_one("#lv-broker", Select)
        bopts = [("Simulated broker", "simulated")]
        for k in cm.brokers:
            paper = cm.meta.get(k, {}).get("paper", True)
            bopts.append((f"{k.upper()} ({'paper' if paper else 'LIVE'})", k))
        cur = select_value(br, "simulated")
        br.set_options(bopts)
        br.value = cur if cur in [v for _, v in bopts] else "simulated"

    @on(Select.Changed, "#lv-source")
    def _source_changed(self, ev: Select.Changed) -> None:
        key = select_value(ev.select, "demo")
        if key != self._lv_last_source:      # only on a real change, never on option refresh
            self._lv_last_source = key
            self.query_one("#lv-symbol", Input).value = DEFAULT_SYMBOL.get(key, "BTC/USD")
            self.query_one("#lv-interval", Input).value = "1" if key == "demo" else "60"
        self._refresh_preview()

    @on(Select.Changed, "#lv-strategy")
    async def _strategy_changed(self, ev: Select.Changed) -> None:
        spec = STRATEGIES[select_value(ev.select)]
        form = self.query_one("#lv-params", ParamForm)
        self.query_one("#lv-desc", Static).update(spec.description)
        if form.spec is not spec:
            await form.set_spec(spec)
        self._refresh_preview()

    @on(Select.Changed, "#lv-sizing")
    def _sizing_changed(self, ev: Select.Changed) -> None:
        self.query_one("#lv-size", Input).value = "20" if select_value(ev.select) == "pct" else "0.01"

    @on(Select.Changed, "#lv-broker")
    @on(Select.Changed, "#lv-mode")
    @on(Input.Changed, "#lv-symbol")
    @on(ParamForm.Changed)
    def _preview_dirty(self) -> None:
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        try:
            src = select_value(self.query_one("#lv-source", Select), "demo")
            broker = select_value(self.query_one("#lv-broker", Select), "simulated")
            cm = self.app.bat.connections
            paper = cm.meta.get(broker, {}).get("paper", True) if broker != "simulated" else True
            t = Text()
            t.append("How a session works\n\n", style="bold")
            t.append("1. Every tick the page fetches the newest bar from the data source.\n"
                     "2. Your strategy is evaluated on the rolling history.\n"
                     "3. The existing LiveTradingEngine places / closes orders through the broker and keeps its own\n"
                     "   state in sync with the broker (the broker is the source of truth).\n"
                     "4. The session runs in the background — you can browse other pages while it trades.\n\n",
                     style="dim")
            t.append("Selected setup\n", style="bold")
            t.append(f"  data    {src}\n  broker  {broker}"
                     f" ({'paper / simulated' if paper else 'REAL MONEY'})\n",
                     style=C.UP if paper else C.DOWN)
            if broker == "simulated":
                t.append("\nSafe to experiment: nothing leaves this machine.\n", style=C.UP)
            elif not paper:
                t.append("\n⚠ This will send real orders. You will be asked to type a confirmation phrase.\n",
                         style=f"bold {C.DOWN}")
            else:
                t.append("\nOrders go to the broker's paper account.\n", style=C.AMBER)
            if src == "demo":
                t.append("\nThe demo market is a regime-switching random walk; use it to see how a strategy behaves "
                         "tick by tick without risk. Lower “Tick every” for faster action.\n", style="dim")
            self.query_one("#lv-preview", Static).update(t)
        except Exception:
            pass

    # ------------------------------------------------------------- start / stop
    def _collect(self) -> Optional[Dict[str, Any]]:
        spec = STRATEGIES[select_value(self.query_one("#lv-strategy", Select))]
        try:
            params = spec.parse(self.query_one("#lv-params", ParamForm).raw_values())
            interval = float(self.query_one("#lv-interval", Input).value)
            lookback = int(float(self.query_one("#lv-lookback", Input).value))
            size = float(self.query_one("#lv-size", Input).value)
            balance = float(self.query_one("#lv-balance", Input).value)
            sizing = select_value(self.query_one("#lv-sizing", Select), "pct")
            if interval < 0.05 or lookback < 50 or size <= 0 or balance <= 0:
                raise ValueError("interval ≥ 0.05s, history ≥ 50 bars, size > 0 and balance > 0 are required")
            if sizing == "pct" and size > 100:
                raise ValueError("% of account must be ≤ 100")
        except ValueError as e:
            self.app.notify_err(str(e))
            return None
        symbol = self.query_one("#lv-symbol", Input).value.strip()
        if not symbol:
            self.app.notify_err("Symbol is required")
            return None
        return dict(spec=spec, params=params, interval=interval, lookback=lookback, size=size, sizing=sizing,
                    balance=balance, symbol=symbol, source=select_value(self.query_one("#lv-source", Select), "demo"),
                    broker=select_value(self.query_one("#lv-broker", Select), "simulated"),
                    mode=select_value(self.query_one("#lv-mode", Select), "long_only"))

    def action_primary(self) -> None:
        live = self.app.bat.live
        if live and live.running:
            self._ask_stop()
        else:
            self._start()

    @on(Button.Pressed, "#lv-start")
    def _start(self) -> None:
        live = self.app.bat.live
        if live and live.running:
            self.app.notify("A session is already running", severity="warning")
            self.query_one("#live-switch", ContentSwitcher).current = "live-dash"
            return
        c = self._collect()
        if not c:
            return
        cm = self.app.bat.connections
        paper = cm.meta.get(c["broker"], {}).get("paper", True) if c["broker"] != "simulated" else True
        summary = Text()
        for k, v in (("Strategy", f"{c['spec'].label} ({', '.join(f'{x:g}' if isinstance(x, float) else str(x) for x in c['params'].values())})"), ("Symbol", c["symbol"]), ("Mode", c["mode"]),
                     ("Data", c["source"]), ("Broker", f"{c['broker']} ({'paper' if paper else 'REAL MONEY'})"),
                     ("Size", f"{c['size']:g}{'% of account' if c['sizing'] == 'pct' else ' units'}"),
                     ("Tick", f"{c['interval']:g}s")):
            summary.append(f"{k:<9}", style="dim")
            summary.append(f"{v}\n")
        if c["broker"] == "simulated":
            self._launch(c)
            return

        def _answer(ok: Optional[bool]) -> None:
            if ok:
                self._launch(c)
        self.app.push_screen(ConfirmModal(
            "Start trading session" + ("" if paper else " — REAL MONEY"), summary,
            confirm="Start trading", danger=not paper, require_text=None if paper else "TRADE LIVE"), _answer)

    def start_demo(self) -> None:
        """Command-palette shortcut: start an offline demo session with the current strategy."""
        self.query_one("#lv-source", Select).value = "demo"
        self.query_one("#lv-broker", Select).value = "simulated"
        self._start()

    def _launch(self, c: Dict[str, Any]) -> None:
        cm, spec = self.app.bat.connections, c["spec"]
        try:
            if c["source"] == "demo":
                feed = DemoFeed(start_price=demo_price(c["symbol"]), seed=None, bar_seconds=60)
                source, source_label = make_bar_source(feed), "demo"
            else:
                source, source_label = make_bar_source(cm.providers[c["source"]]), c["source"]
            broker = cm.broker_for_live(c["broker"], c["balance"])
            if c["broker"] != "simulated":
                acct = broker.get_account() if hasattr(broker, "get_account") else {}
                c["balance"] = float(acct.get("equity", c["balance"]) or c["balance"])
            paper = cm.meta.get(c["broker"], {}).get("paper", True) if c["broker"] != "simulated" else True
            cfg = LiveConfig(strategy=spec.build(c["params"]), symbol=c["symbol"], source=source, broker=broker,
                             trading_mode=c["mode"], position_pct=c["size"] if c["sizing"] == "pct" else None,
                             quantity=c["size"] if c["sizing"] == "qty" else None, interval=c["interval"],
                             lookback=c["lookback"], initial_balance=c["balance"], broker_label=c["broker"],
                             source_label=source_label, paper=paper)
        except Exception as e:  # noqa: BLE001
            self.app.notify_err(f"Cannot start: {e}")
            return
        self.query_one("#lv-log", EventLog).wipe()
        self._lv_trades_n, self._lv_first = -1, True
        session = LiveSession(cfg, self._on_snapshot, self._on_event)
        self.app.bat.live = session
        self.query_one("#live-switch", ContentSwitcher).current = "live-dash"
        self.app.log_event("ok", "live", f"Started {spec.label} on {cfg.symbol} ({cfg.broker_label}, "
                                         f"{'paper' if cfg.paper else 'REAL'})")
        self._set_title(cfg, "starting")
        session.start()

    # thread -> UI bridges -------------------------------------------------------------
    def _on_snapshot(self, snap: LiveSnapshot) -> None:
        try:
            self.app.call_from_thread(self._apply, snap)
        except Exception:
            pass

    def _on_event(self, ev: LiveEvent) -> None:
        try:
            self.app.call_from_thread(self._event, ev)
        except Exception:
            pass

    def _event(self, ev: LiveEvent) -> None:
        self.query_one("#lv-log", EventLog).add(ev.level, ev.text, ts=ev.ts)
        if ev.level in ("trade", "error"):
            self.app.log_event(ev.level, "live", ev.text)

    # ----------------------------------------------------------------- stop / close
    @on(Button.Pressed, "#lv-stop")
    def _ask_stop(self) -> None:
        live = self.app.bat.live
        if not live or not live.running:
            return
        snap = live.last_snapshot
        qty = float((snap.position or {}).get("qty", 0) or 0) if snap else 0.0
        if qty != 0:
            body = Text(f"A position of {qty:g} {live.cfg.symbol} is open.\nStop the session and leave it open at the "
                        f"broker, or close it first?")
            choices = [("Close position & stop", "close", "error"), ("Stop, leave open", "stop", "warning")]
        else:
            body = Text("No open position. Stop the session?")
            choices = [("Stop session", "stop", "error")]

        def _answer(choice: Optional[str]) -> None:
            if choice:
                live.stop(close_position=(choice == "close"))
        self.app.push_screen(ChoiceModal("Stop live session", body, choices), _answer)

    @on(Button.Pressed, "#lv-close")
    def action_close_position(self) -> None:
        live = self.app.bat.live
        if not live or not live.running:
            return
        snap = live.last_snapshot
        qty = float((snap.position or {}).get("qty", 0) or 0) if snap else 0.0
        if qty == 0:
            self.app.notify("No open position", severity="information")
            return
        real = live.cfg.broker_label != "simulated" and not live.cfg.paper

        def _answer(ok: Optional[bool]) -> None:
            if ok:
                live.close_position()
        self.app.push_screen(ConfirmModal("Close position", f"Close {qty:g} {live.cfg.symbol} at market?",
                                          confirm="Close now", danger=real), _answer)

    @on(Button.Pressed, "#lv-tick")
    def _tick_now(self) -> None:
        live = self.app.bat.live
        if live and live.running:
            live.tick_now()

    @on(Button.Pressed, "#lv-back")
    def _back(self) -> None:
        self.query_one("#live-switch", ContentSwitcher).current = "live-config"

    @on(Button.Pressed, "#lv-export")
    def _export(self) -> None:
        live = self.app.bat.live
        if not live or not live.last_snapshot or live.last_snapshot.trades.empty:
            self.app.notify("No trades to export yet", severity="warning")
            return
        out = ROOT / "runs" / "live"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"live_{live.cfg.symbol.replace('/', '')}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        live.last_snapshot.trades.to_csv(path, index=False)
        self.app.notify_ok(f"Exported {path.relative_to(ROOT)}")

    @on(Select.Changed, "#lv-level")
    def _level(self, ev: Select.Changed) -> None:
        self.query_one("#lv-log", EventLog).set_level(select_value(ev.select, "trade"))

    # ----------------------------------------------------------------- rendering
    def _set_title(self, cfg: LiveConfig, state: str) -> None:
        badge = self.query_one("#lv-badge", Static)
        badge.update(state.upper())
        badge.set_classes(f"live-badge {state}")
        self.query_one("#lv-title", Static).update(Text.assemble(
            (f"  {cfg.symbol}", "bold"), (f"  {cfg.strategy.name}  ", ""), (f"{cfg.trading_mode}  ", "dim"),
            (f"{cfg.source_label} → {cfg.broker_label} ", "dim"),
            ("(paper)" if cfg.paper else "(REAL)", C.AMBER if cfg.paper else f"bold {C.DOWN}")))

    def _apply(self, snap: LiveSnapshot) -> None:
        live = self.app.bat.live
        if live is None:
            return
        cfg = live.cfg
        self._set_title(cfg, snap.state)
        is_run = snap.state in ("running", "starting")
        self.query_one("#lv-stop", Button).disabled = not is_run
        self.query_one("#lv-close", Button).disabled = not is_run
        self.query_one("#lv-tick", Button).disabled = not is_run
        if len(snap.bars):
            trades = snap.trades
            markers: List[tuple] = []
            if len(trades):
                bar_ts = pd.to_datetime(snap.bars["timestamp"], utc=True).dt.tz_localize(None).to_numpy()
                tr_ts = pd.to_datetime(trades["time"], utc=True).dt.tz_localize(None).to_numpy()
                idx = np.clip(np.searchsorted(bar_ts, tr_ts), 0, len(bar_ts) - 1)
                for i, a in zip(idx, trades["action"].astype(str)):
                    markers.append((int(i), "buy" if a == "BUY" else "sell" if a == "SELL_SHORT" else "exit"))
            chart = self.query_one("#lv-price", PriceChart)
            chart.set_data(snap.bars, markers=markers, indicators=cfg.strategy.get_indicators(), title=cfg.symbol,
                           follow=chart.follow if not self._lv_first else True, reset_view=self._lv_first,
                           view_len=120)
            self._lv_first = False
            eq = [v for _, v in snap.equity]
            init = cfg.initial_balance
            self.query_one("#lv-equity", Plot).set_builder((lambda w, h: C.line_chart(
                {"Equity": eq}, w, h, colors={"Equity": C.pnl_color(eq[-1] - init) if eq[-1] != init else C.ORANGE},
                hline=init, y_fmt=lambda v: "$" + C.fmt_num(v), title="Account equity (session)",
                x_labels=None)) if len(eq) > 1 else None)
        self._render_side(snap, cfg)
        if len(snap.trades) != self._lv_trades_n:
            self._lv_trades_n = len(snap.trades)
            t = self.query_one("#live-trades", DataTable)
            t.clear()
            for _, r in snap.trades.iloc[::-1].iterrows():
                pnl = r["pnl"]
                t.add_row(f"{pd.Timestamp(r['time']):%H:%M:%S}",
                          Text(r["action"], style=C.UP if r["action"] == "BUY" else C.DOWN if r["action"] == "SELL_SHORT" else C.AMBER),
                          f"{r['price']:,.5g}", f"{r['qty']:.5g}",
                          pnl_text(pnl) if pd.notna(pnl) else Text("open", style="dim"), r["result"])

    def _render_side(self, snap: LiveSnapshot, cfg: LiveConfig) -> None:
        pos, acct, perf = snap.position or {}, snap.account or {}, snap.perf or {}
        qty = float(pos.get("qty", 0) or 0)
        upl = float(pos.get("unrealized_pl", 0) or 0)
        entry = float(pos.get("avg_entry_price", 0) or 0)
        chg = (snap.price / snap.prev_price - 1) * 100 if snap.prev_price and np.isfinite(snap.prev_price) else 0.0

        t = Text()
        t.append("PRICE\n", style="dim")
        t.append(f"{snap.price:,.6g}  ", style="bold")
        t.append(f"{C.fmt_pct(chg, decimals=3)}\n", style=C.pnl_color(chg))
        t.append("\nPOSITION\n", style="dim")
        if qty == 0:
            t.append("FLAT\n", style="bold")
        else:
            side = "LONG" if qty > 0 else "SHORT"
            t.append(f"{side} {abs(qty):.6g}\n", style=f"bold {C.UP if qty > 0 else C.DOWN}")
            t.append(f"entry {entry:,.6g}\n", style="dim")
            t.append(f"unrealized {C.fmt_money(upl, signed=True)}\n", style=C.pnl_color(upl))
        self.query_one("#lv-pos", Static).update(t)

        a = Text()
        a.append("ACCOUNT\n", style="dim")
        eq = float(acct.get("equity", cfg.initial_balance) or cfg.initial_balance)
        a.append(f"{C.fmt_money(eq)}\n", style="bold")
        sp = perf.get("session_pnl", 0.0)
        a.append(f"session {C.fmt_money(sp, signed=True)} ({C.fmt_pct(perf.get('return_pct', 0.0))})\n", style=C.pnl_color(sp))
        if "buying_power" in acct:
            a.append(f"buying power {C.fmt_money(float(acct['buying_power']))}\n", style="dim")
        self.query_one("#lv-acct", Static).update(a)

        p = Text()
        p.append("PERFORMANCE\n", style="dim")
        wr = perf.get("win_rate", float("nan"))
        p.append(f"{perf.get('trades', 0)} closed · ")
        p.append(f"{perf.get('wins', 0)}W", style=C.UP)
        p.append(" / ")
        p.append(f"{perf.get('losses', 0)}L\n", style=C.DOWN)
        p.append(f"win rate {wr:.0f}%\n" if np.isfinite(wr) else "win rate –\n")
        p.append(f"realized {C.fmt_money(perf.get('realized', 0.0), signed=True)}\n", style=C.pnl_color(perf.get("realized", 0.0)))
        p.append(f"tick #{snap.iteration}", style="dim")
        if snap.message:
            p.append(f"\n{snap.message}", style="dim")
        self.query_one("#lv-perf", Static).update(p)

    def _tick_clock(self) -> None:
        live = self.app.bat.live
        if not live or not live.running or not live.last_snapshot:
            return
        s = live.last_snapshot
        left = max(s.next_tick - time.time(), 0.0)
        try:
            badge = self.query_one("#lv-badge", Static)
            badge.update(f"RUNNING  next {left:4.0f}s" if s.state == "running" else s.state.upper())
        except Exception:
            pass
