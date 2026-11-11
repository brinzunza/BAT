"""Home: workspace overview and quick actions."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, DataTable, Static

from src.analytics.prodlive import sync as prodlive_sync
from src.tui import charts as C
from src.tui.datasets import DatasetInfo, human_size, scan_datasets, short_name
from src.tui.services import mlops
from src.tui.services.ticker import TickerFeed
from src.tui.state import LogEntry
from src.tui.widgets.brand import BrandBox
from src.tui.widgets.common import EventLog, Section


class HomePage(Widget):
    def compose(self) -> ComposeResult:
        with Horizontal(id="home-top"):
            yield BrandBox(id="home-logo")
            with Vertical(id="home-intro"):
                yield BrandBox(compact=True, id="home-ticker")
                yield Static("[b]Bruninvestor Algorithmic Trading[/b]\n"
                             "[dim]Backtest strategies, trade live, research parameters, train models and monitor "
                             "production — from one terminal.[/]")
                with Horizontal(id="home-actions"):
                    yield Button("New backtest", id="act-backtest", variant="primary")
                    yield Button("Demo live session", id="act-demo")
                    yield Button("Datasets", id="act-datasets")
                    yield Button("Train model", id="act-train")
                    yield Button("Analytics", id="act-analytics")
        with Horizontal(id="home-cols"):
            with VerticalScroll(id="home-left"):
                yield Section("Datasets")
                yield DataTable(id="home-datasets", cursor_type="row", zebra_stripes=True)
                yield Section("Recent backtests")
                yield DataTable(id="home-backtests", cursor_type="row", zebra_stripes=True)
            with Vertical(id="home-right"):
                yield Section("Workspace")
                yield Static(id="home-workspace")
                yield Section("Activity")
                yield EventLog(id="home-activity")

    def on_mount(self) -> None:
        self.feed = TickerFeed(self.app.bat)
        self.refresh_brand()
        self.set_interval(0.15, self._tick_brand)
        dt = self.query_one("#home-datasets", DataTable)
        dt.add_columns("Dataset", "Bars", "TF", "Range", "Trend", "Δ")
        bt = self.query_one("#home-backtests", DataTable)
        bt.add_columns("#", "Strategy", "Symbol", "Return", "Sharpe", "Max DD", "Trades")
        for e in list(self.app.bat.events)[-30:]:
            self.push(e)
        self.refresh_all()

    # ----- header ticker -----------------------------------------------------------------
    def refresh_brand(self) -> None:
        """Draw one frame now (also used after toggling the animation, to show/hide the "paused" tag)."""
        self._frame = self.feed.step() if not hasattr(self, "_frame") or self.app.brand_anim else self._frame
        for w in self.query(BrandBox):
            w.show(self._frame, paused=not self.app.brand_anim)

    def _tick_brand(self) -> None:
        if self.app.brand_anim and self.display:         # a hidden page costs nothing
            self.refresh_brand()

    def on_show(self) -> None:
        self.refresh_all()

    def refresh_all(self) -> None:
        self._load_datasets()
        self._refresh_backtests()
        self._refresh_workspace()

    # ----- datasets (background so big CSVs never block the UI) ------------------
    @work(thread=True, exclusive=True, group="home-datasets")
    def _load_datasets(self) -> None:
        rows = []
        for info in scan_datasets()[:12]:
            try:
                s = pd.read_csv(info.path, usecols=lambda c: c.strip().lower() == "close")
                close = s.iloc[:, 0].to_numpy(float)
                spark = C.colored_sparkline(close, 16)
                change = close[-1] / close[0] - 1 if close[0] else float("nan")
            except Exception:
                spark, change = Text(""), float("nan")
            rows.append((info, spark, change))
        self.app.call_from_thread(self._fill_datasets, rows)

    def _fill_datasets(self, rows) -> None:
        dt = self.query_one("#home-datasets", DataTable)
        dt.clear()
        if not rows:
            dt.add_row("No datasets yet — download one from Research", "", "", "", "", "")
            return
        for info, spark, change in rows:
            fmt = lambda t: f"{t:%m-%d}" if t is not None and pd.notna(t) else "–"
            dt.add_row(short_name(info.name, 26), f"{info.rows:,}", info.timeframe,
                       f"{fmt(info.start)}→{fmt(info.end)}", spark,
                       Text(C.fmt_pct(change * 100), style=C.pnl_color(change)), key=str(info.path))

    def _refresh_backtests(self) -> None:
        bt = self.query_one("#home-backtests", DataTable)
        bt.clear()
        recs = self.app.bat.backtests[-6:][::-1]
        if not recs:
            bt.add_row("–", "Run one from the Backtest page", "", "", "", "", "")
        for r in recs:
            s = r.report.summary
            bt.add_row(str(r.id), r.label, r.symbol, Text(C.fmt_pct(s["total_return"] * 100),
                                                          style=C.pnl_color(s["total_return"])),
                       f"{s['sharpe']:.2f}" if pd.notna(s["sharpe"]) else "–",
                       C.fmt_pct(s["max_drawdown"] * 100, signed=False), str(s["trades"]))

    def _refresh_workspace(self) -> None:
        bat = self.app.bat
        t = Text()
        for st in bat.connections.status():
            glyph, style = ("●", C.UP) if st.connected else ("○", C.AMBER) if st.configured else ("·", C.MUTED)
            t.append(f" {glyph} ", style=style)
            t.append(f"{st.label:<20}", style="default")
            t.append((st.detail or ("ready" if st.configured else "not configured")) + "\n", style="dim")
        runs, cks = mlops.list_runs(), mlops.list_checkpoints()
        t.append("\n ML      ", style="dim")
        best = min((r.best_val for r in runs if r.kind == "train" and r.best_val == r.best_val), default=None)
        t.append(f"{len(runs)} runs · {len(cks)} checkpoints" + (f" · best val {best:.4f}" if best is not None else ""))
        db = prodlive_sync.LATEST
        t.append("\n Prod DB ", style="dim")
        if Path(db).exists():
            age = time.time() - Path(db).stat().st_mtime
            t.append(f"snapshot {int(age // 3600)}h {int(age % 3600 // 60)}m old")
        else:
            t.append("no snapshot (Analytics → Sync)", style="dim")
        live = bat.live
        t.append("\n Live    ", style="dim")
        t.append("running" if live and live.running else "idle", style=C.UP if live and live.running else "dim")
        self.query_one("#home-workspace", Static).update(t)

    def push(self, entry: LogEntry) -> None:
        try:
            self.query_one("#home-activity", EventLog).add(entry.level, entry.text, entry.source, entry.ts)
        except Exception:
            pass

    # ----- actions ---------------------------------------------------------------------
    @on(Button.Pressed, "#act-backtest")
    def _bt(self) -> None:
        self.app.goto("backtest")

    @on(Button.Pressed, "#act-demo")
    def _demo(self) -> None:
        self.app.cmd_demo_live()

    @on(Button.Pressed, "#act-datasets")
    def _ds(self) -> None:
        self.app.goto("research")

    @on(Button.Pressed, "#act-train")
    def _train(self) -> None:
        self.app.goto("mllab")

    @on(Button.Pressed, "#act-analytics")
    def _an(self) -> None:
        self.app.goto("analytics")
