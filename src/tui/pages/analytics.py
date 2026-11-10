"""Analytics page: production (PRODLIVETRADING) trade database."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, DataTable, Input, Select, Static, TabbedContent, TabPane

from src.analytics.prodlive import queries as Q
from src.analytics.prodlive import sync as S
from src.tui import charts as C
from src.tui.widgets.common import CardGrid, LabeledInput, LabeledSelect, Plot, Section, fmt_td, select_value
from src.tui.widgets.modals import FormModal

CARDS = {"n": "Closed trades", "win": "Win rate", "pnl": "Net P&L", "pf": "Profit factor", "avgr": "Avg R",
         "totr": "Total R", "dd": "Max drawdown", "ret": "Return", "sharpe": "Daily Sharpe", "tpw": "Trades / week",
         "hold": "Avg hold", "slip": "Slippage / trade"}
BY = [("Symbol", "symbol"), ("Side", "side"), ("Entry hour (UTC)", "entry_hour_utc"), ("Outcome", "outcome"),
      ("Weekday", "weekday"), ("Month", "month"), ("Account", "account_id")]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fill_frame(table: DataTable, df: pd.DataFrame, max_rows: int = 500, max_cols: int = 10) -> None:
    table.clear(columns=True)
    if df is None or df.empty:
        table.add_column("(empty)")
        return
    cols = list(df.columns)[:max_cols]
    for c in cols:
        table.add_column(str(c))
    for _, r in df.head(max_rows).iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("–" if not np.isfinite(v) else f"{v:,.4f}".rstrip("0").rstrip("."))
            elif isinstance(v, pd.Timestamp):
                cells.append(f"{v:%Y-%m-%d %H:%M}")
            else:
                cells.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)[:40])
        table.add_row(*cells)


class AnalyticsPage(Widget):
    BINDINGS = [Binding("ctrl+r", "run", "Refresh")]

    def __init__(self) -> None:
        super().__init__()
        self.an_flt: Dict[str, Any] = {"symbol": None, "since": None}
        self.an_data: Dict[str, Any] = {}
        self.an_symbols: tuple = ()
        self.an_busy = False

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(classes="btn-row"):
                yield Static("", id="an-info")
                yield Select([("All symbols", "")], value="", allow_blank=False, id="an-symbol", compact=True)
                yield Input(placeholder="since YYYY-MM-DD", id="an-since", compact=True)
                yield Button("Refresh", id="an-refresh")
                yield Button("Sync from bot…", id="an-sync", variant="primary")
            yield Static("", id="an-empty")
            with TabbedContent(id="an-tabs"):
                with TabPane("Overview", id="an-tab-ov"):
                    with VerticalScroll():
                        yield CardGrid(CARDS, columns=4, id="an-cards")
                        yield Plot(id="an-equity")
                        yield Plot(id="an-dd")
                with TabPane("Breakdown", id="an-tab-by"):
                    yield Select(BY, value="symbol", allow_blank=False, id="an-by", compact=True)
                    yield DataTable(id="an-by-table", cursor_type="row", zebra_stripes=True)
                    yield Plot(id="an-by-plot")
                with TabPane("Periods", id="an-tab-per"):
                    yield Select([("Daily", "D"), ("Weekly", "W"), ("Monthly", "M")], value="W", allow_blank=False,
                                 id="an-freq", compact=True)
                    yield Plot(id="an-per-plot")
                    yield DataTable(id="an-per-table", cursor_type="row", zebra_stripes=True)
                with TabPane("Trades", id="an-tab-tr"):
                    yield DataTable(id="an-trades", cursor_type="row", zebra_stripes=True)
                with TabPane("Live state", id="an-tab-live"):
                    with VerticalScroll():
                        yield Section("Open positions")
                        yield DataTable(id="an-open", zebra_stripes=True)
                        yield Section("Pending orders")
                        yield DataTable(id="an-pending", zebra_stripes=True)
                        yield Section("Recent system events")
                        yield DataTable(id="an-events", zebra_stripes=True)
                with TabPane("vs expectation", id="an-tab-exp"):
                    with Horizontal():
                        with VerticalScroll(classes="config"):
                            yield Section("Backtest expectation")
                            yield LabeledInput("Expected win rate (0-1)", "0.45", input_id="an-exp-wr")
                            yield LabeledInput("Expected avg R", "0.3", input_id="an-exp-r")
                            with Horizontal(classes="btn-row actions"):
                                yield Button("Compare", id="an-exp-go", variant="primary")
                        yield Static("Compares live win rate and avg R against what the backtest promised, with a "
                                     "z-score for each.", id="an-exp-out", classes="results")
                with TabPane("SQL", id="an-tab-sql"):
                    yield Input("select symbol, count(*) n, round(sum(realized_pnl),2) pnl from trades group by 1",
                                id="an-sql", compact=True)
                    yield Static("Read-only: the database is opened with mode=ro, so writes fail.", classes="muted")
                    yield DataTable(id="an-sql-table", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        self.query_one("#an-equity", Plot).styles.height = 14
        self.query_one("#an-dd", Plot).styles.height = 8
        self.query_one("#an-by-plot", Plot).styles.height = 12
        self.query_one("#an-per-plot", Plot).styles.height = 12
        self._reload()

    def on_show(self) -> None:
        if not self.an_data:
            self._reload()

    def action_run(self) -> None:
        self._reload()

    # -------------------------------------------------------------------- data
    @on(Button.Pressed, "#an-refresh")
    def _reload(self) -> None:
        if not Path(S.LATEST).exists():
            self._empty(True)
            return
        self._empty(False)
        self._load(dict(self.an_flt), select_value(self.query_one("#an-by", Select), "symbol"),
                   select_value(self.query_one("#an-freq", Select), "W"))

    def _empty(self, show: bool) -> None:
        self.query_one("#an-empty", Static).display = show
        self.query_one("#an-tabs").display = not show
        info = self.query_one("#an-info", Static)
        if show:
            env = os.getenv("PRODLIVE_DB") or os.getenv("PRODLIVE_SSH")
            self.query_one("#an-empty", Static).update(Text.assemble(
                ("No production snapshot yet\n\n", "bold"),
                "Analytics reads a read-only copy of the PRODLIVETRADING bot's trades.db.\n\n"
                "Press ‘Sync from bot…’ and give either a local path to trades.db or user@host:/path (over SSH).\n"
                "The bot's database is never modified; a consistent snapshot is made with SQLite's backup API.\n\n",
                (f"Detected from environment: {env}\n" if env else "Tip: set PRODLIVE_DB or PRODLIVE_SSH in .env.\n", "dim")))
            info.update("")
        else:
            age = time.time() - Path(S.LATEST).stat().st_mtime
            info.update(Text(f" snapshot {int(age // 3600)}h {int(age % 3600 // 60)}m old", style=C.AMBER if age > 6 * 3600 else "dim"))

    @work(thread=True, exclusive=True, group="an-load")
    def _load(self, flt: Dict[str, Any], by: str, freq: str) -> None:
        try:
            d: Dict[str, Any] = {}
            d["trades_all"] = Q.trades()
            d["trades"] = Q.trades(**flt)
            d["equity"] = Q.equity_curve()
            d["summary"] = Q.summary(**flt)
            d["by"] = Q.breakdown(by, **flt)
            d["by_key"] = by
            d["periods"] = Q.period_returns(freq, **flt)
            d["open"], d["pending"], d["events"] = Q.open_positions(), Q.pending_orders(), Q.events(n=100)
            self.app.call_from_thread(self._apply, d)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.query_one("#an-empty", Static).update, Text(f"✗ {type(e).__name__}: {e}", style=C.DOWN))
            self.app.call_from_thread(self._show_error)

    def _show_error(self) -> None:
        self.query_one("#an-empty", Static).display = True
        self.query_one("#an-tabs").display = False

    def _apply(self, d: Dict[str, Any]) -> None:
        self.an_data = d
        syms = tuple(sorted(d["trades_all"]["symbol"].dropna().unique())) if len(d["trades_all"]) else ()
        if syms != self.an_symbols:
            self.an_symbols = syms
            sel = self.query_one("#an-symbol", Select)
            sel.set_options([("All symbols", "")] + [(s, s) for s in syms])
            sel.value = self.an_flt["symbol"] or ""
        self._cards(d["summary"])
        eq = d["equity"]
        if len(eq):
            ts = pd.to_datetime(eq["timestamp"])
            lab = C.time_labels(ts, 5)
            e, dd = eq["equity"].to_numpy(), eq["drawdown"].to_numpy() * 100
            self.query_one("#an-equity", Plot).set_builder(lambda w, h: C.line_chart(
                {"NAV": e}, w, h, x_labels=lab, y_fmt=lambda v: "$" + C.fmt_num(v), title="Production equity (NAV)"))
            lo = min(float(dd.min()) * 1.15, -0.5)
            self.query_one("#an-dd", Plot).set_builder(lambda w, h: C.line_chart(
                {"Drawdown %": dd}, w, h, colors={"Drawdown %": C.DOWN}, fill_to=0.0, y_range=(lo, 0.0), x_labels=lab,
                y_fmt=lambda v: f"{v:.1f}%", title="Drawdown"))
        else:
            self.query_one("#an-equity", Plot).set_builder(None)
            self.query_one("#an-dd", Plot).set_builder(None)
        self._breakdown(d)
        self._periods(d["periods"])
        t = d["trades"]
        cols = ["symbol", "side", "entry_time", "exit_time", "entry_price", "exit_price", "net_pnl", "r_multiple", "outcome"]
        fill_frame(self.query_one("#an-trades", DataTable), t.tail(300)[[c for c in cols if c in t]].iloc[::-1] if len(t) else t)
        fill_frame(self.query_one("#an-open", DataTable), d["open"])
        fill_frame(self.query_one("#an-pending", DataTable), d["pending"])
        fill_frame(self.query_one("#an-events", DataTable), d["events"], max_cols=6)

    def _cards(self, s: Dict[str, Any]) -> None:
        g = self.query_one("#an-cards", CardGrid)
        if s.get("n", 0) == 0:
            g.reset()
            return
        pc = C.pnl_color
        g.set("n", f"{s['n']:,}", f"{s['open_positions']} open · {s['pending_orders']} pending")
        g.set("win", f"{s['win_rate'] * 100:.1f}%", color=C.UP if s["win_rate"] >= 0.5 else None)
        g.set("pnl", C.fmt_money(s["net_pnl"], signed=True), "after financing & commission", pc(s["net_pnl"]))
        g.set("pf", "∞" if s["profit_factor"] == np.inf else f"{s['profit_factor']:.2f}", color=C.UP if s["profit_factor"] > 1 else C.DOWN)
        g.set("avgr", f"{s['avg_r']:+.2f}R" if np.isfinite(s["avg_r"]) else "–", f"median {s['median_r']:+.2f}R" if np.isfinite(s["median_r"]) else "", pc(s["avg_r"]))
        g.set("totr", f"{s['total_r']:+.1f}R", color=pc(s["total_r"]))
        g.set("dd", C.fmt_pct(s["max_drawdown"] * 100, signed=False) if "max_drawdown" in s else "–", color=C.DOWN if s.get("max_drawdown", 0) < -0.1 else None)
        g.set("ret", C.fmt_pct(s["return_pct"] * 100) if "return_pct" in s else "–", color=pc(s.get("return_pct", 0)))
        sh = s.get("daily_sharpe", float("nan"))
        g.set("sharpe", f"{sh:.2f}" if np.isfinite(sh) else "–", color=pc(sh) if np.isfinite(sh) else None)
        g.set("tpw", f"{s['trades_per_week']:.1f}", f"over {s['days']} days")
        g.set("hold", f"{s['avg_hold_h']:.1f}h" if np.isfinite(s["avg_hold_h"]) else "–")
        g.set("slip", C.fmt_money(s["slippage_per_trade"]), f"total {C.fmt_money(s['slippage_total'])}")

    def _breakdown(self, d: Dict[str, Any]) -> None:
        b = d["by"]
        key = d["by_key"]
        if len(b):
            if key == "weekday":
                b = b.reindex([w for w in WEEKDAYS if w in b.index])
            elif key in ("entry_hour_utc", "month"):
                b = b.sort_index()
            tbl = self.query_one("#an-by-table", DataTable)
            show = b.reset_index()
            fill_frame(tbl, show.round(3))
            labels = [str(i) for i in b.index]
            vals = b["net_pnl"].to_numpy()
            self.query_one("#an-by-plot", Plot).set_builder(lambda w, h: C.bar_chart(
                vals, w, h, labels=labels, y_fmt=lambda v: "$" + C.fmt_num(v), title=f"Net P&L by {key}"))
        else:
            fill_frame(self.query_one("#an-by-table", DataTable), pd.DataFrame())
            self.query_one("#an-by-plot", Plot).set_builder(None)

    def _periods(self, p: pd.DataFrame) -> None:
        if len(p):
            vals = p["net_pnl"].to_numpy()
            labels = [f"{i:%m-%d}" for i in p.index]
            self.query_one("#an-per-plot", Plot).set_builder(lambda w, h: C.bar_chart(
                vals, w, h, labels=labels, y_fmt=lambda v: "$" + C.fmt_num(v), title="Net P&L per period"))
            fill_frame(self.query_one("#an-per-table", DataTable), p.reset_index().iloc[::-1].round(2))
        else:
            self.query_one("#an-per-plot", Plot).set_builder(None)
            fill_frame(self.query_one("#an-per-table", DataTable), pd.DataFrame())

    # ----------------------------------------------------------------- filters
    @on(Select.Changed, "#an-symbol")
    def _symbol(self, ev: Select.Changed) -> None:
        v = select_value(ev.select, "") or None
        if v != self.an_flt["symbol"]:
            self.an_flt["symbol"] = v
            self._reload()

    @on(Input.Submitted, "#an-since")
    def _since(self, ev: Input.Submitted) -> None:
        txt = ev.value.strip()
        try:
            if txt:
                pd.Timestamp(txt)
        except ValueError:
            self.app.notify_err("Use YYYY-MM-DD")
            return
        self.an_flt["since"] = txt or None
        self._reload()

    @on(Select.Changed, "#an-by")
    @on(Select.Changed, "#an-freq")
    def _group_changed(self) -> None:
        if self.an_data:
            self._reload()

    # --------------------------------------------------------------- expectation
    @on(Button.Pressed, "#an-exp-go")
    def _expect(self) -> None:
        try:
            wr = float(self.query_one("#an-exp-wr", Input).value)
            ar = float(self.query_one("#an-exp-r", Input).value)
            if not 0 < wr < 1:
                raise ValueError("win rate must be between 0 and 1")
        except ValueError as e:
            self.app.notify_err(str(e))
            return
        try:
            out = Q.vs_expectation(None, wr, ar, **self.an_flt)
        except Exception as e:  # noqa: BLE001
            self.query_one("#an-exp-out", Static).update(Text(f"✗ {e}", style=C.DOWN))
            return
        t = Text()
        t.append(f"{out['n']} closed trades\n\n", style="dim")
        for key, label in (("win_rate", "Win rate"), ("avg_r", "Avg R")):
            if key in out:
                act, exp, z = out[key]
                verdict = ("consistent with the backtest" if abs(z) < 1.64 else
                           "significantly BETTER than expected" if z > 0 else "significantly WORSE than expected")
                t.append(f"{label:<9}", style="bold")
                t.append(f"live {act:.3f}  vs  expected {exp:.3f}   z = {z:+.2f}  ")
                t.append(verdict + "\n", style=C.UP if abs(z) < 1.64 or z > 0 else C.DOWN)
        t.append("\n|z| < 1.64 ≈ within normal sampling noise (90% two-sided).", style="dim")
        self.query_one("#an-exp-out", Static).update(t)

    # ---------------------------------------------------------------------- sql
    @on(Input.Submitted, "#an-sql")
    def _sql(self, ev: Input.Submitted) -> None:
        tbl = self.query_one("#an-sql-table", DataTable)
        try:
            fill_frame(tbl, Q.sql(ev.value))
        except Exception as e:  # noqa: BLE001
            tbl.clear(columns=True)
            tbl.add_column("error")
            tbl.add_row(Text(str(e), style=C.DOWN))

    # --------------------------------------------------------------------- sync
    @on(Button.Pressed, "#an-sync")
    def _sync(self) -> None:
        def _done(vals):
            if vals is not None:
                self._do_sync(vals.get("src") or None, vals.get("ssh") or None)
        self.app.push_screen(FormModal(
            "Sync production snapshot",
            [("src", "Local trades.db path", os.getenv("PRODLIVE_DB", ""), False),
             ("ssh", "…or SSH  user@host:/path/to/trades.db", os.getenv("PRODLIVE_SSH", ""), False)],
            note="Read-only. Leave both empty to use PRODLIVE_DB / PRODLIVE_SSH from the environment.",
            submit="Sync"), _done)

    @work(thread=True, exclusive=True, group="an-sync")
    def _do_sync(self, src: Optional[str], ssh: Optional[str]) -> None:
        try:
            path = S.sync(src, ssh)
            self.app.call_from_thread(self.app.notify_ok, f"Snapshot saved ({Path(path).name})")
            self.app.call_from_thread(self.app.log_event, "ok", "analytics", f"Production snapshot synced → {path}")
            self.app.call_from_thread(self._reload)
        except SystemExit as e:
            self.app.call_from_thread(self.app.notify_err, f"Sync failed: {e}")
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.app.notify_err, f"Sync failed: {type(e).__name__}: {e}")
