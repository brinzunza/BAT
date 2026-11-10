"""Research page: datasets, downloads and parameter optimisation."""
from __future__ import annotations

import re
import threading
import time
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
from textual.widgets import Button, DataTable, Input, Label, ProgressBar, Select, Static, TabbedContent, TabPane

from src.core.domain.models import BacktestConfig
from src.tui import charts as C
from src.tui.pages.backtest import symbol_from_filename
from src.tui.datasets import (DatasetInfo, dataset_stats, human_size, resample_bars, save_dataset, scan_datasets,
                              short_name)
from src.tui.services.optimizer import MAX_COMBOS, OBJECTIVES, build_grid, parse_range, run_grid
from src.tui.strategies import STRATEGIES, ParamSpec, StrategySpec, strategy_options
from src.tui.widgets.common import LabeledInput, LabeledSelect, Plot, Section, select_value, GuardedTabs, switch_tab
from src.tui.widgets.modals import FormModal
from src.tui.widgets.price import PriceChart

ROOT = Path(__file__).resolve().parents[3]
RESAMPLE = [("Native", "none"), ("5 min", "5min"), ("15 min", "15min"), ("30 min", "30min"), ("1 hour", "1h"),
            ("4 hours", "4h"), ("1 day", "1D")]
BARS_PER_SEC = 55_000.0     # conservative backtest throughput used for time estimates


def default_range(p: ParamSpec) -> str:
    if p.kind == "choice":
        return ",".join(str(c) for c in p.choices[:3])
    step = p.step or (1 if p.kind == "int" else 0.25)
    lo = max(p.min if p.min is not None else -1e9, p.default - 2 * step)
    hi = min(p.max if p.max is not None else 1e9, p.default + 2 * step)
    f = (lambda v: f"{int(v)}") if p.kind == "int" else (lambda v: f"{v:g}")
    return f"{f(lo)}:{f(hi)}:{f(step)}"


class RangeForm(Vertical):
    """One ``start:stop:step`` input per strategy parameter."""

    DEFAULT_CSS = "RangeForm { height: auto; } RangeForm .rg-row { height: auto; margin-bottom: 1; } RangeForm Label { color: $text-muted; }"

    def __init__(self, spec: StrategySpec, **kw):
        super().__init__(**kw)
        self.spec = spec

    def compose(self) -> ComposeResult:
        yield from self._rows()

    def _rows(self):
        if not self.spec.params:
            yield Static("No parameters to sweep.", classes="muted")
        for p in self.spec.params:
            yield Vertical(Label(f"{p.label}  [dim]start:stop:step or a,b,c[/]"),
                           Input(default_range(p), id=f"rg-{p.name}", compact=True), classes="rg-row")

    async def set_spec(self, spec: StrategySpec) -> None:
        self.spec = spec
        await self.remove_children()
        await self.mount_all(list(self._rows()))

    def ranges(self) -> Dict[str, List[Any]]:
        out = {}
        for p in self.spec.params:
            out[p.name] = parse_range(self.query_one(f"#rg-{p.name}", Input).value, p)
        return out


class ResearchPage(Widget):
    BINDINGS = [Binding("ctrl+r", "run", "Run")]

    def __init__(self) -> None:
        super().__init__()
        self._rs_ds: List[DatasetInfo] = []
        self._rs_cancel = threading.Event()
        self._rs_busy = False
        self._rs_results = pd.DataFrame()
        self._rs_ctx: Dict[str, Any] = {}
        self._rs_sort: Dict[str, Any] = {"col": None, "rev": False}

    # ----------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        first = STRATEGIES["mean_reversion"]
        with GuardedTabs(id="rs-tabs"):
            with TabPane("Datasets", id="rs-tab-data"):
                with Horizontal():
                    with Vertical(id="rs-left"):
                        yield DataTable(id="rs-ds", cursor_type="row", zebra_stripes=True)
                        with Horizontal(classes="btn-row"):
                            yield Button("Use in backtest", id="rs-use", variant="primary")
                            yield Button("Optimize", id="rs-opt")
                            yield Button("Resample & save…", id="rs-resample")
                            yield Button("Rescan", id="rs-rescan")
                    with Vertical(id="rs-right"):
                        yield Static("Select a dataset.", id="rs-stats")
                        yield PriceChart(id="rs-chart")
            with TabPane("Optimizer", id="rs-tab-opt"):
                with Horizontal():
                    with VerticalScroll(classes="config"):
                        yield Section("Data")
                        yield LabeledSelect("Dataset", [("(scanning…)", "")], select_id="op-dataset")
                        yield LabeledSelect("Resample to", RESAMPLE, select_id="op-resample", value="none")
                        yield LabeledInput("Use last N bars (0 = all)", "20000", input_id="op-bars")
                        yield Section("Strategy")
                        yield LabeledSelect("Strategy", strategy_options(), select_id="op-strategy", value=first.key)
                        yield RangeForm(first, id="op-ranges")
                        yield Static("", id="op-combos")
                        yield Section("Execution")
                        yield LabeledSelect("Trading mode", [("Long only", "long_only"), ("Long / short", "long_short")],
                                            select_id="op-mode", value="long_only")
                        with Horizontal(classes="row"):
                            yield LabeledInput("Balance", "10000", input_id="op-balance")
                            yield LabeledInput("Per trade %", "100", input_id="op-pos")
                        with Horizontal(classes="row"):
                            yield LabeledInput("Spread (pips)", "1.0", input_id="op-spread")
                            yield LabeledInput("Train %", "70", input_id="op-split")
                        yield LabeledSelect("Rank by (train)", [(v, k) for k, v in OBJECTIVES.items()],
                                            select_id="op-objective", value="sharpe")
                        with Horizontal(classes="btn-row actions"):
                            yield Button("Run sweep ▶", id="op-run", variant="primary")
                            yield Button("Cancel", id="op-cancel", disabled=True)
                    with Vertical(classes="results"):
                        yield Static("Sweep parameters over a train segment, then check each combination on held-out "
                                     "test data. Robust settings stay good out-of-sample.", id="op-status",
                                     classes="status-line")
                        yield ProgressBar(total=100, show_eta=True, id="op-progress")
                        yield DataTable(id="op-results", cursor_type="row", zebra_stripes=True)
                        with Horizontal(classes="btn-row"):
                            yield Select([("x", "x")], id="op-hx", allow_blank=False, compact=True)
                            yield Select([("y", "y")], id="op-hy", allow_blank=False, compact=True)
                            yield Select([("metric", "m")], id="op-hm", allow_blank=False, compact=True)
                        yield Static("", id="op-heat")
                        with Horizontal(classes="btn-row"):
                            yield Button("Apply selected to Backtest", id="op-apply", variant="success")
                            yield Button("Export CSV", id="op-export")
            with TabPane("Download", id="rs-tab-dl"):
                with VerticalScroll(classes="config"):
                    yield Section("Download history")
                    yield LabeledSelect("Provider", [("(connect one in Connections)", "")], select_id="dl-provider")
                    yield LabeledInput("Ticker", "X:BTCUSD", input_id="dl-ticker", placeholder="AAPL · X:BTCUSD · C:EURUSD")
                    yield LabeledSelect("Timespan", [("Minute", "minute"), ("Hour", "hour"), ("Day", "day")],
                                        select_id="dl-span", value="minute")
                    with Horizontal(classes="row"):
                        yield LabeledInput("From", "2025-01-01", input_id="dl-from")
                        yield LabeledInput("To", "2025-02-01", input_id="dl-to")
                    yield LabeledInput("Max bars", "50000", input_id="dl-limit")
                    with Horizontal(classes="btn-row actions"):
                        yield Button("Download ▶", id="dl-go", variant="primary")
                yield Static("Downloaded CSVs are saved to [b]data/raw/[/b] and show up in every dataset picker.",
                             id="dl-status", classes="results")

    def on_mount(self) -> None:
        ds = self.query_one("#rs-ds", DataTable)
        ds.add_columns("Dataset", "Bars", "TF", "From", "To", "Size")
        res = self.query_one("#op-results", DataTable)
        self.query_one("#op-progress", ProgressBar).display = False
        self.query_one("#op-heat").styles.height = "auto"
        self._rescan()
        self._refresh_sources()
        self._update_combos()

    def on_show(self) -> None:
        self._refresh_sources()

    # ---------------------------------------------------------------- datasets
    @work(thread=True, exclusive=True, group="rs-scan")
    def _rescan(self) -> None:
        infos = scan_datasets()
        self.app.call_from_thread(self._set_datasets, infos)

    def _set_datasets(self, infos: List[DatasetInfo]) -> None:
        self._rs_ds = infos
        t = self.query_one("#rs-ds", DataTable)
        t.clear()
        fmt = lambda x: f"{x:%Y-%m-%d}" if x is not None and pd.notna(x) else "–"
        for i in infos:
            t.add_row(short_name(i.name, 30), f"{i.rows:,}", i.timeframe, fmt(i.start), fmt(i.end),
                      human_size(i.size), key=str(i.path))
        opts = [(f"{short_name(i.name, 30)} · {i.rows:,}", str(i.path)) for i in infos] or [("No datasets", "")]
        sel = self.query_one("#op-dataset", Select)
        sel.set_options(opts)
        sel.value = opts[0][1]
        if infos:
            self._load_preview(infos[0])

    @on(Button.Pressed, "#rs-rescan")
    def _rescan_btn(self) -> None:
        self._rescan()

    def _selected_dataset(self) -> Optional[DatasetInfo]:
        t = self.query_one("#rs-ds", DataTable)
        if t.row_count == 0:
            return None
        try:
            key = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        except Exception:
            return None
        return next((i for i in self._rs_ds if str(i.path) == key), None)

    @on(DataTable.RowHighlighted, "#rs-ds")
    def _highlight(self, ev: DataTable.RowHighlighted) -> None:
        info = next((i for i in self._rs_ds if ev.row_key and str(i.path) == ev.row_key.value), None)
        if info:
            self._load_preview(info)

    @work(thread=True, exclusive=True, group="rs-preview")
    def _load_preview(self, info: DatasetInfo) -> None:
        try:
            df = self.app.bat.datasets.get(info.path)
            stats = dataset_stats(df)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.query_one("#rs-stats", Static).update, Text(f"✗ {e}", style=C.DOWN))
            return
        self.app.call_from_thread(self._show_preview, info, df, stats)

    def _show_preview(self, info: DatasetInfo, df: pd.DataFrame, s: Dict[str, Any]) -> None:
        t = Table(show_header=False, box=None, padding=(0, 2), title=info.name, title_justify="left")
        t.add_row("Bars", f"{s['rows']:,}", "Range", f"{s.get('start', '–')}  →  {s.get('end', '–')}")
        t.add_row("Price", f"{C.fmt_num(s['low'])} – {C.fmt_num(s['high'])}", "Change", Text(C.fmt_pct(s["change"] * 100),
                                                                                           style=C.pnl_color(s["change"])))
        t.add_row("Vol / bar", C.fmt_pct(s["vol_per_bar"] * 100, signed=False, decimals=3),
                  "Avg volume", C.fmt_num(s["avg_volume"]))
        issues = []
        if s.get("gaps"):
            issues.append(f"{s['gaps']} gap(s), largest {s['largest_gap']}")
        if s.get("duplicates"):
            issues.append(f"{s['duplicates']} duplicate timestamps")
        if s.get("bad_ohlc"):
            issues.append(f"{s['bad_ohlc']} inconsistent OHLC rows")
        if s.get("nan_cells"):
            issues.append(f"{s['nan_cells']} missing cells")
        t.add_row("Quality", Text("✓ clean" if not issues else "⚠ " + "; ".join(issues),
                                  style=C.UP if not issues else C.AMBER), "Bar size", str(s.get("bar", "–")))
        self.query_one("#rs-stats", Static).update(t)
        self.query_one("#rs-chart", PriceChart).set_data(df, title=info.name[:24], view_len=min(len(df), 200),
                                                         reset_view=True)

    @on(Button.Pressed, "#rs-use")
    async def _use(self) -> None:
        info = self._selected_dataset()
        if info:
            self.app.goto("backtest")
            await self.app.query_one("#backtest").prefill(dataset=str(info.path))

    @on(Button.Pressed, "#rs-opt")
    def _to_opt(self) -> None:
        info = self._selected_dataset()
        if info:
            self.query_one("#op-dataset", Select).value = str(info.path)
            switch_tab(self, "#rs-tabs", "rs-tab-opt", focus="#op-run")

    @on(Button.Pressed, "#rs-resample")
    def _resample(self) -> None:
        info = self._selected_dataset()
        if not info:
            return

        def _done(vals):
            if vals and vals.get("rule"):
                self._do_resample(info, vals["rule"])
        self.app.push_screen(FormModal("Resample & save", [("rule", "Pandas rule (5min, 15min, 1h, 4h, 1D)", "15min", False)],
                                       note="Writes a new CSV under data/processed/.", submit="Resample"), _done)

    @work(thread=True)
    def _do_resample(self, info: DatasetInfo, rule: str) -> None:
        try:
            df = self.app.bat.datasets.get(info.path)
            out = resample_bars(df, rule)
            path = save_dataset(out, f"{info.path.stem}_{rule}.csv", ROOT / "data" / "processed")
            self.app.call_from_thread(self.app.notify_ok, f"Saved {path.relative_to(ROOT)} ({len(out):,} bars)")
            self.app.call_from_thread(self._rescan)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.app.notify_err, f"Resample failed: {e}")

    # ---------------------------------------------------------------- download
    def _refresh_sources(self) -> None:
        sel = self.query_one("#dl-provider", Select)
        opts = self.app.bat.connections.data_sources() or [("(connect one in Connections)", "")]
        cur = select_value(sel, "")
        sel.set_options(opts)
        sel.value = cur if cur in [v for _, v in opts] else opts[0][1]

    @on(Button.Pressed, "#dl-go")
    def _download(self) -> None:
        key = select_value(self.query_one("#dl-provider", Select), "")
        if not key:
            self.app.notify_err("Connect a data provider first (Connections page)")
            return
        try:
            args = dict(ticker=self.query_one("#dl-ticker", Input).value.strip(),
                        timespan=select_value(self.query_one("#dl-span", Select), "minute"),
                        from_date=self.query_one("#dl-from", Input).value.strip(),
                        to_date=self.query_one("#dl-to", Input).value.strip(),
                        limit=int(self.query_one("#dl-limit", Input).value))
            pd.Timestamp(args["from_date"]), pd.Timestamp(args["to_date"])
            if not args["ticker"]:
                raise ValueError("ticker is required")
        except ValueError as e:
            self.app.notify_err(f"Invalid download settings: {e}")
            return
        self.query_one("#dl-status", Static).update("Downloading…")
        self._do_download(key, args)

    @work(thread=True, exclusive=True, group="rs-dl")
    def _do_download(self, key: str, args: Dict[str, Any]) -> None:
        try:
            df = self.app.bat.connections.providers[key].get_data(**args)
            if df is None or df.empty:
                raise ValueError("provider returned no rows")
            clean = re.sub(r"[:/]", "_", args["ticker"])
            path = save_dataset(df, f"{clean}_{args['timespan']}_{args['from_date']}_to_{args['to_date']}.csv")
            msg = f"✓ Saved {len(df):,} bars → {path.relative_to(ROOT)}"
            self.app.call_from_thread(self.query_one("#dl-status", Static).update, Text(msg, style=C.UP))
            self.app.call_from_thread(self.app.log_event, "ok", "research", msg)
            self.app.call_from_thread(self._rescan)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.query_one("#dl-status", Static).update,
                                      Text(f"✗ Download failed: {e}", style=C.DOWN))

    # --------------------------------------------------------------- optimizer
    @on(Select.Changed, "#op-strategy")
    async def _op_strategy(self, ev: Select.Changed) -> None:
        spec = STRATEGIES[select_value(ev.select)]
        form = self.query_one("#op-ranges", RangeForm)
        if form.spec is not spec:
            await form.set_spec(spec)
        self._update_combos()

    @on(Input.Changed)
    def _op_input(self, ev: Input.Changed) -> None:
        if (ev.input.id or "").startswith("rg-"):
            self._update_combos()

    def _update_combos(self) -> Optional[List[Dict[str, Any]]]:
        info = self.query_one("#op-combos", Static)
        spec = self.query_one("#op-ranges", RangeForm).spec
        try:
            combos = build_grid(spec, self.query_one("#op-ranges", RangeForm).ranges())
        except Exception as e:  # noqa: BLE001
            info.update(Text(f"✗ {e}", style=C.DOWN))
            return None
        n = len(combos)
        bars = 20000
        try:
            bars = int(self.query_one("#op-bars", Input).value) or 50000
        except ValueError:
            pass
        est = n * 2 * bars / BARS_PER_SEC
        color = C.DOWN if n > MAX_COMBOS else C.AMBER if est > 120 else C.UP
        info.update(Text(f"{n} combinations · ≈ {est:.0f}s" + (f"  (max {MAX_COMBOS})" if n > MAX_COMBOS else ""),
                         style=color))
        return combos

    def action_run(self) -> None:
        if self.query_one("#rs-tabs", TabbedContent).active == "rs-tab-opt":
            self._run_sweep()

    @on(Button.Pressed, "#op-run")
    def _run_sweep(self) -> None:
        if self._rs_busy:
            return
        combos = self._update_combos()
        if not combos:
            self.app.notify_err("No valid parameter combinations")
            return
        if len(combos) > MAX_COMBOS:
            self.app.notify_err(f"Too many combinations ({len(combos)} > {MAX_COMBOS}); narrow the ranges")
            return
        path = select_value(self.query_one("#op-dataset", Select), "")
        if not path:
            self.app.notify_err("Pick a dataset")
            return
        try:
            ctx = dict(spec=self.query_one("#op-ranges", RangeForm).spec, path=path, combos=combos,
                       resample=select_value(self.query_one("#op-resample", Select), "none"),
                       bars=int(float(self.query_one("#op-bars", Input).value or 0)),
                       cfg=BacktestConfig(initial_cash=float(self.query_one("#op-balance", Input).value),
                                          trading_mode=select_value(self.query_one("#op-mode", Select), "long_only"),
                                          position_fraction=float(self.query_one("#op-pos", Input).value) / 100,
                                          spread_pips=float(self.query_one("#op-spread", Input).value),
                                          symbol=symbol_from_filename(Path(path).name)),
                       split=float(self.query_one("#op-split", Input).value) / 100,
                       objective=select_value(self.query_one("#op-objective", Select), "sharpe"))
            if not 0.1 <= ctx["split"] <= 0.95:
                raise ValueError("Train % must be between 10 and 95")
        except ValueError as e:
            self.app.notify_err(f"Invalid settings: {e}")
            return
        self._rs_ctx = ctx
        self._rs_busy = True
        self._rs_cancel.clear()
        self.query_one("#op-run", Button).disabled = True
        self.query_one("#op-cancel", Button).disabled = False
        bar = self.query_one("#op-progress", ProgressBar)
        bar.display = True
        bar.update(total=len(combos), progress=0)
        self._status(f"Sweeping {len(combos)} combinations…", "busy")
        self._sweep(ctx)

    @work(thread=True, exclusive=True, group="rs-sweep")
    def _sweep(self, ctx: Dict[str, Any]) -> None:
        ui = self.app.call_from_thread
        try:
            kw = {"resample": None if ctx["resample"] == "none" else ctx["resample"]}
            df = self.app.bat.datasets.get(ctx["path"], **kw)
            if ctx["bars"]:
                df = df.tail(ctx["bars"]).reset_index(drop=True)
            res = run_grid(df, ctx["spec"], ctx["combos"], ctx["cfg"], split=ctx["split"], objective=ctx["objective"],
                           progress=lambda i, n, el: ui(self._progress, i, n, el), cancel=self._rs_cancel)
            ui(self._sweep_done, res, len(df))
        except Exception as e:  # noqa: BLE001
            ui(self._sweep_failed, e)

    def _progress(self, i: int, n: int, elapsed: float) -> None:
        self.query_one("#op-progress", ProgressBar).update(total=n, progress=i)
        eta = elapsed / i * (n - i) if i else 0
        self._status(f"{i}/{n} done · {elapsed:.0f}s elapsed · ~{eta:.0f}s left", "busy")

    def _sweep_failed(self, e: Exception) -> None:
        self._finish_sweep()
        self._status(f"✗ {type(e).__name__}: {e}", "error")
        self.app.notify_err(f"Sweep failed: {e}")

    def _finish_sweep(self) -> None:
        self._rs_busy = False
        self.query_one("#op-run", Button).disabled = False
        self.query_one("#op-cancel", Button).disabled = True

    def _sweep_done(self, res: pd.DataFrame, nbars: int) -> None:
        self._finish_sweep()
        cancelled = self._rs_cancel.is_set()
        self._rs_results = res
        self._rs_sort = {"col": None, "rev": False}
        self._status(f"{'Cancelled — partial results: ' if cancelled else 'Done: '}{len(res)} combinations on "
                     f"{nbars:,} bars, ranked by train {OBJECTIVES[self._rs_ctx['objective']]}",
                     "error" if cancelled else "ok")
        self.app.log_event("ok", "research", f"Sweep of {self._rs_ctx['spec'].label}: {len(res)} combos")
        self._fill_results()
        self._setup_heat()

    @on(Button.Pressed, "#op-cancel")
    def _cancel(self) -> None:
        self._rs_cancel.set()

    def _status(self, msg: str, kind: str = "") -> None:
        s = self.query_one("#op-status", Static)
        s.update(msg)
        s.set_classes(f"status-line {kind}".strip())

    # -- results table ---------------------------------------------------------
    def _result_columns(self) -> List[str]:
        spec = self._rs_ctx.get("spec")
        pnames = [p.name for p in spec.params] if spec else []
        return pnames + ["train_total_return", "train_sharpe", "train_profit_factor", "train_max_dd", "train_trades",
                         "test_total_return", "test_sharpe", "test_profit_factor", "test_trades"]

    LABELS = {"train_total_return": "Train ret %", "train_sharpe": "Train Sharpe", "train_profit_factor": "Train PF",
              "train_max_dd": "Train DD %", "train_trades": "Trn #", "test_total_return": "Test ret %",
              "test_sharpe": "Test Sharpe", "test_profit_factor": "Test PF", "test_trades": "Tst #"}

    def _fill_results(self) -> None:
        t = self.query_one("#op-results", DataTable)
        t.clear(columns=True)
        res = self._rs_results
        if res.empty:
            return
        cols = [c for c in self._result_columns() if c in res.columns]
        t.add_column("#", key="rank")
        for c in cols:
            t.add_column(self.LABELS.get(c, c), key=c)
        t.add_column("Robust", key="robust")
        view = res
        if self._rs_sort["col"] in res.columns:
            view = res.sort_values(self._rs_sort["col"], ascending=not self._rs_sort["rev"], na_position="last")
        for rank, (idx, r) in enumerate(view.head(500).iterrows(), 1):
            cells: List[Any] = [str(rank)]
            for c in cols:
                v = r[c]
                if c.endswith("total_return") or c.endswith("max_dd"):
                    cells.append(Text(f"{v:+.2f}" if np.isfinite(v) else "–", style=C.pnl_color(v) if "ret" in c else "default"))
                elif c.endswith("sharpe") or c.endswith("profit_factor"):
                    cells.append("∞" if v == np.inf else f"{v:.2f}" if np.isfinite(v) else "–")
                elif c.endswith("trades"):
                    cells.append(str(int(v)) if np.isfinite(v) else "–")
                else:
                    cells.append(f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v))
            ok = r.get("train_total_return", 0) > 0 and r.get("test_total_return", 0) > 0 and r.get("test_trades", 0) >= 5
            cells.append(Text("✓", style=C.UP) if ok else Text("·", style="dim"))
            t.add_row(*cells, key=str(idx))

    @on(DataTable.HeaderSelected, "#op-results")
    def _sort_results(self, ev: DataTable.HeaderSelected) -> None:
        col = str(ev.column_key.value)
        if col in ("rank", "robust"):
            return
        self._rs_sort["rev"] = (not self._rs_sort["rev"]) if self._rs_sort["col"] == col else True
        self._rs_sort["col"] = col
        self._fill_results()

    # -- heat map ---------------------------------------------------------------
    def _setup_heat(self) -> None:
        spec = self._rs_ctx["spec"]
        res = self._rs_results
        numeric = [p.name for p in spec.params if p.kind != "choice" and res[p.name].nunique() > 1] if len(res) else []
        metrics = [(self.LABELS[c], c) for c in self.LABELS if c in res.columns]
        for sid, opts, val in (("#op-hx", [(n, n) for n in numeric] or [("–", "")], numeric[0] if numeric else ""),
                               ("#op-hy", [(n, n) for n in numeric] or [("–", "")], numeric[1] if len(numeric) > 1 else (numeric[0] if numeric else "")),
                               ("#op-hm", metrics or [("–", "")], f"train_{self._rs_ctx['objective']}" if f"train_{self._rs_ctx['objective']}" in res.columns else (metrics[0][1] if metrics else ""))):
            sel = self.query_one(sid, Select)
            sel.set_options(opts)
            sel.value = val
        self._draw_heat()

    @on(Select.Changed, "#op-hx")
    @on(Select.Changed, "#op-hy")
    @on(Select.Changed, "#op-hm")
    def _draw_heat(self) -> None:
        heat = self.query_one("#op-heat", Static)
        res = self._rs_results
        x = select_value(self.query_one("#op-hx", Select), "")
        y = select_value(self.query_one("#op-hy", Select), "")
        m = select_value(self.query_one("#op-hm", Select), "")
        if res.empty or not x or not m or x not in res or m not in res:
            heat.update("")
            return
        if y == x or y not in res:
            piv = res.groupby(x)[m].mean().to_frame().T
            rows, cols = [m], list(piv.columns)
        else:
            piv = res.pivot_table(index=y, columns=x, values=m, aggfunc="mean")
            rows, cols = list(piv.index), list(piv.columns)
        mat = piv.to_numpy(dtype=float)
        mat = np.where(np.isinf(mat), np.nan, mat)
        diverging = m.endswith("return") or m.endswith("sharpe")
        fmt = (lambda v: f"{v:.1f}") if "return" in m or "dd" in m else (lambda v: f"{v:.2f}")
        title = f"{self.LABELS.get(m, m)} · rows {y or '–'} · cols {x}  (mean over other parameters, ★ = best)"
        heat.update(Text.assemble((title + "\n", "bold"), C.heatmap(
            mat, [f"{r:g}" if isinstance(r, (int, float, np.number)) else str(r) for r in rows],
            [f"{c:g}" if isinstance(c, (int, float, np.number)) else str(c) for c in cols],
            cell_width=8, fmt=fmt, diverging=diverging, mark_best=True)))

    # -- apply / export -------------------------------------------------------------
    def _selected_combo(self) -> Optional[Dict[str, Any]]:
        t = self.query_one("#op-results", DataTable)
        if t.row_count == 0 or self._rs_results.empty:
            return None
        try:
            idx = int(t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value)
        except Exception:
            return None
        spec = self._rs_ctx["spec"]
        row = self._rs_results.loc[idx]
        return {p.name: (int(row[p.name]) if p.kind == "int" else row[p.name]) for p in spec.params}

    @on(Button.Pressed, "#op-apply")
    async def _apply(self) -> None:
        combo = self._selected_combo()
        if combo is None:
            self.app.notify("Select a row in the results first", severity="warning")
            return
        spec = self._rs_ctx["spec"]
        self.app.goto("backtest")
        await self.app.query_one("#backtest").prefill(strategy_key=spec.key, params=combo, dataset=self._rs_ctx["path"])
        self.app.notify_ok("Parameters copied to the Backtest page — press ctrl+r to run")

    @on(Button.Pressed, "#op-export")
    def _export(self) -> None:
        if self._rs_results.empty:
            self.app.notify("Nothing to export yet", severity="warning")
            return
        out = ROOT / "runs" / "optimize"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self._rs_ctx['spec'].key}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self._rs_results.to_csv(path, index=False)
        self.app.notify_ok(f"Exported {path.relative_to(ROOT)}")
