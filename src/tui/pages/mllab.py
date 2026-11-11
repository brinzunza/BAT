"""ML Lab: train, compare, evaluate and run model checkpoints."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import (Button, DataTable, Input, ProgressBar, RichLog, Select, Static, TabbedContent,
                             TabPane)

from src.tui import charts as C
from src.tui.datasets import human_size, scan_datasets, short_name
from src.tui.services import mlops
from src.tui.widgets.common import LabeledInput, LabeledSelect, Plot, Section, select_value, GuardedTabs, switch_tab

ROOT = mlops.ROOT
RESAMPLE = [("Raw bars", "none"), ("5 min", "5min"), ("15 min", "15min"), ("30 min", "30min"), ("1 hour", "1h")]
DEVICES = [("Auto", "auto"), ("CPU", "cpu"), ("CUDA", "cuda"), ("Apple MPS", "mps")]


def _opts(seq):
    return [(str(s), str(s)) for s in seq]


class MLPage(Widget):
    BINDINGS = [Binding("ctrl+r", "run", "Run")]

    def __init__(self) -> None:
        super().__init__()
        self.ml_base = mlops.load_configs()
        self.ml_job: Optional[mlops.Job] = None
        self.ml_eval_job: Optional[mlops.Job] = None
        self.ml_eval_lines: List[str] = []
        self.ml_run_name = ""
        self.ml_runs: List[mlops.RunInfo] = []
        self.ml_picked: set = set()

    # ----------------------------------------------------------------- layout
    def compose(self) -> ComposeResult:
        d, m, t = self.ml_base["data"], self.ml_base["model"], self.ml_base["train"]
        with GuardedTabs(id="ml-tabs"):
            with TabPane("Train", id="ml-tab-train"):
                with Horizontal():
                    with VerticalScroll(classes="config"):
                        yield Section("Run")
                        yield LabeledInput("Run name", f"run_{time.strftime('%m%d_%H%M')}", input_id="ml-name")
                        yield Section("Model")
                        yield LabeledSelect("Architecture", _opts(mlops.MODELS), select_id="ml-model-name",
                                            value=str(m.get("name", "lstm")))
                        with Horizontal(classes="row"):
                            yield LabeledInput("Hidden", str(m.get("hidden", 64)), input_id="ml-model-hidden")
                            yield LabeledInput("Layers", str(m.get("layers", 2)), input_id="ml-model-layers")
                        with Horizontal(classes="row"):
                            yield LabeledInput("Dropout", str(m.get("dropout", 0.2)), input_id="ml-model-dropout")
                            yield LabeledInput("Heads (transf.)", str(m.get("heads", 4)), input_id="ml-model-heads")
                        yield LabeledInput("Kernel (cnn)", str(m.get("kernel", 5)), input_id="ml-model-kernel")
                        yield Section("Data")
                        yield LabeledSelect("Dataset", [("(scanning…)", "")], select_id="ml-data-path")
                        with Horizontal(classes="row"):
                            yield LabeledSelect("Resample", RESAMPLE, select_id="ml-data-resample",
                                                value=str(d.get("resample") or "none"))
                            yield LabeledSelect("Target", [("direction", "direction"), ("return", "return")],
                                                select_id="ml-data-target", value=str(d.get("target", "direction")))
                        with Horizontal(classes="row"):
                            yield LabeledInput("Seq length", str(d.get("seq_len", 64)), input_id="ml-data-seq_len")
                            yield LabeledInput("Horizon", str(d.get("horizon", 4)), input_id="ml-data-horizon")
                        with Horizontal(classes="row"):
                            yield LabeledInput("Embargo", str(d.get("embargo", 4)), input_id="ml-data-embargo")
                            yield LabeledInput("Batch size", str(d.get("batch_size", 256)), input_id="ml-data-batch_size")
                        yield Section("Training")
                        with Horizontal(classes="row"):
                            yield LabeledInput("Epochs", str(t.get("epochs", 30)), input_id="ml-train-epochs")
                            yield LabeledInput("Patience", str(t.get("patience", 6)), input_id="ml-train-patience")
                        with Horizontal(classes="row"):
                            yield LabeledInput("Learning rate", str(t.get("lr", 0.001)), input_id="ml-train-lr")
                            yield LabeledInput("Weight decay", str(t.get("weight_decay", 0.0001)),
                                               input_id="ml-train-weight_decay")
                        with Horizontal(classes="row"):
                            yield LabeledSelect("Loss", _opts(mlops.LOSSES), select_id="ml-train-loss",
                                                value=str(t.get("loss", "bce")))
                            yield LabeledSelect("Optimizer", _opts(mlops.OPTIMIZERS), select_id="ml-train-optimizer",
                                                value=str(t.get("optimizer", "adamw")))
                        with Horizontal(classes="row"):
                            yield LabeledSelect("Scheduler", _opts(mlops.SCHEDULERS), select_id="ml-train-scheduler",
                                                value=str(t.get("scheduler", "cosine")))
                            yield LabeledSelect("Device", DEVICES, select_id="ml-train-device",
                                                value=str(t.get("device", "auto")))
                        with Horizontal(classes="row"):
                            yield LabeledInput("Grad clip", str(t.get("grad_clip", 1.0)), input_id="ml-train-grad_clip")
                            yield LabeledInput("Seed", str(t.get("seed", 42)), input_id="ml-train-seed")
                        with Horizontal(classes="btn-row actions"):
                            yield Button("Train ▶", id="ml-start", variant="primary")
                            yield Button("Stop", id="ml-stop", variant="error", disabled=True)
                    with Vertical(classes="results"):
                        yield Static("Configure a run and press ctrl+r. Training runs as a separate process.",
                                     id="ml-status", classes="status-line")
                        yield Static("", id="ml-cmd", classes="muted")
                        yield ProgressBar(total=100, show_eta=True, id="ml-progress")
                        yield Plot(id="ml-loss")
                        yield RichLog(id="ml-log", wrap=True, markup=False, max_lines=2000, min_width=20)
            with TabPane("Runs", id="ml-tab-runs"):
                with Horizontal(classes="btn-row"):
                    yield Button("Refresh", id="ml-runs-refresh")
                    yield Button("Evaluate this run", id="ml-runs-eval")
                    yield Static("  Enter toggles ✓ to overlay validation curves", classes="muted")
                yield DataTable(id="ml-runs", cursor_type="row", zebra_stripes=True)
                yield Plot(id="ml-runs-plot")
            with TabPane("Evaluate", id="ml-tab-eval"):
                with Horizontal():
                    with VerticalScroll(classes="config"):
                        yield Section("Evaluate a checkpoint")
                        yield LabeledSelect("Checkpoint", [("(none)", "")], select_id="ev-ckpt")
                        yield LabeledSelect("Mode", [("Held-out test split", "test"),
                                                     ("Walk-forward (retrains each fold)", "wf")],
                                            select_id="ev-mode", value="test")
                        with Horizontal(classes="row"):
                            yield LabeledInput("Folds", "5", input_id="ev-folds")
                            yield LabeledInput("Cost / turn", "0.0001", input_id="ev-cost")
                        with Horizontal(classes="btn-row actions"):
                            yield Button("Evaluate ▶", id="ev-run", variant="primary")
                            yield Button("Stop", id="ev-stop", variant="error", disabled=True)
                    with Vertical(classes="results"):
                        yield Static("Pick a checkpoint and press Evaluate.", id="ev-status", classes="status-line")
                        yield DataTable(id="ev-table", cursor_type="row", zebra_stripes=True)
                        yield Plot(id="ev-plot")
                        yield Static("", id="ev-summary")
                        yield RichLog(id="ev-log", wrap=True, markup=False, max_lines=500, min_width=20)
            with TabPane("Predict", id="ml-tab-pred"):
                with Horizontal():
                    with VerticalScroll(classes="config"):
                        yield Section("Score the latest bars")
                        yield LabeledSelect("Checkpoint", [("(none)", "")], select_id="pr-ckpt")
                        yield LabeledSelect("Bars (CSV)", [("(scanning…)", "")], select_id="pr-data")
                        with Horizontal(classes="btn-row actions"):
                            yield Button("Predict ▶", id="pr-run", variant="primary")
                    yield Static("Scores the most recent closed bar using the checkpoint's own resample and scaler.",
                                 id="pr-out", classes="results")
            with TabPane("Checkpoints", id="ml-tab-ck"):
                with Horizontal(classes="btn-row"):
                    yield Button("Refresh", id="ck-refresh")
                    yield Button("Inspect config", id="ck-inspect")
                    yield Static("  Inspecting loads the file with pickle — only open checkpoints you created.",
                                 classes="muted")
                yield DataTable(id="ck-table", cursor_type="row", zebra_stripes=True)
                yield Static("", id="ck-cfg", classes="panel")

    def on_mount(self) -> None:
        self.query_one("#ml-runs", DataTable).add_columns("✓", "Run", "Type", "Epochs", "Best val", "@ epoch",
                                                          "Last train", "Last val", "Modified")
        self.query_one("#ev-table", DataTable)
        self.query_one("#ck-table", DataTable).add_columns("Checkpoint", "Size", "Modified")
        self.query_one("#ml-progress", ProgressBar).display = False
        self.query_one("#ml-loss", Plot).styles.height = 14
        self.query_one("#ml-runs-plot", Plot).styles.height = 14
        self.query_one("#ev-plot", Plot).styles.height = 12
        self._scan_data()
        self._refresh_runs()
        self._refresh_ckpts()
        self._update_cmd()
        self.set_interval(1.0, self._poll_metrics)

    def on_show(self) -> None:
        self._refresh_runs()
        self._refresh_ckpts()

    # ---------------------------------------------------------------- datasets
    @work(thread=True, exclusive=True, group="ml-scan")
    def _scan_data(self) -> None:
        self.app.call_from_thread(self._set_data, scan_datasets())

    def _set_data(self, infos) -> None:
        opts = [(f"{short_name(i.name, 30)} · {i.rows:,}", str(i.path.relative_to(ROOT)) if ROOT in i.path.parents else str(i.path))
                for i in infos] or [("No datasets", "")]
        cfg_path = str(self.ml_base["data"].get("path", ""))
        for sid in ("#ml-data-path", "#pr-data"):
            sel = self.query_one(sid, Select)
            sel.set_options(opts)
            vals = [v for _, v in opts]
            sel.value = cfg_path if (sid == "#ml-data-path" and cfg_path in vals) else opts[0][1]
        self._update_cmd()

    # ------------------------------------------------------------------- train
    def _gather(self) -> Dict[str, Dict[str, Any]]:
        def q(i): return self.query_one(f"#{i}")
        def num(i, cast=float):
            try:
                return cast(float(q(i).value)) if cast is int else cast(q(i).value)
            except ValueError:
                raise ValueError(f"'{q(i).value}' is not a valid number ({i.split('-', 2)[-1]})")
        model = dict(name=select_value(q("ml-model-name")), hidden=num("ml-model-hidden", int),
                     layers=num("ml-model-layers", int), dropout=num("ml-model-dropout"),
                     heads=num("ml-model-heads", int), kernel=num("ml-model-kernel", int))
        rs = select_value(q("ml-data-resample"), "none")
        data = dict(path=select_value(q("ml-data-path"), ""), resample=None if rs == "none" else rs,
                    seq_len=num("ml-data-seq_len", int), horizon=num("ml-data-horizon", int),
                    target=select_value(q("ml-data-target")), embargo=num("ml-data-embargo", int),
                    batch_size=num("ml-data-batch_size", int))
        train = dict(epochs=num("ml-train-epochs", int), patience=num("ml-train-patience", int), lr=num("ml-train-lr"),
                     weight_decay=num("ml-train-weight_decay"), loss=select_value(q("ml-train-loss")),
                     optimizer=select_value(q("ml-train-optimizer")), scheduler=select_value(q("ml-train-scheduler")),
                     device=select_value(q("ml-train-device")), grad_clip=num("ml-train-grad_clip"),
                     seed=num("ml-train-seed", int))
        return dict(data=data, model=model, train=train)

    def _validate(self, cfgs) -> Optional[str]:
        m, d, t = cfgs["model"], cfgs["data"], cfgs["train"]
        if not d["path"]:
            return "Pick a dataset"
        if m["name"] == "transformer" and m["hidden"] % m["heads"] != 0:
            return f"Transformer: hidden ({m['hidden']}) must be divisible by heads ({m['heads']})"
        if d["target"] == "direction" and t["loss"] != "bce":
            return "Direction target needs the 'bce' loss"
        if d["target"] == "return" and t["loss"] == "bce":
            return "Return target needs 'mse' or 'huber'"
        if d["embargo"] < d["horizon"]:
            return f"Embargo ({d['embargo']}) should be ≥ horizon ({d['horizon']}) to avoid label leakage"
        if min(m["hidden"], m["layers"], d["seq_len"], d["horizon"], d["batch_size"], t["epochs"]) < 1:
            return "Sizes, horizon, batch size and epochs must be ≥ 1"
        return None

    @on(Input.Changed)
    @on(Select.Changed)
    def _form_changed(self, ev) -> None:
        wid = getattr(ev, "input", None) or getattr(ev, "select", None)
        if wid is not None and (wid.id or "").startswith("ml-"):
            self._update_cmd()

    def _update_cmd(self) -> None:
        try:
            cfgs = self._gather()
            ov = mlops.overrides_from(cfgs, self.ml_base)
            name = self.query_one("#ml-name", Input).value.strip() or "run"
            self.query_one("#ml-cmd", Static).update(
                Text("$ python train.py --name " + name + (" --override " + " ".join(ov) if ov else ""), style="dim"))
        except Exception:
            pass

    def action_run(self) -> None:
        tab = self.query_one("#ml-tabs", TabbedContent).active
        if tab == "ml-tab-train":
            self._train()
        elif tab == "ml-tab-eval":
            self._evaluate()
        elif tab == "ml-tab-pred":
            self._predict()

    def _status(self, msg: str, kind: str = "") -> None:
        s = self.query_one("#ml-status", Static)
        s.update(msg)
        s.set_classes(f"status-line {kind}".strip())

    @on(Button.Pressed, "#ml-start")
    def _train(self) -> None:
        if self.ml_job and self.ml_job.running:
            self.app.notify("Training already running", severity="warning")
            return
        try:
            cfgs = self._gather()
        except ValueError as e:
            self._status(f"✗ {e}", "error")
            return
        err = self._validate(cfgs)
        if err:
            self._status(f"✗ {err}", "error")
            return
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.query_one("#ml-name", Input).value.strip() or "run")
        self.ml_run_name = name
        argv = mlops.train_argv(name, mlops.overrides_from(cfgs, self.ml_base))
        log = self.query_one("#ml-log", RichLog)
        log.clear()
        self.ml_epochs = cfgs["train"]["epochs"]
        self.query_one("#ml-loss", Plot).set_builder(None)
        bar = self.query_one("#ml-progress", ProgressBar)
        bar.display = True
        bar.update(total=self.ml_epochs, progress=0)
        self.query_one("#ml-start", Button).disabled = True
        self.query_one("#ml-stop", Button).disabled = False
        self._status(f"Training '{name}' …", "busy")
        self.app.log_event("info", "ml", f"Training started: {name}")
        ui = self.app.call_from_thread
        self.ml_job = mlops.Job(argv, on_line=lambda l: ui(self._train_line, l), on_done=lambda rc: ui(self._train_done, rc))
        self.ml_job.start()

    def _train_line(self, line: str) -> None:
        self.query_one("#ml-log", RichLog).write(Text(line, style="dim" if "INFO" in line else "default"))

    @on(Button.Pressed, "#ml-stop")
    def _stop(self) -> None:
        if self.ml_job:
            self.ml_job.cancel()
            self._status("Stopping…", "busy")

    def _train_done(self, rc: int) -> None:
        self.query_one("#ml-start", Button).disabled = False
        self.query_one("#ml-stop", Button).disabled = True
        self._poll_metrics(final=True)
        if rc == 0:
            self._status(f"✓ Training finished: checkpoints/{self.ml_run_name}.pt", "ok")
            self.app.notify_ok(f"Training '{self.ml_run_name}' finished")
            self.app.log_event("ok", "ml", f"Training finished: {self.ml_run_name}")
        elif rc == -15:
            self._status("Training stopped", "error")
            self.app.log_event("warn", "ml", f"Training stopped: {self.ml_run_name}")
        else:
            self._status(f"✗ Training failed (exit {rc}) — see the log below", "error")
            self.app.notify_err("Training failed — see ML Lab log")
            self.app.log_event("error", "ml", f"Training failed ({rc}): {self.ml_run_name}")
        self._refresh_runs()
        self._refresh_ckpts()

    def _poll_metrics(self, final: bool = False) -> None:
        if not (final or (self.ml_job and self.ml_job.running)) or not self.ml_run_name:
            return
        rows = mlops.read_metrics(ROOT / "runs" / self.ml_run_name)
        rows = [r for r in rows if "val_loss" in r]
        if not rows:
            return
        tr = [r["train_loss"] for r in rows]
        va = [r["val_loss"] for r in rows]
        best = int(np.argmin(va))
        self.query_one("#ml-progress", ProgressBar).update(total=self.ml_epochs, progress=len(rows))
        n = len(rows)
        self.query_one("#ml-loss", Plot).set_builder(lambda w, h: C.line_chart(
            {"train": tr, "val": va}, w, h, colors={"train": C.ORANGE, "val": C.AMBER},
            x_labels=["1", str(max(n // 2, 1)), str(n)] if n > 2 else None, y_fmt=lambda v: f"{v:.4f}",
            title=f"Loss · epoch {n}/{self.ml_epochs} · best val {va[best]:.4f} @ {best + 1}"))
        if not final:
            self._status(f"Training '{self.ml_run_name}': epoch {n}/{self.ml_epochs} · train {tr[-1]:.4f} · val {va[-1]:.4f}", "busy")

    # -------------------------------------------------------------------- runs
    def _refresh_runs(self) -> None:
        self.ml_runs = mlops.list_runs()
        if not self.ml_picked:
            self.ml_picked = {r.name for r in self.ml_runs[:3]}
        self._fill_runs()

    def _fill_runs(self) -> None:
        t = self.query_one("#ml-runs", DataTable)
        t.clear()
        for r in self.ml_runs:
            t.add_row(Text("✓", style=C.UP) if r.name in self.ml_picked else "", r.name, r.kind, str(r.epochs),
                      f"{r.best_val:.4f}" if r.best_val == r.best_val else "–", str(r.best_epoch + 1) if r.kind == "train" else "–",
                      f"{r.last_train:.4f}" if r.last_train == r.last_train else "–",
                      f"{r.last_val:.4f}" if r.last_val == r.last_val else "–",
                      time.strftime("%m-%d %H:%M", time.localtime(r.mtime)), key=r.name)
        series = {}
        for r in self.ml_runs:
            if r.name in self.ml_picked and r.kind == "train":
                series[r.name[:22]] = [x["val_loss"] for x in mlops.read_metrics(r.path) if "val_loss" in x]
        self.query_one("#ml-runs-plot", Plot).set_builder((lambda w, h: C.line_chart(
            series, w, h, y_fmt=lambda v: f"{v:.4f}", title="Validation loss by epoch")) if series else None)

    @on(DataTable.RowSelected, "#ml-runs")
    def _toggle_run(self, ev: DataTable.RowSelected) -> None:
        name = ev.row_key.value
        self.ml_picked.symmetric_difference_update({name})
        self._fill_runs()

    @on(Button.Pressed, "#ml-runs-refresh")
    def _runs_btn(self) -> None:
        self._refresh_runs()

    @on(Button.Pressed, "#ml-runs-eval")
    def _runs_eval(self) -> None:
        t = self.query_one("#ml-runs", DataTable)
        try:
            name = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        except Exception:
            return
        ck = ROOT / "checkpoints" / f"{name}.pt"
        if not ck.exists():
            self.app.notify_err(f"No checkpoint named {name}.pt")
            return
        self.query_one("#ev-ckpt", Select).value = str(ck)
        switch_tab(self, "#ml-tabs", "ml-tab-eval", focus="#ev-run")

    # ------------------------------------------------------------- checkpoints
    def _refresh_ckpts(self) -> None:
        cks = mlops.list_checkpoints()
        t = self.query_one("#ck-table", DataTable)
        t.clear()
        for c in cks:
            t.add_row(c.name, human_size(c.size), time.strftime("%Y-%m-%d %H:%M", time.localtime(c.mtime)), key=str(c.path))
        opts = [(c.name, str(c.path)) for c in cks] or [("(no checkpoints yet — train one)", "")]
        for sid in ("#ev-ckpt", "#pr-ckpt"):
            sel = self.query_one(sid, Select)
            cur = select_value(sel, "")
            sel.set_options(opts)
            sel.value = cur if cur in [v for _, v in opts] else opts[0][1]

    @on(Button.Pressed, "#ck-refresh")
    def _ck_btn(self) -> None:
        self._refresh_ckpts()

    @on(Button.Pressed, "#ck-inspect")
    def _ck_inspect(self) -> None:
        t = self.query_one("#ck-table", DataTable)
        if t.row_count == 0:
            return
        key = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        self.query_one("#ck-cfg", Static).update("Loading…")
        self._load_ck(Path(key))

    @work(thread=True, exclusive=True, group="ml-ck")
    def _load_ck(self, path: Path) -> None:
        try:
            info = mlops.checkpoint_config(path)
            text = yaml.safe_dump(info["cfg"], sort_keys=False, default_flow_style=False)
            self.app.call_from_thread(self.query_one("#ck-cfg", Static).update, Text(f"{path.name}\n\n{text}"))
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.query_one("#ck-cfg", Static).update, Text(f"✗ {e}", style=C.DOWN))

    # ---------------------------------------------------------------- evaluate
    @on(Button.Pressed, "#ev-run")
    def _evaluate(self) -> None:
        if self.ml_eval_job and self.ml_eval_job.running:
            return
        ck = select_value(self.query_one("#ev-ckpt", Select), "")
        if not ck:
            self.app.notify_err("Pick a checkpoint")
            return
        wf = select_value(self.query_one("#ev-mode", Select), "test") == "wf"
        try:
            folds = int(self.query_one("#ev-folds", Input).value)
            cost = float(self.query_one("#ev-cost", Input).value)
        except ValueError:
            self.app.notify_err("Folds and cost must be numbers")
            return
        self.ml_eval_lines = []
        self.query_one("#ev-log", RichLog).clear()
        self.query_one("#ev-status", Static).update(Text("Evaluating… walk-forward retrains each fold and can take a while."
                                                         if wf else "Evaluating…", style=C.AMBER))
        self.query_one("#ev-run", Button).disabled = True
        self.query_one("#ev-stop", Button).disabled = False
        ui = self.app.call_from_thread

        def on_line(l):
            self.ml_eval_lines.append(l)
            ui(self.query_one("#ev-log", RichLog).write, l)
        self.ml_eval_job = mlops.Job(mlops.evaluate_argv(ck, wf, cost, folds), on_line=on_line,
                                     on_done=lambda rc: ui(self._eval_done, rc, wf))
        self.ml_eval_job.start()

    @on(Button.Pressed, "#ev-stop")
    def _eval_stop(self) -> None:
        if self.ml_eval_job:
            self.ml_eval_job.cancel()

    def _eval_done(self, rc: int, wf: bool) -> None:
        self.query_one("#ev-run", Button).disabled = False
        self.query_one("#ev-stop", Button).disabled = True
        rows = mlops.parse_result_lines(self.ml_eval_lines)
        status = self.query_one("#ev-status", Static)
        if rc != 0 or not rows:
            status.update(Text(f"✗ Evaluation failed (exit {rc}) — see log", style=C.DOWN))
            return
        status.update(Text(f"✓ {'Walk-forward' if wf else 'Test split'}: {len(rows)} result row(s)", style=C.UP))
        t = self.query_one("#ev-table", DataTable)
        t.clear(columns=True)
        cols = list(rows[0].keys())
        for c in cols:
            t.add_column(c)
        for r in rows:
            t.add_row(*[(f"{v:.4f}" if isinstance(v, float) else str(v)) for v in r.values()])
        plot, summ = self.query_one("#ev-plot", Plot), self.query_one("#ev-summary", Static)
        if wf and len(rows) > 1 and "sharpe" in rows[0]:
            sharpe = [r["sharpe"] for r in rows]
            acc = [r["acc"] for r in rows]
            ic = [r.get("ic", float("nan")) for r in rows]
            labels = [f"F{r['fold']}" for r in rows]
            plot.set_builder(lambda w, h: C.bar_chart(sharpe, w, h, labels=labels, title="Sharpe by fold (after costs)"))
            summ.update(Text.assemble(
                ("mean accuracy ", "dim"), f"{np.nanmean(acc):.3f}   ", ("mean IC ", "dim"), f"{np.nanmean(ic):.3f}   ",
                ("mean Sharpe ", "dim"), (f"{np.nanmean(sharpe):.2f}", C.pnl_color(np.nanmean(sharpe))),
                ("   folds > 0 ", "dim"), f"{sum(1 for s in sharpe if s > 0)}/{len(sharpe)}"))
        else:
            plot.set_builder(None)
            summ.update(Text(f"directional accuracy {rows[0].get('acc', float('nan')):.3f} over {rows[0].get('n', '?')} samples "
                             f"(50% = coin flip)", style=C.pnl_color(rows[0].get("acc", 0.5) - 0.5)))

    # ----------------------------------------------------------------- predict
    @on(Button.Pressed, "#pr-run")
    def _predict(self) -> None:
        ck = select_value(self.query_one("#pr-ckpt", Select), "")
        data = select_value(self.query_one("#pr-data", Select), "")
        if not ck or not data:
            self.app.notify_err("Pick a checkpoint and a bars file")
            return
        self.query_one("#pr-out", Static).update("Scoring…")
        self._do_predict(ck, data)

    @work(thread=True, exclusive=True, group="ml-pred")
    def _do_predict(self, ck: str, data: str) -> None:
        try:
            from src.data.preprocess import load_ohlcv
            from src.inference.predict import predict_latest
            cfg = mlops.checkpoint_config(Path(ck))["cfg"]
            bars = load_ohlcv(str(ROOT / data) if not Path(data).is_absolute() else data, cfg["data"].get("resample"))
            out = predict_latest(bars, ck)
            self.app.call_from_thread(self._show_pred, out, cfg)
        except Exception as e:  # noqa: BLE001
            self.app.call_from_thread(self.query_one("#pr-out", Static).update, Text(f"✗ {type(e).__name__}: {e}", style=C.DOWN))

    def _show_pred(self, out: Dict[str, Any], cfg: Dict[str, Any]) -> None:
        t = Text()
        t.append(f"As of {out['as_of']}\n\n", style="dim")
        if out.get("prob_up") is not None:
            p = out["prob_up"]
            t.append(f"P(up over next {cfg['data']['horizon']} bars) = ", style="dim")
            t.append(f"{p * 100:.1f}%\n\n", style=f"bold {C.UP if p >= 0.5 else C.DOWN}")
            t.append_text(C.gauge(p, 0.0, 1.0, 50, "bearish ◀──▶ bullish"))
        else:
            t.append("Predicted forward log-return: ", style="dim")
            t.append(f"{out['score'] * 100:+.3f}%", style=f"bold {C.pnl_color(out['score'])}")
        t.append(f"\n\nraw model output {out['score']:+.4f}", style="dim")
        self.query_one("#pr-out", Static).update(t)
