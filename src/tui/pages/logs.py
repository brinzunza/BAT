"""Global event log."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Button, Select, Static

from src.tui.state import LogEntry
from src.tui.widgets.common import EventLog, select_value

LEVELS = [("Everything", "detail"), ("Info and above", "info"), ("Signals and trades", "signal"), ("Errors only", "error")]


class LogsPage(Widget):
    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(classes="btn-row"):
                yield Select(LEVELS, value="info", allow_blank=False, id="log-level", compact=True)
                yield Button("Clear", id="log-clear")
                yield Button("Save…", id="log-save")
            yield EventLog(id="log-view")

    def on_mount(self) -> None:
        view = self.query_one("#log-view", EventLog)
        for e in self.app.bat.events:
            view.add(e.level, e.text, e.source, e.ts)

    def push(self, e: LogEntry) -> None:
        self.query_one("#log-view", EventLog).add(e.level, e.text, e.source, e.ts)

    @on(Select.Changed, "#log-level")
    def _level(self, ev: Select.Changed) -> None:
        self.query_one("#log-view", EventLog).set_level(select_value(ev.select, "info"))

    @on(Button.Pressed, "#log-clear")
    def _clear(self) -> None:
        self.app.bat.events.clear()
        self.query_one("#log-view", EventLog).wipe()

    @on(Button.Pressed, "#log-save")
    def _save(self) -> None:
        out = Path("runs") / "logs"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"bat_{datetime.now():%Y%m%d_%H%M%S}.log"
        path.write_text("\n".join(f"{e.ts:%Y-%m-%d %H:%M:%S} [{e.level}] {e.source}: {e.text}"
                                  for e in self.app.bat.events))
        self.app.notify_ok(f"Saved {path}")
