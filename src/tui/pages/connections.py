"""Connections: configure and test data providers and brokers."""
from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Grid, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Static

from src.tui.services.connections import SPECS, ConnSpec
from src.tui.widgets.common import Section
from src.tui.widgets.modals import FormModal


class ConnCard(Static):
    def __init__(self, spec: ConnSpec):
        super().__init__(id=f"conn-{spec.key}")
        self.spec = spec

    def compose(self) -> ComposeResult:
        yield Static(id=f"conn-{self.spec.key}-info")
        yield Button("Connect with environment", id=f"env-{self.spec.key}", variant="primary")
        yield Button("Enter credentials…", id=f"form-{self.spec.key}")

    def update_state(self, connected: bool, configured: bool, detail: str) -> None:
        self.set_class(connected, "-connected")
        mark = "[green]● connected[/]" if connected else "[yellow]○ credentials found[/]" if configured \
            else "[dim]· not configured[/]"
        env = f"[dim]env: {', '.join(self.spec.env)}[/]" if self.spec.env else ""
        self.query_one(f"#conn-{self.spec.key}-info", Static).update(
            f"[b]{self.spec.label}[/b]  {mark}\n[dim]{self.spec.note}[/]\n{detail}\n{env}")
        self.query_one(f"#env-{self.spec.key}", Button).disabled = not configured and bool(self.spec.env)


class ConnectionsPage(Widget):
    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("[dim]Credentials are read from the environment (.env) or typed here. They stay in memory "
                         "for this session only and are never written to disk.[/]", classes="muted")
            yield Section("Always available")
            yield Static("[green]●[/] [b]Simulated broker[/b] — in-memory fills at the last price, any balance\n"
                         "[green]●[/] [b]Demo market[/b] — offline random-walk feed for trying strategies safely\n"
                         "[green]●[/] [b]Local datasets[/b] — CSVs under data/raw for backtesting", classes="panel")
            yield Section("Data providers & brokers")
            with Grid(id="conn-grid"):
                for spec in SPECS.values():
                    yield ConnCard(spec)

    def on_mount(self) -> None:
        self.refresh_cards()

    def on_show(self) -> None:
        self.refresh_cards()

    def refresh_cards(self) -> None:
        by = {s.key: s for s in self.app.bat.connections.status()}
        for card in self.query(ConnCard):
            s = by[card.spec.key]
            card.update_state(s.connected, s.configured, s.detail)

    # ----- connect ------------------------------------------------------------
    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid.startswith("env-"):
            self._connect(bid[4:], {})
        elif bid.startswith("form-"):
            key = bid[5:]
            spec = SPECS[key]
            defaults = self.app.bat.connections.env_defaults(key)
            fields = [(n, lbl, defaults.get(n, ""), secret) for n, lbl, secret in spec.fields]
            switches = [("paper", "Paper trading (recommended)", True)] if key == "alpaca" else []
            note = "Values are used for this session only."
            if key == "oanda":
                note += " Environment defaults to the practice account."

            def _done(values):
                if values is not None:
                    paper = values.pop("paper", True)
                    if key == "alpaca":
                        values["paper"] = "true" if paper else "false"
                    self._connect(key, values)
            self.app.push_screen(FormModal(f"Connect · {spec.label}", fields, switches, note), _done)

    @work(thread=True, exclusive=True, group="connect")
    def _connect(self, key: str, creds: dict) -> None:
        self.app.call_from_thread(self.app.log_event, "info", "connect", f"Connecting to {SPECS[key].label}…")
        ok, msg = self.app.bat.connections.connect(key, **creds)
        self.app.call_from_thread(self._connected, key, ok, msg)

    def _connected(self, key: str, ok: bool, msg: str) -> None:
        self.app.log_event("ok" if ok else "error", "connect", f"{SPECS[key].label}: {msg}")
        (self.app.notify_ok if ok else self.app.notify_err)(f"{SPECS[key].label}: {msg}")
        self.refresh_cards()
