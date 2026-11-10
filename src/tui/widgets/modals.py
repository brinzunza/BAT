"""Modal dialogs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import RenderableType
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, Switch


class ConfirmModal(ModalScreen[bool]):
    """Yes/no dialog. ``require_text`` forces the user to type a phrase (used for real-money actions)."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, body: RenderableType, *, confirm: str = "Confirm", danger: bool = False,
                 require_text: Optional[str] = None):
        super().__init__()
        self._title, self._body, self._confirm, self._danger, self._req = title, body, confirm, danger, require_text

    def compose(self) -> ComposeResult:
        with Vertical(id="modal", classes="danger" if self._danger else ""):
            yield Label(self._title, id="modal-title")
            yield Static(self._body, id="modal-body")
            if self._req:
                yield Label(f"Type [b]{self._req}[/b] to continue:", classes="muted")
                yield Input(id="confirm-text", compact=True)
            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button(self._confirm, id="ok", variant="error" if self._danger else "primary",
                             disabled=bool(self._req))

    @on(Input.Changed, "#confirm-text")
    def _typed(self, event: Input.Changed) -> None:
        self.query_one("#ok", Button).disabled = event.value.strip() != self._req

    @on(Input.Submitted, "#confirm-text")
    def _submit(self, event: Input.Submitted) -> None:
        if event.value.strip() == self._req:
            self.dismiss(True)

    @on(Button.Pressed, "#ok")
    def _ok(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(False)


class FormModal(ModalScreen[Optional[Dict[str, Any]]]):
    """Generic form: fields are ``(key, label, default, secret)``; ``switches`` are ``(key, label, default)``."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, fields: Sequence[Tuple[str, str, str, bool]],
                 switches: Sequence[Tuple[str, str, bool]] = (), note: str = "", submit: str = "Connect"):
        super().__init__()
        self._title, self._fields, self._switches, self._note, self._submit_label = title, fields, switches, note, submit

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Label(self._title, id="modal-title")
            if self._note:
                yield Static(self._note, classes="muted")
            with VerticalScroll(id="modal-form"):
                for key, label, default, secret in self._fields:
                    yield Label(label, classes="muted")
                    yield Input(default, id=f"f-{key}", password=secret, compact=True)
                for key, label, default in self._switches:
                    with Horizontal(classes="switch-row"):
                        yield Switch(default, id=f"s-{key}")
                        yield Label(label)
            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button(self._submit_label, id="ok", variant="primary")

    @on(Button.Pressed, "#ok")
    @on(Input.Submitted)
    def _ok(self) -> None:
        out: Dict[str, Any] = {k: self.query_one(f"#f-{k}", Input).value.strip() for k, *_ in self._fields}
        out.update({k: self.query_one(f"#s-{k}", Switch).value for k, *_ in self._switches})
        self.dismiss(out)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class HelpModal(ModalScreen[None]):
    BINDINGS = [Binding("escape,question_mark,q", "close", "Close")]

    def __init__(self, text: RenderableType):
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="modal", classes="wide"):
            yield Label("Keyboard & mouse", id="modal-title")
            with VerticalScroll():
                yield Static(self._text)
            with Horizontal(id="modal-buttons"):
                yield Button("Close", id="close", variant="primary")

    @on(Button.Pressed)
    def action_close(self) -> None:
        self.dismiss(None)


class ChoiceModal(ModalScreen[Optional[str]]):
    """Pick one of several actions; escape returns ``None``."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, body: RenderableType, choices: Sequence[Tuple[str, str, str]]):
        super().__init__()
        self._title, self._body, self._choices = title, body, choices   # (label, value, variant)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Label(self._title, id="modal-title")
            yield Static(self._body, id="modal-body")
            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="choice-cancel")
                for i, (label, _, variant) in enumerate(self._choices):
                    yield Button(label, id=f"choice-{i}", variant=variant)

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "choice-cancel":
            self.dismiss(None)
        else:
            self.dismiss(self._choices[int(bid.split("-")[1])][1])

    def action_cancel(self) -> None:
        self.dismiss(None)
