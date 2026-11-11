"""Small building blocks shared by every page."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, RichLog, Select, Static, Switch, TabbedContent

from src.tui import charts as C
from src.tui.strategies import ParamSpec, StrategySpec

LEVEL_STYLE = {"info": "#9b9b9b", "detail": "#6b6b6b", "signal": "#ffd23f", "trade": "#00e676",
               "error": "#ff4d4d", "warn": "#ffa028", "ok": "#00e676", "loss": "#ff4d4d"}


def select_value(sel: Select, default: Any = None) -> Any:
    """Current value of a Select, or ``default`` when blank."""
    v = sel.value
    return default if v is Select.NULL or v is None else v


def pnl_text(v: float, fmt: Callable[[float], str] = lambda x: C.fmt_money(x, signed=True)) -> Text:
    return Text(fmt(v), style=C.pnl_color(v), justify="right")


class Section(Static):
    """Dimmed, uppercase section header used inside forms."""

    def __init__(self, title: str, **kw):
        super().__init__(title.upper(), classes="section", **kw)


class Plot(Widget):
    """Draws whatever ``builder(width, height) -> Text`` returns, re-rendering on resize."""

    DEFAULT_CSS = "Plot { height: 12; }"

    def __init__(self, builder: Optional[Callable[[int, int], Text]] = None, **kw):
        super().__init__(**kw)
        self.builder = builder

    def set_builder(self, builder: Optional[Callable[[int, int], Text]]) -> None:
        self.builder = builder
        self.refresh()

    def render(self) -> Text:
        w, h = self.size.width, self.size.height
        if not self.builder or w < 8 or h < 2:
            return Text("")
        try:
            return self.builder(w, h)
        except Exception as e:  # noqa: BLE001 - a bad chart must never crash the app
            return Text(f"chart error: {type(e).__name__}: {e}", style=C.DOWN)


class Card(Static):
    """Label + big value + subtitle. Colour follows the value's sign when ``tone`` is set."""

    DEFAULT_CSS = """
    Card { height: 4; padding: 0 1; border: round $panel-lighten-1; background: $surface; }
    Card .lbl { color: $text-muted; }
    """

    def __init__(self, label: str, value: str = "–", sub: str = "", **kw):
        super().__init__(**kw)
        self._label, self._value, self._sub, self._color = label, value, sub, None

    def set(self, value: str, sub: str = "", color: Optional[str] = None) -> None:
        self._value, self._sub, self._color = value, sub, color
        self.refresh()

    def render(self) -> Text:
        t = Text(no_wrap=True, overflow="ellipsis")
        t.append(self._label + "\n", style="dim")
        t.append(self._value, style=f"bold {self._color}" if self._color else "bold")
        if self._sub:
            t.append("\n" + self._sub, style="dim")
        return t


class CardGrid(Grid):
    """A grid of ``Card`` widgets keyed by name."""

    DEFAULT_CSS = """
    CardGrid { grid-size: 4; grid-gutter: 0 1; height: auto; padding: 0 0 1 0; }
    """

    def __init__(self, labels: Dict[str, str], columns: int = 4, **kw):
        super().__init__(**kw)
        self._labels = labels
        self._columns = columns          # kept for documentation; sizing is done in CSS (responsive)

    def compose(self) -> ComposeResult:
        for key, label in self._labels.items():
            yield Card(label, id=f"card-{key}")

    def set(self, key: str, value: str, sub: str = "", color: Optional[str] = None) -> None:
        try:
            self.query_one(f"#card-{key}", Card).set(value, sub, color)
        except Exception:
            pass

    def reset(self) -> None:
        for c in self.query(Card):
            c.set("–", "")


class EventLog(RichLog):
    """Timestamped, level-coloured log with a minimum-level filter."""

    ORDER = {"detail": 0, "info": 1, "signal": 2, "trade": 3, "warn": 3, "ok": 3, "loss": 3, "error": 4}

    def __init__(self, **kw):
        kw.setdefault("highlight", False)
        kw.setdefault("markup", False)
        kw.setdefault("wrap", True)
        kw.setdefault("max_lines", 3000)
        kw.setdefault("min_width", 20)        # RichLog defaults to 78 columns -> needless horizontal scrolling
        super().__init__(**kw)
        self.min_level = "info"
        self._history: List[Tuple[datetime, str, str, str]] = []

    def add(self, level: str, text: str, source: str = "", ts: Optional[datetime] = None) -> None:
        ts = ts or datetime.now()
        self._history.append((ts, level, source, text))
        self._history = self._history[-3000:]
        if self.ORDER.get(level, 1) >= self.ORDER[self.min_level]:
            self.write(self._line(ts, level, source, text))

    @staticmethod
    def _line(ts, level, source, text) -> Text:
        t = Text(no_wrap=False)
        t.append(f"{ts:%H:%M:%S} ", style="dim")
        if source:
            t.append(f"{source:<9}", style="#ffa028")
        t.append(text, style=LEVEL_STYLE.get(level, "default"))
        return t

    def set_level(self, level: str) -> None:
        self.min_level = level
        self.clear()
        for ts, lv, src, text in self._history:
            if self.ORDER.get(lv, 1) >= self.ORDER[level]:
                self.write(self._line(ts, lv, src, text))

    def wipe(self) -> None:
        self._history.clear()
        self.clear()


class LabeledInput(Vertical):
    """Compact label above an input."""

    DEFAULT_CSS = """
    LabeledInput { height: auto; margin-bottom: 1; }
    LabeledInput > Label { color: $text-muted; }
    """

    def __init__(self, label: str, value: str = "", *, placeholder: str = "", input_id: str,
                 password: bool = False, **kw):
        super().__init__(**kw)
        self._label, self._value, self._ph, self._iid, self._pw = label, value, placeholder, input_id, password

    def compose(self) -> ComposeResult:
        yield Label(self._label)
        yield Input(self._value, placeholder=self._ph, id=self._iid, password=self._pw, compact=True)


class LabeledSelect(Vertical):
    DEFAULT_CSS = """
    LabeledSelect { height: auto; margin-bottom: 1; }
    LabeledSelect > Label { color: $text-muted; }
    """

    def __init__(self, label: str, options: Sequence[Tuple[str, Any]], *, select_id: str, value: Any = None, **kw):
        super().__init__(**kw)
        self._label, self._options, self._sid, self._value = label, list(options), select_id, value

    def compose(self) -> ComposeResult:
        yield Label(self._label)
        kwargs = {}
        if self._value is not None:
            kwargs["value"] = self._value
        yield Select(self._options, id=self._sid, allow_blank=False, compact=True, **kwargs)


class ParamForm(Vertical):
    """Builds one input per strategy parameter and reads/validates them back."""

    DEFAULT_CSS = """
    ParamForm { height: auto; }
    ParamForm .param-row { height: auto; margin-bottom: 1; }
    ParamForm .param-row Label { color: $text-muted; }
    """

    class Changed(Message):
        pass

    def __init__(self, spec: Optional[StrategySpec] = None, prefix: str = "pf", **kw):
        super().__init__(**kw)
        self.spec, self.prefix = spec, prefix

    def _wid(self, name: str) -> str:
        return f"{self.prefix}-{name}"

    def compose(self) -> ComposeResult:
        yield from self._rows()

    def _rows(self) -> Iterable[Widget]:
        # plain constructors (not ``with`` blocks): this is also called when remounting outside compose()
        if not self.spec or not self.spec.params:
            yield Static("This strategy has no parameters.", classes="muted")
            return
        for p in self.spec.params:
            label = Label(p.label + (f"  [dim]{p.help}[/]" if p.help else ""))
            if p.kind == "choice":
                field: Widget = Select([(str(c), str(c)) for c in p.choices], value=str(p.default), allow_blank=False,
                                       id=self._wid(p.name), compact=True)
            else:
                field = Input(str(p.default), id=self._wid(p.name), compact=True,
                              type="integer" if p.kind == "int" else "number")
            yield Vertical(label, field, classes="param-row")

    async def set_spec(self, spec: StrategySpec, values: Optional[Dict[str, Any]] = None) -> None:
        self.spec = spec
        await self.remove_children()
        await self.mount_all(list(self._rows()))
        if values:
            self.set_values(values)

    def raw_values(self) -> Dict[str, str]:
        out = {}
        for p in (self.spec.params if self.spec else ()):
            w = self.query_one(f"#{self._wid(p.name)}")
            out[p.name] = str(w.value if not isinstance(w, Select) else select_value(w, p.default))
        return out

    def set_values(self, values: Dict[str, Any]) -> None:
        for p in (self.spec.params if self.spec else ()):
            if p.name in values:
                try:
                    w = self.query_one(f"#{self._wid(p.name)}")
                    w.value = str(values[p.name])
                except Exception:
                    pass

    @on(Input.Changed)
    @on(Select.Changed)
    def _changed(self, event) -> None:
        event.stop()
        self.post_message(self.Changed())


def fmt_td(td) -> str:
    """Compact duration (Timedelta) for tables."""
    try:
        s = int(td.total_seconds())
    except Exception:
        return "–"
    if s >= 86400:
        return f"{s // 86400}d {s % 86400 // 3600}h"
    if s >= 3600:
        return f"{s // 3600}h {s % 3600 // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s}s"


class GuardedTabs(TabbedContent):
    """``TabbedContent`` that can briefly ignore *focus-driven* tab changes.

    By default a tab re-activates itself whenever a widget inside it receives focus. After a click handler switches
    tabs, two such focus events arrive late (the click focusing the button, and Textual moving focus off the
    now-hidden button) and flip the tab straight back. ``guard()`` makes the tab bar ignore them for a moment.
    Clicking a tab header is unaffected.
    """

    _guard_until = 0.0

    def guard(self, seconds: float = 1.0) -> None:
        self._guard_until = time.monotonic() + seconds

    def _on_tab_pane_focused(self, event) -> None:
        if time.monotonic() < self._guard_until:
            event.prevent_default()           # skip TabbedContent's own handler
            event.stop()


def switch_tab(owner: Widget, tabs_selector: str, tab_id: str, focus: Optional[str] = None) -> None:
    """Activate a tab from inside an event handler and keep it active (see :class:`GuardedTabs`)."""
    tabs = owner.query_one(tabs_selector, TabbedContent)
    if isinstance(tabs, GuardedTabs):
        tabs.guard()
    tabs.active = tab_id
    if focus:
        owner.call_after_refresh(lambda: owner.query_one(focus).focus())
    else:
        owner.screen.set_focus(None)          # do not leave focus on a widget inside the tab we just left
