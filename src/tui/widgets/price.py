"""Interactive candlestick chart: pan, zoom, hover read-out."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget

from src.tui import charts as C
from src.tui.strategies import split_indicators


class PriceChart(Widget, can_focus=True):
    DEFAULT_CSS = """
    PriceChart { height: 1fr; min-height: 12; }
    PriceChart:focus { background: $surface-lighten-1 5%; }
    """
    BINDINGS = [
        Binding("left,h", "pan(-1)", "Pan ◀", show=False),
        Binding("right,l", "pan(1)", "Pan ▶", show=False),
        Binding("plus,equals_sign,up", "zoom(0.7)", "Zoom +", show=False),
        Binding("minus,down", "zoom(1.4)", "Zoom −", show=False),
        Binding("home", "jump('start')", "Start", show=False),
        Binding("end", "jump('end')", "End", show=False),
        Binding("0", "jump('all')", "All", show=False),
    ]

    class ViewChanged(Message):
        def __init__(self, start: int, end: int, total: int):
            super().__init__()
            self.start, self.end, self.total = start, end, total

    def __init__(self, **kw):
        super().__init__(**kw)
        self.df: Optional[pd.DataFrame] = None
        self.markers: List[Tuple[int, str]] = []
        self.overlay_cols: List[str] = []
        self.pane_cols: List[str] = []
        self.bands: Sequence[float] = ()
        self.title = ""
        self.view_start = 0
        self.view_len = 0
        self.follow = False
        self._hover: Optional[int] = None
        self._lay: Optional[C.Layout] = None
        self._gutter = 0
        self.y_fmt = C.fmt_num

    # ----- data -------------------------------------------------------------
    def set_data(self, df: pd.DataFrame, *, markers: Optional[List[Tuple[int, str]]] = None,
                 indicators: Sequence[str] = (), title: str = "", follow: bool = False,
                 view_len: Optional[int] = None, reset_view: bool = True, y_fmt=None) -> None:
        self.df = df.reset_index(drop=True) if df is not None else None
        self.markers = markers or []
        self.title = title
        if y_fmt:
            self.y_fmt = y_fmt
        if self.df is not None and len(self.df):
            parts = split_indicators(self.df, indicators)
            self.overlay_cols, self.pane_cols = parts["overlay"], parts["pane"]
            self.bands = (30, 70) if any(c.lower().startswith("rsi") for c in self.pane_cols) else ()
            n = len(self.df)
            if reset_view or self.view_len == 0:
                self.view_len = min(view_len or n, n)
                self.view_start = max(n - self.view_len, 0) if follow else 0
                self.follow = follow
            else:
                self.view_len = min(self.view_len, n)
                if self.follow:
                    self.view_start = max(n - self.view_len, 0)
                self.view_start = int(np.clip(self.view_start, 0, max(n - self.view_len, 0)))
        self.refresh()

    # ----- view control ---------------------------------------------------------
    def _clamp(self) -> None:
        n = len(self.df) if self.df is not None else 0
        self.view_len = int(np.clip(self.view_len, min(20, n), max(n, 1)))
        self.view_start = int(np.clip(self.view_start, 0, max(n - self.view_len, 0)))
        if n:
            self.post_message(self.ViewChanged(self.view_start, self.view_start + self.view_len, n))
        self.refresh()

    def action_pan(self, direction: int) -> None:
        self.follow = False
        self.view_start += int(direction * max(self.view_len // 5, 1))
        self._clamp()

    def action_zoom(self, factor: float) -> None:
        if self.df is None:
            return
        mid = self.view_start + self.view_len // 2
        self.view_len = int(round(self.view_len * float(factor)))
        self.view_start = mid - self.view_len // 2
        self._clamp()

    def action_jump(self, where: str) -> None:
        n = len(self.df) if self.df is not None else 0
        if where == "start":
            self.view_start, self.follow = 0, False
        elif where == "end":
            self.view_start, self.follow = max(n - self.view_len, 0), True
        else:
            self.view_start, self.view_len, self.follow = 0, n, False
        self._clamp()

    def focus_bar(self, bar: int, span: int = 80) -> None:
        if self.df is None:
            return
        self.follow = False
        self.view_len = int(np.clip(span, 20, max(len(self.df), 20)))
        self.view_start = bar - self.view_len // 2
        self._clamp()

    # ----- mouse ----------------------------------------------------------------
    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._lay is None or self.df is None:
            return
        local = self._lay.bar_at_cell(event.x - self._gutter)
        new = None if local is None else self.view_start + local
        if new != self._hover:
            self._hover = new
            self.refresh()

    def on_leave(self, event: events.Leave) -> None:
        if self._hover is not None:
            self._hover = None
            self.refresh()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.stop()
        self.action_zoom(0.8)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.stop()
        self.action_zoom(1.25)

    # ----- render ---------------------------------------------------------------
    def render(self) -> Text:
        if self.df is None or len(self.df) == 0:
            return Text("No price data yet.", style=C.MUTED)
        w, h = self.size.width, self.size.height
        if w < 24 or h < 8:
            return Text("")
        a, b = self.view_start, self.view_start + self.view_len
        view = self.df.iloc[a:b].reset_index(drop=True)
        overlays = {c: self.df[c].to_numpy(float)[a:b] for c in self.overlay_cols}
        markers = [(i - a, k) for i, k in self.markers if a <= i < b]
        hover = None if self._hover is None else self._hover - a
        title = f"{self.title}  " if self.title else ""
        title += f"[{a + 1}–{b} / {len(self.df)}]"
        try:
            text, self._lay, self._gutter = C.price_stack(view, w, h, markers=markers, overlays=overlays,
                                                          panes=self.pane_cols, bands=self.bands, hover=hover,
                                                          title=title, y_fmt=self.y_fmt)
        except Exception as e:  # noqa: BLE001
            return Text(f"chart error: {type(e).__name__}: {e}", style=C.DOWN)
        return text
