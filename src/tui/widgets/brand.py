"""Header brand block: the BAT logo with an animated sparkline and PnL ticker underneath (or a one-line version)."""
from __future__ import annotations

import numpy as np
from rich.text import Text
from textual.widgets import Static

from src.tui import charts as C
from src.tui.services.ticker import TickerFrame

_SOURCE_STYLE = {"LIVE": C.UP, "REPLAY": C.AMBER, "DEMO": C.MUTED}


def spark_text(values, width: int, direction: int) -> Text:
    """Block sparkline whose colour fades up to the newest bar, which is green/red by the last tick."""
    v = np.asarray(values, dtype=float)[-width:]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Text("")
    lo, hi = float(v.min()), float(v.max())
    idx = np.zeros(v.size, dtype=int) if hi - lo < 1e-9 else np.round((v - lo) / (hi - lo) * 7).astype(int)
    t = Text()
    for i, k in enumerate(idx):
        last = i == v.size - 1
        color = (C.UP if direction >= 0 else C.DOWN) if last else C.lerp_color("#4a2f08", C.ORANGE, i / max(v.size - 1, 1))
        t.append("▁▂▃▄▅▆▇█"[int(k)], style=color)
    return t


def _pnl(frame: TickerFrame) -> Text:
    t = Text()
    t.append(C.fmt_money(frame.pnl, signed=True), style=f"bold {C.pnl_color(frame.pnl)}")
    arrow, color = ("▲", C.UP) if frame.direction >= 0 else ("▼", C.DOWN)
    t.append(f" {arrow} ", style=color)
    t.append(C.fmt_pct(frame.pct), style=color)
    return t


def render_full(frame: TickerFrame, paused: bool = False) -> Text:
    t = C.ascii_logo()
    t.append_text(spark_text(frame.values, len(C.LOGO_ROWS[0]), frame.direction))
    t.append("\n")
    t.append_text(_pnl(frame))
    t.append("\n")
    tag = f"{frame.source} · {frame.label}" + ("  (paused)" if paused else "")
    t.append(tag, style=_SOURCE_STYLE.get(frame.source, C.MUTED))
    return t


def render_compact(frame: TickerFrame, paused: bool = False) -> Text:
    t = Text()
    t.append("BAT ", style=f"bold {C.ORANGE}")
    t.append_text(spark_text(frame.values, 12, frame.direction))
    t.append("  ")
    t.append_text(_pnl(frame))
    t.append(f"  {frame.source}" + (" (paused)" if paused else ""), style=_SOURCE_STYLE.get(frame.source, C.MUTED))
    return t


class BrandBox(Static):
    def __init__(self, compact: bool = False, **kw) -> None:
        super().__init__("", **kw)
        self.compact = compact

    def show(self, frame: TickerFrame, paused: bool = False) -> None:
        self.update((render_compact if self.compact else render_full)(frame, paused))
