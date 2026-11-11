"""Terminal chart rendering.

Everything here is a pure function from data to a ``rich.text.Text`` so it can
be unit-tested without a terminal and drawn by any widget. Charts use

* braille dots (2x4 sub-cell resolution) for lines, areas and bars,
* half-block glyphs for candlestick bodies,
* background-coloured cells for heat maps.

Colours are fixed hex values that read well on both dark and light themes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rich.text import Text

# Bloomberg-style palette: pure black ground, amber for labels and the primary series, green/red for up/down.
UP = "#00e676"
DOWN = "#ff4d4d"
ORANGE = "#ffa028"          # primary: headers, labels, the strategy line
AMBER = "#ffd23f"           # highlights, warnings, signals
BLUE = "#3d9bff"
PURPLE = "#b48cff"
CYAN = "#19d3da"
PINK = "#f778ba"
MUTED = "#8f8f8f"
GRID = "#3a3a3a"
SERIES_COLORS = [ORANGE, CYAN, AMBER, BLUE, PINK, UP]

_BITS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))
_SPARKS = "▁▂▃▄▅▆▇█"


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_num(v: float, signed: bool = False) -> str:
    """Compact, magnitude-aware number format used for axis labels."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    a = abs(v)
    sign = "+" if signed and v > 0 else ""
    if a >= 1e9:
        s = f"{v / 1e9:.2f}B"
    elif a >= 1e6:
        s = f"{v / 1e6:.2f}M"
    elif a >= 1e4:
        s = f"{v / 1e3:.1f}K"
    elif a >= 1e3:
        s = f"{v:,.0f}"
    elif a >= 100:
        s = f"{v:.1f}"
    elif a >= 1:
        s = f"{v:.2f}"
    elif a >= 0.01:
        s = f"{v:.4f}"
    elif a == 0:
        s = "0"
    else:
        s = f"{v:.2e}"
    return sign + s


def fmt_money(v: float, signed: bool = False, decimals: Optional[int] = None) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    if decimals is None:
        decimals = 2 if abs(v) >= 1 else 4
    sign = "+" if signed and v > 0 else ""
    return f"{sign}{'-' if v < 0 else ''}${abs(v):,.{decimals}f}"


def fmt_pct(v: float, signed: bool = True, decimals: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    return f"{'+' if signed and v > 0 else ''}{v:.{decimals}f}%"


def pnl_color(v: float) -> str:
    if v is None or not np.isfinite(v) or v == 0:
        return MUTED
    return UP if v > 0 else DOWN


def dim(color: str, factor: float = 0.45) -> str:
    """Blend ``color`` toward black; used for area fills under a line."""
    c = color.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(int(r * factor), int(g * factor), int(b * factor))


def lerp_color(c0: str, c1: str, t: float) -> str:
    t = float(np.clip(t, 0, 1))
    a = [int(c0.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(c1.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#{:02x}{:02x}{:02x}".format(*(int(round(x + (y - x) * t)) for x, y in zip(a, b)))


# --------------------------------------------------------------------------- #
# Sparklines and gauges
# --------------------------------------------------------------------------- #
def sparkline(values: Sequence[float], width: Optional[int] = None) -> str:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ""
    if width and v.size > width:
        idx = np.linspace(0, v.size - 1, width).astype(int)
        v = v[idx]
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return _SPARKS[3] * len(v)
    scaled = (v - lo) / (hi - lo) * (len(_SPARKS) - 1)
    return "".join(_SPARKS[int(round(s))] for s in scaled)


def colored_sparkline(values: Sequence[float], width: Optional[int] = None) -> Text:
    s = sparkline(values, width)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    color = UP if v.size > 1 and v[-1] >= v[0] else DOWN
    return Text(s, style=color)


def gauge(value: float, lo: float, hi: float, width: int, label: str = "") -> Text:
    """Diverging gauge centred on the middle of [lo, hi] (e.g. a probability)."""
    width = max(width, 5)
    pos = int(round(np.clip((value - lo) / max(hi - lo, 1e-12), 0, 1) * (width - 1)))
    t = Text()
    for i in range(width):
        if i == pos:
            t.append("◆", style="bold white")
        else:
            t.append("━", style=lerp_color(DOWN, UP, i / (width - 1)))
    if label:
        t.append(" " + label, style=MUTED)
    return t


# --------------------------------------------------------------------------- #
# Braille canvas
# --------------------------------------------------------------------------- #
class BrailleCanvas:
    """A ``cols`` x ``rows`` cell canvas addressed in dots (2 wide, 4 tall per cell)."""

    def __init__(self, cols: int, rows: int):
        self.cols, self.rows = max(cols, 1), max(rows, 1)
        self.w, self.h = self.cols * 2, self.rows * 4
        self.bits = np.zeros((self.rows, self.cols), dtype=np.uint16)
        self.cell_color = np.full((self.rows, self.cols), -1, dtype=np.int16)
        self.palette: List[str] = []

    def color_id(self, color: str) -> int:
        if color not in self.palette:
            self.palette.append(color)
        return self.palette.index(color)

    def set(self, x: int, y: int, cid: int) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            r, c = y // 4, x // 2
            self.bits[r, c] |= _BITS[x % 2][y % 4]
            self.cell_color[r, c] = cid

    def line(self, x0: int, y0: int, x1: int, y1: int, cid: int) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        for _ in range(dx - dy + 2 + 8):  # hard cap protects against bad input
            self.set(x0, y0, cid)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def vfill(self, x: int, y0: int, y1: int, cid: int) -> None:
        for y in range(min(y0, y1), max(y0, y1) + 1):
            self.set(x, y, cid)

    def hline(self, y: int, cid: int, dash: int = 3) -> None:
        for x in range(0, self.w, dash * 2):
            for k in range(dash):
                self.set(x + k, y, cid)

    def char_at(self, r: int, c: int) -> str:
        return chr(0x2800 + int(self.bits[r, c]))


def _line_to_text(cells: Sequence[Tuple[str, Optional[str]]]) -> Text:
    """Run-length group (char, style) cells into one Text row."""
    t = Text(no_wrap=True, overflow="crop")
    buf, cur = [], None
    for ch, st in cells:
        if st != cur and buf:
            t.append("".join(buf), style=cur)
            buf = []
        cur = st
        buf.append(ch)
    if buf:
        t.append("".join(buf), style=cur)
    return t


def _canvas_rows(cv: BrailleCanvas) -> List[List[Tuple[str, Optional[str]]]]:
    rows = []
    for r in range(cv.rows):
        row = []
        for c in range(cv.cols):
            if cv.bits[r, c]:
                row.append((cv.char_at(r, c), cv.palette[cv.cell_color[r, c]]))
            else:
                row.append((" ", None))
        rows.append(row)
    return rows


def _nice_range(lo: float, hi: float, pad: float = 0.04) -> Tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi - lo < 1e-12:
        d = abs(hi) * 0.01 or 1.0
        return lo - d, hi + d
    p = (hi - lo) * pad
    return lo - p, hi + p


def _resample(y: np.ndarray, n: int) -> np.ndarray:
    """Resample ``y`` to ``n`` points (NaN-aware linear interpolation)."""
    if len(y) == n:
        return y
    xs = np.linspace(0, len(y) - 1, n)
    good = np.isfinite(y)
    if good.sum() < 2:
        return np.full(n, np.nan)
    out = np.interp(xs, np.arange(len(y))[good], y[good])
    # keep gaps (NaN regions) as NaN
    nearest = np.clip(np.round(xs).astype(int), 0, len(y) - 1)
    out[~good[nearest]] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Axes frame shared by braille charts
# --------------------------------------------------------------------------- #
def _frame(body: List[List[Tuple[str, Optional[str]]]], lo: float, hi: float, y_fmt: Callable,
           x_labels: Optional[Sequence[str]], legend: Optional[Text], gutter: int,
           ticks: Optional[Dict[int, float]] = None) -> List[Text]:
    rows = len(body)
    out: List[Text] = []
    if legend is not None:
        lg = Text(" " * gutter)
        lg.append_text(legend)
        out.append(lg)
    if ticks is None:
        ticks = {0: hi, rows // 2: (lo + hi) / 2, rows - 1: lo} if rows > 2 else {0: hi, rows - 1: lo}
    for r, cells in enumerate(body):
        label = y_fmt(ticks[r]) if r in ticks else ""
        t = Text(label.rjust(gutter - 2), style=MUTED, no_wrap=True)
        t.append(" ┤" if r in ticks else " │", style=GRID)
        t.append_text(_line_to_text(cells))
        out.append(t)
    if x_labels:
        width = len(body[0]) if body else 0
        axis = Text(" " * (gutter - 1) + "└" + "─" * width, style=GRID, no_wrap=True)
        out.append(axis)
        out.append(_x_label_row(x_labels, width, gutter))
    return out


def _x_label_row(labels: Sequence[str], width: int, gutter: int) -> Text:
    row = [" "] * width
    n = len(labels)
    for i, lab in enumerate(labels):
        start = 0 if n == 1 else int(round(i * (width - len(lab)) / (n - 1)))
        start = max(0, min(start, width - len(lab)))
        for k, ch in enumerate(lab):
            if start + k < width:
                row[start + k] = ch
    return Text(" " * gutter + "".join(row), style=MUTED, no_wrap=True)


def time_labels(index: Sequence, n: int = 4) -> List[str]:
    """Evenly spaced, compact time labels for an x axis."""
    idx = pd.Index(index)
    if len(idx) == 0:
        return []
    try:
        ts = pd.to_datetime(idx)
    except Exception:
        return [str(idx[int(round(i))]) for i in np.linspace(0, len(idx) - 1, n)]
    span = ts[-1] - ts[0]
    fmt = "%H:%M" if span <= pd.Timedelta(days=2) else ("%m-%d" if span <= pd.Timedelta(days=400) else "%Y-%m")
    pick = np.unique(np.linspace(0, len(ts) - 1, min(n, len(ts))).round().astype(int))
    return [ts[i].strftime(fmt) for i in pick]


# --------------------------------------------------------------------------- #
# Line / area chart
# --------------------------------------------------------------------------- #
def line_chart(series: Dict[str, Sequence[float]], width: int, height: int, *,
               colors: Optional[Dict[str, str]] = None, x_labels: Optional[Sequence[str]] = None,
               y_fmt: Callable[[float], str] = fmt_num, hline: Optional[float] = None,
               fill_to: Optional[float] = None, legend: bool = True,
               y_range: Optional[Tuple[float, float]] = None, title: Optional[str] = None) -> Text:
    """Multi-series braille line chart. ``fill_to`` shades under the first series."""
    width, height = max(int(width), 12), max(int(height), 4)
    arrays = {k: np.asarray(v, dtype=float) for k, v in series.items() if len(v)}
    if not arrays:
        return Text("no data", style=MUTED)
    finite = np.concatenate([a[np.isfinite(a)] for a in arrays.values()] or [np.array([0.0])])
    if finite.size == 0:
        return Text("no data", style=MUTED)
    lo, hi = y_range if y_range else _nice_range(float(finite.min()), float(finite.max()))
    if hline is not None and not y_range:
        lo, hi = min(lo, hline), max(hi, hline)
    gutter = max(len(y_fmt(lo)), len(y_fmt(hi)), len(y_fmt((lo + hi) / 2))) + 3
    x_rows = 2 if x_labels else 0
    leg_rows = 1 if (legend or title) else 0
    plot_cols = max(width - gutter, 4)
    plot_rows = max(height - x_rows - leg_rows, 2)
    cv = BrailleCanvas(plot_cols, plot_rows)

    def ypix(v: float) -> int:
        return int(round((1 - (v - lo) / (hi - lo)) * (cv.h - 1)))

    if hline is not None:
        cv.hline(int(np.clip(ypix(hline), 0, cv.h - 1)), cv.color_id(GRID))

    palette = dict(colors or {})
    names = list(arrays)
    for i, name in enumerate(names):
        palette.setdefault(name, SERIES_COLORS[i % len(SERIES_COLORS)])

    for i, name in enumerate(names):
        y = _resample(arrays[name], cv.w) if len(arrays[name]) > cv.w else arrays[name]
        xs = np.linspace(0, cv.w - 1, len(y)) if len(y) > 1 else np.array([cv.w // 2])
        cid = cv.color_id(palette[name])
        if fill_to is not None and i == 0:
            fid = cv.color_id(dim(palette[name]))
            base = int(np.clip(ypix(fill_to), 0, cv.h - 1))
            dense = _resample(arrays[name], cv.w) if len(arrays[name]) != cv.w else arrays[name]
            for x, v in enumerate(dense):
                if np.isfinite(v):
                    cv.vfill(x, int(np.clip(ypix(v), 0, cv.h - 1)), base, fid)
        prev = None
        for x, v in zip(xs, y):
            if not np.isfinite(v):
                prev = None
                continue
            pt = (int(round(x)), int(np.clip(ypix(v), 0, cv.h - 1)))
            if prev is None:
                cv.set(*pt, cid)
            else:
                cv.line(prev[0], prev[1], pt[0], pt[1], cid)
            prev = pt

    leg = None
    if legend or title:
        leg = Text(no_wrap=True)
        if title:
            leg.append(title + "  ", style="bold")
        if legend and (len(names) > 1 or not title):
            for name in names:
                last = arrays[name][np.isfinite(arrays[name])]
                leg.append("━ ", style=palette[name])
                leg.append(f"{name} ", style=MUTED)
                if last.size:
                    leg.append(f"{y_fmt(float(last[-1]))}  ", style=palette[name])
    lines = _frame(_canvas_rows(cv), lo, hi, y_fmt, x_labels, leg, gutter)
    return Text("\n").join(lines)


# --------------------------------------------------------------------------- #
# Binning: align candles, volume and indicator panes on the same columns
# --------------------------------------------------------------------------- #
@dataclass
class Layout:
    """Maps ``n`` bars onto ``cols`` terminal cells."""
    n: int
    cols: int
    step: int          # cells per candle slot (1 or 2)
    slots: int         # number of candle slots on screen
    edges: np.ndarray  # bar index boundaries for each slot, len slots+1

    @classmethod
    def build(cls, n: int, cols: int, prefer_gap: bool = True) -> "Layout":
        cols = max(cols, 4)
        if prefer_gap and n * 2 <= cols:
            step, slots = 2, n
        else:
            step, slots = 1, min(n, cols)
        edges = np.linspace(0, n, slots + 1).round().astype(int)
        return cls(n, cols, step, slots, edges)

    def slot_of_bar(self, i: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges, i, side="right") - 1, 0, self.slots - 1)

    def left_pad(self) -> int:
        return max((self.cols - self.slots * self.step) // 2, 0) if self.step == 2 else 0

    def cell_of_slot(self, s: np.ndarray) -> np.ndarray:
        return self.left_pad() + np.asarray(s) * self.step

    def bar_at_cell(self, col: int) -> Optional[int]:
        """Last bar of the slot drawn at terminal column ``col`` (None when outside)."""
        rel = col - self.left_pad()
        if rel < 0:
            return None
        slot = rel // self.step
        if slot >= self.slots:
            return None
        return int(max(self.edges[slot + 1] - 1, self.edges[slot]))


def _aggregate_ohlcv(df: pd.DataFrame, lay: Layout) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    v = df["Volume"].to_numpy(float) if "Volume" in df else np.zeros(len(df))
    rows = []
    for a, b in zip(lay.edges[:-1], lay.edges[1:]):
        b = max(b, a + 1)
        rows.append((o[a], h[a:b].max(), l[a:b].min(), c[b - 1], v[a:b].sum()))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"])


# --------------------------------------------------------------------------- #
# Candlesticks
# --------------------------------------------------------------------------- #
MARKER_GLYPH = {"buy": ("▲", UP), "sell": ("▼", DOWN), "exit": ("◆", AMBER)}


def candlestick_chart(df: pd.DataFrame, width: int, height: int, *,
                      markers: Optional[Iterable[Tuple[int, str]]] = None,
                      overlays: Optional[Dict[str, Sequence[float]]] = None,
                      x_labels: Optional[Sequence[str]] = None, y_fmt: Callable[[float], str] = fmt_num,
                      last_price: bool = True, legend: Optional[Text] = None,
                      layout: Optional[Layout] = None, gutter: Optional[int] = None,
                      y_range: Optional[Tuple[float, float]] = None) -> Text:
    """Half-block candlesticks with braille indicator overlays and trade markers.

    ``markers`` is an iterable of ``(bar_index, kind)`` with kind in buy/sell/exit.
    ``overlays`` maps names to full-length series drawn in price space.
    """
    if df is None or len(df) == 0:
        return Text("no data", style=MUTED)
    width, height = max(int(width), 16), max(int(height), 6)
    overlays = overlays or {}
    lo = float(df["Low"].min())
    hi = float(df["High"].max())
    for s in overlays.values():
        a = np.asarray(s, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    lo, hi = y_range if y_range else _nice_range(lo, hi)
    if gutter is None:
        gutter = max(len(y_fmt(lo)), len(y_fmt(hi))) + 3
    x_rows = 2 if x_labels else 0
    leg_rows = 1 if legend is not None else 0
    cols = max(width - gutter, 8)
    rows = max(height - x_rows - leg_rows, 4)

    lay = layout or Layout.build(len(df), cols)
    agg = _aggregate_ohlcv(df, lay)
    half = rows * 2  # half-cell levels

    def lvl(v: float) -> float:  # 0 at top
        return (1 - (v - lo) / (hi - lo)) * (half - 1)

    grid = [[(" ", None)] * cols for _ in range(rows)]
    for s in range(lay.slots):
        o, h, l, c = (float(agg.at[s, k]) for k in ("Open", "High", "Low", "Close"))
        col = int(lay.cell_of_slot(s))
        if col >= cols:
            continue
        color = UP if c >= o else DOWN
        top_b, bot_b = lvl(max(o, c)), lvl(min(o, c))
        w_top, w_bot = lvl(h), lvl(l)
        b0, b1 = int(round(top_b)), int(round(bot_b))
        for r in range(rows):
            t_half, b_half = 2 * r, 2 * r + 1
            has_t = b0 <= t_half <= b1
            has_b = b0 <= b_half <= b1
            wick = int(round(w_top)) <= 2 * r + 1 and int(round(w_bot)) >= 2 * r
            if has_t and has_b:
                ch = "█"
            elif has_t:
                ch = "▀"
            elif has_b:
                ch = "▄"
            elif wick:
                ch = "│"
            else:
                continue
            grid[r][col] = (ch, color)

    # indicator overlays (braille) only where no candle glyph is present
    if overlays:
        cv = BrailleCanvas(cols, rows)
        for i, (name, s) in enumerate(overlays.items()):
            a = np.asarray(s, dtype=float)
            if len(a) != len(df):
                continue
            cid = cv.color_id(SERIES_COLORS[(i + 1) % len(SERIES_COLORS)] if i else AMBER)
            prev = None
            for slot in range(lay.slots):
                seg = a[lay.edges[slot]:max(lay.edges[slot + 1], lay.edges[slot] + 1)]
                seg = seg[np.isfinite(seg)]
                if not seg.size:
                    prev = None
                    continue
                x = int(lay.cell_of_slot(slot)) * 2 + (1 if lay.step == 2 else 0)
                y = int(np.clip((1 - (seg[-1] - lo) / (hi - lo)) * (cv.h - 1), 0, cv.h - 1))
                if prev is None:
                    cv.set(x, y, cid)
                else:
                    cv.line(prev[0], prev[1], x, y, cid)
                prev = (x, y)
        for r in range(rows):
            for c in range(cols):
                if cv.bits[r, c] and grid[r][c][0] == " ":
                    grid[r][c] = (cv.char_at(r, c), cv.palette[cv.cell_color[r, c]])

    # trade markers
    if markers:
        for bar, kind in markers:
            if not 0 <= bar < len(df) or kind not in MARKER_GLYPH:
                continue
            slot = int(lay.slot_of_bar(np.array([bar]))[0])
            col = int(lay.cell_of_slot(slot))
            if col >= cols:
                continue
            glyph, color = MARKER_GLYPH[kind]
            hi_r = int(np.clip(round(lvl(float(agg.at[slot, "High"])) / 2), 0, rows - 1))
            lo_r = int(np.clip(round(lvl(float(agg.at[slot, "Low"])) / 2), 0, rows - 1))
            r = min(lo_r + 1, rows - 1) if kind == "buy" else max(hi_r - 1, 0)
            if kind == "exit":
                r = min(lo_r + 1, rows - 1)
            grid[r][col] = (glyph, "bold " + color)

    if last_price:
        lastv = float(agg["Close"].iloc[-1])
        r = int(np.clip(round(lvl(lastv) / 2), 0, rows - 1))
        for c in range(cols - 1, -1, -1):
            if grid[r][c][0] == " ":
                grid[r][c] = ("·", GRID)

    lines = _frame(grid, lo, hi, y_fmt, x_labels, legend, gutter)
    return Text("\n").join(lines)


def volume_chart(df: pd.DataFrame, width: int, height: int, *, layout: Layout, gutter: int = 0) -> Text:
    """Volume bars aligned with ``candlestick_chart`` columns."""
    if df is None or len(df) == 0 or "Volume" not in df:
        return Text("")
    agg = _aggregate_ohlcv(df, layout)
    cols = layout.cols
    rows = max(int(height), 1)
    vmax = float(agg["Volume"].max()) or 1.0
    grid = [[(" ", None)] * cols for _ in range(rows)]
    for s in range(layout.slots):
        col = int(layout.cell_of_slot(s))
        if col >= cols:
            continue
        color = dim(UP if agg.at[s, "Close"] >= agg.at[s, "Open"] else DOWN, 0.7)
        eighths = int(round(agg.at[s, "Volume"] / vmax * rows * 8))
        for r in range(rows):
            filled = eighths - (rows - 1 - r) * 8
            if filled >= 8:
                grid[r][col] = ("█", color)
            elif filled > 0:
                grid[r][col] = ("▁▂▃▄▅▆▇"[filled - 1], color)
    lines = []
    for cells in grid:
        t = Text(" " * gutter, no_wrap=True)
        t.append_text(_line_to_text(cells))
        lines.append(t)
    return Text("\n").join(lines)


def indicator_pane(df: pd.DataFrame, columns: Sequence[str], width: int, height: int, *, layout: Layout,
                   gutter: int, bands: Sequence[float] = (), y_fmt: Callable[[float], str] = fmt_num) -> Text:
    """Line chart of oscillator-style indicators aligned to the candle columns."""
    cols = layout.cols
    rows = max(int(height), 2)
    cv = BrailleCanvas(cols, rows)
    arrays = {c: df[c].to_numpy(float) for c in columns if c in df}
    if not arrays:
        return Text("")
    allv = np.concatenate([a[np.isfinite(a)] for a in arrays.values()] or [np.array([0.0])])
    if allv.size == 0:
        return Text("")
    lo, hi = float(allv.min()), float(allv.max())
    for b in bands:
        lo, hi = min(lo, b), max(hi, b)
    lo, hi = _nice_range(lo, hi, 0.08)
    g = max(gutter, 6)
    for b in bands:
        cv.hline(int(np.clip((1 - (b - lo) / (hi - lo)) * (cv.h - 1), 0, cv.h - 1)), cv.color_id(GRID))
    for i, (name, a) in enumerate(arrays.items()):
        cid = cv.color_id(SERIES_COLORS[i % len(SERIES_COLORS)])
        prev = None
        for slot in range(layout.slots):
            seg = a[layout.edges[slot]:max(layout.edges[slot + 1], layout.edges[slot] + 1)]
            seg = seg[np.isfinite(seg)]
            if not seg.size:
                prev = None
                continue
            x = int(layout.cell_of_slot(slot)) * 2 + (1 if layout.step == 2 else 0)
            y = int(np.clip((1 - (seg[-1] - lo) / (hi - lo)) * (cv.h - 1), 0, cv.h - 1))
            if prev is None:
                cv.set(x, y, cid)
            else:
                cv.line(prev[0], prev[1], x, y, cid)
            prev = (x, y)
    body = _canvas_rows(cv)
    out = []
    for r, cells in enumerate(body):
        label = {0: y_fmt(hi), rows - 1: y_fmt(lo)}.get(r, "")
        t = Text(label.rjust(g - 2), style=MUTED, no_wrap=True)
        t.append(" ┤" if label else " │", style=GRID)
        t.append_text(_line_to_text(cells))
        out.append(t)
    return Text("\n").join(out)


# --------------------------------------------------------------------------- #
# Bars, histograms, heat maps
# --------------------------------------------------------------------------- #
def bar_chart(values: Sequence[float], width: int, height: int, *, labels: Optional[Sequence[str]] = None,
              y_fmt: Callable[[float], str] = fmt_num, colors: Tuple[str, str] = (UP, DOWN),
              title: Optional[str] = None) -> Text:
    """Vertical bars with a zero baseline; positives and negatives coloured separately."""
    v = np.asarray(values, dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    if v.size == 0:
        return Text("no data", style=MUTED)
    width, height = max(int(width), 12), max(int(height), 4)
    lo, hi = min(float(v.min()), 0.0), max(float(v.max()), 0.0)
    lo, hi = _nice_range(lo, hi, 0.05) if hi != lo else (lo - 1, hi + 1)
    gutter = max(len(y_fmt(lo)), len(y_fmt(hi))) + 3
    lab_rows = 1 if labels else 0
    leg_rows = 1 if title else 0
    cols = max(width - gutter, 4)
    rows = max(height - lab_rows - leg_rows, 3)
    cv = BrailleCanvas(cols, rows)
    pos_id, neg_id = cv.color_id(colors[0]), cv.color_id(colors[1])
    zero = int(round((1 - (0 - lo) / (hi - lo)) * (cv.h - 1)))
    per = cv.w / len(v)
    bw = max(int(per) - (1 if per >= 3 else 0), 1)
    for i, x in enumerate(v):
        x0 = int(i * per + (per - bw) / 2)
        y = int(np.clip(round((1 - (x - lo) / (hi - lo)) * (cv.h - 1)), 0, cv.h - 1))
        for dx in range(bw):
            cv.vfill(x0 + dx, y, zero, pos_id if x >= 0 else neg_id)
    cv.hline(int(np.clip(zero, 0, cv.h - 1)), cv.color_id(GRID), dash=1)
    for i, x in enumerate(v):  # redraw bars so the baseline does not tint them
        x0 = int(i * per + (per - bw) / 2)
        y = int(np.clip(round((1 - (x - lo) / (hi - lo)) * (cv.h - 1)), 0, cv.h - 1))
        for dx in range(bw):
            cv.vfill(x0 + dx, y, zero, pos_id if x >= 0 else neg_id)
    leg = Text(title or "", style="bold", no_wrap=True) if title else None
    zero_row = int(np.clip(zero // 4, 0, rows - 1))
    ticks = {0: hi, rows - 1: lo}
    if zero_row not in (0, rows - 1):
        ticks[zero_row] = 0.0
    lines = _frame(_canvas_rows(cv), lo, hi, y_fmt, None, leg, gutter, ticks)
    if labels:
        lines.append(_x_label_row(_thin_labels(labels, cols), cols, gutter))
    return Text("\n").join(lines)


def _thin_labels(labels: Sequence[str], cols: int) -> List[str]:
    labels = [str(x) for x in labels]
    maxlen = max(len(x) for x in labels) + 1
    keep = max(cols // maxlen, 1)
    if len(labels) <= keep:
        return labels
    idx = np.unique(np.linspace(0, len(labels) - 1, keep).round().astype(int))
    return [labels[i] for i in idx]


def histogram(values: Sequence[float], width: int, height: int, *, bins: int = 24,
              title: Optional[str] = None, y_fmt: Callable[[float], str] = lambda v: f"{int(v)}") -> Text:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return Text("no data", style=MUTED)
    bins = int(min(bins, max(width // 2, 4)))
    counts, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    # colour bars by sign of their centre; bar_chart colours by value sign, so build manually
    gutter = len(y_fmt(counts.max())) + 3
    cols = max(width - gutter, 4)
    rows = max(height - (1 if title else 0) - 1, 3)
    cv = BrailleCanvas(cols, rows)
    up, dn = cv.color_id(UP), cv.color_id(DOWN)
    per = cv.w / bins
    bw = max(int(per) - (1 if per >= 3 else 0), 1)
    cmax = max(int(counts.max()), 1)
    for i, c in enumerate(counts):
        x0 = int(i * per + (per - bw) / 2)
        y = int(round((1 - c / cmax) * (cv.h - 1)))
        for dx in range(bw):
            if c:
                cv.vfill(x0 + dx, y, cv.h - 1, up if centers[i] >= 0 else dn)
    leg = Text(title, style="bold") if title else None
    lines = _frame(_canvas_rows(cv), 0, cmax, y_fmt, None, leg, gutter)
    axis = _x_label_row([fmt_num(edges[0]), fmt_num(0.0) if edges[0] < 0 < edges[-1] else "", fmt_num(edges[-1])],
                        cols, gutter)
    lines.append(axis)
    return Text("\n").join(lines)


def heatmap(matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], *, cell_width: int = 7,
            fmt: Callable[[float], str] = lambda v: f"{v:.1f}", diverging: bool = True,
            vmin: Optional[float] = None, vmax: Optional[float] = None, mark_best: bool = False) -> Text:
    """Background-coloured grid. NaN cells are left blank."""
    m = np.asarray(matrix, dtype=float)
    if m.size == 0:
        return Text("no data", style=MUTED)
    finite = m[np.isfinite(m)]
    if finite.size == 0:
        return Text("no data", style=MUTED)
    vmin = float(finite.min()) if vmin is None else vmin
    vmax = float(finite.max()) if vmax is None else vmax
    lab_w = max((len(str(r)) for r in row_labels), default=0) + 1
    t = Text(no_wrap=True)
    t.append(" " * lab_w)
    for c in col_labels:
        t.append(str(c)[:cell_width].center(cell_width), style=MUTED)
    t.append("\n")
    best = np.unravel_index(np.nanargmax(m), m.shape) if mark_best and finite.size else None
    for i, rl in enumerate(row_labels):
        t.append(str(rl).rjust(lab_w - 1) + " ", style=MUTED)
        for j in range(m.shape[1]):
            val = m[i, j]
            if not np.isfinite(val):
                t.append(" " * cell_width)
                continue
            if diverging:
                bound = max(abs(vmin), abs(vmax), 1e-12)
                x = val / bound
                bg = lerp_color("#242424", UP if x >= 0 else DOWN, abs(x) * 0.85)
            else:
                bg = lerp_color("#2a1800", ORANGE, (val - vmin) / max(vmax - vmin, 1e-12))
            txt = fmt(val).center(cell_width)
            if best is not None and (i, j) == best:
                txt = ("★" + fmt(val)).center(cell_width)
            t.append(txt, style=f"white on {bg}")
        t.append("\n")
    return t


LOGO_ROWS = [
    "█████▄   ▄████▄  ▀█████▀",
    "█    ██  █▀  ▀█     █   ",
    "█████▀   ██████     █   ",
    "█    ██  █    █     █   ",
    "█████▀   █    █     █   ",
]
LOGO_TOP, LOGO_BOTTOM = "#ffc457", "#ff7a00"      # light amber -> deep orange


def ascii_logo() -> Text:
    t = Text()
    for i, line in enumerate(LOGO_ROWS):
        t.append(line + "\n", style=lerp_color(LOGO_TOP, LOGO_BOTTOM, i / (len(LOGO_ROWS) - 1)))
    return t


# --------------------------------------------------------------------------- #
# Composite price view: candles + volume + indicator pane on shared columns
# --------------------------------------------------------------------------- #
def price_gutter(df: pd.DataFrame, overlays: Optional[Dict[str, Sequence[float]]] = None,
                 y_fmt: Callable[[float], str] = fmt_num) -> Tuple[int, Tuple[float, float]]:
    lo, hi = float(df["Low"].min()), float(df["High"].max())
    for s in (overlays or {}).values():
        a = np.asarray(s, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    lo, hi = _nice_range(lo, hi)
    return max(len(y_fmt(lo)), len(y_fmt(hi))) + 3, (lo, hi)


def price_stack(df: pd.DataFrame, width: int, height: int, *, markers=None, overlays=None,
                panes: Sequence[str] = (), bands: Sequence[float] = (), y_fmt: Callable[[float], str] = fmt_num,
                hover: Optional[int] = None, title: str = "") -> Tuple[Text, Optional[Layout], int]:
    """Render candles, volume and (optionally) an oscillator pane as one block.

    Returns ``(text, layout, gutter)`` so a widget can map mouse columns back to bars.
    """
    if df is None or len(df) == 0:
        return Text("no data", style=MUTED), None, 0
    width, height = max(int(width), 24), max(int(height), 8)
    overlays = {k: v for k, v in (overlays or {}).items() if len(v) == len(df)}
    gutter, yr = price_gutter(df, overlays, y_fmt)
    cols = width - gutter
    lay = Layout.build(len(df), cols)
    has_pane = bool(panes) and any(c in df for c in panes)
    vol_h = 3 if height >= 14 else 0
    pane_h = max(height // 4, 4) if has_pane and height >= 16 else 0
    xr = 2
    price_h = max(height - vol_h - pane_h - xr - 1, 4)

    legend = Text(no_wrap=True)
    if title:
        legend.append(title + "  ", style="bold")
    i = len(df) - 1 if hover is None else int(np.clip(hover, 0, len(df) - 1))
    r = df.iloc[i]
    col = UP if r["Close"] >= r["Open"] else DOWN
    if "timestamp" in df:
        legend.append(f"{pd.Timestamp(r['timestamp']):%Y-%m-%d %H:%M}  ", style=MUTED)
    for k in ("Open", "High", "Low", "Close"):
        legend.append(f"{k[0]} ", style=MUTED)
        legend.append(f"{y_fmt(float(r[k]))}  ", style=col)
    if "Volume" in df:
        legend.append("V ", style=MUTED)
        legend.append(fmt_num(float(r["Volume"])), style=MUTED)
    names = list(overlays)
    for j, n in enumerate(names[:4]):
        legend.append(f"  ━{n}", style=AMBER if j == 0 else SERIES_COLORS[(j + 1) % len(SERIES_COLORS)])

    xl = time_labels(df["timestamp"], 5) if "timestamp" in df else None
    parts = [candlestick_chart(df, width, price_h + 1 + xr, markers=markers, overlays=overlays, x_labels=xl,
                               y_fmt=y_fmt, legend=legend, layout=lay, gutter=gutter, y_range=yr)]
    if vol_h:
        parts.append(volume_chart(df, width, vol_h, layout=lay, gutter=gutter))
    if pane_h:
        parts.append(indicator_pane(df, [c for c in panes if c in df], width, pane_h, layout=lay, gutter=gutter,
                                    bands=bands))
    return Text("\n").join(parts), lay, gutter
