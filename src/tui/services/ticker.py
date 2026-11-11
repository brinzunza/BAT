"""Data behind the animated PnL ticker in the Home header.

Pure logic (no widgets): each ``step()`` returns the next frame. The source is chosen on every step, most real first:

* ``LIVE``   - a running live/demo session: its equity curve and session PnL.
* ``REPLAY`` - the most recent backtest's equity curve, replayed in a loop.
* ``DEMO``   - a seeded-or-random walk. Always labelled, so it can never be mistaken for real money.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

DEMO_START = 100_000.0


@dataclass(frozen=True)
class TickerFrame:
    values: List[float]          # recent equity values, oldest first (the sparkline)
    pnl: float                   # equity - starting equity
    pct: float                   # pnl as a percentage of starting equity
    direction: int               # +1 / -1 / 0 versus the previous step
    source: str                  # 'LIVE' | 'REPLAY' | 'DEMO'
    label: str                   # short caption, e.g. the strategy name


class TickerFeed:
    def __init__(self, state=None, rng: Optional[np.random.Generator] = None, window: int = 28, replay_steps: int = 400):
        self.state = state
        self.rng = rng if rng is not None else np.random.default_rng()
        self.window = window
        self.replay_steps = replay_steps            # a replay loop lasts about this many steps whatever the curve length
        self._demo = [DEMO_START] * window
        for _ in range(window * 3):                      # warm up so the very first frame already has a chart
            self._demo_curve()
        self._replay_id: Optional[int] = None
        self._cursor = 0.0
        self._last: Optional[float] = None

    # ----- source selection -------------------------------------------------
    def step(self) -> TickerFrame:
        live = self._live_curve()
        if live is not None:
            values, start, label, source = live[0], live[1], "live session", "LIVE"
        else:
            replay = self._replay_curve()
            if replay is not None:
                values, start, label = replay
                source = "REPLAY"
            else:
                values, start, label, source = self._demo_curve(), DEMO_START, "simulated", "DEMO"
        cur = values[-1]
        direction = 0 if self._last is None or cur == self._last else (1 if cur > self._last else -1)
        self._last = cur
        pnl = cur - start
        return TickerFrame(list(values[-self.window:]), pnl, pnl / start * 100 if start else 0.0, direction, source, label)

    # ----- sources ----------------------------------------------------------
    def _live_curve(self):
        live = getattr(self.state, "live", None)
        if live is None or not getattr(live, "running", False):
            return None
        eq = [float(v[1]) for v in getattr(live, "equity", [])]
        if len(eq) < 2:
            return None
        start = float(getattr(getattr(live, "cfg", None), "initial_balance", eq[0]) or eq[0])
        return eq, start, "live session"

    def _replay_curve(self):
        recs = getattr(self.state, "backtests", None)
        if not recs:
            return None
        rec = recs[-1]
        eq = np.asarray(rec.report.equity, dtype=float)
        eq = eq[np.isfinite(eq)]
        if eq.size < 2:
            return None
        if rec.id != self._replay_id:                 # a new backtest finished: restart its replay
            self._replay_id, self._cursor = rec.id, 0.0
        stride = max(eq.size / self.replay_steps, 1.0)
        self._cursor = (self._cursor + stride) % eq.size
        end = max(self._cursor, 2.0)
        idx = np.clip(np.round(end - stride * np.arange(self.window - 1, -1, -1)), 0, eq.size - 1).astype(int)
        return list(eq[idx]), float(rec.report.initial_cash or eq[0]), rec.label

    def _demo_curve(self) -> List[float]:
        cur = self._demo[-1]
        drift = (DEMO_START - cur) * 0.002                     # gentle pull back so the PnL stays near zero
        cur += drift + self.rng.normal(0, DEMO_START * 0.0004)
        self._demo = self._demo[1:] + [cur]
        return self._demo
