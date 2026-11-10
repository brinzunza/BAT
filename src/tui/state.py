"""Process-wide UI state shared by every page."""
from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from src.tui.datasets import DatasetCache
from src.tui.report import BacktestReport
from src.tui.services.connections import ConnectionManager
from src.tui.services.live import LiveSession

_ids = itertools.count(1)


@dataclass
class BacktestRecord:
    label: str
    strategy_key: str
    params: Dict[str, Any]
    symbol: str
    mode: str
    source: str
    report: BacktestReport
    config: Dict[str, Any]
    id: int = field(default_factory=lambda: next(_ids))
    created: datetime = field(default_factory=datetime.now)


@dataclass
class LogEntry:
    ts: datetime
    level: str
    source: str
    text: str


class BatState:
    def __init__(self) -> None:
        self.connections = ConnectionManager()
        self.datasets = DatasetCache()
        self.backtests: List[BacktestRecord] = []
        self.live: Optional[LiveSession] = None
        self.events: Deque[LogEntry] = deque(maxlen=5000)

    def add_event(self, level: str, source: str, text: str) -> LogEntry:
        e = LogEntry(datetime.now(), level, source, text)
        self.events.append(e)
        return e
