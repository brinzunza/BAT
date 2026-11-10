"""Offline market simulator so live trading can be exercised without any account.

Prices follow a regime-switching geometric random walk; each ``latest`` call
advances simulated time by one bar, so a session produces fresh candles every
tick regardless of the wall clock.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class DemoFeed:
    name = "Demo market (offline)"

    def __init__(self, start_price: float = 60_000.0, seed: Optional[int] = 7, bar_seconds: int = 60):
        self.rng = np.random.RandomState(seed)
        self.price = float(start_price)
        self.bar = pd.Timedelta(seconds=bar_seconds)
        self.now = pd.Timestamp("2025-01-01 09:30:00")
        self.vol = 0.0006
        self._trend = 0.0

    def _next_bar(self) -> dict:
        if self.rng.rand() < 0.03:                       # regime change
            self.vol = float(self.rng.choice([0.0003, 0.0006, 0.0012]))
            self._trend = float(self.rng.choice([-1, 0, 1])) * 0.00008
        o = self.price
        path = o * np.exp(np.cumsum(self.rng.normal(self._trend, self.vol, 6)))
        c = float(path[-1])
        h, l = float(max(o, path.max())), float(min(o, path.min()))
        self.price = c
        self.now = self.now + self.bar
        return dict(timestamp=self.now, Open=o, High=h, Low=l, Close=c, Volume=float(self.rng.uniform(1, 30)))

    def history(self, symbol: str, n: int = 300) -> pd.DataFrame:
        return pd.DataFrame([self._next_bar() for _ in range(n)])

    def latest(self, symbol: str) -> dict:
        return self._next_bar()
