"""The one documented way to run a backtest.

    from src.core.api import run_backtest, BacktestConfig
    outcome = run_backtest(bars, strategy, BacktestConfig(initial_cash=10_000, trading_mode="long_short"))
    outcome.summary["sharpe"], outcome.trades, outcome.equity
    outcome.to_dict()            # JSON-safe, size-capped: safe to hand to an agent or an API response

Inputs
    bars      DataFrame with at least ``Close`` and ``timestamp`` (strategies usually also need Open/High/Low/Volume).
    strategy  anything satisfying ``src.core.ports.trading.Strategy`` (see docs/engine_api.md).
    config    ``BacktestConfig`` (validated on construction).
The call is deterministic, performs no I/O, reads no environment variables and never mutates ``bars``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core.analytics.report import BacktestReport, build_report
from src.core.application.backtest import BacktestResult, BacktestService
from src.core.domain.models import BacktestConfig
from src.core.ports.trading import Strategy

__all__ = ["BacktestConfig", "BacktestOutcome", "run_backtest", "json_safe"]


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars, NaN/inf, timestamps and timedeltas into plain JSON types."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, timedelta)):
        return float(value.total_seconds())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    return str(value)


@dataclass(frozen=True)
class BacktestOutcome:
    """Everything a caller needs after a run: raw engine result plus derived analytics."""

    result: BacktestResult
    report: BacktestReport
    config: BacktestConfig

    @property
    def trades(self) -> pd.DataFrame:           # one row per execution (open or close)
        return self.result.trades

    @property
    def signal_data(self) -> pd.DataFrame:      # strategy output (bars + indicators + signal columns)
        return self.result.signal_data

    @property
    def summary(self) -> Dict[str, object]:     # headline metrics (return, sharpe, drawdown, win rate, ...)
        return self.report.summary

    @property
    def equity(self) -> pd.Series:              # mark-to-market equity, one value per bar
        return self.report.equity

    @property
    def round_trips(self) -> pd.DataFrame:      # one row per completed trade
        return self.report.round_trips

    def to_dict(self, max_equity_points: int = 300, max_trades: int = 200) -> Dict[str, Any]:
        """JSON-safe digest. Large series are down-sampled and trades truncated so the payload stays small."""
        eq = self.equity
        if len(eq) > max_equity_points > 0:
            eq = eq.iloc[np.linspace(0, len(eq) - 1, max_equity_points).astype(int)]
        rt = self.round_trips.drop(columns=["duration"], errors="ignore")
        return json_safe({
            "config": self.config.__dict__,
            "summary": self.summary,
            "equity": [[ts, v] for ts, v in zip(eq.index, eq.to_numpy())],
            "round_trips": rt.head(max_trades).to_dict(orient="records"),
            "round_trips_truncated": max(0, len(rt) - max_trades),
            "open_trade": self.report.open_trade,
        })


def run_backtest(bars: pd.DataFrame, strategy: Strategy, config: Optional[BacktestConfig] = None) -> BacktestOutcome:
    """Run ``strategy`` over ``bars`` and return trades plus analytics. Raises ``ValueError`` on contract violations."""
    config = config or BacktestConfig()
    result = BacktestService().run(bars, strategy, config)
    report = build_report(result.trades, result.signal_data, config.initial_cash)
    return BacktestOutcome(result=result, report=report, config=config)
