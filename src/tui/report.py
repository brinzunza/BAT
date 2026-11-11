"""Backward-compatible re-export: the analytics moved to ``src.core.analytics.report`` (pure pandas/numpy, no UI)."""
from src.core.analytics.report import (  # noqa: F401
    CLOSE_ACTIONS, OPEN_ACTIONS, BacktestReport, _periodic_returns, build_report, equity_curve, pair_round_trips,
)
