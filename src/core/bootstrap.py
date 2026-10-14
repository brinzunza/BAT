"""The composition root: dependencies are selected here, never in the domain."""

from src.core.application.backtest import BacktestService


def build_backtest_service() -> BacktestService:
    return BacktestService()
