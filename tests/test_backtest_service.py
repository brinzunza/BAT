import pandas as pd
import pytest

from src.core.application.backtest import BacktestService
from src.core.domain.models import BacktestConfig


class ScriptedStrategy:
    name = "scripted"

    def __init__(self, buys, sells):
        self.buys, self.sells = buys, sells

    def generate_signals(self, data):
        data["buy"] = self.buys
        data["sell"] = self.sells
        return data

    def get_signal_names(self):
        return {"buy": "buy", "sell": "sell"}

    def get_required_lookback(self):
        return 1


def bars(prices):
    return pd.DataFrame({"Close": prices, "timestamp": pd.date_range("2026-01-01", periods=len(prices), freq="h")})


def test_long_only_realizes_a_profitable_trade():
    result = BacktestService().run(bars([100, 100, 110]), ScriptedStrategy([False, True, False], [False, False, True]), BacktestConfig())
    assert result.portfolio.cash == pytest.approx(11_000)
    assert result.portfolio.realized_profit == pytest.approx(1_000)
    assert result.trades["Action"].tolist() == ["BUY", "CLOSE"]


def test_long_short_reverses_a_long_position():
    result = BacktestService().run(bars([100, 100, 110, 100]), ScriptedStrategy([False, True, False, False], [False, False, True, True]), BacktestConfig(trading_mode="long_short"))
    assert result.trades["Action"].tolist() == ["BUY", "CLOSE", "SELL_SHORT"]
    assert result.portfolio.quantity < 0


def test_forex_spread_is_applied_on_entry_and_exit():
    result = BacktestService().run(bars([1.0, 1.0, 1.0]), ScriptedStrategy([False, True, False], [False, False, True]), BacktestConfig(symbol="C:EURUSD", spread_pips=1))
    assert result.portfolio.realized_profit < 0


def test_rejects_missing_market_data_contract():
    with pytest.raises(ValueError, match="Close"):
        BacktestService().run(pd.DataFrame({"timestamp": []}), ScriptedStrategy([], []), BacktestConfig())


def test_legacy_engine_is_a_compatible_inbound_adapter():
    from src.evaluation.backtest import BacktestEngine

    engine = BacktestEngine()
    result = engine.backtest(bars([100, 100, 110]), ScriptedStrategy([False, True, False], [False, False, True]))
    assert engine.analyze_results(result)["final_balance"] == pytest.approx(11_000)
    assert "Index" in result


def test_run_for_symbol_uses_the_market_data_port():
    class MarketData:
        def historical_bars(self, symbol, **kwargs):
            assert symbol == "TEST"
            assert kwargs == {"timespan": "day"}
            return bars([100, 100, 110])

    result = BacktestService().run_for_symbol(
        MarketData(), "TEST", ScriptedStrategy([False, True, False], [False, False, True]),
        BacktestConfig(), timespan="day",
    )
    assert result.portfolio.realized_profit == pytest.approx(1_000)


def test_legacy_data_provider_adapter_translates_the_provider_contract():
    from src.core.adapters.legacy import LegacyDataProviderAdapter

    class LegacyProvider:
        def get_data(self, **kwargs):
            assert kwargs == {"ticker": "TEST", "timespan": "day"}
            return bars([1])

    assert len(LegacyDataProviderAdapter(LegacyProvider()).historical_bars("TEST", timespan="day")) == 1
