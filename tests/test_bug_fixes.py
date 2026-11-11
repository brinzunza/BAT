"""Regression tests for bugs found in the original platform (each test fails on the pre-fix code)."""
import inspect

import numpy as np
import pandas as pd
import pytest

from src.core.api import BacktestConfig, run_backtest
from src.execution.live_engine import LiveTradingEngine
from src.strategies.macd_strategy import MACDStrategy


def bars(n=400, seed=3):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.01, n)))
    return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="h"), "Open": c, "High": c * 1.001,
                         "Low": c * 0.999, "Close": c, "Volume": 1.0})


def test_macd_strategy_produces_numeric_indicators_and_signals():
    out = MACDStrategy().generate_signals(bars())
    for col in ("macd_line", "signal_line", "histogram"):
        assert pd.api.types.is_float_dtype(out[col]), f"{col} must be numeric, got {out[col].dtype}"
    assert out["Buy Signal"].sum() > 0 and out["Sell Signal"].sum() > 0
    assert not (out["Buy Signal"] & out["Sell Signal"]).any()


def _broker_classes():
    from src.data.providers import alpaca_provider as ap
    classes = [ap.AlpacaBroker, ap.SimulatedBroker]
    try:
        from src.execution.ib_broker import IBBroker
        classes.append(IBBroker)
    except Exception:       # ibapi not installed
        pass
    from src.execution.brokers import SimulatedBroker
    return classes + [SimulatedBroker]


@pytest.mark.parametrize("cls", _broker_classes(), ids=lambda c: f"{c.__module__}.{c.__name__}")
def test_every_broker_accepts_the_current_price_hint_the_engine_sends(cls):
    for method in ("buy", "sell", "close_position"):
        assert "current_price" in inspect.signature(getattr(cls, method)).parameters, f"{cls.__name__}.{method}"


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_failed_alpaca_order_is_reported_as_failed_not_empty(side):
    from src.data.providers.alpaca_provider import AlpacaBroker
    broker = object.__new__(AlpacaBroker)

    class Api:
        def submit_order(self, **kw):
            raise RuntimeError("insufficient buying power")
    broker.trade_api = Api()
    result = getattr(broker, side)("AAPL", 1, current_price=10.0)
    # The live engine only treats status == 'failed' as a failure, so an empty dict used to count as a fill.
    assert result.get("status") == "failed" and "insufficient" in result["error"]


class Both:
    name = "both"

    def generate_signals(self, df):
        out = df.copy()
        out["Buy Signal"] = [i % 10 == 5 for i in range(len(df))]
        out["Sell Signal"] = [i % 10 == 5 or i % 10 == 8 for i in range(len(df))]   # bar 5: conflicting signals
        return out

    def get_signal_names(self):
        return {"buy": "Buy Signal", "sell": "Sell Signal"}

    def get_required_lookback(self):
        return 1


def test_backtest_ignores_bars_with_conflicting_signals_like_the_live_engine():
    df = bars(60)
    for mode in ("long_only", "long_short"):
        t = run_backtest(df, Both(), BacktestConfig(trading_mode=mode)).trades
        assert t.empty or not set(t["Index"]) & {i for i in range(60) if i % 10 == 5}, \
            f"{mode}: traded on a bar where buy and sell fired together"
    t = run_backtest(df, Both(), BacktestConfig(trading_mode="long_short")).trades
    assert set(t["Action"]) <= {"SELL_SHORT"}, "only the unambiguous sell bars should have traded"


def test_pending_broker_status_is_not_described_as_an_error():
    msg = LiveTradingEngine._order_problem({"status": "pending"})
    assert "not confirmed" in msg and "Unknown error" not in msg
    assert LiveTradingEngine._order_problem({"status": "failed", "error": "boom"}) == "boom"
    assert LiveTradingEngine._order_problem({"status": "failed"}) == "Unknown error"


def test_simulated_brokers_spell_cancelled_the_same_way():
    from src.data.providers.alpaca_provider import SimulatedBroker as A
    from src.execution.brokers import SimulatedBroker as B
    a = object.__new__(A)
    a.orders = {"o1": {"status": "pending", "side": "buy", "qty": 1, "symbol": "AAPL"}}
    assert a.cancel_order("o1")["status"] == "cancelled"
    assert B(10_000).cancel_order("o1")["status"] == "cancelled"
