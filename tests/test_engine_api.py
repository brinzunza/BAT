"""The engine facade (src/core/api.py) and the output-contract hardening in BacktestService."""
import json

import numpy as np
import pandas as pd
import pytest

from src.core.api import BacktestConfig, run_backtest
from src.core.application.backtest import BacktestService


def bars(prices):
    n = len(prices)
    return pd.DataFrame({"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": 1.0,
                         "timestamp": pd.date_range("2026-01-01", periods=n, freq="D")})


class Scripted:
    name = "scripted"

    def __init__(self, buys, sells, rows=None):
        self.buys, self.sells, self.rows = buys, sells, rows

    def generate_signals(self, data):
        data = data.copy()
        data["buy"], data["sell"] = self.buys, self.sells
        return data if self.rows is None else data.iloc[: self.rows]

    def get_signal_names(self):
        return {"buy": "buy", "sell": "sell"}

    def get_required_lookback(self):
        return 1


PRICES = [100, 100, 110, 110, 120]
BUYS = [False, True, False, False, False]
SELLS = [False, False, True, False, False]


def test_facade_matches_the_service_and_adds_analytics():
    b = bars(PRICES)
    out = run_backtest(b, Scripted(BUYS, SELLS), BacktestConfig())
    direct = BacktestService().run(b, Scripted(BUYS, SELLS), BacktestConfig())
    pd.testing.assert_frame_equal(out.trades, direct.trades)
    assert out.summary["final"] == pytest.approx(direct.portfolio.cash)
    assert out.summary["trades"] == 1 and len(out.equity) == len(PRICES)
    assert list(out.round_trips["side"]) == ["LONG"]


def test_facade_does_not_mutate_the_input_bars():
    b = bars(PRICES)
    before = b.copy()
    run_backtest(b, Scripted(BUYS, SELLS))
    pd.testing.assert_frame_equal(b, before)


def test_to_dict_is_strict_json_and_size_capped():
    n = 2000
    prices = list(100 + np.sin(np.arange(n) / 5) * 10)
    buys = [i % 40 == 1 for i in range(n)]
    sells = [i % 40 == 21 for i in range(n)]
    d = run_backtest(bars(prices), Scripted(buys, sells)).to_dict(max_equity_points=100, max_trades=5)
    json.dumps(d, allow_nan=False)                       # no NaN/inf/Timestamp leaks
    assert len(d["equity"]) == 100 and len(d["round_trips"]) == 5 and d["round_trips_truncated"] > 0
    assert d["config"]["trading_mode"] == "long_only"


def test_to_dict_handles_runs_with_no_trades():
    d = run_backtest(bars(PRICES), Scripted([False] * 5, [False] * 5)).to_dict()
    json.dumps(d, allow_nan=False)
    assert d["summary"]["trades"] == 0 and d["round_trips"] == []


def test_nan_signals_are_not_signals():
    buys = [np.nan, np.nan, np.nan, np.nan, np.nan]
    sells = [None] * 5
    out = run_backtest(bars(PRICES), Scripted(buys, sells), BacktestConfig(trading_mode="long_short"))
    assert len(out.trades) == 0
    assert out.signal_data["buy"].dtype == bool and not out.signal_data["buy"].any()


def test_numeric_signal_columns_are_coerced():
    out = run_backtest(bars(PRICES), Scripted([0, 1, 0, 0, 0], [0.0, 0.0, 1.0, np.nan, 0.0]))
    assert out.trades["Action"].tolist() == ["BUY", "CLOSE"]


def test_rejects_misaligned_strategy_output():
    with pytest.raises(ValueError, match="aligned"):
        run_backtest(bars(PRICES), Scripted(BUYS, SELLS, rows=3))


def test_rejects_non_dataframe_output():
    class Bad(Scripted):
        def generate_signals(self, data):
            return [1, 2, 3]

    with pytest.raises(ValueError, match="DataFrame"):
        run_backtest(bars(PRICES), Bad(BUYS, SELLS))


def test_rejects_strategy_that_drops_required_columns():
    class Dropper(Scripted):
        def generate_signals(self, data):
            out = super().generate_signals(data)
            return out.drop(columns=["timestamp"])

    with pytest.raises(ValueError, match="dropped required columns"):
        run_backtest(bars(PRICES), Dropper(BUYS, SELLS))


def test_rejects_malformed_signal_names():
    class Bad(Scripted):
        def get_signal_names(self):
            return {"buy": "buy"}

    with pytest.raises(ValueError, match="declared buy and sell"):
        run_backtest(bars(PRICES), Bad(BUYS, SELLS))


def test_ui_report_module_still_exports_the_same_objects():
    from src.core.analytics import report as core
    from src.tui import report as ui
    assert ui.build_report is core.build_report and ui.BacktestReport is core.BacktestReport
