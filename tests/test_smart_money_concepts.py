import pandas as pd

from src.strategies.smart_money_concepts import SmartMoneyConceptsStrategy
from src.core.application.backtest import BacktestService
from src.core.domain.models import BacktestConfig


def minute_bars(length=180):
    close = [100 + ((index % 17) - 8) * 0.2 for index in range(length)]
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=length, freq="min", tz="UTC"),
        "Open": close, "High": [value + 0.1 for value in close],
        "Low": [value - 0.1 for value in close], "Close": close, "Volume": [100] * length,
    })


def test_smc_produces_the_standard_bat_signal_contract():
    result = SmartMoneyConceptsStrategy().generate_signals(minute_bars())
    assert {"Buy Signal", "Sell Signal", "HTF Bias", "POI Low", "LTF CHOCH Down"}.issubset(result.columns)
    assert result["Buy Signal"].dtype == bool
    assert result["Sell Signal"].dtype == bool


def test_smc_confirms_fractals_only_after_the_right_hand_window():
    strategy = SmartMoneyConceptsStrategy(swing_window=2)
    highs = pd.Series([1, 2, 5, 2, 1, 1])
    lows = pd.Series([0, 0, 0, 0, 0, 0])
    confirmed_highs, _ = strategy._confirmed_swings(highs, lows)
    assert pd.isna(confirmed_highs.iloc[2])
    assert confirmed_highs.iloc[4] == 5


def test_smc_rejects_missing_ohlcv_contract():
    try:
        SmartMoneyConceptsStrategy().generate_signals(pd.DataFrame({"Close": [1]}))
    except ValueError as error:
        assert "Invalid data format" in str(error)
    else:
        raise AssertionError("expected invalid data to be rejected")


def test_smc_runs_through_the_hexagonal_backtest_use_case():
    result = BacktestService().run(minute_bars(), SmartMoneyConceptsStrategy(), BacktestConfig(trading_mode="long_short"))
    assert len(result.signal_data) == 180
    assert result.portfolio.cash > 0
