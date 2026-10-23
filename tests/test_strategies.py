import numpy as np, pandas as pd, pytest
from src.data.features import candlestick_flags, bollinger_bands
from src.evaluation.backtest import BacktestEngine
from src.strategies.bollinger_bands_strategy import BollingerBandsStrategy
from src.strategies.candlestick_strategy import CandlestickPatternsStrategy
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.mean_reversion import MeanReversionExtremeStrategy
from src.strategies.smart_money_concepts import SmartMoneyConceptsStrategy
def market(n=800, seed=1):
    rng = np.random.RandomState(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, .01, n))); o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(dict(timestamp=pd.date_range("2024-01-01", periods=n, freq="h"), Open=o, High=np.maximum(o, c) * 1.003, Low=np.minimum(o, c) * .997, Close=c, Volume=1000.0))
def test_bollinger_signals_use_the_bands():
    d = market(); out = BollingerBandsStrategy(window=20, num_std=2).generate_signals(d); b = bollinger_bands(d.Close, 20, 2)
    assert np.allclose(out.bb_upper.dropna(), b["upper"].dropna()) and (out.bb_lower.dropna() <= out.bb_middle.dropna()).all()
    assert (out["Buy Signal"] == (d.Close <= b["lower"])).all() and out["Buy Signal"].any() and out["Sell Signal"].any()
def _flags(rows):
    o, h, l, c = (pd.Series([r[i] for r in rows], dtype=float) for i in range(4)); return candlestick_flags(o, h, l, c)
def test_hammer_after_decline_and_hanging_man_after_rise():
    down = [(10, 10.1, 9.9, 10), (9.8, 9.9, 9.6, 9.7), (9.5, 9.6, 9.3, 9.4), (9.2, 9.22, 8.4, 9.22)]    # last bar: tiny body, long lower shadow, after falling closes
    f = _flags(down); assert f.loc[3, "hammer"] and not f.loc[3, "hanging_man"]
    up = [(10, 10.2, 9.9, 10.1), (10.2, 10.4, 10.1, 10.3), (10.4, 10.6, 10.3, 10.5), (10.6, 10.62, 9.8, 10.62)]  # same shape after rising closes
    f = _flags(up); assert f.loc[3, "hanging_man"] and not f.loc[3, "hammer"]
def test_shooting_star_after_rise():
    rows = [(10, 10.2, 9.9, 10.1), (10.2, 10.4, 10.1, 10.3), (10.4, 10.6, 10.3, 10.5), (10.55, 11.4, 10.54, 10.58)]
    assert _flags(rows).loc[3, "shooting_star"]
def test_engulfing_patterns():
    bull = [(10.0, 10.1, 9.4, 9.5), (9.4, 10.3, 9.3, 10.2)]; bear = [(9.5, 10.1, 9.4, 10.0), (10.1, 10.2, 9.3, 9.4)]
    assert _flags(bull).loc[1, "bullish_engulfing"] and not _flags(bull).loc[1, "bearish_engulfing"]
    assert _flags(bear).loc[1, "bearish_engulfing"] and not _flags(bear).loc[1, "bullish_engulfing"]
    small = [(10.0, 10.1, 9.4, 9.5), (9.6, 9.9, 9.55, 9.8)]; assert not _flags(small).loc[1, "bullish_engulfing"]    # body not engulfing
def test_candlestick_flags_are_causal():
    d = market(); full = candlestick_flags(d.Open, d.High, d.Low, d.Close); cut = candlestick_flags(d.Open[:400], d.High[:400], d.Low[:400], d.Close[:400])
    assert (full.iloc[:400].values == cut.values).all()
def test_candlestick_strategy_runs():
    out = CandlestickPatternsStrategy().generate_signals(market(3000)); assert out["Buy Signal"].any() and out["Sell Signal"].any()
@pytest.mark.parametrize("S", [MovingAverageStrategy, RSIStrategy, MACDStrategy, BollingerBandsStrategy, MeanReversionExtremeStrategy, CandlestickPatternsStrategy, SmartMoneyConceptsStrategy])
@pytest.mark.parametrize("mode", ["long_only", "long_short"])
def test_every_strategy_backtests(S, mode):
    e = BacktestEngine(10000, mode); r = e.backtest(market(1500), S()); assert np.isfinite(e.analyze_results(r)["final_balance"])
