"""Run-log diagnostics: every way a backtest can silently produce nothing must be explained."""
import numpy as np
import pandas as pd
import pytest

from src.tui.diagnostics import (check_config, check_data, check_result, check_signals, explain_exception)

NAMES = {"buy": "Buy Signal", "sell": "Sell Signal"}


def bars(n=300):
    c = 100 + np.sin(np.arange(n) / 10)
    return pd.DataFrame(dict(timestamp=pd.date_range("2025-01-01", periods=n, freq="min"), Open=c, High=c + 0.2,
                             Low=c - 0.2, Close=c, Volume=1.0))


def texts(findings, level=None):
    return [f.text for f in findings if level is None or f.level == level]


def test_data_missing_columns_and_empty_block_the_run():
    f = check_data(bars().drop(columns=["Volume", "timestamp"]))
    assert f[0].blocking and "Volume" in f[0].text and "timestamp" in f[0].text
    assert check_data(None)[0].blocking and check_data(bars().iloc[0:0])[0].blocking


def test_data_shorter_than_strategy_lookback_blocks():
    f = check_data(bars(30), lookback=50)
    assert len(f) == 1 and f[0].blocking and "30" in f[0].text and "50" in f[0].text


def test_data_quality_warnings():
    df = bars(200)
    df.loc[5, "Close"] = np.nan
    df.loc[6, "High"] = df.loc[6, "Low"] - 1
    df = pd.concat([df, df.iloc[[10]]])                       # duplicate timestamp
    df = df.drop(index=range(50, 80))                         # a gap
    out = check_data(df.reset_index(drop=True))
    joined = " ".join(texts(out, "warn") + texts(out, "info"))
    assert not any(f.blocking for f in out)
    for needle in ("NaN", "inconsistent OHLC", "duplicate", "gap", "not in increasing order"):
        assert needle in joined
    assert texts(out, "ok")[0].startswith("Loaded 171 bars")   # 200 + 1 duplicate - 30 dropped; summary is always there


def test_short_sample_warning():
    assert any("Very short" in t for t in texts(check_data(bars(60)), "warn"))


def test_text_valued_indicators_are_reported_as_a_strategy_bug():
    sig = bars(100)
    sig["macd_line"] = "macd"                                  # what the MACD strategy currently produces
    sig["Buy Signal"], sig["Sell Signal"] = False, False
    out = check_signals(sig, NAMES, ["macd_line"], "long_only")
    errors = texts(out, "error")
    assert errors and "macd_line" in errors[0] and "bug in the strategy" in errors[0]
    assert any("0 buy/sell signals" in t for t in texts(out, "warn"))


@pytest.mark.parametrize("mode,buy,sell,expect", [
    ("long_only", 0, 5, "needs Buy signals"),
    ("long_only", 5, 0, "held to the end"),
    ("long_short", 0, 0, "0 buy/sell signals"),
])
def test_signal_findings(mode, buy, sell, expect):
    sig = bars(100)
    sig["Buy Signal"] = False
    sig["Sell Signal"] = False
    sig.loc[sig.index[:buy], "Buy Signal"] = True
    sig.loc[sig.index[50:50 + sell], "Sell Signal"] = True
    out = check_signals(sig, NAMES, [], mode)
    assert any(expect in f.text for f in out)


def test_warmup_all_nan_and_conflicting_signals():
    sig = bars(100)
    sig["sma"] = np.r_[np.full(20, np.nan), np.ones(80)]
    sig["dead"] = np.nan
    sig["Buy Signal"], sig["Sell Signal"] = False, False
    sig.loc[30:32, ["Buy Signal", "Sell Signal"]] = True
    out = check_signals(sig, NAMES, ["sma", "dead"], "long_short")
    joined = " ".join(f.text for f in out)
    assert "first 20 bars" in joined and "dead" in joined and "together on 3 bars" in joined


def test_result_findings_explain_empty_and_open_positions():
    none_summary = dict(trades=0, days=100.0)
    assert any("no signals" in t for t in texts(check_result(none_summary, 0, 0, "long_only", None), "warn"))
    assert any("never closed" in t for t in texts(check_result(none_summary, 3, 0, "long_only",
                                                               dict(side="LONG", unrealized=-5.0)), "warn"))
    assert any("none turned into a trade" in t for t in texts(check_result(none_summary, 0, 4, "long_only", None), "warn"))
    ok = check_result(dict(trades=5, days=10.0), 5, 5, "long_only", dict(side="LONG", unrealized=1.5))
    assert any("Short sample" in f.text for f in ok) and any("still open" in f.text for f in ok)
    assert any("no commissions" in f.text for f in ok)       # the optimism caveat is always shown


def test_config_notes():
    assert any("ignored" in f.text for f in check_config("X:BTCUSD", 1.0, "long_only", 100))
    assert any("applied" in f.text for f in check_config("C:EURUSD", 1.0, "long_only", 100))
    assert any("compound" in f.text for f in check_config("AAPL", 0, "long_short", 100))
    assert check_config("AAPL", 0, "long_only", 100) == []


def test_explain_exception_gives_actionable_hints():
    assert "Rescan" in explain_exception(FileNotFoundError("x.csv"))
    assert "Connections" in explain_exception(RuntimeError("Connection timeout talking to api"))
    assert "credentials" in explain_exception(RuntimeError("API request failed with status code 401"))
    assert explain_exception(ValueError("bad value")) == "bad value"
