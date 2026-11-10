"""Pure-logic tests for the terminal UI: charts, report, registry, optimizer, services."""
import threading
import time

import numpy as np
import pandas as pd
import pytest

from src.core.application.backtest import BacktestService
from src.core.domain.models import BacktestConfig
from src.tui import charts as C
from src.tui.datasets import dataset_stats, inspect_csv, load_dataset, resample_bars, short_name
from src.tui.report import build_report, equity_curve, pair_round_trips
from src.tui.services import mlops
from src.tui.services.capture import capture_thread_output, classify
from src.tui.services.connections import BrokerShim
from src.tui.services.demo import DemoFeed
from src.tui.services.live import LiveConfig, LiveSession, make_bar_source, round_trips_from_engine
from src.tui.services.optimizer import build_grid, parse_range, run_grid
from src.tui.strategies import STRATEGIES, ParamSpec, split_indicators


@pytest.fixture
def ohlcv():
    rng = np.random.RandomState(3)
    n = 1500
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(dict(
        timestamp=pd.date_range("2025-03-03", periods=n, freq="5min"), Open=o, Close=c,
        High=np.maximum(o, c) * 1.001, Low=np.minimum(o, c) * 0.999, Volume=rng.uniform(1, 9, n)))


# ------------------------------------------------------------------ charts
def test_fmt_helpers():
    assert C.fmt_num(93_800) == "93.8K" and C.fmt_num(0) == "0" and C.fmt_num(float("nan")) == "–"
    assert C.fmt_money(-1234.5, signed=True) == "-$1,234.50" and C.fmt_pct(1.234) == "+1.23%"
    assert C.sparkline([1, 2, 3, 4], 4) == "▁▃▆█" and C.sparkline([]) == ""


def test_line_chart_has_requested_size():
    t = C.line_chart({"a": np.sin(np.linspace(0, 6, 200)), "b": np.cos(np.linspace(0, 6, 200))}, 80, 12,
                     x_labels=["a", "b", "c"], hline=0.0)
    lines = t.plain.split("\n")
    assert len(lines) == 12 and all(len(l) <= 80 for l in lines)
    assert any(0x2800 < ord(ch) <= 0x28FF for ch in t.plain)          # braille dots were drawn


def test_line_chart_handles_nan_and_flat_series():
    assert "no data" in C.line_chart({}, 40, 8).plain
    flat = C.line_chart({"x": [5.0] * 50}, 40, 8)
    assert len(flat.plain.split("\n")) == 8
    C.line_chart({"x": [1, float("nan"), 3, 4]}, 40, 8)                # must not raise


def test_candlestick_and_price_stack(ohlcv):
    df = ohlcv.head(60)
    t = C.candlestick_chart(df, 90, 16, markers=[(10, "buy"), (20, "sell"), (30, "exit")])
    assert len(t.plain.split("\n")) == 16
    assert "▲" in t.plain and "▼" in t.plain and "◆" in t.plain
    stack, lay, gutter = C.price_stack(ohlcv.head(300).assign(rsi=50.0), 100, 30, panes=["rsi"], hover=5, title="T")
    assert len(stack.plain.split("\n")) == 30 and lay.slots <= 100 - gutter and gutter > 3
    assert lay.bar_at_cell(gutter + 0) is not None


def test_layout_maps_bars_to_columns():
    lay = C.Layout.build(40, 100)                    # few bars -> spaced candles
    assert lay.step == 2 and lay.slots == 40
    dense = C.Layout.build(1000, 100)                # many bars -> aggregated, one per column
    assert dense.step == 1 and dense.slots == 100 and dense.edges[0] == 0 and dense.edges[-1] == 1000
    assert dense.bar_at_cell(500) is None


def test_bar_chart_labels_zero_and_histogram_heatmap():
    t = C.bar_chart([5, -3, 2], 40, 8, labels=["a", "b", "c"], title="t")
    assert "0" in t.plain
    assert "no data" in C.histogram([], 40, 8).plain
    hm = C.heatmap(np.array([[1.0, -2.0], [np.nan, 3.0]]), ["r1", "r2"], ["c1", "c2"], mark_best=True)
    assert "★" in hm.plain


# ------------------------------------------------------------------ report
def _run(ohlcv, mode="long_only"):
    strat = STRATEGIES["rsi"].build({"window": 14, "oversold_threshold": 40, "overbought_threshold": 60})
    res = BacktestService().run(ohlcv, strat, BacktestConfig(trading_mode=mode, symbol="X:TEST"))
    return res, strat


@pytest.mark.parametrize("mode", ["long_only", "long_short"])
def test_report_reconciles_with_engine(ohlcv, mode):
    res, _ = _run(ohlcv, mode)
    assert len(res.trades) > 4
    rep = build_report(res.trades, res.signal_data, 10_000.0)
    last = res.trades.groupby("Index").tail(1)
    assert np.allclose(rep.equity.to_numpy()[last["Index"].to_numpy()], last["Total_Account_Worth"].to_numpy())
    rt_pnl = rep.round_trips["pnl"].sum()
    assert rt_pnl == pytest.approx(res.portfolio.realized_profit, abs=1e-6)
    s = rep.summary
    assert s["trades"] == len(rep.round_trips) and -1 <= s["max_drawdown"] <= 0
    assert s["final"] == pytest.approx(rep.equity.iloc[-1]) and 0 <= s["exposure"] <= 1


def test_report_without_trades(ohlcv):
    rep = build_report(pd.DataFrame(), ohlcv, 5_000.0)
    assert rep.summary["trades"] == 0 and rep.summary["final"] == 5_000.0
    assert (rep.equity == 5_000.0).all() and rep.round_trips.empty and rep.open_trade is None


def test_round_trips_pair_and_flag_open_trade(ohlcv):
    res, _ = _run(ohlcv, "long_only")
    rt, open_trade = pair_round_trips(res.trades, res.signal_data)
    closes = res.trades["Action"].isin(["CLOSE", "CLOSE_SHORT"]).sum()
    assert len(rt) == closes
    assert (open_trade is not None) == (res.portfolio.quantity != 0)


# --------------------------------------------------------------- registry
def test_every_strategy_builds_with_defaults(ohlcv):
    for key, spec in STRATEGIES.items():
        strat = spec.build()
        out = strat.generate_signals(ohlcv.copy())
        assert {"Buy Signal", "Sell Signal"} <= set(out.columns), key


def test_param_validation_messages():
    rsi, macd = STRATEGIES["rsi"], STRATEGIES["macd"]
    with pytest.raises(ValueError, match="Oversold"):
        rsi.parse({"window": 14, "oversold_threshold": 70, "overbought_threshold": 30})
    with pytest.raises(ValueError, match="Fast"):
        macd.parse({"fast": 30, "slow": 10, "signal": 9})
    with pytest.raises(ValueError, match="whole number"):
        rsi.parse({"window": "2.5", "oversold_threshold": 30, "overbought_threshold": 70})
    with pytest.raises(ValueError, match="not a valid"):
        rsi.parse({"window": "abc"})
    assert STRATEGIES["moving_average"].warnings({"short_window": 9, "medium_window": 5, "long_window": 25})


def test_split_indicators_classifies_columns(ohlcv):
    df = STRATEGIES["bollinger"].build().generate_signals(ohlcv.copy())
    parts = split_indicators(df, STRATEGIES["bollinger"].build().get_indicators())
    assert set(parts["overlay"]) == {"bb_upper", "bb_middle", "bb_lower"}
    df = STRATEGIES["rsi"].build().generate_signals(ohlcv.copy())
    assert split_indicators(df, ["rsi"])["pane"] == ["rsi"]
    df = STRATEGIES["candlestick"].build().generate_signals(ohlcv.copy())
    assert split_indicators(df, STRATEGIES["candlestick"].build().get_indicators())["flags"]


# --------------------------------------------------------------- optimizer
def test_parse_range_and_grid():
    p = ParamSpec("w", "Window", "int", 20, 1, 100, 5)
    assert parse_range("10:30:10", p) == [10, 20, 30] and parse_range("5,7,9", p) == [5, 7, 9]
    assert parse_range("", p) == [20]
    with pytest.raises(ValueError):
        parse_range("10:30:0", p)
    with pytest.raises(ValueError, match="≤"):
        parse_range("90:120:10", p)
    macd = STRATEGIES["macd"]
    grid = build_grid(macd, {"fast": [5, 30], "slow": [10, 20], "signal": [9]})
    assert all(g["fast"] < g["slow"] for g in grid) and len(grid) == 2         # (30, *) dropped


def test_run_grid_train_test_and_cancel(ohlcv):
    spec = STRATEGIES["rsi"]
    combos = build_grid(spec, {"window": [7, 14], "oversold_threshold": [30, 40], "overbought_threshold": [60, 70]})
    cfg = BacktestConfig(symbol="X:TEST")
    seen = []
    res = run_grid(ohlcv, spec, combos, cfg, split=0.7, objective="total_return",
                   progress=lambda i, n, el: seen.append((i, n)))
    assert len(res) == len(combos) == seen[-1][0] and {"train_sharpe", "test_total_return", "test_trades"} <= set(res.columns)
    assert res["train_total_return"].is_monotonic_decreasing                    # ranked by the objective
    stop = threading.Event()
    stop.set()
    assert run_grid(ohlcv, spec, combos, cfg, cancel=stop).empty


# ---------------------------------------------------------------- services
def test_broker_shim_drops_unsupported_kwargs():
    class Alpacaish:                       # mirrors AlpacaBroker: no current_price kwarg
        def buy(self, symbol, quantity, order_type="market", limit_price=None):
            return {"status": "ok", "symbol": symbol, "qty": quantity, "type": order_type}

        def close_position(self, symbol, cancel_orders=True):
            return {"status": "closed", "symbol": symbol}

        def get_account(self):
            return {"equity": 1}

    class Flexible:
        def sell(self, symbol, quantity, **kw):
            return kw

    shim = BrokerShim(Alpacaish())
    assert shim.buy("BTC", 1, order_type="market", current_price=5.0)["status"] == "ok"
    assert shim.close_position("BTC", current_price=5.0)["status"] == "closed"
    assert shim.get_account() == {"equity": 1} and not hasattr(shim, "get_account_api")
    assert BrokerShim(Flexible()).sell("X", 1, current_price=7) == {"current_price": 7}


def test_capture_routes_only_registered_threads(capsys):
    got = []

    def worker():
        with capture_thread_output(got.append):
            print("hello from worker")
            print("two\nlines")

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    print("main thread output")
    assert got == ["hello from worker", "two", "lines"]
    assert "main thread output" in capsys.readouterr().out


def test_classify_levels():
    assert classify("     Symbol: BTC") == "detail" and classify("BUY ORDER FAILED - x") == "error"
    assert classify("[BUY] 1 BTC @ 1") == "trade" and classify("🔵 BUY SIGNAL - go") == "signal"


def test_demo_feed_is_deterministic_and_advances():
    a, b = DemoFeed(seed=1), DemoFeed(seed=1)
    h1, h2 = a.history("X", 50), b.history("X", 50)
    assert h1.equals(h2) and (h1["High"] >= h1["Low"]).all()
    nxt = a.latest("X")
    assert nxt["timestamp"] > h1["timestamp"].iloc[-1] and nxt["Open"] == pytest.approx(h1["Close"].iloc[-1])


def test_live_session_runs_headless_and_accounts_consistently():
    from src.execution.brokers import SimulatedBroker
    snaps, events = [], []
    cfg = LiveConfig(strategy=STRATEGIES["mean_reversion"].build(), symbol="BTC/USD", source=make_bar_source(DemoFeed(seed=3)),
                     broker=BrokerShim(SimulatedBroker(10_000)), trading_mode="long_short", position_pct=20,
                     interval=0.001, lookback=300, initial_balance=10_000, max_iterations=120)
    s = LiveSession(cfg, snaps.append, events.append)
    s.start()
    deadline = time.time() + 60
    while s.running and time.time() < deadline:
        time.sleep(0.05)
    assert s.state == "stopped"
    last = snaps[-1]
    assert last.iteration == 120 and len(last.bars) > 100 and len(last.equity) > 100
    assert any("Loaded" in e.text for e in events)
    unreal = float(last.position.get("unrealized_pl", 0) or 0)
    equity = float(last.account["equity"])
    assert equity - 10_000 == pytest.approx(last.perf["realized"] + unreal, abs=1e-6)
    assert last.perf["trades"] == last.perf["wins"] + last.perf["losses"]


def test_live_session_close_and_stop_actions():
    from src.execution.brokers import SimulatedBroker
    snaps, events = [], []
    cfg = LiveConfig(strategy=STRATEGIES["rsi"].build({"window": 5, "oversold_threshold": 45, "overbought_threshold": 55}),
                     symbol="BTC/USD", source=make_bar_source(DemoFeed(seed=5)), broker=BrokerShim(SimulatedBroker(10_000)),
                     position_pct=20, interval=0.05, lookback=300, initial_balance=10_000)
    s = LiveSession(cfg, snaps.append, events.append)
    s.start()
    deadline = time.time() + 30
    while time.time() < deadline and not any(float((x.position or {}).get("qty", 0) or 0) != 0 for x in snaps):
        time.sleep(0.05)
    s.stop(close_position=True)
    while s.running and time.time() < deadline:
        time.sleep(0.05)
    assert not s.running and s.state == "stopped"
    assert float(snaps[-1].position.get("qty", 0) or 0) == 0                    # closed before stopping


def test_round_trips_from_engine_pairs_shorts():
    trades = [dict(timestamp=1, action="sell_short", price=100.0, quantity=2),
              dict(timestamp=2, action="close_short", price=90.0, quantity=2)]
    rt = round_trips_from_engine(trades)
    assert rt["pnl"].dropna().tolist() == [20.0] and rt["result"].tolist() == ["OPEN", "Win"]


# ---------------------------------------------------------------- datasets
def test_dataset_inspect_load_stats(tmp_path):
    ts = pd.date_range("2025-01-01", periods=300, freq="min")
    ts = ts.delete(range(100, 130))                                             # a gap
    df = pd.DataFrame(dict(timestamp=ts, Open=1.0, High=1.2, Low=0.9, Close=1.1, Volume=5.0))
    p = tmp_path / "X_TEST_minute_2025-01-01_to_2025-01-02.csv"
    df.to_csv(p, index=False)
    info = inspect_csv(p)
    assert info.rows == len(df) and info.timeframe == "1m" and info.start == ts[0]
    loaded = load_dataset(p)
    assert list(loaded.columns[:6]) == ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    st = dataset_stats(loaded)
    assert st["gaps"] == 1 and st["duplicates"] == 0 and st["bad_ohlc"] == 0
    hourly = resample_bars(loaded, "1h")
    assert len(hourly) <= 6 and hourly["Volume"].sum() == loaded["Volume"].sum()
    assert load_dataset(p, start="2025-01-01", end="2025-01-01").shape[0] == len(loaded)
    assert short_name("a" * 50, 20).count("…") == 1 and len(short_name("a" * 50, 20)) == 20


def test_inspect_csv_rejects_non_ohlcv(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    assert inspect_csv(p) is None


# ------------------------------------------------------------------- mlops
def test_overrides_only_contain_changes():
    base = {"data": {"resample": "15min", "seq_len": 64}, "model": {"name": "lstm"}, "train": {"epochs": 30}}
    new = {"data": {"resample": None, "seq_len": 64}, "model": {"name": "transformer"}, "train": {"epochs": 30}}
    assert sorted(mlops.overrides_from(new, base)) == ["data.resample=null", "model.name=transformer"]
    assert mlops.overrides_from(base, base) == []


def test_parse_result_lines_and_run_listing(tmp_path):
    rows = mlops.parse_result_lines(["noise", "{'fold': 0, 'acc': 0.5}", "{bad", "{'fold': 1, 'acc': 0.6}"])
    assert [r["fold"] for r in rows] == [0, 1]
    run = tmp_path / "r1"
    run.mkdir()
    (run / "metrics.jsonl").write_text('{"epoch":0,"train_loss":0.9,"val_loss":0.8}\n{"epoch":1,"train_loss":0.7,"val_loss":0.85}\n')
    (tmp_path / "empty").mkdir()
    runs = mlops.list_runs(tmp_path)
    assert len(runs) == 1 and runs[0].best_val == 0.8 and runs[0].best_epoch == 0 and runs[0].epochs == 2


def test_job_streams_and_cancels(tmp_path):
    lines, done = [], []
    job = mlops.Job(["python3", "-c", "print('a'); print('b')"], lines.append, done.append, cwd=tmp_path)
    job.start()
    t0 = time.time()
    while not done and time.time() - t0 < 20:
        time.sleep(0.05)
    assert lines == ["a", "b"] and done == [0]
    lines.clear(), done.clear()
    slow = mlops.Job(["python3", "-c", "import time; print('x', flush=True); time.sleep(30)"], lines.append, done.append, cwd=tmp_path)
    slow.start()
    time.sleep(0.5)
    slow.cancel()
    t0 = time.time()
    while not done and time.time() - t0 < 10:
        time.sleep(0.05)
    assert done == [-15]
