"""End-to-end tests that drive the real Textual app with its pilot (no terminal needed)."""
import asyncio
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("textual")

from textual.widgets import Button, DataTable, Input, Select, TabbedContent  # noqa: E402

from src.analytics.prodlive import queries as Q  # noqa: E402
from src.analytics.prodlive import sync as S  # noqa: E402
from src.tui import datasets  # noqa: E402
from src.tui.app import BatApp  # noqa: E402
from src.tui.pages.analytics import AnalyticsPage  # noqa: E402
from src.tui.widgets.modals import ConfirmModal  # noqa: E402
from src.tui.widgets.price import PriceChart  # noqa: E402
from tests.test_prodlive_analytics import SCHEMA  # noqa: E402

SIZE = (150, 46)


def run(coro):
    return asyncio.run(coro)


async def wait_for(pilot, cond, timeout=30.0, step=0.1):
    waited = 0.0
    while not cond():
        await pilot.pause(step)
        waited += step
        if waited > timeout:
            raise AssertionError("condition not met in time")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolate datasets and the production DB from the developer's real files."""
    d = tmp_path / "raw"
    d.mkdir()
    rng = np.random.RandomState(8)
    n = 2500
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.r_[c[0], c[:-1]]
    pd.DataFrame(dict(timestamp=pd.date_range("2025-03-03", periods=n, freq="5min"), Open=o, High=np.maximum(o, c) * 1.001,
                      Low=np.minimum(o, c) * 0.999, Close=c, Volume=rng.uniform(1, 9, n))
                 ).to_csv(d / "X_TEST_minute_2025-03-03_to_2025-03-12.csv", index=False)
    monkeypatch.setattr(datasets, "DATASET_DIRS", [d])
    missing = tmp_path / "none.db"
    monkeypatch.setattr(S, "LATEST", missing)
    monkeypatch.setattr(Q, "LATEST", missing)
    return tmp_path


def test_every_page_opens(workspace):
    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            assert app.theme == "bat"
            for i, page in enumerate(app.pages, 1):
                await pilot.press(str(i))
                await pilot.pause(0.15)
                assert app.active_page == page.id
            await pilot.press("question_mark")
            await pilot.pause(0.2)
            assert app.screen.__class__.__name__ == "HelpModal"
            await pilot.press("escape")
    run(go())


def test_backtest_flow_through_the_ui(workspace):
    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("2")
            page = app.query_one("#backtest")
            await wait_for(pilot, lambda: select_has_value(app, "#bt-dataset"))
            assert app.query_one("#bt-ticker", Input).value == "X:TEST"          # derived from the file name
            app.query_one("#bt-replay", Select).value = "0"                      # no animation: keep the test fast
            await pilot.pause(0.2)
            await pilot.click("#bt-run")                                         # the real button, not the action
            await wait_for(pilot, lambda: len(app.bat.backtests) == 1)
            rec = app.bat.backtests[0]
            assert rec.report.summary["bars"] == 2500 and rec.symbol == "X:TEST"
            await pilot.pause(0.3)
            card = app.query_one("#card-trades")
            assert card._value == f"{rec.report.summary['trades']:,}"
            tabs = app.query_one("#bt-tabs", TabbedContent)
            tabs.active = "tab-price"
            await pilot.pause(0.3)
            chart = app.query_one("#bt-price", PriceChart)
            assert "chart error" not in chart.render().plain and len(chart.render().plain) > 200
            chart.action_pan(1)
            chart.action_zoom(0.5)
            assert chart.view_len >= 20
            tabs.active = "tab-trades"
            await pilot.pause(0.3)
            table = app.query_one("#bt-trips", DataTable)
            assert table.row_count == len(rec.report.round_trips)
            # sort by P&L descending: header click must reorder rows
            page._sort_trips(type("E", (), {"column_key": type("K", (), {"value": "pnl"})()})())
            first = table.get_row_at(0)[7].plain
            assert first.startswith("+")
            tabs.active = "tab-analysis"
            await pilot.pause(0.2)
            tabs.active = "tab-history"
            await pilot.pause(0.2)
            assert app.query_one("#bt-history", DataTable).row_count == 1
    run(go())


def select_has_value(app, sel_id):
    v = app.query_one(sel_id, Select).value
    return isinstance(v, str) and v != ""


def test_backtest_validation_blocks_bad_params(workspace):
    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("2")
            await wait_for(pilot, lambda: select_has_value(app, "#bt-dataset"))
            app.query_one("#bt-strategy", Select).value = "rsi"
            await pilot.pause(0.4)
            app.query_one("#bt-p-oversold_threshold", Input).value = "80"        # >= overbought (70)
            await pilot.pause(0.2)
            app.query_one("#backtest").action_run()
            await pilot.pause(0.3)
            assert not app.bat.backtests and not app.query_one("#backtest")._bt_busy
            assert "Fix the strategy" in str(app.query_one("#bt-status").content)
    run(go())


def test_demo_live_session_trades_and_stops(workspace):
    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("3")
            app.query_one("#lv-interval", Input).value = "0.05"
            app.query_one("#lv-strategy", Select).value = "mean_reversion"
            await pilot.pause(0.4)
            app.query_one("#live").start_demo()
            await wait_for(pilot, lambda: app.bat.live and app.bat.live.last_snapshot and app.bat.live.last_snapshot.iteration >= 40,
                           timeout=60)
            snap = app.bat.live.last_snapshot
            assert snap.state == "running" and len(snap.bars) > 100
            chart = app.query_one("#lv-price", PriceChart)
            assert chart.follow and len(chart.df) == len(snap.bars)
            app.bat.live.stop()
            await wait_for(pilot, lambda: not app.bat.live.running)
            await pilot.pause(0.3)
            assert str(app.query_one("#lv-badge").content).startswith("STOPPED")
            assert app.query_one("#lv-stop", Button).disabled
    run(go())


def test_real_money_broker_needs_typed_confirmation(workspace):
    from src.execution.brokers import SimulatedBroker

    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            cm = app.bat.connections
            cm.brokers["alpaca"], cm.meta["alpaca"] = SimulatedBroker(1000), {"paper": False}
            await pilot.press("3")
            page = app.query_one("#live")
            page._refresh_sources()
            await pilot.pause(0.3)
            app.query_one("#lv-broker", Select).value = "alpaca"
            await pilot.pause(0.2)
            page._start()
            await pilot.pause(0.4)
            modal = app.screen
            assert isinstance(modal, ConfirmModal)
            ok = modal.query_one("#ok", Button)
            assert ok.disabled and "REAL MONEY" in str(modal._title)
            modal.query_one("#confirm-text", Input).value = "nope"
            await pilot.pause(0.1)
            assert ok.disabled
            modal.query_one("#confirm-text", Input).value = "TRADE LIVE"
            await pilot.pause(0.1)
            assert not ok.disabled
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert app.bat.live is None                                         # cancelling must not start anything
    run(go())


def test_analytics_empty_then_populated(workspace, monkeypatch):
    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("6")
            await pilot.pause(0.4)
            assert app.query_one("#an-empty").display and not app.query_one("#an-tabs").display
        # populate a snapshot, new app instance
        db = workspace / "latest.db"
        c = sqlite3.connect(db)
        c.executescript(SCHEMA)
        t0 = pd.Timestamp("2026-08-01", tz="UTC")
        for i in range(30):
            et, xt = t0 + pd.Timedelta(hours=6 * i), t0 + pd.Timedelta(hours=6 * i + 2)
            win = i % 3 != 0
            c.execute("insert into trades(account_id,side,oanda_trade_id,symbol,entry_time,entry_price,qty,exit_time,exit_price,realized_pnl,"
                      "financing,commission,entry_slippage_cost,exit_slippage_cost,outcome,r_multiple,holding_duration_seconds,created_at,"
                      "updated_at,entry_hour_utc) values('A','buy',?,?,?,1.1,1,?,1.1,?,0,0.1,0,0,?,?,7200,'x','x',?)",
                      (str(i), "EUR_USD" if i % 2 else "GBP_USD", et.isoformat(), xt.isoformat(), 10.0 if win else -10.0,
                       "TP" if win else "SL", 1.0 if win else -1.0, et.hour))
            c.execute("insert into account_snapshots values(?,?,?,?,0)", (i, xt.isoformat(), 1000 + i, 1000 + i))
        c.commit()
        c.close()
        monkeypatch.setattr(S, "LATEST", db)
        monkeypatch.setattr(Q, "LATEST", db)
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("6")
            await wait_for(pilot, lambda: app.query_one("#analytics").an_data)
            assert app.query_one("#an-tabs").display
            assert app.query_one("#card-n")._value == "30"
            assert app.query_one("#an-by-table", DataTable).row_count == 2
            app.query_one("#an-symbol", Select).value = "EUR_USD"
            await wait_for(pilot, lambda: len(app.query_one("#analytics").an_data["trades"]) == 15)
            app.query_one("#an-sql", Input).value = "select count(*) as n from trades"
            app.query_one("#analytics")._sql(type("E", (), {"value": "select count(*) as n from trades"})())
            await pilot.pause(0.2)
            assert app.query_one("#an-sql-table", DataTable).row_count == 1
            app.query_one("#analytics")._sql(type("E", (), {"value": "delete from trades"})())
            await pilot.pause(0.2)
            assert "readonly" in str(app.query_one("#an-sql-table", DataTable).get_row_at(0)[0]).lower()
    run(go())


def test_command_palette_actions_and_theme_toggle(workspace):
    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            app.cmd_toggle_theme()
            await pilot.pause(0.1)
            assert app.theme == "bat-light"
            app.cmd_toggle_theme()
            assert app.theme == "bat"
            app.action_toggle_sidebar()
            assert app.query_one("#nav").has_class("collapsed")
    run(go())


def test_run_button_streams_a_replay_and_logs_every_step(workspace):
    from src.tui.widgets.common import EventLog

    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("2")
            page = app.query_one("#backtest")
            await wait_for(pilot, lambda: select_has_value(app, "#bt-dataset"))
            app.query_one("#bt-strategy", Select).value = "moving_average"
            await pilot.pause(0.4)
            app.query_one("#bt-replay", Select).value = "2"
            await pilot.pause(0.2)
            await pilot.click("#bt-run")
            await wait_for(pilot, lambda: page._bt_busy)
            assert app.query_one("#bt-run", Button).disabled                      # cannot double-start
            await wait_for(pilot, lambda: page._bt_frames >= 3)                   # frames arrive while running
            assert not app.bat.backtests and "Replaying" in str(app.query_one("#bt-status").content)
            assert not app.query_one("#bt-skip", Button).disabled
            await wait_for(pilot, lambda: len(app.bat.backtests) == 1, timeout=60)
            assert page._bt_frames >= 10 and not app.query_one("#bt-run", Button).disabled
            text = "\n".join(t for _, _, _, t in app.query_one("#bt-runlog", EventLog)._history)
            for needle in ("Loaded 2,500 bars", "Signals:", "Simulated", "Replaying the simulation", "Done"):
                assert needle in text, needle
            # the replay's last frame must equal the final report shown afterwards
            rec = app.bat.backtests[0]
            assert app.query_one("#card-trades")._value == f"{rec.report.summary['trades']:,}"
    run(go())


def test_skip_jumps_to_the_final_result(workspace):
    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("2")
            page = app.query_one("#backtest")
            await wait_for(pilot, lambda: select_has_value(app, "#bt-dataset"))
            app.query_one("#bt-replay", Select).value = "15"                      # slow replay...
            await pilot.pause(0.2)
            await pilot.click("#bt-run")
            await wait_for(pilot, lambda: page._bt_frames >= 2)
            await pilot.click("#bt-skip")                                         # ...then skip it
            await wait_for(pilot, lambda: len(app.bat.backtests) == 1, timeout=10)
            assert page._bt_frames < 40
    run(go())


def test_data_problems_are_reported_not_swallowed(workspace):
    from src.tui.widgets.common import EventLog

    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("2")
            page = app.query_one("#backtest")
            await wait_for(pilot, lambda: select_has_value(app, "#bt-dataset"))
            app.query_one("#bt-resample", Select).value = "1D"                    # 2,500 5-min bars -> ~9 daily bars
            await pilot.pause(0.2)
            await pilot.click("#bt-run")
            await wait_for(pilot, lambda: str(app.query_one("#bt-status").content).startswith("✗"))
            log = "\n".join(t for _, lv, _, t in app.query_one("#bt-runlog", EventLog)._history if lv == "error")
            assert "needs more than" in log and "stopped" in log
            assert not app.bat.backtests and not page._bt_busy and not app.query_one("#bt-run", Button).disabled
    run(go())


def test_bad_form_input_is_rejected_visibly(workspace):
    from src.tui.widgets.common import EventLog

    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("2")
            await wait_for(pilot, lambda: select_has_value(app, "#bt-dataset"))
            app.query_one("#bt-balance", Input).value = "abc"
            await pilot.pause(0.1)
            await pilot.click("#bt-run")
            await pilot.pause(0.3)
            assert "not a number" in str(app.query_one("#bt-status").content)
            assert any("not a number" in t for _, _, _, t in app.query_one("#bt-runlog", EventLog)._history)
            assert not app.bat.backtests
    run(go())


def _handles_button(cls, bid):
    hs = getattr(cls, "_decorated_handlers", {}).get(Button.Pressed, [])
    for _, spec in hs:
        sels = spec.get("control") or ()
        if not sels or any(f"'#{bid}'" in str(sel) for sel in sels):
            return True
    return hasattr(cls, "on_button_pressed")


def test_every_page_button_has_a_handler(workspace):
    """Regression: the Backtest 'Run' button once existed but was wired to nothing."""
    async def go():
        app = BatApp()
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause(0.3)
            dead = [(p.id, b.id) for p in app.pages for b in app.query_one(f"#{p.id}").query(Button)
                    if not _handles_button(type(app.query_one(f"#{p.id}")), b.id or "")]
            assert dead == []
    run(go())


@pytest.mark.parametrize("size", [(100, 30), (120, 34), (170, 52)])
def test_primary_action_buttons_are_visible_without_scrolling(workspace, size):
    """Regression: long forms pushed 'Run' below the fold on ordinary terminal sizes."""
    checks = [("backtest", "#bt-run", None), ("live", "#lv-start", None), ("mllab", "#ml-start", None),
              ("research", "#op-run", ("#rs-tabs", "rs-tab-opt"))]

    async def go():
        app = BatApp()
        async with app.run_test(size=size) as pilot:
            for page_id, button, tab in checks:
                app.goto(page_id)
                await pilot.pause(0.3)
                if tab:
                    app.query_one(tab[0], TabbedContent).active = tab[1]
                    await pilot.pause(0.3)
                region = app.query_one(button).region
                # status bar + footer occupy the last two rows
                assert region.height >= 1 and region.y >= 1 and region.bottom <= size[1] - 2, (page_id, region)
    run(go())
