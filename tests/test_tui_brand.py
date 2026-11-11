"""Home header: the PnL ticker's data sources, its rendering, and its animation controls."""
import types

import numpy as np
import pandas as pd
import pytest
from rich.text import Text

pytest.importorskip("textual")

from src.tui import charts as C  # noqa: E402
from src.tui.app import BatApp  # noqa: E402
from src.tui.services.ticker import DEMO_START, TickerFeed, TickerFrame  # noqa: E402
from src.tui.widgets.brand import BrandBox, render_compact, render_full, spark_text  # noqa: E402
from tests.test_tui_app import run, wait_for, workspace  # noqa: E402,F401


def backtest_record(rec_id=1, label="SMA cross", n=2000, initial=10_000.0, seed=2):
    eq = pd.Series(initial + np.cumsum(np.random.default_rng(seed).normal(0, 20, n)))
    return types.SimpleNamespace(id=rec_id, label=label, report=types.SimpleNamespace(equity=eq, initial_cash=initial))


def live_session(equity, initial=10_000.0, running=True):
    # The real status bar reads this object too, so it carries the attributes a LiveSession has.
    return types.SimpleNamespace(running=running, equity=[(i, float(v)) for i, v in enumerate(equity)], state="running",
                                 last_snapshot=None, cfg=types.SimpleNamespace(initial_balance=initial, symbol="TEST"))


def state(live=None, backtests=()):
    return types.SimpleNamespace(live=live, backtests=list(backtests))


# ---------------------------------------------------------------------------------------------------------- data feed
def test_with_nothing_to_show_the_feed_is_a_clearly_labelled_demo():
    feed = TickerFeed(state(), rng=np.random.default_rng(1))
    frames = [feed.step() for _ in range(400)]
    assert {f.source for f in frames} == {"DEMO"} and frames[0].label == "simulated"
    assert all(len(f.values) == 28 for f in frames)
    assert max(abs(f.pnl) for f in frames) < DEMO_START * 0.05, "the random walk must stay a plausible size"
    assert len({round(f.pnl, 2) for f in frames}) > 100, "it has to actually move"


def test_demo_has_history_on_the_very_first_frame():
    first = TickerFeed(state(), rng=np.random.default_rng(3)).step()
    assert len(set(first.values)) > 5, "a flat first frame would draw an empty chart"


def test_same_seed_gives_the_same_demo():
    a, b = (TickerFeed(state(), rng=np.random.default_rng(5)) for _ in range(2))
    assert [a.step().pnl for _ in range(20)] == [b.step().pnl for _ in range(20)]


def test_latest_backtest_is_replayed_with_its_own_label_and_starting_cash():
    old, new = backtest_record(1, "old"), backtest_record(2, "new strategy", initial=50_000.0)
    f = TickerFeed(state(backtests=[old, new])).step()
    assert f.source == "REPLAY" and f.label == "new strategy"
    assert abs(f.pnl - (f.values[-1] - 50_000.0)) < 1e-6, "PnL is measured from that run's initial cash"


def test_replay_walks_through_the_curve_and_loops():
    feed = TickerFeed(state(backtests=[backtest_record(n=1000)]), replay_steps=50)
    last = [feed.step().values[-1] for _ in range(120)]
    assert len(set(last)) > 30, "the replay must advance"
    assert feed._cursor < 1000, "and wrap around instead of running off the end"


def test_a_new_backtest_restarts_the_replay():
    st = state(backtests=[backtest_record(1)])
    feed = TickerFeed(st, replay_steps=40)
    for _ in range(30):
        feed.step()
    mid = feed._cursor
    st.backtests.append(backtest_record(2, "second"))
    feed.step()
    assert feed._cursor < mid and feed._replay_id == 2


def test_a_running_live_session_wins_over_replay_and_demo():
    f = TickerFeed(state(live_session([10_000, 10_050, 10_020]), [backtest_record()])).step()
    assert f.source == "LIVE" and f.pnl == 20.0 and f.values == [10_000.0, 10_050.0, 10_020.0]
    assert f.pct == pytest.approx(0.2)


@pytest.mark.parametrize("live", [live_session([10_000, 10_010], running=False), live_session([10_000]), None])
def test_a_stopped_or_empty_live_session_falls_back_to_replay(live):
    assert TickerFeed(state(live, [backtest_record()])).step().source == "REPLAY"


def test_direction_follows_the_last_tick():
    st = state(live_session([100.0, 101.0]), [])
    feed = TickerFeed(st)
    assert feed.step().direction == 0, "nothing to compare with on the first frame"
    st.live.equity.append((2, 102.0))
    assert feed.step().direction == 1
    st.live.equity.append((3, 101.0))
    assert feed.step().direction == -1


# --------------------------------------------------------------------------------------------------------- rendering
def frame(pnl=179.54, direction=1, source="DEMO", label="simulated", values=None):
    return TickerFrame(values or list(np.linspace(1, 8, 28)), pnl, pnl / 1000, direction, source, label)


def test_full_render_is_logo_then_sparkline_then_pnl_then_source():
    lines = render_full(frame()).plain.split("\n")
    assert lines[:5] == C.LOGO_ROWS and len(lines) == 8
    assert set(lines[5]) <= set("▁▂▃▄▅▆▇█") and lines[5]
    assert "+$179.54" in lines[6] and "▲" in lines[6]
    assert lines[7] == "DEMO · simulated"


def test_the_source_is_always_visible_so_demo_money_cannot_pass_for_real():
    for src in ("DEMO", "REPLAY", "LIVE"):
        assert src in render_full(frame(source=src)).plain and src in render_compact(frame(source=src)).plain


def test_pnl_and_arrow_colours_follow_the_numbers():
    def style_at(text: Text, needle: str) -> str:
        i = text.plain.index(needle)
        return " ".join(str(s.style) for s in text.spans if s.start <= i < s.end).lower()
    down = render_full(frame(pnl=-50.0, direction=-1))
    assert "-$50.00" in down.plain and "#ff4d4d" in style_at(down, "-$50.00") and "#ff4d4d" in style_at(down, "▼")
    up = render_full(frame(pnl=50.0, direction=1))
    assert "#00e676" in style_at(up, "+$50.00") and "#00e676" in style_at(up, "▲")


def test_compact_is_a_single_line_and_marks_paused():
    t = render_compact(frame(), paused=True)
    assert "\n" not in t.plain and "(paused)" in t.plain and t.plain.startswith("BAT ")
    assert "(paused)" in render_full(frame(), paused=True).plain


def test_sparkline_handles_flat_and_empty_data():
    assert spark_text([], 10, 1).plain == ""
    assert spark_text([5.0] * 6, 10, 1).plain == "▁" * 6
    assert len(spark_text(list(range(100)), 10, 1).plain) == 10
    assert spark_text([1, 2, float("nan"), 3], 10, 1).plain


# ------------------------------------------------------------------------------------------------------- in the app
def brand_text(app, wid="#home-logo"):
    return str(app.query_one(wid, BrandBox).render())


def test_the_header_animates_while_home_is_open(workspace, monkeypatch):
    monkeypatch.setenv("BAT_NO_ANIM", "0")

    async def go():
        app = BatApp()
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause(0.3)
            seen = set()
            for _ in range(12):
                seen.add(brand_text(app))
                await pilot.pause(0.2)
            assert len(seen) > 4, "the logo block never changed, so nothing is animating"
            assert all("DEMO" in s for s in seen)
    run(go())


def test_it_costs_nothing_while_another_page_is_showing(workspace, monkeypatch):
    monkeypatch.setenv("BAT_NO_ANIM", "0")

    async def go():
        app = BatApp()
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause(0.3)
            home = app.query_one("#home")
            calls = []
            real = home.feed.step
            home.feed.step = lambda: (calls.append(1), real())[1]
            await pilot.press("2")
            await pilot.pause(0.2)
            calls.clear()
            await pilot.pause(1.0)
            assert not calls, "the feed kept running behind a hidden page"
            await pilot.press("1")
            await pilot.pause(0.6)
            assert calls, "it should resume when Home is shown again"
    run(go())


def test_toggle_command_pauses_and_resumes(workspace, monkeypatch):
    monkeypatch.setenv("BAT_NO_ANIM", "0")

    async def go():
        app = BatApp()
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause(0.3)
            assert app.brand_anim is True
            app.cmd_toggle_brand_anim()
            await pilot.pause(0.1)
            frozen = brand_text(app)
            assert "(paused)" in frozen
            await pilot.pause(0.8)
            assert brand_text(app) == frozen, "still moving after being paused"
            app.cmd_toggle_brand_anim()
            await pilot.pause(0.1)
            assert "(paused)" not in brand_text(app)
    run(go())


def test_env_var_starts_paused_and_the_command_is_in_the_palette(workspace, monkeypatch):
    monkeypatch.setenv("BAT_NO_ANIM", "1")

    async def go():
        app = BatApp()
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause(0.3)
            assert app.brand_anim is False and "(paused)" in brand_text(app)
            from src.tui.app import BatCommands
            from textual.screen import Screen
            cmds = BatCommands(Screen(), match_style=None)._commands()
            assert "Toggle logo animation" in [c[0] for c in cmds]
    run(go())


def test_wide_terminals_show_the_big_logo_and_narrow_ones_the_one_line_ticker(workspace):
    async def go(size):
        app = BatApp()
        async with app.run_test(size=size) as pilot:
            await pilot.pause(0.4)
            return app.query_one("#home-logo").display, app.query_one("#home-ticker").region.height
    wide_logo, wide_ticker = run(go((150, 46)))
    narrow_logo, narrow_ticker = run(go((110, 34)))
    assert wide_logo and wide_ticker == 0
    assert not narrow_logo and narrow_ticker == 1


def test_the_header_shows_live_numbers_when_a_session_is_running(workspace, monkeypatch):
    monkeypatch.setenv("BAT_NO_ANIM", "0")

    async def go():
        app = BatApp()
        async with app.run_test(size=(150, 46)) as pilot:
            await pilot.pause(0.3)
            app.bat.live = live_session([10_000, 10_100, 10_250])
            await wait_for(pilot, lambda: "LIVE" in brand_text(app), timeout=5)
            assert "+$250.00" in brand_text(app)
    run(go())
