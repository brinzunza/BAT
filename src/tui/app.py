"""BAT terminal application: navigation shell, status bar, command palette."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Type

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.theme import Theme
from textual.widgets import ContentSwitcher, Footer, Header, Static

from src.tui import charts as C
from src.tui.state import BatState
from src.tui.widgets.modals import ConfirmModal, HelpModal

BAT_DARK = Theme(
    name="bat", primary="#58a6ff", secondary="#bc8cff", accent="#39c5cf", warning="#d29922", error="#f85149",
    success="#3fb950", foreground="#c9d1d9", background="#0d1117", surface="#161b22", panel="#21262d", dark=True,
)
BAT_LIGHT = Theme(
    name="bat-light", primary="#0969da", secondary="#8250df", accent="#1b7c83", warning="#9a6700", error="#cf222e",
    success="#1a7f37", foreground="#1f2328", background="#ffffff", surface="#f6f8fa", panel="#eaeef2", dark=False,
)


@dataclass(frozen=True)
class PageDef:
    id: str
    label: str
    icon: str
    key: str
    factory: Callable[[], object]


def page_defs() -> List[PageDef]:
    from src.tui.pages.analytics import AnalyticsPage
    from src.tui.pages.backtest import BacktestPage
    from src.tui.pages.connections import ConnectionsPage
    from src.tui.pages.home import HomePage
    from src.tui.pages.live import LivePage
    from src.tui.pages.logs import LogsPage
    from src.tui.pages.mllab import MLPage
    from src.tui.pages.research import ResearchPage
    return [
        PageDef("home", "Home", "◈", "1", HomePage),
        PageDef("backtest", "Backtest", "◧", "2", BacktestPage),
        PageDef("live", "Live Trading", "◉", "3", LivePage),
        PageDef("research", "Research", "◫", "4", ResearchPage),
        PageDef("mllab", "ML Lab", "◬", "5", MLPage),
        PageDef("analytics", "Analytics", "◭", "6", AnalyticsPage),
        PageDef("connections", "Connections", "◍", "7", ConnectionsPage),
        PageDef("logs", "Logs", "≡", "8", LogsPage),
    ]


class NavItem(Static, can_focus=True):
    class Picked(Message):
        def __init__(self, page_id: str):
            super().__init__()
            self.page_id = page_id

    BINDINGS = [Binding("enter", "pick", "Open", show=False)]

    def __init__(self, page: PageDef):
        super().__init__(f"{page.icon}  {page.label:<14}{page.key}", id=f"nav-{page.id}")
        self.page_id = page.id

    def on_click(self) -> None:
        self.post_message(self.Picked(self.page_id))

    def action_pick(self) -> None:
        self.post_message(self.Picked(self.page_id))


class StatusBar(Static):
    """One-line global status: connections, live session, clock."""

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh)

    def render(self) -> Text:
        app: BatApp = self.app  # type: ignore[assignment]
        bat = app.bat
        t = Text(no_wrap=True, overflow="ellipsis")
        for st in bat.connections.status():
            glyph, style = ("●", C.UP) if st.connected else ("○", C.AMBER) if st.configured else ("·", C.MUTED)
            t.append(glyph + " ", style=style)
            t.append(st.key + "  ", style="dim")
        t.append("│ ", style="dim")
        live = bat.live
        if live and live.running and live.last_snapshot:
            s = live.last_snapshot
            t.append("LIVE ", style="bold " + C.UP)
            t.append(f"{live.cfg.symbol} ", style="bold")
            t.append(C.fmt_money(s.perf.get("session_pnl", 0.0), signed=True), style=C.pnl_color(s.perf.get("session_pnl", 0.0)))
            t.append(f"  #{s.iteration}", style="dim")
        elif live and live.state == "error":
            t.append("LIVE ERROR", style="bold " + C.DOWN)
        else:
            t.append("no live session", style="dim")
        t.append("  │  ", style="dim")
        n = len(bat.backtests)
        t.append(f"{n} backtest{'s' if n != 1 else ''}", style="dim")
        t.append("   ? help · ctrl+p commands", style="dim")
        return t


class BatCommands(Provider):
    """Command palette entries (ctrl+p)."""

    def _commands(self):
        app: BatApp = self.app  # type: ignore[assignment]
        cmds = [(f"Go to {p.label}", (lambda pid=p.id: app.goto(pid)), f"Open the {p.label} page") for p in app.pages]
        cmds += [
            ("Backtest: run with current settings", app.cmd_run_backtest, "Runs the form on the Backtest page"),
            ("Live: start demo session", app.cmd_demo_live, "Offline simulated market + simulated broker"),
            ("Toggle light / dark theme", app.cmd_toggle_theme, "Switch between BAT dark and light"),
            ("Toggle sidebar", app.action_toggle_sidebar, "Show or hide the navigation"),
            ("Show keyboard help", app.action_help, "Shortcuts for every page"),
            ("Quit BAT", app.action_quit_app, "Exit (asks first if live trading is running)"),
        ]
        return cmds

    async def discover(self) -> Hits:
        for text, cb, help_ in self._commands():
            yield DiscoveryHit(text, cb, help=help_)

    async def search(self, query: str) -> Hits:
        m = self.matcher(query)
        for text, cb, help_ in self._commands():
            score = m.match(text)
            if score > 0:
                yield Hit(score, m.highlight(text), cb, help=help_)


HELP = """\
[b]Navigation[/b]
  1-8 / F1-F8       switch page            ctrl+b   toggle sidebar
  ctrl+p            command palette        esc      leave a text field
  ?                 this help              ctrl+q   quit

[b]Backtest[/b]
  ctrl+r            run                    tabs     Overview · Price · Trades · Analysis · History
  Price chart       ←/→ pan · +/− or wheel zoom · Home/End/0 jump · hover for OHLC
  Trades table      click a header to sort · Enter jumps the chart to that trade

[b]Live trading[/b]
  ctrl+s            start / stop session   x        close open position
  The session runs in the background: browse other pages and it keeps trading.

[b]Research · ML Lab · Analytics[/b]
  ctrl+r            run the action on the active tab
  Optimizer         ranges as start:stop:step (10:40:5) or a list (10,20,30)

[b]Safety[/b]
  Live (non-paper) brokers require typing a confirmation phrase before a session starts.
  Credentials are kept in memory only; they are never written to disk.
"""


class BatApp(App):
    TITLE = "BAT"
    SUB_TITLE = "Bruninvestor Algorithmic Trading"
    CSS_PATH = "bat.tcss"
    COMMANDS = App.COMMANDS | {BatCommands}
    ENABLE_COMMAND_PALETTE = True
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (125, "-normal"), (170, "-wide")]
    BINDINGS = [
        *[Binding(str(i + 1), f"goto_index({i})", show=False) for i in range(8)],
        *[Binding(f"f{i + 1}", f"goto_index({i})", show=False) for i in range(8)],
        Binding("ctrl+r", "run_active", "Run", priority=True),
        Binding("ctrl+s", "primary_active", "Start/Stop", priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar"),
        Binding("question_mark", "help", "Help"),
        Binding("escape", "blur", show=False),
        Binding("ctrl+q", "quit_app", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.bat = BatState()
        self.pages: List[PageDef] = []
        self.active_page = "home"

    # ----- layout ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        self.pages = page_defs()
        yield Header(show_clock=True)
        with Horizontal(id="shell"):
            with Vertical(id="nav"):
                yield Static("◢◣ BAT", id="nav-brand")
                for p in self.pages:
                    yield NavItem(p)
                yield Static("", id="nav-sep")
                yield Static("[dim]ctrl+p  palette\n?  help[/]", id="nav-foot")
            with ContentSwitcher(initial="home", id="pages"):
                for p in self.pages:
                    w = p.factory()
                    w.id = p.id
                    w.add_class("page")
                    yield w
        yield StatusBar()
        yield Footer()

    def on_mount(self) -> None:
        from src.tui.services.connections import load_env
        load_env()
        self.register_theme(BAT_DARK)
        self.register_theme(BAT_LIGHT)
        self.theme = "bat"
        self._mark_active("home")
        self.log_event("info", "app", "BAT started")

    # ----- navigation -----------------------------------------------------------
    def goto(self, page_id: str) -> None:
        if page_id not in {p.id for p in self.pages}:
            return
        self.query_one("#pages", ContentSwitcher).current = page_id
        self._mark_active(page_id)
        self.active_page = page_id
        page = self.query_one(f"#{page_id}")
        if hasattr(page, "on_show"):
            page.on_show()

    def _mark_active(self, page_id: str) -> None:
        for item in self.query(NavItem):
            item.set_class(item.page_id == page_id, "-active")

    def action_goto_index(self, i: int) -> None:
        if i < len(self.pages):
            self.goto(self.pages[i].id)

    @on(NavItem.Picked)
    def _picked(self, event: NavItem.Picked) -> None:
        self.goto(event.page_id)

    def action_run_active(self) -> None:
        page = self.query_one(f"#{self.active_page}")
        if hasattr(page, "action_run"):
            page.action_run()

    def action_primary_active(self) -> None:
        page = self.query_one(f"#{self.active_page}")
        if hasattr(page, "action_primary"):
            page.action_primary()

    def action_toggle_sidebar(self) -> None:
        # on narrow terminals the sidebar is hidden by default and opens as an overlay column
        self.query_one("#nav").toggle_class("open" if "-narrow" in self.screen.classes else "collapsed")

    def action_blur(self) -> None:
        self.screen.set_focus(None)

    def action_help(self) -> None:
        self.push_screen(HelpModal(HELP))

    # ----- commands ---------------------------------------------------------------
    def cmd_run_backtest(self) -> None:
        self.goto("backtest")
        self.query_one("#backtest").action_run()

    def cmd_demo_live(self) -> None:
        self.goto("live")
        self.query_one("#live").start_demo()

    def cmd_toggle_theme(self) -> None:
        self.theme = "bat-light" if self.theme != "bat-light" else "bat"

    def action_quit_app(self) -> None:
        live = self.bat.live
        if live and live.running:
            def _done(ok: Optional[bool]) -> None:
                if ok:
                    live.stop(close_position=False)
                    self.exit()
            self.push_screen(ConfirmModal(
                "Live session is running",
                "Quitting stops the session [b]without[/b] closing any open position.\n"
                "Stop trading and exit anyway?", confirm="Stop & exit", danger=True), _done)
        else:
            self.exit()

    # ----- events ---------------------------------------------------------------------
    def log_event(self, level: str, source: str, text: str) -> None:
        entry = self.bat.add_event(level, source, text)
        try:
            self.query_one("#logs").push(entry)
        except Exception:
            pass
        try:
            self.query_one("#home").push(entry)
        except Exception:
            pass

    def notify_ok(self, msg: str) -> None:
        self.notify(msg, severity="information", timeout=4)

    def notify_err(self, msg: str) -> None:
        self.notify(msg, severity="error", timeout=8)


def run() -> None:
    BatApp().run()


if __name__ == "__main__":
    run()
