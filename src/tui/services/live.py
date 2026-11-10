"""Headless live-trading session.

Drives the existing ``LiveTradingEngine`` (order logic, sizing, position sync)
from a worker thread and publishes structured snapshots instead of drawing a
matplotlib window. The engine itself is unchanged.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.execution.live_engine import LiveTradingEngine
from src.tui.services.capture import capture_thread_output, classify

MAX_BARS = 600


# --------------------------------------------------------------------------- #
# Bar sources: one tiny interface over the very different provider APIs
# --------------------------------------------------------------------------- #
class BarSource:
    name = "bars"

    def history(self, symbol: str, n: int) -> pd.DataFrame:
        raise NotImplementedError

    def latest(self, symbol: str) -> Optional[dict]:
        raise NotImplementedError


class _Demo(BarSource):
    def __init__(self, feed):
        self.feed, self.name = feed, feed.name

    def history(self, symbol, n):
        return self.feed.history(symbol, n)

    def latest(self, symbol):
        return self.feed.latest(symbol)


class _Oanda(BarSource):
    name = "OANDA"

    def __init__(self, p):
        self.p = p

    def history(self, symbol, n):
        return self.p.get_data(ticker=symbol, timespan="M1", limit=n)

    def latest(self, symbol):
        c = self.p.get_latest_candle(symbol)
        if not c:
            return None
        ts = pd.to_datetime(c.get("time", datetime.now()))
        return dict(timestamp=ts, Open=c["open"], High=c["high"], Low=c["low"], Close=c["close"],
                    Volume=c.get("volume", 0))


class _Alpaca(BarSource):
    name = "Alpaca"

    def __init__(self, p):
        self.p = p

    def history(self, symbol, n):
        return self.p.get_recent_bars_public(symbol, limit=n)

    def latest(self, symbol):
        return self.p.get_latest_bar(symbol)


class _Generic(BarSource):
    """Providers that only return the current candle (e.g. Synth)."""

    def __init__(self, p, name="provider"):
        self.p, self.name = p, name

    def history(self, symbol, n):
        return self.p.get_live_data(symbol)

    def latest(self, symbol):
        df = self.p.get_live_data(symbol)
        return df.iloc[0].to_dict() if len(df) else None


def make_bar_source(provider: Any) -> BarSource:
    from src.data.providers.oanda_provider import OandaProvider
    from src.tui.services.demo import DemoFeed
    if isinstance(provider, DemoFeed):
        return _Demo(provider)
    if isinstance(provider, OandaProvider):
        return _Oanda(provider)
    if hasattr(provider, "get_recent_bars_public") and hasattr(provider, "get_latest_bar"):
        return _Alpaca(provider)
    return _Generic(provider, type(provider).__name__)


# --------------------------------------------------------------------------- #
@dataclass
class LiveConfig:
    strategy: Any
    symbol: str
    source: BarSource
    broker: Any
    trading_mode: str = "long_only"
    position_pct: Optional[float] = 20.0       # None -> fixed quantity
    quantity: Optional[float] = None
    interval: float = 60.0
    lookback: int = 300
    initial_balance: float = 10_000.0
    broker_label: str = "simulated"
    source_label: str = "demo"
    paper: bool = True
    max_iterations: Optional[int] = None


@dataclass
class LiveEvent:
    ts: datetime
    level: str
    text: str


@dataclass
class LiveSnapshot:
    iteration: int
    ts: datetime
    state: str                         # starting | running | stopped | error
    price: float = float("nan")
    prev_price: float = float("nan")
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    position: Dict[str, Any] = field(default_factory=dict)
    account: Dict[str, Any] = field(default_factory=dict)
    equity: List[tuple] = field(default_factory=list)       # (timestamp, equity)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    perf: Dict[str, Any] = field(default_factory=dict)
    next_tick: float = 0.0
    message: str = ""


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def round_trips_from_engine(trades: List[dict]) -> pd.DataFrame:
    """Pair the engine's open/close records into rows with P&L."""
    rows, open_rec = [], None
    for t in trades:
        a = t.get("action")
        if a in ("buy_long", "sell_short"):
            open_rec = t
            rows.append(dict(time=t["timestamp"], action="BUY" if a == "buy_long" else "SELL_SHORT",
                             price=t["price"], qty=t["quantity"], pnl=np.nan, result="OPEN"))
        elif a in ("close_position", "close_long", "close_short") and open_rec is not None:
            long = open_rec["action"] == "buy_long"
            pnl = ((t["price"] - open_rec["price"]) if long else (open_rec["price"] - t["price"])) * abs(t["quantity"])
            rows.append(dict(time=t["timestamp"], action="CLOSE" if long else "CLOSE_SHORT", price=t["price"],
                             qty=abs(t["quantity"]), pnl=pnl, result="Win" if pnl > 0 else "Loss"))
            open_rec = None
    return pd.DataFrame(rows, columns=["time", "action", "price", "qty", "pnl", "result"])


class LiveSession:
    """Runs the trading loop in a thread; the UI only consumes callbacks."""

    def __init__(self, cfg: LiveConfig, on_snapshot: Callable[[LiveSnapshot], None],
                 on_event: Callable[[LiveEvent], None]):
        self.cfg, self.on_snapshot, self.on_event = cfg, on_snapshot, on_event
        self.engine: Optional[LiveTradingEngine] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.RLock()
        self._pending: List[str] = []           # queued user actions
        self.history = pd.DataFrame()
        self.equity: List[tuple] = []
        self.iteration = 0
        self.state = "idle"
        self.last_snapshot: Optional[LiveSnapshot] = None
        self._last_price = float("nan")

    # ----- public API (UI thread) -----------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="bat-live", daemon=True)
        self._thread.start()

    def stop(self, close_position: bool = False) -> None:
        if close_position:
            self._pending.append("close")
        self._pending.append("stop")
        self._wake.set()

    def close_position(self) -> None:
        self._pending.append("close")
        self._wake.set()

    def tick_now(self) -> None:
        self._wake.set()

    # ----- internals ------------------------------------------------------
    def _emit(self, level: str, text: str) -> None:
        self.on_event(LiveEvent(datetime.now(), level, text))

    def _sink(self, line: str) -> None:
        self._emit(classify(line), line.strip())

    def _min_bars(self) -> int:
        try:
            return max(int(self.cfg.strategy.get_required_lookback()) + 10, 50)
        except Exception:
            return 50

    def _bootstrap(self) -> None:
        c = self.cfg
        hist = c.source.history(c.symbol, max(c.lookback, self._min_bars()))
        if hist is None or hist.empty:
            raise RuntimeError("data source returned no history")
        hist = hist.copy()
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        self.history = hist.tail(MAX_BARS).reset_index(drop=True)
        self._emit("info", f"Loaded {len(self.history)} bars from {c.source.name}")

    def _ingest_latest(self) -> None:
        bar = self.cfg.source.latest(self.cfg.symbol)
        if not bar:
            return
        ts = pd.to_datetime(bar["timestamp"])
        last = self.history["timestamp"].iloc[-1] if len(self.history) else None
        self._last_price = _f(bar.get("Close"), self._last_price)
        if last is not None:
            if getattr(ts, "tzinfo", None) is None and getattr(last, "tzinfo", None) is not None:
                ts = ts.tz_localize(last.tzinfo)
            elif getattr(ts, "tzinfo", None) is not None and getattr(last, "tzinfo", None) is None:
                last = last.tz_localize(ts.tzinfo)
        if last is None or ts > last:
            row = {k: bar.get(k) for k in ("Open", "High", "Low", "Close", "Volume")}
            row["timestamp"] = pd.to_datetime(bar["timestamp"])
            self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True).tail(MAX_BARS)

    def _quantity(self) -> Optional[float]:
        return None if self.cfg.position_pct is not None else self.cfg.quantity

    def _account(self) -> Dict[str, Any]:
        try:
            return dict(self.engine.get_alpaca_account())
        except Exception:
            return {}

    def _snapshot(self, state: str, message: str = "", next_tick: float = 0.0) -> LiveSnapshot:
        eng, c = self.engine, self.cfg
        acct = self._account()
        equity = _f(acct.get("equity"), c.initial_balance)
        if state != "starting" and (not self.equity or self.equity[-1][1] != equity or state == "running"):
            self.equity.append((datetime.now(), equity))
            self.equity = self.equity[-2000:]
        try:
            pos = dict(eng.get_alpaca_position(c.symbol))
        except Exception:
            pos = {}
        try:
            processed = c.strategy.generate_signals(self.history.copy())
        except Exception:
            processed = self.history
        trades = round_trips_from_engine(eng.trades) if eng else pd.DataFrame()
        closed = trades[trades["result"].isin(["Win", "Loss"])] if len(trades) else trades
        wins = int((closed["result"] == "Win").sum()) if len(closed) else 0
        perf = dict(trades=len(closed), wins=wins, losses=len(closed) - wins,
                    win_rate=wins / len(closed) * 100 if len(closed) else float("nan"),
                    realized=float(closed["pnl"].sum()) if len(closed) else 0.0,
                    session_pnl=equity - c.initial_balance,
                    return_pct=(equity / c.initial_balance - 1) * 100 if c.initial_balance else float("nan"))
        prev = float(self.history["Close"].iloc[-2]) if len(self.history) > 1 else float("nan")
        price = _f(self.history["Close"].iloc[-1]) if len(self.history) else float("nan")
        if hasattr(c.broker, "set_current_price"):
            try:
                c.broker.set_current_price(c.symbol, price)
            except Exception:
                pass
        snap = LiveSnapshot(self.iteration, datetime.now(), state, price, prev, processed.tail(MAX_BARS),
                            pos, acct, list(self.equity), trades, perf, next_tick, message)
        self.last_snapshot = snap
        return snap

    def _handle_pending(self) -> bool:
        """Process queued user actions. Returns True when the loop should end."""
        stop = False
        while self._pending:
            act = self._pending.pop(0)
            if act == "close":
                try:
                    price = _f(self.history["Close"].iloc[-1]) if len(self.history) else None
                    res = self.engine.close_position(self.cfg.symbol, current_price=price)
                    if res.get("status") == "failed":
                        self._emit("error", f"Close failed: {res.get('error', 'unknown')}")
                    else:
                        qty = _f(res.get("qty"), 0.0)
                        self.engine.trades.append(dict(
                            timestamp=self.history["timestamp"].iloc[-1] if len(self.history) else datetime.now(),
                            action="close_position", price=_f(res.get("avg_fill_price"), price or 0.0),
                            quantity=qty, order_details=res))
                        self._emit("trade", f"Position closed manually @ {price}")
                except Exception as e:  # noqa: BLE001
                    self._emit("error", f"Close failed: {e}")
            elif act == "stop":
                stop = True
        return stop

    def _run(self) -> None:
        c = self.cfg
        with capture_thread_output(self._sink):
            try:
                self.state = "starting"
                self.engine = LiveTradingEngine(
                    data_provider=None, broker_interface=c.broker, initial_balance=c.initial_balance,
                    trading_mode=c.trading_mode, position_percentage=c.position_pct or 100.0, verbose_mode=False)
                self.engine.quiet_mode = True
                self._emit("info", f"Session starting · {c.strategy.name} · {c.symbol} · {c.trading_mode} · "
                                   f"{c.broker_label} ({'paper' if c.paper else 'LIVE'})")
                self.on_snapshot(self._snapshot("starting", "loading history"))
                self._bootstrap()
                self.state = "running"
                while not self._stop.is_set():
                    started = time.time()
                    with self._lock:
                        if self._handle_pending():
                            break
                        try:
                            self.iteration += 1
                            if self.iteration > 1:
                                self._ingest_latest()
                            if len(self.history) >= self._min_bars():
                                self.engine.process_signals(self.history, c.strategy, c.symbol, self._quantity())
                            else:
                                self._emit("info", f"Warming up: {len(self.history)}/{self._min_bars()} bars")
                        except Exception as e:  # noqa: BLE001 - keep the session alive
                            self._emit("error", f"Iteration {self.iteration} failed: {e}")
                        snap_msg = ""
                        self.on_snapshot(self._snapshot("running", snap_msg, started + c.interval))
                    if c.max_iterations and self.iteration >= c.max_iterations:
                        break
                    deadline = started + c.interval
                    while time.time() < deadline and not self._stop.is_set():
                        self._wake.wait(timeout=min(0.2, max(deadline - time.time(), 0.01)))
                        if self._wake.is_set():
                            self._wake.clear()
                            if self._pending:
                                with self._lock:
                                    if self._handle_pending():
                                        self._stop.set()
                                    self.on_snapshot(self._snapshot("running", "", deadline))
                            else:
                                break  # tick_now
                self.state = "stopped"
                self.on_snapshot(self._snapshot("stopped", "session ended"))
                self._emit("info", "Session stopped")
            except Exception as e:  # noqa: BLE001
                self.state = "error"
                self._emit("error", f"Session crashed: {e}")
                self._emit("error", traceback.format_exc().splitlines()[-2].strip())
                self.on_snapshot(self._snapshot("error", str(e)))
