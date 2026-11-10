"""Route ``print`` output of worker threads into the UI.

The legacy engines ``print`` their trade messages. ``contextlib.redirect_stdout``
is process-global, so instead we install one router that dispatches by thread:
threads that registered a sink get their lines delivered to it, everyone else
(including Textual itself) keeps writing to the original stream.
"""
from __future__ import annotations

import io
import sys
import threading
from contextlib import contextmanager
from typing import Callable, Dict


class _Router(io.TextIOBase):
    def __init__(self, original):
        self.original = original
        self.sinks: Dict[int, Callable[[str], None]] = {}
        self._buf: Dict[int, str] = {}

    def writable(self):  # pragma: no cover - trivial
        return True

    def write(self, s: str) -> int:
        sink = self.sinks.get(threading.get_ident())
        if sink is None:
            return self.original.write(s)
        tid = threading.get_ident()
        buf = self._buf.get(tid, "") + s.replace("\r", "\n")
        *lines, rest = buf.split("\n")
        self._buf[tid] = rest
        for line in lines:
            if line.strip():
                sink(line.rstrip())
        return len(s)

    def flush(self):
        if threading.get_ident() not in self.sinks:
            try:
                self.original.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def __getattr__(self, name):  # delegate encoding, fileno, ...
        return getattr(self.original, name)


def _install(stream_name: str) -> _Router:
    cur = getattr(sys, stream_name)
    if isinstance(cur, _Router):
        return cur
    router = _Router(cur)
    setattr(sys, stream_name, router)
    return router


@contextmanager
def capture_thread_output(sink: Callable[[str], None]):
    """Deliver stdout+stderr lines written by the *calling thread* to ``sink``."""
    routers = [_install("stdout"), _install("stderr")]
    tid = threading.get_ident()
    for r in routers:
        r.sinks[tid] = sink
    try:
        yield
    finally:
        for r in routers:
            r.sinks.pop(tid, None)
            rest = r._buf.pop(tid, "")
            if rest.strip():
                sink(rest.strip())


def classify(line: str) -> str:
    """Map a legacy engine message to a log level for colouring."""
    u = line.upper()
    if line.startswith("    ") or "POSITION CHECK" in u or u.strip().startswith(("SYMBOL:", "ACCOUNT:")):
        return "detail"      # continuation lines of multi-line engine messages
    if any(k in u for k in ("FAILED", "ERROR", "TRACEBACK", "REJECTED", "INSUFFICIENT")):
        return "error"
    if any(k in u for k in ("BUY ORDER FILLED", "POSITION CLOSED", "SHORT", "[BUY]", "[CLOSE]", "[SELL]", "FILLED")):
        return "trade"
    if any(k in u for k in ("SIGNAL", "WARNING", "IGNORED", "SYNC", "CANCEL")):
        return "signal"
    return "info"
