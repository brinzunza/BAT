"""Backward-compatible alias: this module moved to ``src.data.diagnostics``.

The old import path is made *the same module object*, so ``monkeypatch.setattr(src.tui.diagnostics, ...)`` still works.
"""
import sys

from src.data import diagnostics as _moved

sys.modules[__name__] = _moved
