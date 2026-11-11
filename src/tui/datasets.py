"""Backward-compatible alias: this module moved to ``src.data.local_datasets``.

The old import path is made *the same module object*, so ``monkeypatch.setattr(src.tui.datasets, ...)`` still works.
"""
import sys

from src.data import local_datasets as _moved

sys.modules[__name__] = _moved
