"""Re-export shim: this function's canonical home is now `quant_engine.stats`. Kept here,
unchanged in signature and behavior, so every existing script that does
`from mc_true_median import true_median_and_stats` keeps working without modification.
See `quant_engine/stats.py` for the full docstring and rationale.
"""
import sys

sys.path.insert(0, "/Users/brunoinzunza/Documents/GitHub/tradingResearch")

from quant_engine.stats import true_median_and_stats

__all__ = ["true_median_and_stats"]
