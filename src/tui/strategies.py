"""Strategy registry: parameter specs the TUI turns into forms.

Adding a strategy to the UI = adding one ``StrategySpec`` here. The spec holds
defaults, bounds and cross-field validation so forms, the optimiser and tests
all share one source of truth (replacing ~100 lines of hand-written prompts in
the classic CLI).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.strategies.bollinger_bands_strategy import BollingerBandsStrategy
from src.strategies.candlestick_strategy import CandlestickPatternsStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.mean_reversion import MeanReversionExtremeStrategy
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.smart_money_concepts import SmartMoneyConceptsStrategy


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: str                        # "int" | "float" | "choice"
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None     # default optimiser step
    choices: Sequence[str] = ()
    help: str = ""

    def parse(self, raw: Any) -> Any:
        if self.kind == "choice":
            if str(raw) not in [str(c) for c in self.choices]:
                raise ValueError(f"{self.label}: choose one of {', '.join(map(str, self.choices))}")
            return str(raw)
        try:
            v = int(float(raw)) if self.kind == "int" else float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{self.label}: '{raw}' is not a valid {self.kind}")
        if self.kind == "int" and float(raw) != v:
            raise ValueError(f"{self.label}: must be a whole number")
        if self.min is not None and v < self.min:
            raise ValueError(f"{self.label}: must be ≥ {self.min:g}")
        if self.max is not None and v > self.max:
            raise ValueError(f"{self.label}: must be ≤ {self.max:g}")
        return v


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    cls: type
    params: Sequence[ParamSpec] = ()
    description: str = ""
    validate: Optional[Callable[[Dict[str, Any]], List[str]]] = None   # hard errors
    warn: Optional[Callable[[Dict[str, Any]], List[str]]] = None       # soft warnings
    fixed: Dict[str, Any] = field(default_factory=dict)

    def defaults(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def parse(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Parse + validate raw (string) values. Raises ``ValueError`` with all messages."""
        errors, out = [], {}
        for p in self.params:
            try:
                out[p.name] = p.parse(raw.get(p.name, p.default))
            except ValueError as e:
                errors.append(str(e))
        if not errors and self.validate:
            errors += self.validate(out)
        if errors:
            raise ValueError("; ".join(errors))
        return out

    def warnings(self, values: Dict[str, Any]) -> List[str]:
        return self.warn(values) if self.warn else []

    def build(self, values: Optional[Dict[str, Any]] = None):
        vals = self.parse(values if values is not None else self.defaults())
        return self.cls(**vals, **self.fixed)


def _rsi_validate(v):
    return [f"Oversold ({v['oversold_threshold']:g}) must be below overbought ({v['overbought_threshold']:g})"] \
        if v["oversold_threshold"] >= v["overbought_threshold"] else []


def _macd_validate(v):
    return [f"Fast ({v['fast']}) must be shorter than slow ({v['slow']})"] if v["fast"] >= v["slow"] else []


def _ma_warn(v):
    return ["Windows should increase: short ≤ medium ≤ long"] \
        if not (v["short_window"] <= v["medium_window"] <= v["long_window"]) else []


STRATEGIES: Dict[str, StrategySpec] = {s.key: s for s in [
    StrategySpec(
        "mean_reversion", "Mean Reversion", MeanReversionExtremeStrategy,
        (ParamSpec("window", "Window", "int", 20, 1, 5000, 5, help="Rolling window for the mean/std"),
         ParamSpec("num_std", "Std devs", "float", 2.0, 0.1, 10, 0.25, help="Band width in standard deviations")),
        "Fades moves beyond N standard deviations from a rolling mean."),
    StrategySpec(
        "moving_average", "Moving Average", MovingAverageStrategy,
        (ParamSpec("short_window", "Short window", "int", 1, 1, 5000, 1),
         ParamSpec("medium_window", "Medium window", "int", 5, 1, 5000, 1),
         ParamSpec("long_window", "Long window", "int", 25, 1, 5000, 5)),
        "Trades alignment/crossovers of three moving averages.", warn=_ma_warn),
    StrategySpec(
        "rsi", "RSI", RSIStrategy,
        (ParamSpec("window", "RSI window", "int", 14, 2, 500, 2),
         ParamSpec("oversold_threshold", "Oversold <", "float", 30, 0, 100, 5, help="Buy below this level"),
         ParamSpec("overbought_threshold", "Overbought >", "float", 70, 0, 100, 5, help="Sell above this level")),
        "Buys oversold readings and sells overbought ones.", validate=_rsi_validate),
    StrategySpec(
        "macd", "MACD", MACDStrategy,
        (ParamSpec("fast", "Fast EMA", "int", 12, 1, 500, 2),
         ParamSpec("slow", "Slow EMA", "int", 26, 2, 1000, 2),
         ParamSpec("signal", "Signal", "int", 9, 1, 500, 1)),
        "Trades MACD / signal-line crossovers.", validate=_macd_validate),
    StrategySpec(
        "bollinger", "Bollinger Bands", BollingerBandsStrategy,
        (ParamSpec("window", "Window", "int", 20, 2, 5000, 5),
         ParamSpec("num_std", "Std devs", "float", 2.0, 0.1, 10, 0.25)),
        "Trades band touches around a moving average."),
    StrategySpec(
        "candlestick", "Candlestick Patterns", CandlestickPatternsStrategy, (),
        "Hammer / engulfing / shooting-star pattern signals (no parameters)."),
    StrategySpec(
        "smc", "Smart Money Concepts", SmartMoneyConceptsStrategy,
        (ParamSpec("swing_window", "Swing window", "int", 2, 1, 100, 1, help="Bars to confirm a swing"),
         ParamSpec("liquidity_lookback", "Liquidity lookback", "int", 60, 2, 5000, 10),
         ParamSpec("poi_expiry_bars", "POI expiry (bars)", "int", 60, 1, 5000, 10),
         ParamSpec("structure_timeframe", "Structure TF", "choice", "15min",
                   choices=("5min", "15min", "30min", "1h", "4h"), help="Higher-timeframe used for structure")),
        "HTF bias + liquidity sweep + LTF change-of-character entries."),
]}


def get_strategy(key: str) -> StrategySpec:
    return STRATEGIES[key]


def strategy_options() -> List[tuple]:
    return [(s.label, s.key) for s in STRATEGIES.values()]


# --------------------------------------------------------------------------- #
# Indicator placement for charts
# --------------------------------------------------------------------------- #
def split_indicators(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, List[str]]:
    """Classify a strategy's indicator columns.

    * ``overlay``: numeric series living in price space (moving averages, bands)
    * ``pane``:    numeric series on their own scale (RSI, MACD, ...)
    * ``flags``:   boolean/pattern columns (candlestick patterns, SMC events)
    """
    close = df["Close"].to_numpy(float)
    ref = np.nanmedian(close) if close.size else 1.0
    overlay, pane, flags = [], [], []
    for c in columns:
        if c not in df:
            continue
        s = df[c]
        if s.dtype == bool or set(pd.unique(s.dropna())) <= {0, 1, True, False}:
            flags.append(c)
            continue
        if not pd.api.types.is_numeric_dtype(s):
            continue
        med = np.nanmedian(s.to_numpy(float))
        if np.isfinite(med) and ref and 0.5 * ref <= med <= 2 * ref:
            overlay.append(c)
        else:
            pane.append(c)
    return {"overlay": overlay, "pane": pane, "flags": flags}
