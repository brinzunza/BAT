"""Framework-free value objects used by the trading domain."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class PositionSide(str, Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    quantity: float
    timestamp: datetime
    price: Optional[float] = None


@dataclass(frozen=True)
class Fill:
    order: OrderRequest
    price: float
    fees: float = 0.0


@dataclass(frozen=True)
class TradeRecord:
    timestamp: datetime
    action: str
    price: float
    quantity: float
    position: PositionSide
    cash: float
    equity: float
    bar_index: int
    realized_profit: Optional[float] = None


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    trading_mode: str = "long_only"
    position_fraction: float = 1.0
    spread_pips: float = 0.0
    symbol: str = ""

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.trading_mode not in {"long_only", "long_short"}:
            raise ValueError("trading_mode must be 'long_only' or 'long_short'")
        if not 0 < self.position_fraction <= 1:
            raise ValueError("position_fraction must be in (0, 1]")
        if self.spread_pips < 0:
            raise ValueError("spread_pips cannot be negative")
