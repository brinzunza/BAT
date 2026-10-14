"""Deterministic portfolio accounting with no IO dependencies."""

from dataclasses import dataclass
from .models import PositionSide


@dataclass
class Portfolio:
    initial_cash: float
    cash: float = 0.0
    quantity: float = 0.0
    entry_price: float = 0.0
    realized_profit: float = 0.0

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = self.initial_cash

    @property
    def side(self) -> PositionSide:
        if self.quantity > 0:
            return PositionSide.LONG
        if self.quantity < 0:
            return PositionSide.SHORT
        return PositionSide.FLAT

    def equity(self, mark_price: float) -> float:
        return self.cash + self.quantity * mark_price

    def open_long(self, quantity: float, price: float) -> None:
        if self.side is not PositionSide.FLAT:
            raise ValueError("cannot open a long while a position exists")
        if quantity <= 0 or quantity * price > self.cash + 1e-9:
            raise ValueError("insufficient cash or invalid quantity")
        self.cash -= quantity * price
        self.quantity = quantity
        self.entry_price = price

    def close_long(self, price: float) -> float:
        if self.side is not PositionSide.LONG:
            raise ValueError("no long position to close")
        profit = self.quantity * (price - self.entry_price)
        self.cash += self.quantity * price
        self.realized_profit += profit
        self.quantity = 0.0
        self.entry_price = 0.0
        return profit

    def open_short(self, quantity: float, price: float) -> None:
        if self.side is not PositionSide.FLAT:
            raise ValueError("cannot open a short while a position exists")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self.cash += quantity * price
        self.quantity = -quantity
        self.entry_price = price

    def close_short(self, price: float) -> float:
        if self.side is not PositionSide.SHORT:
            raise ValueError("no short position to close")
        quantity = -self.quantity
        profit = quantity * (self.entry_price - price)
        self.cash -= quantity * price
        self.realized_profit += profit
        self.quantity = 0.0
        self.entry_price = 0.0
        return profit
