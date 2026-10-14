"""Backtest use case. It is deterministic and deliberately free of broker SDKs."""

from dataclasses import dataclass
from typing import Any
import pandas as pd

from src.core.domain.models import BacktestConfig, PositionSide, TradeRecord
from src.core.domain.portfolio import Portfolio
from src.core.ports.trading import MarketDataProvider, Strategy


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    signal_data: pd.DataFrame
    portfolio: Portfolio


class BacktestService:
    """Run a strategy against bars using the same domain accounting every time."""

    REQUIRED_COLUMNS = {"Close", "timestamp"}

    def run_for_symbol(self, market_data: MarketDataProvider, symbol: str, strategy: Strategy,
                       config: BacktestConfig, **query: object) -> BacktestResult:
        """Fetch data through an outbound port, then execute the use case."""
        return self.run(market_data.historical_bars(symbol, **query), strategy, config)

    def run(self, bars: pd.DataFrame, strategy: Strategy, config: BacktestConfig) -> BacktestResult:
        missing = self.REQUIRED_COLUMNS - set(bars.columns)
        if missing:
            raise ValueError(f"bars are missing required columns: {sorted(missing)}")
        if bars.empty:
            return BacktestResult(pd.DataFrame(), bars.copy(), Portfolio(config.initial_cash))

        signal_data = strategy.generate_signals(bars.copy())
        names = strategy.get_signal_names()
        if not {"buy", "sell"}.issubset(names) or not {names["buy"], names["sell"]}.issubset(signal_data.columns):
            raise ValueError("strategy must produce declared buy and sell signal columns")

        portfolio = Portfolio(config.initial_cash)
        records: list[TradeRecord] = []
        for index in range(1, len(signal_data)):
            row = signal_data.iloc[index]
            price = float(row["Close"])
            if price <= 0:
                continue
            timestamp = pd.Timestamp(row["timestamp"]).to_pydatetime()
            buy, sell = bool(row[names["buy"]]), bool(row[names["sell"]])
            self._apply_signals(portfolio, buy, sell, price, timestamp, index, config, records)

        return BacktestResult(self._to_dataframe(records, config.initial_cash), signal_data, portfolio)

    @staticmethod
    def _spread_adjusted(price: float, buying: bool, config: BacktestConfig) -> float:
        if not config.symbol.startswith("C:") or config.spread_pips == 0:
            return price
        pip = 0.01 if "JPY" in config.symbol else 0.0001
        return price + config.spread_pips * pip if buying else price - config.spread_pips * pip

    def _apply_signals(self, portfolio: Portfolio, buy: bool, sell: bool, price: float, timestamp: Any, index: int,
                       config: BacktestConfig, records: list[TradeRecord]) -> None:
        if config.trading_mode == "long_only":
            if buy and portfolio.side is PositionSide.FLAT:
                entry = self._spread_adjusted(price, True, config)
                quantity = portfolio.cash * config.position_fraction / entry
                portfolio.open_long(quantity, entry)
                self._record(records, timestamp, "BUY", entry, quantity, portfolio, price, index)
            elif sell and portfolio.side is PositionSide.LONG:
                exit_price = self._spread_adjusted(price, False, config)
                quantity, profit = portfolio.quantity, portfolio.close_long(exit_price)
                self._record(records, timestamp, "CLOSE", exit_price, quantity, portfolio, price, index, profit)
            return

        if buy and portfolio.side is PositionSide.SHORT:
            exit_price = self._spread_adjusted(price, True, config)
            quantity, profit = -portfolio.quantity, portfolio.close_short(exit_price)
            self._record(records, timestamp, "CLOSE_SHORT", exit_price, quantity, portfolio, price, index, profit)
        if sell and portfolio.side is PositionSide.LONG:
            exit_price = self._spread_adjusted(price, False, config)
            quantity, profit = portfolio.quantity, portfolio.close_long(exit_price)
            self._record(records, timestamp, "CLOSE", exit_price, quantity, portfolio, price, index, profit)
        if buy and portfolio.side is PositionSide.FLAT:
            entry = self._spread_adjusted(price, True, config)
            quantity = portfolio.cash * config.position_fraction / entry
            portfolio.open_long(quantity, entry)
            self._record(records, timestamp, "BUY", entry, quantity, portfolio, price, index)
        elif sell and portfolio.side is PositionSide.FLAT:
            entry = self._spread_adjusted(price, False, config)
            quantity = portfolio.cash * config.position_fraction / entry
            portfolio.open_short(quantity, entry)
            self._record(records, timestamp, "SELL_SHORT", entry, quantity, portfolio, price, index)

    @staticmethod
    def _record(records: list[TradeRecord], timestamp: Any, action: str, price: float, quantity: float,
                portfolio: Portfolio, mark_price: float, bar_index: int, realized_profit: float | None = None) -> None:
        records.append(TradeRecord(timestamp, action, price, quantity, portfolio.side, portfolio.cash,
                                   portfolio.equity(mark_price), bar_index, realized_profit))

    @staticmethod
    def _to_dataframe(records: list[TradeRecord], initial_cash: float) -> pd.DataFrame:
        rows = []
        for record in records:
            row = {
                "Time": record.timestamp, "Price": record.price, "Position": {PositionSide.LONG: 1, PositionSide.SHORT: -1, PositionSide.FLAT: 0}[record.position], "Index": record.bar_index,
                "Action": record.action, "Shares": record.quantity, "Balance": record.cash,
                "Total_Account_Worth": record.equity, "Total_Profit": record.equity - initial_cash,
                "Last_Trade_Realized": record.realized_profit or 0.0, "Profit": record.realized_profit, "Result": None,
                "Trade_Result": "OPEN" if record.realized_profit is None else ("Win" if record.realized_profit > 0 else "Loss"),
            }
            if record.realized_profit is not None:
                row.update({"Result": "Win" if record.realized_profit > 0 else "Loss"})
            rows.append(row)
        return pd.DataFrame(rows)
