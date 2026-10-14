"""Anti-corruption adapters for the pre-hexagonal provider and broker classes."""

from typing import Any
import pandas as pd

from src.core.domain.models import OrderRequest


class LegacyDataProviderAdapter:
    def __init__(self, provider: Any):
        self.provider = provider

    def historical_bars(self, symbol: str, **kwargs: object) -> pd.DataFrame:
        return self.provider.get_data(ticker=symbol, **kwargs)

    def latest_bars(self, symbol: str, **kwargs: object) -> pd.DataFrame:
        return self.provider.get_live_data(ticker=symbol, **kwargs)


class LegacyBrokerAdapter:
    def __init__(self, broker: Any):
        self.broker = broker

    def submit(self, order: OrderRequest) -> dict:
        method = self.broker.buy if order.side.value == "buy" else self.broker.sell
        result = method(order.symbol, order.quantity, current_price=order.price)
        return result if isinstance(result, dict) else {"status": "submitted", "details": result}

    def position(self, symbol: str) -> dict:
        return self.broker.get_position_for_symbol(symbol)

    def account(self) -> dict:
        if hasattr(self.broker, "get_account"):
            return self.broker.get_account()
        return self.broker.get_account_info()
