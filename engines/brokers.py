from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BrokerInterface(ABC):
    """Base class for broker interfaces"""
    
    @abstractmethod
    def buy(self, symbol: str, quantity: float) -> str:
        """Execute buy order"""
        pass
    
    @abstractmethod
    def sell(self, symbol: str, quantity: float) -> str:
        """Execute sell order"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass


class AlpacaBroker(BrokerInterface):
    """Alpaca broker interface based on the existing alpacaTest.py"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.trade_api = None
        
        # Initialize Alpaca API
        try:
            import alpaca_trade_api as tradeapi
            self.trade_api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        except ImportError:
            raise ImportError("alpaca_trade_api package not installed. Install with: pip install alpaca-trade-api")
    
    def buy(self, symbol: str = 'BTCUSD', quantity: float = 1) -> str:
        """Execute buy order"""
        try:
            self.trade_api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            return f"Bought {quantity} shares of {symbol}"
        except Exception as e:
            raise Exception(f"Failed to execute buy order: {e}")
    
    def sell(self, symbol: str = 'BTCUSD', quantity: float = 1) -> str:
        """Execute sell order"""
        try:
            self.trade_api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            return f"Sold {quantity} shares of {symbol}"
        except Exception as e:
            raise Exception(f"Failed to execute sell order: {e}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            account = self.trade_api.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
        except Exception as e:
            raise Exception(f"Failed to get account info: {e}")


class SimulatedBroker(BrokerInterface):
    """Simulated broker for testing without real trades"""
    
    def __init__(self, initial_balance: float = 10000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.trade_count = 0
    
    def buy(self, symbol: str, quantity: float) -> str:
        """Simulate buy order"""
        self.trade_count += 1
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += quantity
        return f"SIMULATED: Bought {quantity} of {symbol} (Total: {self.positions[symbol]})"
    
    def sell(self, symbol: str, quantity: float) -> str:
        """Simulate sell order"""
        self.trade_count += 1
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] -= quantity
        return f"SIMULATED: Sold {quantity} of {symbol} (Total: {self.positions[symbol]})"
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get simulated account information"""
        return {
            'equity': self.balance,
            'cash': self.balance,
            'buying_power': self.balance,
            'portfolio_value': self.balance,
            'positions': self.positions.copy(),
            'trade_count': self.trade_count
        }