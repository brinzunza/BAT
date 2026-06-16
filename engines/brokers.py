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
    """Simulated broker for testing without real trades - tracks positions with proper price data"""

    def __init__(self, initial_balance: float = 10000):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.cash = initial_balance
        # positions format: {symbol: {'qty': float, 'avg_entry_price': float, 'side': str}}
        self.positions = {}
        self.current_prices = {}  # Track current market prices
        self.trade_count = 0

    def set_current_price(self, symbol: str, price: float):
        """Update current market price for a symbol"""
        self.current_prices[symbol] = price

    def buy(self, symbol: str, quantity: float, order_type: str = "market", limit_price: float = None, current_price: float = None) -> dict:
        """Simulate buy order at current market price"""
        self.trade_count += 1

        # Determine fill price
        if current_price is not None:
            fill_price = current_price
        elif order_type == "limit" and limit_price is not None:
            fill_price = limit_price
        else:
            fill_price = self.current_prices.get(symbol, 0)

        # Update current price
        if fill_price > 0:
            self.current_prices[symbol] = fill_price

        # Update position
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = {
                'qty': quantity,
                'avg_entry_price': fill_price,
                'side': 'long'
            }
        else:
            # Adding to existing position
            pos = self.positions[symbol]
            old_qty = pos['qty']
            old_avg_price = pos['avg_entry_price']

            new_qty = old_qty + quantity
            # Calculate new average entry price
            if new_qty != 0:
                new_avg_price = ((old_qty * old_avg_price) + (quantity * fill_price)) / new_qty
                pos['qty'] = new_qty
                pos['avg_entry_price'] = new_avg_price
                pos['side'] = 'long' if new_qty > 0 else 'short' if new_qty < 0 else 'flat'

        # Update cash (deduct cost)
        self.cash -= quantity * fill_price

        return {
            'status': 'filled',
            'symbol': symbol,
            'qty': quantity,
            'side': 'buy',
            'avg_fill_price': fill_price,
            'message': f"SIMULATED: Bought {quantity} of {symbol} @ ${fill_price:.2f}"
        }

    def sell(self, symbol: str, quantity: float, order_type: str = "market", limit_price: float = None, current_price: float = None) -> dict:
        """Simulate sell order at current market price"""
        self.trade_count += 1

        # Determine fill price
        if current_price is not None:
            fill_price = current_price
        elif order_type == "limit" and limit_price is not None:
            fill_price = limit_price
        else:
            fill_price = self.current_prices.get(symbol, 0)

        # Update current price
        if fill_price > 0:
            self.current_prices[symbol] = fill_price

        # Update position
        if symbol not in self.positions:
            # New short position
            self.positions[symbol] = {
                'qty': -quantity,
                'avg_entry_price': fill_price,
                'side': 'short'
            }
        else:
            # Reducing/closing existing position or opening short
            pos = self.positions[symbol]
            old_qty = pos['qty']
            old_avg_price = pos['avg_entry_price']

            new_qty = old_qty - quantity

            if abs(new_qty) < 0.0001:  # Position closed
                pos['qty'] = 0
                pos['side'] = 'flat'
            elif new_qty > 0:  # Still long but reduced
                pos['qty'] = new_qty
                pos['side'] = 'long'
            elif new_qty < 0:  # Now short
                # If flipping from long to short, calculate new avg price
                if old_qty > 0:
                    # Closed long and opened short
                    short_qty = abs(new_qty)
                    pos['qty'] = new_qty
                    pos['avg_entry_price'] = fill_price
                    pos['side'] = 'short'
                else:
                    # Adding to short position
                    new_avg_price = ((abs(old_qty) * old_avg_price) + (quantity * fill_price)) / abs(new_qty)
                    pos['qty'] = new_qty
                    pos['avg_entry_price'] = new_avg_price
                    pos['side'] = 'short'

        # Update cash (add proceeds)
        self.cash += quantity * fill_price

        return {
            'status': 'filled',
            'symbol': symbol,
            'qty': quantity,
            'side': 'sell',
            'avg_fill_price': fill_price,
            'message': f"SIMULATED: Sold {quantity} of {symbol} @ ${fill_price:.2f}"
        }

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total = self.cash
        for symbol, pos in self.positions.items():
            if pos['qty'] != 0:
                current_price = self.current_prices.get(symbol, pos['avg_entry_price'])
                market_value = pos['qty'] * current_price
                total += market_value
        return total

    def get_account_info(self) -> Dict[str, Any]:
        """Get simulated account information"""
        portfolio_value = self._calculate_portfolio_value()
        return {
            'equity': portfolio_value,
            'cash': self.cash,
            'buying_power': self.cash,
            'portfolio_value': portfolio_value,
            'positions': self.positions.copy(),
            'trade_count': self.trade_count
        }

    def get_account(self) -> Dict[str, Any]:
        """Alias for get_account_info to match broker interface"""
        return self.get_account_info()

    def get_position_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Get position information for a specific symbol"""
        if symbol not in self.positions or self.positions[symbol]['qty'] == 0:
            return {
                'symbol': symbol,
                'qty': '0',
                'side': 'flat',
                'avg_entry_price': '0',
                'market_value': '0',
                'unrealized_pl': '0'
            }

        pos = self.positions[symbol]
        qty = pos['qty']
        avg_entry = pos['avg_entry_price']
        current_price = self.current_prices.get(symbol, avg_entry)

        market_value = qty * current_price

        # Calculate unrealized P&L
        if qty > 0:  # Long position
            unrealized_pl = (current_price - avg_entry) * qty
        else:  # Short position
            unrealized_pl = (avg_entry - current_price) * abs(qty)

        return {
            'symbol': symbol,
            'qty': str(qty),
            'side': pos['side'],
            'avg_entry_price': str(avg_entry),
            'market_value': str(market_value),
            'unrealized_pl': str(unrealized_pl)
        }

    def close_position(self, symbol: str, current_price: float = None) -> dict:
        """Close position for a symbol"""
        if symbol not in self.positions or self.positions[symbol]['qty'] == 0:
            return {'status': 'failed', 'error': 'No position to close'}

        pos = self.positions[symbol]
        qty = pos['qty']

        # Determine fill price
        if current_price is not None:
            fill_price = current_price
        else:
            fill_price = self.current_prices.get(symbol, pos['avg_entry_price'])

        self.current_prices[symbol] = fill_price

        # Close the position
        if qty > 0:
            # Close long
            self.cash += qty * fill_price
        else:
            # Close short
            self.cash -= abs(qty) * fill_price

        self.positions[symbol]['qty'] = 0
        self.positions[symbol]['side'] = 'flat'

        return {
            'status': 'filled',
            'symbol': symbol,
            'qty': abs(qty),
            'side': 'sell' if qty > 0 else 'buy',
            'avg_fill_price': fill_price,
            'message': f"SIMULATED: Closed position for {symbol} @ ${fill_price:.2f}"
        }

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an order (no-op for simple simulated broker)"""
        return {'status': 'cancelled', 'order_id': order_id}