"""
Interactive Brokers broker interface for live trading
Uses the IB TWS API for order execution on forex pairs
"""

import time
import threading
from typing import Optional, Dict
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order


class IBBroker(EWrapper, EClient):
    """Interactive Brokers broker interface"""

    def __init__(self):
        EClient.__init__(self, self)
        self.next_order_id = None
        self.positions = {}
        self.account_info = {}
        self.connected = False
        self.connection_event = threading.Event()
        self.position_event = threading.Event()
        self.account_event = threading.Event()
        self.order_status = {}

    # IB API Callbacks
    def nextValidId(self, orderId: int):
        """Called when connection is established"""
        super().nextValidId(orderId)
        self.next_order_id = orderId
        self.connected = True
        self.connection_event.set()
        print(f"✓ Connected to IB TWS. Next order ID: {orderId}")

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson="", errorTime=""):
        """Error callback"""
        if errorCode in [2104, 2106, 2158]:  # Informational messages
            pass
        elif errorCode == 502:
            print(f"✗ IB Connection error: {errorString}")
            self.connected = False
        else:
            print(f"IB Error {errorCode}: {errorString}")

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Position callback"""
        symbol = contract.symbol
        if contract.secType == 'CASH':
            symbol = f"{contract.symbol}_{contract.currency}"

        self.positions[symbol] = {
            'qty': float(position),
            'avg_entry_price': float(avgCost),
            'contract': contract
        }

    def positionEnd(self):
        """Called when all positions received"""
        self.position_event.set()

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """Account summary callback"""
        self.account_info[tag] = {'value': value, 'currency': currency}

    def accountSummaryEnd(self, reqId: int):
        """Called when account summary complete"""
        self.account_event.set()

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId,
                    parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        """Order status callback"""
        self.order_status[orderId] = {
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avg_fill_price': avgFillPrice
        }
        print(f"IB Order {orderId}: {status} | Filled: {filled} | Remaining: {remaining} | Avg Price: {avgFillPrice}")

    # Connection methods
    def connect_to_tws(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1) -> bool:
        """
        Connect to TWS or IB Gateway

        Args:
            host: TWS host
            port: TWS port (7497 for paper, 7496 for live)
            client_id: Client ID

        Returns:
            True if connected successfully
        """
        self.connect(host, port, client_id)

        # Start the client thread
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

        # Wait for connection
        if self.connection_event.wait(timeout=5):
            time.sleep(0.5)  # Give extra time for initialization
            return self.connected
        else:
            print("✗ Failed to connect to IB TWS")
            return False

    def disconnect_from_tws(self):
        """Disconnect from TWS"""
        if self.connected:
            self.disconnect()
            self.connected = False
            print("✓ Disconnected from IB TWS")

    # Trading methods
    def create_forex_contract(self, symbol: str) -> Contract:
        """
        Create forex contract for EUR/USD

        Args:
            symbol: Currency pair (e.g., 'EURUSD' or 'EUR_USD')

        Returns:
            Contract object
        """
        # Handle different formats
        if '_' in symbol:
            base = symbol.split('_')[0]
            quote = symbol.split('_')[1]
        else:
            base = symbol[:3]
            quote = symbol[3:]

        contract = Contract()
        contract.symbol = base
        contract.secType = 'CASH'
        contract.currency = quote
        contract.exchange = 'IDEALPRO'
        return contract

    def buy(self, symbol: str, quantity: float, order_type: str = "market", limit_price: float = None) -> dict:
        """
        Place buy order

        Args:
            symbol: Currency pair
            quantity: Quantity in base currency units (20000 = 20K)
            order_type: 'market' or 'limit'
            limit_price: Limit price (for limit orders)

        Returns:
            Order details
        """
        if not self.connected or not self.next_order_id:
            return {'status': 'failed', 'error': 'Not connected to IB TWS'}

        contract = self.create_forex_contract(symbol)

        order = Order()
        order.action = "BUY"
        order.totalQuantity = quantity
        order.orderType = "MKT" if order_type.lower() == "market" else "LMT"

        if order_type.lower() == "limit" and limit_price:
            order.lmtPrice = limit_price

        order_id = self.next_order_id
        self.next_order_id += 1

        self.placeOrder(order_id, contract, order)
        time.sleep(1)  # Wait for order submission

        return {
            'id': str(order_id),
            'status': 'submitted',
            'symbol': symbol,
            'side': 'buy',
            'qty': quantity,
            'order_type': order_type
        }

    def sell(self, symbol: str, quantity: float, order_type: str = "market", limit_price: float = None) -> dict:
        """
        Place sell order

        Args:
            symbol: Currency pair
            quantity: Quantity in base currency units
            order_type: 'market' or 'limit'
            limit_price: Limit price (for limit orders)

        Returns:
            Order details
        """
        if not self.connected or not self.next_order_id:
            return {'status': 'failed', 'error': 'Not connected to IB TWS'}

        contract = self.create_forex_contract(symbol)

        order = Order()
        order.action = "SELL"
        order.totalQuantity = quantity
        order.orderType = "MKT" if order_type.lower() == "market" else "LMT"

        if order_type.lower() == "limit" and limit_price:
            order.lmtPrice = limit_price

        order_id = self.next_order_id
        self.next_order_id += 1

        self.placeOrder(order_id, contract, order)
        time.sleep(1)  # Wait for order submission

        return {
            'id': str(order_id),
            'status': 'submitted',
            'symbol': symbol,
            'side': 'sell',
            'qty': quantity,
            'order_type': order_type
        }

    def close_position(self, symbol: str) -> dict:
        """
        Close position for symbol

        Args:
            symbol: Currency pair

        Returns:
            Order details
        """
        position = self.get_position_for_symbol(symbol)
        qty = float(position.get('qty', 0))

        if qty == 0:
            return {'status': 'failed', 'error': 'No position to close'}

        # Close position by placing opposite order
        if qty > 0:
            return self.sell(symbol, abs(qty), order_type="market")
        else:
            return self.buy(symbol, abs(qty), order_type="market")

    def cancel_order(self, order_id: str) -> dict:
        """
        Cancel order

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation status
        """
        try:
            self.cancelOrder(int(order_id))
            return {'status': 'cancelled', 'order_id': order_id}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    # Account information methods
    def get_position_for_symbol(self, symbol: str) -> dict:
        """
        Get position for specific symbol

        Args:
            symbol: Currency pair

        Returns:
            Position details
        """
        # Normalize symbol format
        if '_' not in symbol and len(symbol) == 6:
            symbol = f"{symbol[:3]}_{symbol[3:]}"

        self.positions = {}
        self.position_event.clear()
        self.reqPositions()

        self.position_event.wait(timeout=3)
        self.cancelPositions()

        position = self.positions.get(symbol, {})
        return {
            'qty': str(position.get('qty', 0)),
            'avg_entry_price': str(position.get('avg_entry_price', 0)),
            'side': 'long' if position.get('qty', 0) > 0 else 'short',
            'market_value': '0',
            'unrealized_pl': '0'
        }

    def get_account(self) -> dict:
        """
        Get account information

        Returns:
            Account details
        """
        self.account_info = {}
        self.account_event.clear()
        self.reqAccountSummary(9001, "All", "NetLiquidation,TotalCashValue,BuyingPower")

        self.account_event.wait(timeout=3)
        self.cancelAccountSummary(9001)

        net_liq = float(self.account_info.get('NetLiquidation', {}).get('value', 0))
        buying_power = float(self.account_info.get('BuyingPower', {}).get('value', 0))

        return {
            'equity': net_liq,
            'buying_power': buying_power,
            'portfolio_value': net_liq
        }

    def get_account_api(self) -> dict:
        """Alias for get_account"""
        return self.get_account()