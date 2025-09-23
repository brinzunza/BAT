import time
import pandas as pd
from typing import Dict, Any, Optional, Callable, List
from strategies.base_strategy import BaseStrategy
from data_providers.base_provider import BaseDataProvider
from datetime import datetime
import logging


class LiveTradingEngine:
    """Live trading engine for executing strategies in real-time"""
    
    def __init__(self,
                 data_provider: BaseDataProvider,
                 broker_interface: Optional[object] = None,
                 initial_balance: float = 10000,
                 trading_mode: str = "long_only"):
        self.data_provider = data_provider
        self.broker_interface = broker_interface
        self.initial_balance = initial_balance
        self.trading_mode = trading_mode
        self.reset()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def reset(self):
        """Reset engine state"""
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0
        self.realized_gains = 0
        self.trades = []
        self.running = False
        self.current_balance = self.initial_balance
        self.use_alpaca_data = True  # Flag to use only Alpaca data

        # Order tracking for limit orders
        self.pending_orders = {}  # Track pending limit orders by order_id
        self.order_timestamps = {}  # Track order placement times
    
    def set_broker_interface(self, broker_interface):
        """Set the broker interface for live trading"""
        self.broker_interface = broker_interface
    
    def execute_buy_order(self, symbol: str, quantity: float = 1, order_type: str = "market", limit_price: float = None) -> dict:
        """Execute buy order and return order details from Alpaca"""
        try:
            if self.broker_interface:
                if order_type == "limit" and limit_price is not None:
                    # For limit orders, pass the limit price to the broker interface
                    result = self.broker_interface.buy(symbol, quantity, order_type="limit", limit_price=limit_price)
                else:
                    result = self.broker_interface.buy(symbol, quantity, order_type=order_type)

                # Track pending limit orders
                if order_type == "limit" and isinstance(result, dict) and 'id' in result:
                    self.pending_orders[result['id']] = {
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': quantity,
                        'limit_price': limit_price,
                        'order_type': order_type
                    }
                    self.order_timestamps[result['id']] = datetime.now()

                return result if isinstance(result, dict) else {'status': 'executed', 'details': result}
            else:
                return {'status': 'simulated', 'symbol': symbol, 'qty': quantity, 'side': 'buy', 'order_type': order_type}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def execute_sell_order(self, symbol: str, quantity: float = 1, order_type: str = "market", limit_price: float = None) -> dict:
        """Execute sell order and return order details from Alpaca"""
        try:
            if self.broker_interface:
                if order_type == "limit" and limit_price is not None:
                    # For limit orders, pass the limit price to the broker interface
                    result = self.broker_interface.sell(symbol, quantity, order_type="limit", limit_price=limit_price)
                else:
                    result = self.broker_interface.sell(symbol, quantity, order_type=order_type)

                # Track pending limit orders
                if order_type == "limit" and isinstance(result, dict) and 'id' in result:
                    self.pending_orders[result['id']] = {
                        'symbol': symbol,
                        'side': 'sell',
                        'quantity': quantity,
                        'limit_price': limit_price,
                        'order_type': order_type
                    }
                    self.order_timestamps[result['id']] = datetime.now()

                return result if isinstance(result, dict) else {'status': 'executed', 'details': result}
            else:
                return {'status': 'simulated', 'symbol': symbol, 'qty': quantity, 'side': 'sell', 'order_type': order_type}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def close_position(self, symbol: str) -> dict:
        """Close position using Alpaca close position API - always use market orders"""
        try:
            if self.broker_interface and hasattr(self.broker_interface, 'close_position'):
                result = self.broker_interface.close_position(symbol)
                return result if isinstance(result, dict) else {'status': 'executed', 'details': result}
            else:
                return {'status': 'simulated', 'symbol': symbol, 'action': 'close_position'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending order"""
        try:
            if self.broker_interface and hasattr(self.broker_interface, 'cancel_order'):
                result = self.broker_interface.cancel_order(order_id)
                # Remove from tracking
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                if order_id in self.order_timestamps:
                    del self.order_timestamps[order_id]
                return result
            else:
                return {'status': 'simulated', 'order_id': order_id, 'action': 'cancel'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def check_and_cancel_expired_orders(self, symbol: str, timeout_minutes: int = 1):
        """Check for expired limit orders and cancel them"""
        current_time = datetime.now()
        expired_orders = []

        for order_id, order_time in self.order_timestamps.items():
            if order_id in self.pending_orders:
                order_info = self.pending_orders[order_id]
                if order_info['symbol'] == symbol:
                    time_diff = (current_time - order_time).total_seconds() / 60
                    if time_diff >= timeout_minutes:
                        expired_orders.append(order_id)

        # Cancel expired orders
        for order_id in expired_orders:
            order_info = self.pending_orders.get(order_id, {})
            print(f"\n⏰ CANCELING EXPIRED ORDER - {order_info.get('side', 'unknown').upper()} order for {symbol} (expired after {timeout_minutes} minute(s))")
            result = self.cancel_order(order_id)
            if result.get('status') != 'failed':
                print(f"✅ ORDER CANCELED - Order ID: {order_id}")
            else:
                print(f"❌ CANCEL FAILED - {result.get('error', 'Unknown error')}")

        return len(expired_orders)

    def _cancel_pending_buy_orders(self, symbol: str):
        """Cancel all pending buy orders for a symbol"""
        orders_to_cancel = []
        for order_id, order_info in self.pending_orders.items():
            if order_info['symbol'] == symbol and order_info['side'] == 'buy':
                orders_to_cancel.append(order_id)

        for order_id in orders_to_cancel:
            print(f"🚫 CANCELING PENDING BUY ORDER - Order ID: {order_id}")
            self.cancel_order(order_id)

    def _cancel_pending_sell_orders(self, symbol: str):
        """Cancel all pending sell orders for a symbol"""
        orders_to_cancel = []
        for order_id, order_info in self.pending_orders.items():
            if order_info['symbol'] == symbol and order_info['side'] == 'sell':
                orders_to_cancel.append(order_id)

        for order_id in orders_to_cancel:
            print(f"🚫 CANCELING PENDING SELL ORDER - Order ID: {order_id}")
            self.cancel_order(order_id)

    def get_alpaca_position(self, symbol: str) -> dict:
        """Get current position from Alpaca API"""
        try:
            if self.broker_interface and hasattr(self.broker_interface, 'get_position_for_symbol'):
                return self.broker_interface.get_position_for_symbol(symbol)
            return {'qty': '0', 'side': 'long', 'avg_entry_price': '0', 'market_value': '0', 'unrealized_pl': '0'}
        except Exception as e:
            self.logger.error(f"Error getting Alpaca position: {e}")
            return {'qty': '0', 'side': 'long', 'avg_entry_price': '0', 'market_value': '0', 'unrealized_pl': '0'}

    def get_alpaca_account(self) -> dict:
        """Get account information from Alpaca API"""
        try:
            if self.broker_interface and hasattr(self.broker_interface, 'get_account_api'):
                return self.broker_interface.get_account_api()
            elif self.broker_interface and hasattr(self.broker_interface, 'get_account'):
                return self.broker_interface.get_account()
            return {'equity': self.current_balance, 'buying_power': self.current_balance, 'portfolio_value': self.current_balance}
        except Exception as e:
            self.logger.error(f"Error getting Alpaca account: {e}")
            return {'equity': self.current_balance, 'buying_power': self.current_balance, 'portfolio_value': self.current_balance}

    def _validate_signals(self, df: pd.DataFrame, strategy: BaseStrategy) -> bool:
        """Validate signals before executing trades (silent)"""
        try:
            # Ensure we have enough data for reliable signals
            if len(df) < strategy.get_required_lookback():
                return False

            # Get the latest few signals to check for consistency
            signal_names = strategy.get_signal_names()
            latest_row = df.iloc[-1]
            buy_signal = latest_row[signal_names['buy']]
            sell_signal = latest_row[signal_names['sell']]

            # Check if signals are not conflicting (both buy and sell at same time)
            if buy_signal and sell_signal:
                return False

            # Additional strategy-specific validations
            if hasattr(strategy, 'validate_signal_conditions'):
                if not strategy.validate_signal_conditions(df):
                    return False

            return True

        except Exception:
            return False

    def _confirm_trade_execution(self, action: str, symbol: str, quantity: float,
                               current_price: float, position_data: dict) -> bool:
        """Final confirmation before executing trades (silent)"""
        try:
            # Get account info for buying power check
            account_info = self.get_alpaca_account()
            buying_power = float(account_info.get('buying_power', 0))

            # For buy orders, check buying power
            if action == 'BUY':
                trade_value = quantity * current_price
                if trade_value > buying_power:
                    return False

            # For close position, just check that position exists (handled in main logic)
            return True

        except Exception:
            return False

    def process_signals(self,
                       df: pd.DataFrame,
                       strategy: BaseStrategy,
                       symbol: str,
                       quantity: float = 1):
        """Process trading signals from strategy"""
        if len(df) == 0:
            return
        
        # Get signals from strategy
        df_with_signals = strategy.generate_signals(df)
        signal_names = strategy.get_signal_names()
        
        # Get latest signals
        latest_row = df_with_signals.iloc[-1]
        buy_signal = latest_row[signal_names['buy']]
        sell_signal = latest_row[signal_names['sell']]
        
        current_price = latest_row['Close']
        timestamp = latest_row['timestamp']

        # Check for pending limit order fills (for SimulatedBroker)
        if hasattr(self.broker_interface, 'check_pending_orders'):
            filled_orders = self.broker_interface.check_pending_orders()
            if filled_orders > 0:
                print(f"🎯 {filled_orders} limit order(s) filled")

        # Check for expired orders and cancel them (for long-only mode: 1 minute timeout)
        if self.trading_mode == "long_only":
            self.check_and_cancel_expired_orders(symbol, timeout_minutes=1)

        # Get current position from Alpaca
        alpaca_position = self.get_alpaca_position(symbol)
        current_qty = float(alpaca_position['qty'])

        # Debug output for signal detection
        if buy_signal or sell_signal:
            print(f"\n🔍 SIGNAL DETECTED:")
            print(f"    📈 Buy Signal: {buy_signal}")
            print(f"    📉 Sell Signal: {sell_signal}")
            print(f"    📊 Current Position: {current_qty} {symbol}")
            print(f"    💰 Current Price: ${current_price:.2f}")

        # Validate signals before acting
        signals_valid = self._validate_signals(df_with_signals, strategy)
        if not signals_valid:
            if buy_signal or sell_signal:
                print(f"    ❌ SIGNAL VALIDATION FAILED - Signal rejected")
            return

        # Process signals based on trading mode
        if self.trading_mode == "long_only":
            self._process_long_only_signals(buy_signal, sell_signal, current_qty, alpaca_position,
                                          symbol, quantity, current_price, timestamp)
        else:  # long_short mode
            self._process_long_short_signals(buy_signal, sell_signal, current_qty, alpaca_position,
                                           symbol, quantity, current_price, timestamp)

        # Update position info from Alpaca - ensure sync after manual changes
        previous_position = self.position
        if current_qty > 0:
            self.position = 1
            self.entry_price = float(alpaca_position['avg_entry_price'])
        elif current_qty < 0:
            self.position = -1
            self.entry_price = float(alpaca_position['avg_entry_price'])
        else:
            self.position = 0
            self.entry_price = 0

        # Debug output if position changed without engine action (manual closure)
        if previous_position != self.position and not (buy_signal or sell_signal):
            print(f"\n🔄 POSITION SYNC - Position changed externally")
            print(f"    📊 Previous: {previous_position} → Current: {self.position}")
            print(f"    💰 Alpaca Qty: {current_qty}")
            if self.position == 0 and previous_position != 0:
                print(f"    ✂️ Position was manually closed - ready for new signals")

    def _process_long_only_signals(self, buy_signal, sell_signal, current_qty, alpaca_position,
                                 symbol, quantity, current_price, timestamp):
        """Process signals for long-only trading mode"""
        # Get account info for display
        account_info = self.get_alpaca_account()
        account_balance = float(account_info.get('equity', 0))
        unrealized_pnl = float(alpaca_position.get('unrealized_pl', 0))
        session_pnl = account_balance - self.initial_balance

        # Process buy signal - only buy if no position exists
        if buy_signal and current_qty == 0:
            print(f"\n🔵 BUY SIGNAL - Attempting to buy {quantity} {symbol} at ${current_price:.2f}")
            print(f"    💰 Account: ${account_balance:.2f} | Unrealized: ${unrealized_pnl:.2f} | Session: ${session_pnl:.2f}")

            # Final trade confirmation (silent)
            if self._confirm_trade_execution('BUY', symbol, quantity, current_price, alpaca_position):
                # Open long position with LIMIT order at current price
                result = self.execute_buy_order(symbol, quantity, order_type="limit", limit_price=current_price)
                if result.get('status') != 'failed':
                    # Get updated account info after trade
                    updated_account = self.get_alpaca_account()
                    updated_balance = float(updated_account.get('equity', account_balance))
                    updated_session_pnl = updated_balance - self.initial_balance

                    print(f"✅ BUY ORDER FILLED - {quantity} {symbol} at ${current_price:.2f}")
                    print(f"    💰 Updated Account: ${updated_balance:.2f} | Session P&L: ${updated_session_pnl:.2f}")

                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'buy_long',
                        'price': current_price,
                        'quantity': quantity,
                        'order_details': result
                    })
                else:
                    print(f"❌ BUY ORDER FAILED - {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ BUY SIGNAL REJECTED - Insufficient funds or invalid conditions")

        # Debug output when buy signal is ignored due to existing position
        elif buy_signal and current_qty != 0:
            print(f"\n🔵 BUY SIGNAL IGNORED - Already have position: {current_qty} {symbol}")
            print(f"    💰 Current Position Value: ${float(alpaca_position.get('market_value', 0)):.2f}")
            print(f"    📈 Unrealized P&L: ${unrealized_pnl:.2f}")

        # Process sell signal - close position if it exists
        elif sell_signal and current_qty > 0:
            print(f"\n🔴 SELL SIGNAL - Attempting to close position for {symbol} at ${current_price:.2f}")
            print(f"    💰 Account: ${account_balance:.2f} | Unrealized: ${unrealized_pnl:.2f} | Session: ${session_pnl:.2f}")

            # Close the existing long position
            result = self.close_position(symbol)
            if result.get('status') != 'failed':
                # Get updated account info after trade
                updated_account = self.get_alpaca_account()
                updated_balance = float(updated_account.get('equity', account_balance))
                updated_session_pnl = updated_balance - self.initial_balance

                print(f"✅ POSITION CLOSED - {current_qty} {symbol} at market price")
                print(f"    💰 Updated Account: ${updated_balance:.2f} | Session P&L: ${updated_session_pnl:.2f}")

                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'close_position',
                    'price': current_price,
                    'quantity': current_qty,
                    'order_details': result
                })
            else:
                print(f"❌ CLOSE POSITION FAILED - {result.get('error', 'Unknown error')}")

    def _process_long_short_signals(self, buy_signal, sell_signal, current_qty, alpaca_position,
                                  symbol, quantity, current_price, timestamp):
        """Process signals for long/short trading mode - uses persistent limit orders"""
        # Get account info for display
        account_info = self.get_alpaca_account()
        account_balance = float(account_info.get('equity', 0))
        unrealized_pnl = float(alpaca_position.get('unrealized_pl', 0))
        session_pnl = account_balance - self.initial_balance

        # Cancel any conflicting pending orders for opposite direction
        if buy_signal:
            self._cancel_pending_sell_orders(symbol)
        elif sell_signal:
            self._cancel_pending_buy_orders(symbol)

        # Process buy signal
        if buy_signal:
            if current_qty < 0:  # Currently short, close short position first
                print(f"\n🔵 BUY SIGNAL - Closing short position for {symbol} at ${current_price:.2f}")
                print(f"    💰 Account: ${account_balance:.2f} | Unrealized: ${unrealized_pnl:.2f} | Session: ${session_pnl:.2f}")

                result = self.close_position(symbol)
                if result.get('status') != 'failed':
                    print(f"✅ SHORT POSITION CLOSED - {abs(current_qty)} {symbol}")
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'close_short',
                        'price': current_price,
                        'quantity': abs(current_qty),
                        'order_details': result
                    })

            elif current_qty == 0:  # No position, open long
                print(f"\n🔵 BUY SIGNAL - Attempting to buy {quantity} {symbol} at ${current_price:.2f} (LIMIT ORDER)")
                print(f"    💰 Account: ${account_balance:.2f} | Unrealized: ${unrealized_pnl:.2f} | Session: ${session_pnl:.2f}")

                if self._confirm_trade_execution('BUY', symbol, quantity, current_price, alpaca_position):
                    result = self.execute_buy_order(symbol, quantity, order_type="limit", limit_price=current_price)
                    if result.get('status') != 'failed':
                        updated_account = self.get_alpaca_account()
                        updated_balance = float(updated_account.get('equity', account_balance))
                        updated_session_pnl = updated_balance - self.initial_balance

                        print(f"✅ BUY ORDER FILLED - {quantity} {symbol} at ${current_price:.2f}")
                        print(f"    💰 Updated Account: ${updated_balance:.2f} | Session P&L: ${updated_session_pnl:.2f}")

                        self.trades.append({
                            'timestamp': timestamp,
                            'action': 'buy_long',
                            'price': current_price,
                            'quantity': quantity,
                            'order_details': result
                        })
                    else:
                        print(f"❌ BUY ORDER FAILED - {result.get('error', 'Unknown error')}")
                else:
                    print(f"❌ BUY SIGNAL REJECTED - Insufficient funds or invalid conditions")

        # Process sell signal
        elif sell_signal:
            if current_qty > 0:  # Currently long, close long position first
                print(f"\n🔴 SELL SIGNAL - Closing long position for {symbol} at ${current_price:.2f}")
                print(f"    💰 Account: ${account_balance:.2f} | Unrealized: ${unrealized_pnl:.2f} | Session: ${session_pnl:.2f}")

                result = self.close_position(symbol)
                if result.get('status') != 'failed':
                    print(f"✅ LONG POSITION CLOSED - {current_qty} {symbol}")
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'close_long',
                        'price': current_price,
                        'quantity': current_qty,
                        'order_details': result
                    })

            elif current_qty == 0:  # No position, open short
                print(f"\n🔴 SELL SIGNAL - Attempting to short {quantity} {symbol} at ${current_price:.2f} (LIMIT ORDER)")
                print(f"    💰 Account: ${account_balance:.2f} | Unrealized: ${unrealized_pnl:.2f} | Session: ${session_pnl:.2f}")

                if self._confirm_trade_execution('SELL', symbol, quantity, current_price, alpaca_position):
                    result = self.execute_sell_order(symbol, quantity, order_type="limit", limit_price=current_price)
                    if result.get('status') != 'failed':
                        updated_account = self.get_alpaca_account()
                        updated_balance = float(updated_account.get('equity', account_balance))
                        updated_session_pnl = updated_balance - self.initial_balance

                        print(f"✅ SHORT ORDER FILLED - {quantity} {symbol} at ${current_price:.2f}")
                        print(f"    💰 Updated Account: ${updated_balance:.2f} | Session P&L: ${updated_session_pnl:.2f}")

                        self.trades.append({
                            'timestamp': timestamp,
                            'action': 'sell_short',
                            'price': current_price,
                            'quantity': quantity,
                            'order_details': result
                        })
                    else:
                        print(f"❌ SHORT ORDER FAILED - {result.get('error', 'Unknown error')}")
                else:
                    print(f"❌ SELL SIGNAL REJECTED - Invalid conditions")

    def run_strategy(self, 
                    strategy: BaseStrategy,
                    symbol: str,
                    quantity: float = 1,
                    sleep_interval: int = 60,
                    max_iterations: Optional[int] = None):
        """
        Run strategy continuously
        
        Args:
            strategy: Strategy instance
            symbol: Trading symbol
            quantity: Position size
            sleep_interval: Seconds between iterations
            max_iterations: Maximum iterations (None for infinite)
        """
        self.running = True
        iteration = 0
        
        self.logger.info(f"Starting live trading for {strategy.name} on {symbol}")
        
        try:
            while self.running:
                # Check max iterations
                if max_iterations and iteration >= max_iterations:
                    self.logger.info("Max iterations reached, stopping...")
                    break
                
                try:
                    # Get latest data
                    df = self.data_provider.get_live_data(symbol)
                    
                    # Process signals
                    self.process_signals(df, strategy, symbol, quantity)

                    # Display trading stats every iteration
                    self._display_trading_stats(iteration + 1, symbol)
                    
                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration + 1}: {e}")
                
                iteration += 1
                
                # Wait before next iteration
                if self.running:
                    time.sleep(sleep_interval)
        
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, stopping...")
            self.stop()
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        self.logger.info("Live trading engine stopped")
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        return pd.DataFrame(self.trades)
    
    def _get_completed_trades(self) -> List[Dict]:
        """Parse trade history to identify completed trade pairs (buy/close cycles)"""
        trade_df = self.get_trade_history()
        if len(trade_df) == 0:
            return []

        completed_trades = []
        open_positions = {}  # Track open positions by symbol

        for _, trade in trade_df.iterrows():
            action = trade['action']

            # Opening position
            if action in ['buy_long', 'sell_short']:
                # If we already have an open position for this symbol, something's wrong
                # but we'll treat it as a new position
                open_positions[action] = trade

            # Closing position
            elif action in ['close_position', 'close_long', 'close_short']:
                # Find the matching open position
                entry_trade = None
                if action in ['close_position', 'close_long'] and 'buy_long' in open_positions:
                    entry_trade = open_positions.pop('buy_long')
                elif action == 'close_short' and 'sell_short' in open_positions:
                    entry_trade = open_positions.pop('sell_short')

                if entry_trade is not None:
                    # Calculate profit/loss
                    entry_price = entry_trade['price']
                    exit_price = trade['price']

                    if entry_trade['action'] == 'buy_long':
                        profit = (exit_price - entry_price) * trade['quantity']
                    else:  # sell_short
                        profit = (entry_price - exit_price) * trade['quantity']

                    completed_trades.append({
                        'entry_time': entry_trade['timestamp'],
                        'exit_time': trade['timestamp'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': trade['quantity'],
                        'profit': profit,
                        'action_type': entry_trade['action'],
                        'is_win': profit > 0
                    })

        return completed_trades

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary using Alpaca account data and completed trades"""
        # Get account info from Alpaca
        account_info = self.get_alpaca_account()

        current_balance = float(account_info.get('equity', self.current_balance))
        portfolio_value = float(account_info.get('portfolio_value', current_balance))

        # Get completed trades (buy/close pairs)
        completed_trades = self._get_completed_trades()

        if len(completed_trades) == 0:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'current_balance': current_balance,
                'portfolio_value': portfolio_value,
                'total_return': current_balance - self.initial_balance,
                'percent_return': (current_balance / self.initial_balance - 1) * 100,
                'current_position': self.position
            }

        # Calculate trade performance from completed trades
        profitable_trades = sum(1 for trade in completed_trades if trade['is_win'])
        losing_trades = len(completed_trades) - profitable_trades

        return {
            'total_trades': len(completed_trades),
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': profitable_trades / len(completed_trades) * 100 if len(completed_trades) > 0 else 0,
            'current_balance': current_balance,
            'portfolio_value': portfolio_value,
            'total_return': current_balance - self.initial_balance,
            'percent_return': (current_balance / self.initial_balance - 1) * 100,
            'current_position': self.position
        }

    def _display_trading_stats(self, iteration: int, symbol: str):
        """Display current trading statistics using Alpaca data"""
        performance = self.get_performance_summary()
        alpaca_position = self.get_alpaca_position(symbol)

        # Get unrealized P&L directly from Alpaca
        unrealized_pnl = float(alpaca_position['unrealized_pl'])
        current_qty = float(alpaca_position['qty'])
        avg_entry_price = float(alpaca_position['avg_entry_price'])
        market_value = float(alpaca_position['market_value'])

        # Format position status using Alpaca data
        position_status = "FLAT"
        if current_qty > 0:
            position_status = f"LONG {current_qty} @ ${avg_entry_price:.2f}"
        elif current_qty < 0:
            position_status = f"SHORT {abs(current_qty)} @ ${avg_entry_price:.2f}"

        # Print stats
        print("\n" + "=" * 60)
        print(f"📊 LIVE TRADING STATS - Iteration {iteration} (Alpaca Data)")
        print("=" * 60)
        print(f"Symbol: {symbol}")
        print(f"Position: {position_status}")
        print(f"Market Value: ${market_value:.2f}")
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Win Rate: {performance['win_rate']:.1f}%")
        print(f"Account Equity: ${performance['current_balance']:.2f}")
        print(f"Portfolio Value: ${performance['portfolio_value']:.2f}")
        print(f"Total Return: ${performance['total_return']:.2f}")
        print(f"Unrealized P&L: ${unrealized_pnl:.2f}")
        print(f"Total P&L: ${performance['total_return'] + unrealized_pnl:.2f}")
        print(f"Percent Return: {performance['percent_return']:.2f}%")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)