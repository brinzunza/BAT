import time
import pandas as pd
from typing import Dict, Any, Optional, Callable
from strategies.base_strategy import BaseStrategy
from data_providers.base_provider import BaseDataProvider
from datetime import datetime
import logging


class LiveTradingEngine:
    """Live trading engine for executing strategies in real-time"""
    
    def __init__(self, 
                 data_provider: BaseDataProvider,
                 broker_interface: Optional[object] = None,
                 initial_balance: float = 10000):
        self.data_provider = data_provider
        self.broker_interface = broker_interface
        self.initial_balance = initial_balance
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
    
    def set_broker_interface(self, broker_interface):
        """Set the broker interface for live trading"""
        self.broker_interface = broker_interface
    
    def execute_buy_order(self, symbol: str, quantity: float = 1) -> dict:
        """Execute buy order and return order details from Alpaca"""
        try:
            if self.broker_interface:
                result = self.broker_interface.buy(symbol, quantity)
                return result if isinstance(result, dict) else {'status': 'executed', 'details': result}
            else:
                return {'status': 'simulated', 'symbol': symbol, 'qty': quantity, 'side': 'buy'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def execute_sell_order(self, symbol: str, quantity: float = 1) -> dict:
        """Execute sell order and return order details from Alpaca"""
        try:
            if self.broker_interface:
                result = self.broker_interface.sell(symbol, quantity)
                return result if isinstance(result, dict) else {'status': 'executed', 'details': result}
            else:
                return {'status': 'simulated', 'symbol': symbol, 'qty': quantity, 'side': 'sell'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

    def close_position(self, symbol: str) -> dict:
        """Close position using Alpaca close position API"""
        try:
            if self.broker_interface and hasattr(self.broker_interface, 'close_position'):
                result = self.broker_interface.close_position(symbol)
                return result if isinstance(result, dict) else {'status': 'executed', 'details': result}
            else:
                return {'status': 'simulated', 'symbol': symbol, 'action': 'close_position'}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

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

        # Get current position from Alpaca
        alpaca_position = self.get_alpaca_position(symbol)
        current_qty = float(alpaca_position['qty'])

        # Validate signals before acting (silent validation)
        signals_valid = self._validate_signals(df_with_signals, strategy)
        if not signals_valid:
            return

        # Process buy signal - only buy if no position exists
        if buy_signal and current_qty == 0:
            # Get account info for display
            account_info = self.get_alpaca_account()
            account_balance = float(account_info.get('equity', 0))
            unrealized_pnl = float(alpaca_position.get('unrealized_pl', 0))
            session_pnl = account_balance - self.initial_balance

            print(f"\n🔵 BUY SIGNAL - Attempting to buy {quantity} {symbol} at ${current_price:.2f}")
            print(f"    💰 Account: ${account_balance:.2f} | Unrealized: ${unrealized_pnl:.2f} | Session: ${session_pnl:.2f}")

            # Final trade confirmation (silent)
            if self._confirm_trade_execution('BUY', symbol, quantity, current_price, alpaca_position):
                # Open long position
                result = self.execute_buy_order(symbol, quantity)
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

        # Process sell signal - close position if it exists
        elif sell_signal and current_qty > 0:
            # Get account info for display
            account_info = self.get_alpaca_account()
            account_balance = float(account_info.get('equity', 0))
            unrealized_pnl = float(alpaca_position.get('unrealized_pl', 0))
            session_pnl = account_balance - self.initial_balance

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

        else:
            # Update position info from Alpaca (silent)
            if current_qty > 0:
                self.position = 1
                self.entry_price = float(alpaca_position['avg_entry_price'])
            elif current_qty < 0:
                self.position = -1
                self.entry_price = float(alpaca_position['avg_entry_price'])
            else:
                self.position = 0
                self.entry_price = 0
    
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
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary using Alpaca account data"""
        # Get account info from Alpaca
        account_info = self.get_alpaca_account()

        current_balance = float(account_info.get('equity', self.current_balance))
        portfolio_value = float(account_info.get('portfolio_value', current_balance))

        trade_df = self.get_trade_history()

        if len(trade_df) == 0:
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

        # Calculate trade performance from historical data
        profitable_trades = 0
        losing_trades = 0

        # For trades that have profit info (legacy trades)
        if 'profit' in trade_df.columns:
            profitable_trades = len(trade_df[trade_df['profit'] > 0])
            losing_trades = len(trade_df[trade_df['profit'] < 0])

        return {
            'total_trades': len(trade_df),
            'profitable_trades': profitable_trades,
            'losing_trades': losing_trades,
            'win_rate': profitable_trades / len(trade_df) * 100 if len(trade_df) > 0 else 0,
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