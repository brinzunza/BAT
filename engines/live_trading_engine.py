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
    
    def set_broker_interface(self, broker_interface):
        """Set the broker interface for live trading"""
        self.broker_interface = broker_interface
    
    def execute_buy_order(self, symbol: str, quantity: float = 1) -> bool:
        """Execute buy order"""
        try:
            if self.broker_interface:
                result = self.broker_interface.buy(symbol, quantity)
                self.logger.info(f"Buy order executed: {result}")
                return True
            else:
                self.logger.info(f"SIMULATION: Buy {quantity} of {symbol}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to execute buy order: {e}")
            return False
    
    def execute_sell_order(self, symbol: str, quantity: float = 1) -> bool:
        """Execute sell order"""
        try:
            if self.broker_interface:
                result = self.broker_interface.sell(symbol, quantity)
                self.logger.info(f"Sell order executed: {result}")
                return True
            else:
                self.logger.info(f"SIMULATION: Sell {quantity} of {symbol}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to execute sell order: {e}")
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
        
        # Process buy signal
        if buy_signal and self.position != 1:
            self.logger.info(f"Buy signal detected at {timestamp}, price: {current_price}")
            
            # Close short position if exists
            if self.position == -1:
                if self.execute_sell_order(symbol, quantity):  # Cover short
                    profit = self.entry_price - current_price
                    self.realized_gains += profit
                    self.current_balance += profit
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'cover_short',
                        'price': current_price,
                        'profit': profit,
                        'balance': self.current_balance
                    })
            
            # Open long position
            if self.execute_buy_order(symbol, quantity):
                self.position = 1
                self.entry_price = current_price
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'buy_long',
                    'price': current_price,
                    'profit': 0,
                    'balance': self.current_balance
                })
        
        # Process sell signal
        elif sell_signal and self.position != -1:
            self.logger.info(f"Sell signal detected at {timestamp}, price: {current_price}")
            
            # Close long position if exists
            if self.position == 1:
                if self.execute_sell_order(symbol, quantity):
                    profit = current_price - self.entry_price
                    self.realized_gains += profit
                    self.current_balance += profit
                    
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'sell_long',
                        'price': current_price,
                        'profit': profit,
                        'balance': self.current_balance
                    })
            
            # Open short position (if broker supports it)
            if self.execute_sell_order(symbol, quantity):  # Short sell
                self.position = -1
                self.entry_price = current_price
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'sell_short',
                    'price': current_price,
                    'profit': 0,
                    'balance': self.current_balance
                })
        
        else:
            self.logger.info(f"No position change. Current position: {self.position}")
    
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
                    
                    # Log current status
                    self.logger.info(f"Iteration {iteration + 1} completed. "
                                   f"Position: {self.position}, "
                                   f"Balance: {self.current_balance:.5f}")
                    
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
        """Get performance summary"""
        trade_df = self.get_trade_history()
        
        if len(trade_df) == 0:
            return {
                'total_trades': 0,
                'current_balance': self.current_balance,
                'total_return': 0,
                'percent_return': 0
            }
        
        profitable_trades = trade_df[trade_df['profit'] > 0]
        losing_trades = trade_df[trade_df['profit'] < 0]
        
        return {
            'total_trades': len(trade_df),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(profitable_trades) / len(trade_df) * 100 if len(trade_df) > 0 else 0,
            'current_balance': self.current_balance,
            'total_return': self.current_balance - self.initial_balance,
            'percent_return': (self.current_balance / self.initial_balance - 1) * 100,
            'current_position': self.position
        }