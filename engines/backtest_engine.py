import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from strategies.base_strategy import BaseStrategy
import matplotlib.pyplot as plt


class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset engine state"""
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0
        self.realized_gains = 0
        self.trades = []
        self.balance_history = []
    
    def backtest(self, df: pd.DataFrame, strategy: BaseStrategy) -> pd.DataFrame:
        """
        Run backtest with given data and strategy
        
        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy instance
            
        Returns:
            DataFrame with trade results
        """
        self.reset()
        
        # Generate signals
        df_with_signals = strategy.generate_signals(df)
        signal_names = strategy.get_signal_names()
        
        buy_signal_col = signal_names['buy']
        sell_signal_col = signal_names['sell']
        
        # Process each bar
        for i in range(1, len(df_with_signals)):
            current_row = df_with_signals.iloc[i]
            
            trade_data = {}
            
            # Check for buy signal
            if current_row[buy_signal_col] and self.position != 1:
                trade_data['Time'] = current_row['timestamp']
                trade_data['Price'] = current_row['Close']
                trade_data['Position'] = 1
                
                # Close short position if exists
                if self.position == -1:
                    profit = self.entry_price - current_row['Close']
                    trade_data['Profit'] = profit
                    self.realized_gains += profit
                    trade_data['Realized'] = self.realized_gains
                    trade_data['Result'] = "Win" if profit > 0 else "Loss"
                else:
                    trade_data['Realized'] = 0
                
                trade_data['Balance'] = self.initial_balance + self.realized_gains
                self.trades.append(trade_data)
                self.balance_history.append(trade_data['Balance'])
                
                self.position = 1
                self.entry_price = current_row['Close']
            
            # Check for sell signal
            elif current_row[sell_signal_col] and self.position != -1:
                trade_data['Time'] = current_row['timestamp']
                trade_data['Price'] = current_row['Close']
                trade_data['Position'] = -1
                
                # Close long position if exists
                if self.position == 1:
                    profit = current_row['Close'] - self.entry_price
                    trade_data['Profit'] = profit
                    self.realized_gains += profit
                    trade_data['Realized'] = self.realized_gains
                    trade_data['Result'] = "Win" if profit > 0 else "Loss"
                else:
                    trade_data['Realized'] = 0
                
                trade_data['Balance'] = self.initial_balance + self.realized_gains
                self.trades.append(trade_data)
                self.balance_history.append(trade_data['Balance'])
                
                self.position = -1
                self.entry_price = current_row['Close']
        
        return pd.DataFrame(self.trades)
    
    def analyze_results(self, trade_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze backtest results"""
        if len(trade_df) == 0:
            return {
                'num_trades': 0,
                'winrate': 0,
                'final_balance': self.initial_balance,
                'net_returns': 0,
                'percent_return': 1.0,
                'avg_profit_per_trade': 0,
                'largest_win': 0,
                'largest_loss': 0
            }
        
        num_trades = len(trade_df)
        wins = trade_df[trade_df['Result'] == "Win"]
        winrate = len(wins) / num_trades * 100
        final_balance = trade_df['Balance'].iloc[-1]
        net_returns = final_balance - self.initial_balance
        percent_return = final_balance / self.initial_balance
        avg_profit_per_trade = net_returns / num_trades
        
        profits = trade_df['Profit'].dropna()
        largest_win = profits.max() if len(profits) > 0 else 0
        largest_loss = profits.min() if len(profits) > 0 else 0
        
        return {
            'num_trades': num_trades,
            'winrate': winrate,
            'final_balance': final_balance,
            'net_returns': net_returns,
            'percent_return': percent_return,
            'avg_profit_per_trade': avg_profit_per_trade,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def print_analysis(self, trade_df: pd.DataFrame):
        """Print analysis results"""
        analysis = self.analyze_results(trade_df)
        
        print(f"Winrate: {analysis['winrate']:.2f}%")
        print(f"Final Balance: {analysis['final_balance']:.5f}")
        print(f"Net Returns: {analysis['net_returns']:.5f}")
        print(f"Percentage Returns: {analysis['percent_return']:.8f}%")
        print(f"Total Trades: {analysis['num_trades']}")
        print(f"Average Profit per Trade: {analysis['avg_profit_per_trade']:.8f}")
        print(f"Largest Win: {analysis['largest_win']:.8f}")
        print(f"Largest Loss: {analysis['largest_loss']:.8f}")
    
    def plot_results(self, trade_df: pd.DataFrame):
        """Plot balance over time"""
        if len(trade_df) == 0:
            print("No trades to plot")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(trade_df['Time'], trade_df['Balance'], marker='o', linestyle='-')
        plt.title("Balance vs Time")
        plt.xlabel("Time")
        plt.ylabel("Balance ($)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()