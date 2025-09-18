import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from strategies.base_strategy import BaseStrategy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

        # Store the dataframe with signals for charting
        self.df_with_signals = df_with_signals
        self.strategy = strategy

        # Process each bar
        for i in range(1, len(df_with_signals)):
            current_row = df_with_signals.iloc[i]

            trade_data = {}

            # Check for buy signal
            if current_row[buy_signal_col] and self.position != 1:
                trade_data['Time'] = current_row['timestamp']
                trade_data['Price'] = current_row['Close']
                trade_data['Position'] = 1
                trade_data['Index'] = i

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
                trade_data['Index'] = i

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

    def plot_interactive_chart(self, trade_df: pd.DataFrame):
        """Plot interactive candlestick chart with trade markers and strategy indicators using Plotly"""
        if not hasattr(self, 'df_with_signals') or not hasattr(self, 'strategy'):
            print("No strategy data available for plotting")
            return

        df = self.df_with_signals.copy()

        # Prepare data for Plotly
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        else:
            df['datetime'] = df.index

        # Create subplots with secondary y-axis for volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[f'{self.strategy.name} Strategy - Price Chart', 'Volume'],
            row_heights=[0.7, 0.3]
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC',
                showlegend=False
            ),
            row=1, col=1
        )

        # Get strategy-specific indicators and add them to the chart
        indicators = self.strategy.get_indicators() if hasattr(self.strategy, 'get_indicators') else []

        # Define colors for indicators
        indicator_colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        for i, indicator in enumerate(indicators):
            if indicator in df.columns:
                color = indicator_colors[i % len(indicator_colors)]
                fig.add_trace(
                    go.Scatter(
                        x=df['datetime'],
                        y=df[indicator],
                        mode='lines',
                        name=indicator,
                        line=dict(color=color, width=2),
                        opacity=0.7
                    ),
                    row=1, col=1
                )

        # Add trade markers
        if len(trade_df) > 0:
            buy_trades = trade_df[trade_df['Position'] == 1]
            sell_trades = trade_df[trade_df['Position'] == -1]

            # Add buy markers
            if len(buy_trades) > 0:
                buy_times = []
                buy_prices = []
                for _, trade in buy_trades.iterrows():
                    if 'Index' in trade and trade['Index'] < len(df):
                        buy_times.append(df.iloc[trade['Index']]['datetime'])
                        buy_prices.append(trade['Price'])

                if buy_times:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_times,
                            y=buy_prices,
                            mode='markers',
                            name='Buy Signals',
                            marker=dict(
                                symbol='triangle-up',
                                size=12,
                                color='green',
                                line=dict(width=2, color='darkgreen')
                            )
                        ),
                        row=1, col=1
                    )

            # Add sell markers
            if len(sell_trades) > 0:
                sell_times = []
                sell_prices = []
                for _, trade in sell_trades.iterrows():
                    if 'Index' in trade and trade['Index'] < len(df):
                        sell_times.append(df.iloc[trade['Index']]['datetime'])
                        sell_prices.append(trade['Price'])

                if sell_times:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_times,
                            y=sell_prices,
                            mode='markers',
                            name='Sell Signals',
                            marker=dict(
                                symbol='triangle-down',
                                size=12,
                                color='red',
                                line=dict(width=2, color='darkred')
                            )
                        ),
                        row=1, col=1
                    )

        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=df['datetime'],
                y=df['Volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.8)',
                showlegend=False
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'Interactive Chart - {self.strategy.name} Strategy',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Update y-axes
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)

        # Show the plot
        try:
            fig.show()
        except Exception as e:
            print(f"Error creating interactive chart: {e}")
            print("Falling back to basic matplotlib chart...")
            self._plot_basic_chart(trade_df)

    def _plot_basic_chart(self, trade_df: pd.DataFrame):
        """Fallback basic chart using matplotlib"""
        if not hasattr(self, 'df_with_signals'):
            print("No data available for plotting")
            return

        df = self.df_with_signals

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Plot candlesticks (simplified as OHLC)
        ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1)

        # Add strategy indicators
        if hasattr(self.strategy, 'get_indicators'):
            indicators = self.strategy.get_indicators()
            for indicator in indicators:
                if indicator in df.columns:
                    ax1.plot(df.index, df[indicator], label=indicator, alpha=0.7)

        # Add trade markers
        if len(trade_df) > 0:
            buy_trades = trade_df[trade_df['Position'] == 1]
            sell_trades = trade_df[trade_df['Position'] == -1]

            buy_label_added = False
            sell_label_added = False

            for _, trade in buy_trades.iterrows():
                if 'Index' in trade and trade['Index'] < len(df):
                    label = 'Buy' if not buy_label_added else ""
                    ax1.scatter(trade['Index'], trade['Price'], color='green', marker='^', s=100, label=label, zorder=5)
                    buy_label_added = True

            for _, trade in sell_trades.iterrows():
                if 'Index' in trade and trade['Index'] < len(df):
                    label = 'Sell' if not sell_label_added else ""
                    ax1.scatter(trade['Index'], trade['Price'], color='red', marker='v', s=100, label=label, zorder=5)
                    sell_label_added = True

        ax1.set_title(f'{self.strategy.name} Strategy - Price Chart with Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot volume
        ax2.bar(df.index, df['Volume'], alpha=0.6, color='blue')
        ax2.set_title('Volume')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time')

        plt.tight_layout()
        plt.show()