import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from strategies.base_strategy import BaseStrategy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BacktestEngine:
    """Backtesting engine for trading strategies supporting stocks and crypto"""

    def __init__(self, initial_balance: float = 10000, trading_mode: str = "long_only", symbol: str = ""):
        self.initial_balance = initial_balance
        self.trading_mode = trading_mode
        self.symbol = symbol
        self.is_crypto = '/' in symbol  # Determine if it's crypto based on symbol format
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

        # Process each bar based on trading mode
        for i in range(1, len(df_with_signals)):
            current_row = df_with_signals.iloc[i]
            buy_signal = current_row[buy_signal_col]
            sell_signal = current_row[sell_signal_col]

            if self.trading_mode == "long_only":
                self._process_long_only_signals(current_row, buy_signal, sell_signal, i)
            else:  # long_short mode
                self._process_long_short_signals(current_row, buy_signal, sell_signal, i)

        return pd.DataFrame(self.trades)

    def _process_long_only_signals(self, current_row, buy_signal, sell_signal, i):
        """Process signals for long-only trading mode"""
        trade_data = {}

        # Buy signal - only buy if no position exists
        if buy_signal and self.position == 0:
            trade_data['Time'] = current_row['timestamp']
            trade_data['Price'] = current_row['Close']
            trade_data['Position'] = 1
            trade_data['Index'] = i
            trade_data['Action'] = 'BUY'
            trade_data['Realized'] = 0
            trade_data['Balance'] = self.initial_balance + self.realized_gains

            self.trades.append(trade_data)
            self.balance_history.append(trade_data['Balance'])

            self.position = 1
            self.entry_price = current_row['Close']

        # Sell signal - close position if it exists
        elif sell_signal and self.position == 1:
            trade_data['Time'] = current_row['timestamp']
            trade_data['Price'] = current_row['Close']
            trade_data['Position'] = 0
            trade_data['Index'] = i
            trade_data['Action'] = 'CLOSE'

            # Calculate profit from closing long position
            profit = current_row['Close'] - self.entry_price
            trade_data['Profit'] = profit
            self.realized_gains += profit
            trade_data['Realized'] = self.realized_gains
            trade_data['Result'] = "Win" if profit > 0 else "Loss"
            trade_data['Balance'] = self.initial_balance + self.realized_gains

            self.trades.append(trade_data)
            self.balance_history.append(trade_data['Balance'])

            self.position = 0
            self.entry_price = 0

    def _process_long_short_signals(self, current_row, buy_signal, sell_signal, i):
        """Process signals for long/short trading mode"""
        trade_data = {}

        # Buy signal
        if buy_signal and self.position != 1:
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
                trade_data['Action'] = 'CLOSE_SHORT_BUY_LONG'
            else:
                trade_data['Realized'] = 0
                trade_data['Action'] = 'BUY'

            trade_data['Balance'] = self.initial_balance + self.realized_gains
            self.trades.append(trade_data)
            self.balance_history.append(trade_data['Balance'])

            self.position = 1
            self.entry_price = current_row['Close']

        # Sell signal
        elif sell_signal and self.position != -1:
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
                trade_data['Action'] = 'CLOSE_LONG_SELL_SHORT'
            else:
                trade_data['Realized'] = 0
                trade_data['Action'] = 'SELL_SHORT'

            trade_data['Balance'] = self.initial_balance + self.realized_gains
            self.trades.append(trade_data)
            self.balance_history.append(trade_data['Balance'])

            self.position = -1
            self.entry_price = current_row['Close']
    
    def analyze_results(self, trade_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze backtest results - only count completed trades with profit/loss"""
        if len(trade_df) == 0:
            return {
                'num_trades': 0,
                'winrate': 0.0,
                'final_balance': float(self.initial_balance),
                'net_returns': 0.0,
                'percent_return': 0.0,
                'avg_profit_per_trade': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }

        # Only count completed trades (those that have Profit and Result columns)
        completed_trades = trade_df.dropna(subset=['Profit', 'Result'])
        num_trades = len(completed_trades)

        if num_trades == 0:
            # No completed trades yet
            final_balance = float(trade_df['Balance'].iloc[-1]) if len(trade_df) > 0 else float(self.initial_balance)
            net_returns = final_balance - self.initial_balance
            percent_return = (final_balance - self.initial_balance) / self.initial_balance * 100
            return {
                'num_trades': 0,
                'winrate': 0.0,
                'final_balance': float(final_balance),
                'net_returns': float(net_returns),
                'percent_return': float(percent_return),
                'avg_profit_per_trade': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }

        wins = completed_trades[completed_trades['Result'] == "Win"]
        winrate = len(wins) / num_trades * 100
        final_balance = float(trade_df['Balance'].iloc[-1])
        net_returns = final_balance - self.initial_balance
        percent_return = (final_balance - self.initial_balance) / self.initial_balance * 100
        avg_profit_per_trade = net_returns / num_trades if num_trades > 0 else 0

        profits = completed_trades['Profit']
        largest_win = float(profits.max()) if len(profits) > 0 else 0.0
        largest_loss = float(profits.min()) if len(profits) > 0 else 0.0

        return {
            'num_trades': int(num_trades),
            'winrate': float(winrate),
            'final_balance': float(final_balance),
            'net_returns': float(net_returns),
            'percent_return': float(percent_return),
            'avg_profit_per_trade': float(avg_profit_per_trade),
            'largest_win': float(largest_win),
            'largest_loss': float(largest_loss)
        }
    
    def print_analysis(self, trade_df: pd.DataFrame):
        """Print analysis results with appropriate formatting"""
        analysis = self.analyze_results(trade_df)

        print(f"📊 Strategy Performance Analysis")
        print("=" * 35)
        print(f"Win Rate: {analysis['winrate']:.1f}%")

        # Use different decimal places based on asset type
        if self.is_crypto:
            # Crypto often has smaller values, so use more decimal places
            print(f"Final Balance: ${analysis['final_balance']:.6f}")
            print(f"Net Returns: ${analysis['net_returns']:.6f}")
            print(f"Percentage Return: {analysis['percent_return']:.2f}%")
            print(f"Total Completed Trades: {analysis['num_trades']}")
            if analysis['num_trades'] > 0:
                print(f"Average Profit per Trade: ${analysis['avg_profit_per_trade']:.8f}")
                print(f"Largest Win: ${analysis['largest_win']:.8f}")
                print(f"Largest Loss: ${analysis['largest_loss']:.8f}")
        else:
            # Stocks typically use 2 decimal places
            print(f"Final Balance: ${analysis['final_balance']:.2f}")
            print(f"Net Returns: ${analysis['net_returns']:.2f}")
            print(f"Percentage Return: {analysis['percent_return']:.2f}%")
            print(f"Total Completed Trades: {analysis['num_trades']}")
            if analysis['num_trades'] > 0:
                print(f"Average Profit per Trade: ${analysis['avg_profit_per_trade']:.4f}")
                print(f"Largest Win: ${analysis['largest_win']:.4f}")
                print(f"Largest Loss: ${analysis['largest_loss']:.4f}")

        # Performance rating
        if analysis['percent_return'] > 20:
            print("🎉 Excellent performance!")
        elif analysis['percent_return'] > 10:
            print("✅ Good performance!")
        elif analysis['percent_return'] > 0:
            print("👍 Positive returns!")
        else:
            print("📉 Strategy needs improvement.")
    
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
        symbol_display = self.symbol if self.symbol else "Asset"
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[f'{symbol_display} - {self.strategy.name} Strategy', 'Volume'],
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
            close_trades = trade_df[trade_df['Action'] == 'CLOSE']

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

            # Add close markers
            if len(close_trades) > 0:
                close_times = []
                close_prices = []
                for _, trade in close_trades.iterrows():
                    if 'Index' in trade and trade['Index'] < len(df):
                        close_times.append(df.iloc[trade['Index']]['datetime'])
                        close_prices.append(trade['Price'])

                if close_times:
                    fig.add_trace(
                        go.Scatter(
                            x=close_times,
                            y=close_prices,
                            mode='markers',
                            name='Close Position',
                            marker=dict(
                                symbol='x',
                                size=12,
                                color='orange',
                                line=dict(width=2, color='darkorange')
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
        symbol_display = self.symbol if self.symbol else "Asset"
        fig.update_layout(
            title=f'{symbol_display} - {self.strategy.name} Strategy Backtest',
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
            close_trades = trade_df[trade_df['Action'] == 'CLOSE']

            buy_label_added = False
            sell_label_added = False
            close_label_added = False

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

            for _, trade in close_trades.iterrows():
                if 'Index' in trade and trade['Index'] < len(df):
                    label = 'Close' if not close_label_added else ""
                    ax1.scatter(trade['Index'], trade['Price'], color='orange', marker='x', s=100, label=label, zorder=5)
                    close_label_added = True

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