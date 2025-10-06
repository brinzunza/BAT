import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BacktestEngine:
    """Backtesting engine for trading strategies supporting stocks and crypto"""

    def __init__(self, initial_balance: float = 10000, trading_mode: str = "long_only", symbol: str = "", position_percentage: float = 100.0, spread_pips: float = 0.0):
        self.initial_balance = initial_balance
        self.trading_mode = trading_mode
        self.symbol = symbol
        self.position_percentage = position_percentage / 100.0  # Convert to decimal
        self.is_crypto = '/' in symbol  # Determine if it's crypto based on symbol format
        self.is_forex = symbol.startswith('C:')  # Determine if it's forex based on symbol format
        self.spread_pips = spread_pips  # Spread in pips for forex trading
        self.reset()
    
    def reset(self):
        """Reset engine state"""
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = 0
        self.realized_gains = 0
        self.trades = []
        self.balance_history = []
        self.current_balance = self.initial_balance
        self.shares_held = 0  # Track actual shares/units held

    def _calculate_account_worth_realized_only(self):
        """Calculate total account worth based on REALIZED gains/losses only"""
        # Total worth = initial balance + all realized gains/losses
        return self.initial_balance + self.realized_gains

    def _calculate_account_worth(self, current_price=None):
        """Calculate total account worth including open positions (for final analysis only)"""
        if self.position == 0:
            # No open positions
            return self.current_balance
        elif self.position == 1:
            # Long position: cash + position value
            if current_price is None:
                current_price = self.entry_price  # Fallback
            return self.current_balance + (self.shares_held * current_price)
        elif self.position == -1:
            # Short position: cash - liability (current value of borrowed shares)
            if current_price is None:
                current_price = self.entry_price  # Fallback
            liability_value = self.shares_held * current_price
            return self.current_balance - liability_value
        else:
            return self.current_balance
    
    def backtest(self, df: pd.DataFrame, strategy) -> pd.DataFrame:
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
        """Process signals for long-only trading mode with percentage-based position sizing"""
        trade_data = {}

        # Buy signal - only buy if no position exists and account has funds
        if buy_signal and self.position == 0 and self.current_balance > 0:
            # Calculate position size based on percentage of current balance
            trade_amount = self.current_balance * self.position_percentage
            price = current_row['Close']

            # Apply spread for forex (buy at ask price)
            if self.is_forex and self.spread_pips > 0:
                pip_value = 0.0001 if 'JPY' not in self.symbol else 0.01
                spread_cost = self.spread_pips * pip_value
                price = price + spread_cost  # Buy at ask price

            # Ensure we have enough funds for the trade
            if trade_amount > self.current_balance:
                trade_amount = self.current_balance

            # Calculate shares/units to buy
            shares_to_buy = trade_amount / price
            actual_cost = shares_to_buy * price

            # Check if we have enough for at least minimal trade
            if actual_cost > 0 and self.current_balance >= actual_cost:
                # Update account balance first
                self.current_balance -= actual_cost

                # For open positions, use realized-only calculation
                total_account_worth = self._calculate_account_worth_realized_only()
                total_profit = total_account_worth - self.initial_balance

                trade_data['Time'] = current_row['timestamp']
                trade_data['Price'] = price
                trade_data['Position'] = 1
                trade_data['Index'] = i
                trade_data['Action'] = 'BUY'
                trade_data['Shares'] = shares_to_buy
                trade_data['Cost'] = actual_cost
                trade_data['Last_Trade_Realized'] = 0
                trade_data['Balance'] = self.current_balance
                trade_data['Total_Account_Worth'] = total_account_worth
                trade_data['Total_Profit'] = total_profit
                trade_data['Trade_Result'] = 'OPEN'

                self.trades.append(trade_data)
                self.balance_history.append(trade_data['Balance'])

                self.position = 1
                self.entry_price = price
                self.shares_held = shares_to_buy

        # Sell signal - close position if it exists
        elif sell_signal and self.position == 1:
            price = current_row['Close']

            # Apply spread for forex (sell at bid price)
            if self.is_forex and self.spread_pips > 0:
                pip_value = 0.0001 if 'JPY' not in self.symbol else 0.01
                spread_cost = self.spread_pips * pip_value
                price = price - spread_cost  # Sell at bid price

            proceeds = self.shares_held * price

            trade_data['Time'] = current_row['timestamp']
            trade_data['Price'] = price
            trade_data['Position'] = 0
            trade_data['Index'] = i
            trade_data['Action'] = 'CLOSE'
            trade_data['Shares'] = self.shares_held
            trade_data['Proceeds'] = proceeds

            # Calculate profit from closing long position
            cost_basis = self.shares_held * self.entry_price
            profit = proceeds - cost_basis

            # Update account balance
            self.current_balance += proceeds

            # Update realized gains for closed position
            self.realized_gains += profit

            # Calculate total account worth with new realized gains
            total_account_worth = self._calculate_account_worth_realized_only()
            total_profit = total_account_worth - self.initial_balance

            trade_data['Profit'] = profit
            trade_data['Last_Trade_Realized'] = profit
            trade_data['Result'] = "Win" if profit > 0 else "Loss"
            trade_data['Balance'] = self.current_balance
            trade_data['Total_Account_Worth'] = total_account_worth
            trade_data['Total_Profit'] = total_profit
            trade_data['Trade_Result'] = "Win" if profit > 0 else "Loss"

            self.trades.append(trade_data)
            self.balance_history.append(trade_data['Balance'])

            self.position = 0
            self.entry_price = 0
            self.shares_held = 0

    def _process_long_short_signals(self, current_row, buy_signal, sell_signal, i):
        """Process signals for long/short trading mode with percentage-based position sizing"""
        trade_data = {}
        price = current_row['Close']

        # Buy signal - enter long (or reverse from short to long)
        if buy_signal and self.position != 1:
            # Close short position if exists (using ALL shares held)
            if self.position == -1 and self.shares_held > 0:
                # Apply spread for forex when closing short (buy at ask price)
                close_price = price
                if self.is_forex and self.spread_pips > 0:
                    pip_value = 0.0001 if 'JPY' not in self.symbol else 0.01
                    spread_cost = self.spread_pips * pip_value
                    close_price = price + spread_cost  # Buy at ask price

                # Close short: buy back shares at current price to return them
                cost_to_close = self.shares_held * close_price  # Cost to buy back shares
                profit = self.shares_held * (self.entry_price - close_price)  # Short profit calculation

                # Pay to buy back the shares (this removes the liability)
                self.current_balance -= cost_to_close

                # Update realized gains for closed position
                self.realized_gains += profit

                # Calculate total account worth with new realized gains
                total_account_worth = self._calculate_account_worth_realized_only()
                total_profit = total_account_worth - self.initial_balance

                trade_data['Time'] = current_row['timestamp']
                trade_data['Price'] = close_price
                trade_data['Position'] = 0  # Flat first
                trade_data['Index'] = i
                trade_data['Action'] = 'CLOSE_SHORT'
                trade_data['Shares'] = self.shares_held  # Close ALL shares
                trade_data['Profit'] = profit
                trade_data['Last_Trade_Realized'] = profit
                trade_data['Result'] = "Win" if profit > 0 else "Loss"
                trade_data['Balance'] = self.current_balance
                trade_data['Total_Account_Worth'] = total_account_worth
                trade_data['Total_Profit'] = total_profit
                trade_data['Trade_Result'] = "Win" if profit > 0 else "Loss"

                self.trades.append(trade_data.copy())
                self.balance_history.append(trade_data['Balance'])

                self.position = 0
                self.shares_held = 0

            # Now open long position if we have funds
            if self.current_balance > 0:
                # Apply spread for forex when opening long (buy at ask price)
                buy_price = price
                if self.is_forex and self.spread_pips > 0:
                    pip_value = 0.0001 if 'JPY' not in self.symbol else 0.01
                    spread_cost = self.spread_pips * pip_value
                    buy_price = price + spread_cost  # Buy at ask price

                trade_amount = self.current_balance * self.position_percentage
                if trade_amount > self.current_balance:
                    trade_amount = self.current_balance

                shares_to_buy = trade_amount / buy_price
                actual_cost = shares_to_buy * buy_price

                if actual_cost > 0 and self.current_balance >= actual_cost:
                    self.current_balance -= actual_cost

                    # For open positions, use realized-only calculation
                    total_account_worth = self._calculate_account_worth_realized_only()
                    total_profit = total_account_worth - self.initial_balance

                    trade_data = {}  # Reset for new trade
                    trade_data['Time'] = current_row['timestamp']
                    trade_data['Price'] = buy_price
                    trade_data['Position'] = 1
                    trade_data['Index'] = i
                    trade_data['Action'] = 'BUY'
                    trade_data['Shares'] = shares_to_buy
                    trade_data['Cost'] = actual_cost
                    trade_data['Last_Trade_Realized'] = 0
                    trade_data['Balance'] = self.current_balance
                    trade_data['Total_Account_Worth'] = total_account_worth
                    trade_data['Total_Profit'] = total_profit
                    trade_data['Trade_Result'] = 'OPEN'

                    self.trades.append(trade_data)
                    self.balance_history.append(trade_data['Balance'])

                    self.position = 1
                    self.entry_price = buy_price
                    self.shares_held = shares_to_buy

        # Sell signal - enter short (or reverse from long to short)
        elif sell_signal and self.position != -1:
            # Close long position if exists (using ALL shares held)
            if self.position == 1 and self.shares_held > 0:
                # Apply spread for forex when closing long (sell at bid price)
                sell_price = price
                if self.is_forex and self.spread_pips > 0:
                    pip_value = 0.0001 if 'JPY' not in self.symbol else 0.01
                    spread_cost = self.spread_pips * pip_value
                    sell_price = price - spread_cost  # Sell at bid price

                proceeds = self.shares_held * sell_price
                cost_basis = self.shares_held * self.entry_price
                profit = proceeds - cost_basis

                self.current_balance += proceeds

                # Update realized gains for closed position
                self.realized_gains += profit

                # Calculate total account worth with new realized gains
                total_account_worth = self._calculate_account_worth_realized_only()
                total_profit = total_account_worth - self.initial_balance

                trade_data['Time'] = current_row['timestamp']
                trade_data['Price'] = sell_price
                trade_data['Position'] = 0  # Flat first
                trade_data['Index'] = i
                trade_data['Action'] = 'CLOSE_LONG'
                trade_data['Shares'] = self.shares_held  # Close ALL shares
                trade_data['Proceeds'] = proceeds
                trade_data['Profit'] = profit
                trade_data['Last_Trade_Realized'] = profit
                trade_data['Result'] = "Win" if profit > 0 else "Loss"
                trade_data['Balance'] = self.current_balance
                trade_data['Total_Account_Worth'] = total_account_worth
                trade_data['Total_Profit'] = total_profit
                trade_data['Trade_Result'] = "Win" if profit > 0 else "Loss"

                self.trades.append(trade_data.copy())
                self.balance_history.append(trade_data['Balance'])

                self.position = 0
                self.shares_held = 0

            # Now open short position if we have funds for margin
            if self.current_balance > 0:
                # Apply spread for forex when opening short (sell at bid price)
                short_price = price
                if self.is_forex and self.spread_pips > 0:
                    pip_value = 0.0001 if 'JPY' not in self.symbol else 0.01
                    spread_cost = self.spread_pips * pip_value
                    short_price = price - spread_cost  # Sell at bid price

                trade_amount = self.current_balance * self.position_percentage
                shares_to_short = trade_amount / short_price

                if shares_to_short > 0:
                    # For short selling: receive cash but create liability
                    proceeds_from_short = shares_to_short * short_price
                    self.current_balance += proceeds_from_short  # Add cash from sale

                    # Set position details BEFORE calculating account worth
                    self.position = -1
                    self.entry_price = short_price
                    self.shares_held = shares_to_short

                    # For open positions, use realized-only calculation
                    total_account_worth = self._calculate_account_worth_realized_only()
                    total_profit = total_account_worth - self.initial_balance

                    trade_data = {}  # Reset for new trade
                    trade_data['Time'] = current_row['timestamp']
                    trade_data['Price'] = price
                    trade_data['Position'] = -1
                    trade_data['Index'] = i
                    trade_data['Action'] = 'SELL_SHORT'
                    trade_data['Shares'] = shares_to_short
                    trade_data['Proceeds'] = proceeds_from_short
                    trade_data['Last_Trade_Realized'] = 0
                    trade_data['Balance'] = self.current_balance
                    trade_data['Total_Account_Worth'] = total_account_worth
                    trade_data['Total_Profit'] = total_profit
                    trade_data['Trade_Result'] = 'OPEN'

                    self.trades.append(trade_data)
                    self.balance_history.append(trade_data['Balance'])
    
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

        # Use the Total_Account_Worth from the last trade entry (matches the detailed trade overview)
        # This ensures consistency between the performance analysis and the detailed trade table
        if 'Total_Account_Worth' in trade_df.columns and len(trade_df) > 0:
            final_balance = float(trade_df['Total_Account_Worth'].iloc[-1])
        else:
            # Fallback: calculate final balance including any open positions
            final_balance = self.current_balance
            if self.position != 0 and self.shares_held > 0:
                # Add value of current position (using last price from dataframe)
                last_price = self.df_with_signals.iloc[-1]['Close'] if hasattr(self, 'df_with_signals') else 0
                if self.position == 1:  # Long position
                    final_balance += self.shares_held * last_price
                elif self.position == -1:  # Short position
                    # For short positions: we have cash but owe shares
                    # Net worth = cash - current liability (current value of borrowed shares)
                    liability_value = self.shares_held * last_price
                    final_balance = self.current_balance - liability_value

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

        print(f" Strategy Performance Analysis")
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
            print("Excellent performance!")
        elif analysis['percent_return'] > 10:
            print("Good performance!")
        elif analysis['percent_return'] > 0:
            print("Positive returns!")
        else:
            print("Strategy needs improvement.")
    
    def plot_results(self, trade_df: pd.DataFrame):
        """Plot total worth vs trades placed"""
        if len(trade_df) == 0:
            print("No trades to plot")
            return

        # Create trade numbers (x-axis)
        trade_numbers = list(range(1, len(trade_df) + 1))

        plt.figure(figsize=(12, 8))

        # Main plot - Total Worth vs Trades
        plt.subplot(2, 1, 1)
        plt.plot(trade_numbers, trade_df['Total_Account_Worth'],
                marker='o', linestyle='-', linewidth=2, markersize=6, color='blue')

        # Add horizontal line for initial balance
        plt.axhline(y=self.initial_balance, color='red', linestyle='--', alpha=0.7,
                   label=f'Initial Balance: ${self.initial_balance:,.0f}')

        # Color code markers based on trade result
        for i, (idx, trade) in enumerate(trade_df.iterrows()):
            if trade.get('Trade_Result') == 'Win':
                plt.scatter(i+1, trade['Total_Account_Worth'], color='green', s=100, zorder=5)
            elif trade.get('Trade_Result') == 'Loss':
                plt.scatter(i+1, trade['Total_Account_Worth'], color='red', s=100, zorder=5)
            else:  # OPEN positions
                plt.scatter(i+1, trade['Total_Account_Worth'], color='gray', s=60, zorder=5)

        plt.title(f"Account Worth vs Trades Placed - {self.strategy.name if hasattr(self, 'strategy') else 'Strategy'}")
        plt.xlabel("Trade Number")
        plt.ylabel("Total Account Worth ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Format y-axis as currency
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Profit/Loss bar chart
        plt.subplot(2, 1, 2)

        # Extract profits/losses for completed trades only
        profits = []
        trade_nums = []
        colors = []

        for i, (idx, trade) in enumerate(trade_df.iterrows()):
            if 'Last_Trade_Realized' in trade and trade['Last_Trade_Realized'] != 0:
                profits.append(trade['Last_Trade_Realized'])
                trade_nums.append(i+1)
                colors.append('green' if trade['Last_Trade_Realized'] > 0 else 'red')

        if profits:
            plt.bar(trade_nums, profits, color=colors, alpha=0.7)
            plt.title("Individual Trade P&L (Realized Only)")
            plt.xlabel("Trade Number")
            plt.ylabel("Profit/Loss ($)")
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Format y-axis as currency
            ax2 = plt.gca()
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        else:
            plt.text(0.5, 0.5, 'No completed trades to show P&L',
                    transform=plt.gca().transAxes, ha='center', va='center')
            plt.title("Individual Trade P&L (Realized Only)")

        plt.tight_layout()

        # Add text instruction for easy closing
        plt.figtext(0.5, 0.02, 'Press ESC or Q to close, or simply close the window',
                   ha='center', va='bottom', fontsize=10, style='italic')

        # Add event handlers for safe closing
        def on_key_press(event):
            if event.key in ['escape', 'q']:
                plt.close('all')

        def on_close(event):
            plt.close('all')

        fig = plt.gcf()
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('close_event', on_close)

        try:
            plt.show(block=False)
            print("Chart displayed. Press ESC or Q to close, or simply close the window.")
        except Exception as e:
            print(f"Error displaying chart: {e}")
            plt.close('all')

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

        # Add text instruction for easy closing
        plt.figtext(0.5, 0.02, 'Press ESC or Q to close, or simply close the window',
                   ha='center', va='bottom', fontsize=10, style='italic')

        # Add event handlers for safe closing
        def on_key_press(event):
            if event.key in ['escape', 'q']:
                plt.close('all')

        def on_close(event):
            plt.close('all')

        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.mpl_connect('close_event', on_close)

        try:
            plt.show(block=False)
            print("Chart displayed. Press ESC or Q to close, or simply close the window.")
        except Exception as e:
            print(f"Error displaying chart: {e}")
            plt.close('all')