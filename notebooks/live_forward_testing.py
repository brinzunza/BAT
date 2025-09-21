import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from datetime import datetime
import pandas as pd
import numpy as np
from strategies.bollinger_bands_strategy import BollingerBandsStrategy
from indicators.technical_indicators import bollinger_bands

class LiveForwardTesting:
    def __init__(self, strategy=None, update_interval=60):
        self.url = "https://data.alpaca.markets/v1beta3/crypto/us/latest/bars?symbols=BTC%2FUSD"
        self.headers = {"accept": "application/json"}
        self.data = []
        self.update_interval = update_interval

        # Initialize strategy
        self.strategy = strategy or BollingerBandsStrategy(window=20, num_std=2)

        # Portfolio tracking
        self.portfolio_value = 10000  # Starting with $10,000
        self.btc_holdings = 0
        self.cash = self.portfolio_value
        self.trades = []
        self.portfolio_history = []

        # Set up the plot with subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])

        # Main candlestick chart
        self.ax1.set_title('BTC/USD Live Forward Testing with Bollinger Bands')
        self.ax1.set_ylabel('Price (USD)')
        self.ax1.grid(True, alpha=0.3)

        # Portfolio value chart
        self.ax2.set_title('Portfolio Value')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Value (USD)')
        self.ax2.grid(True, alpha=0.3)

        plt.tight_layout()

    def fetch_data(self):
        try:
            response = requests.get(self.url, headers=self.headers)
            response_data = json.loads(response.text)

            if 'bars' in response_data and 'BTC/USD' in response_data['bars']:
                bar_data = response_data['bars']['BTC/USD']
                timestamp = datetime.now()

                ohlc = {
                    'timestamp': timestamp,
                    'Open': float(bar_data['o']),
                    'High': float(bar_data['h']),
                    'Low': float(bar_data['l']),
                    'Close': float(bar_data['c']),
                    'Volume': float(bar_data['v'])
                }

                self.data.append(ohlc)

                # Keep only last 100 data points for better performance
                if len(self.data) > 100:
                    self.data.pop(0)

                # Calculate current portfolio value
                current_price = ohlc['Close']
                current_portfolio_value = self.cash + (self.btc_holdings * current_price)
                self.portfolio_history.append({
                    'timestamp': timestamp,
                    'value': current_portfolio_value
                })

                # Keep portfolio history manageable
                if len(self.portfolio_history) > 100:
                    self.portfolio_history.pop(0)

                print(f"{timestamp.strftime('%H:%M:%S')} - Price: ${current_price:.2f}, Portfolio: ${current_portfolio_value:.2f}")

                # Generate trading signals if we have enough data
                if len(self.data) >= self.strategy.window:
                    self.process_trading_signals()

        except Exception as e:
            print(f"Error fetching data: {e}")

    def process_trading_signals(self):
        # Convert data to DataFrame for strategy processing
        df = pd.DataFrame(self.data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        try:
            # Generate signals using the strategy
            df_with_signals = self.strategy.generate_signals(df)

            # Get the latest signal
            latest = df_with_signals.iloc[-1]
            current_price = latest['Close']
            timestamp = latest['timestamp']

            signal_names = self.strategy.get_signal_names()
            buy_signal = latest.get(signal_names['buy'], False)
            sell_signal = latest.get(signal_names['sell'], False)

            # Execute trades based on signals
            if buy_signal and self.cash > 0:
                # Buy BTC with all available cash
                btc_to_buy = self.cash / current_price
                self.btc_holdings += btc_to_buy
                self.cash = 0

                trade = {
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': current_price,
                    'amount': btc_to_buy,
                    'total_value': btc_to_buy * current_price
                }
                self.trades.append(trade)
                print(f"🟢 BUY: {btc_to_buy:.6f} BTC at ${current_price:.2f}")

            elif sell_signal and self.btc_holdings > 0:
                # Sell all BTC holdings
                cash_received = self.btc_holdings * current_price
                self.cash = cash_received
                btc_sold = self.btc_holdings
                self.btc_holdings = 0

                trade = {
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': current_price,
                    'amount': btc_sold,
                    'total_value': cash_received
                }
                self.trades.append(trade)
                print(f"🔴 SELL: {btc_sold:.6f} BTC at ${current_price:.2f}")

        except Exception as e:
            print(f"Error processing signals: {e}")

    def draw_candlesticks_with_indicators(self):
        self.ax1.clear()
        self.ax1.set_title('BTC/USD Live Forward Testing with Bollinger Bands')
        self.ax1.set_ylabel('Price (USD)')
        self.ax1.grid(True, alpha=0.3)

        if len(self.data) < 1:
            return

        # Convert to DataFrame for indicator calculation
        df = pd.DataFrame(self.data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate Bollinger Bands if we have enough data
        bb_upper, bb_middle, bb_lower = None, None, None
        if len(df) >= self.strategy.window:
            try:
                df_with_indicators = self.strategy.generate_signals(df)
                bb_upper = df_with_indicators['bb_upper'].values
                bb_middle = df_with_indicators['bb_middle'].values
                bb_lower = df_with_indicators['bb_lower'].values
            except:
                pass

        # Draw candlesticks
        for i, candle in enumerate(self.data):
            open_price = candle['Open']
            high_price = candle['High']
            low_price = candle['Low']
            close_price = candle['Close']

            # Determine candle color
            color = 'green' if close_price >= open_price else 'red'

            # Draw the wick
            self.ax1.plot([i, i], [low_price, high_price], color='black', linewidth=1)

            # Draw the body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)

            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, edgecolor='black', alpha=0.8)
            self.ax1.add_patch(rect)

        # Plot Bollinger Bands if available
        if bb_upper is not None and len(bb_upper) == len(self.data):
            x_indices = list(range(len(self.data)))

            # Plot the bands
            self.ax1.plot(x_indices, bb_upper, 'b--', alpha=0.7, label='BB Upper', linewidth=1)
            self.ax1.plot(x_indices, bb_middle, 'orange', alpha=0.7, label='BB Middle (SMA)', linewidth=1)
            self.ax1.plot(x_indices, bb_lower, 'b--', alpha=0.7, label='BB Lower', linewidth=1)

            # Fill between bands for better visualization
            self.ax1.fill_between(x_indices, bb_upper, bb_lower, alpha=0.1, color='blue')

        # Mark trade signals
        for i, trade in enumerate(self.trades):
            # Find the index of this trade in our current data
            trade_time = trade['timestamp']
            for j, candle in enumerate(self.data):
                if abs((candle['timestamp'] - trade_time).total_seconds()) < 30:  # Within 30 seconds
                    if trade['action'] == 'BUY':
                        self.ax1.scatter(j, trade['price'], color='green', marker='^', s=100, zorder=5)
                    else:
                        self.ax1.scatter(j, trade['price'], color='red', marker='v', s=100, zorder=5)
                    break

        # Set x-axis labels
        if len(self.data) > 0:
            timestamps = [d['timestamp'].strftime('%H:%M:%S') for d in self.data]
            tick_positions = list(range(len(self.data)))

            show_every = max(1, len(timestamps) // 10)
            tick_labels = [timestamps[i] if i % show_every == 0 else '' for i in range(len(timestamps))]

            self.ax1.set_xticks(tick_positions)
            self.ax1.set_xticklabels(tick_labels, rotation=45)

        # Add legend
        self.ax1.legend(loc='upper left')

        # Auto-scale
        self.ax1.relim()
        self.ax1.autoscale_view()

    def draw_portfolio_chart(self):
        self.ax2.clear()
        self.ax2.set_title('Portfolio Value Over Time')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Value (USD)')
        self.ax2.grid(True, alpha=0.3)

        if len(self.portfolio_history) > 1:
            timestamps = [p['timestamp'] for p in self.portfolio_history]
            values = [p['value'] for p in self.portfolio_history]

            self.ax2.plot(timestamps, values, 'g-', linewidth=2, label='Portfolio Value')
            self.ax2.axhline(y=self.portfolio_value, color='gray', linestyle='--', alpha=0.5, label='Initial Value')

            # Format x-axis
            self.ax2.tick_params(axis='x', rotation=45)
            self.ax2.legend()

            # Show current portfolio stats
            current_value = values[-1] if values else self.portfolio_value
            pnl = current_value - self.portfolio_value
            pnl_pct = (pnl / self.portfolio_value) * 100

            stats_text = f"Current: ${current_value:.2f}\nP&L: ${pnl:.2f} ({pnl_pct:.2f}%)\nTrades: {len(self.trades)}"
            self.ax2.text(0.02, 0.98, stats_text, transform=self.ax2.transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def animate(self, frame):
        self.fetch_data()
        self.draw_candlesticks_with_indicators()
        self.draw_portfolio_chart()
        return []

    def start_live_testing(self):
        # Fetch initial data
        print("Starting live forward testing...")
        print(f"Strategy: {self.strategy.name}")
        print(f"Initial Portfolio Value: ${self.portfolio_value:.2f}")
        print("Fetching data and generating signals...")

        # Start the animation
        ani = animation.FuncAnimation(
            self.fig, self.animate, interval=self.update_interval * 1000,
            blit=False, cache_frame_data=False
        )

        plt.tight_layout()
        plt.show()
        return ani

if __name__ == "__main__":
    # Initialize with Bollinger Bands strategy
    bb_strategy = BollingerBandsStrategy(window=20, num_std=2)

    # Start live forward testing
    tester = LiveForwardTesting(strategy=bb_strategy, update_interval=60)
    tester.start_live_testing()