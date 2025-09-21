#!/usr/bin/env python3
"""
Demo script for the Live Trading System
This shows how the system works without requiring real API keys
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from live_trading_chart import LiveTradingChart
from strategies.bollinger_bands_strategy import BollingerBandsStrategy


class MockAlpacaProvider:
    """Mock data provider for demo purposes"""

    def __init__(self, api_key=None, secret_key=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.current_time = datetime.now()

    def get_live_data(self, ticker: str, lookback_minutes: int = 100) -> pd.DataFrame:
        """Generate mock BTC/USD data"""
        # Create realistic BTC price data
        base_price = 45000  # Starting BTC price

        # Generate timestamps
        end_time = self.current_time
        start_time = end_time - timedelta(minutes=lookback_minutes)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')

        # Generate realistic price movements
        np.random.seed(42)  # For reproducible demo
        returns = np.random.normal(0, 0.001, len(timestamps))  # 0.1% volatility per minute

        # Create price series with trend
        prices = [base_price]
        for i, ret in enumerate(returns[1:]):
            # Add some trend and mean reversion
            trend = 0.0001 * np.sin(i / 50)  # Sine wave trend
            new_price = prices[-1] * (1 + ret + trend)
            prices.append(new_price)

        # Create OHLCV data
        data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            # Create realistic OHLC from close price
            volatility = 0.002  # 0.2% intraday volatility
            high = price * (1 + np.random.uniform(0, volatility))
            low = price * (1 - np.random.uniform(0, volatility))
            open_price = prices[i-1] if i > 0 else price

            data.append({
                'timestamp': ts,
                'Open': open_price,
                'High': max(open_price, high, price),
                'Low': min(open_price, low, price),
                'Close': price,
                'Volume': np.random.uniform(100, 1000)  # Mock volume
            })

        df = pd.DataFrame(data)
        self.current_time += timedelta(minutes=1)  # Advance time for next call

        return df


def demo_live_trading():
    """Run a demo of the live trading system"""
    print("🚀 BTC/USD Live Trading System - DEMO MODE")
    print("=" * 45)
    print("This demo shows the live trading system using simulated data")
    print("No real API keys or trading will occur\n")

    # Create strategy
    strategy = BollingerBandsStrategy(window=20, num_std=2)
    print(f"📊 Using strategy: {strategy.name}")

    # Create mock live chart (we'll modify it to use mock data)
    print("🔧 Setting up demo environment...")

    # Create the chart but replace the data provider
    live_chart = LiveTradingChart(
        strategy=strategy,
        api_key=None,  # No real API key
        secret_key=None,
        symbol="BTC/USD",
        paper_trading=True,
        initial_balance=10000,
        quantity=0.01
    )

    # Replace with mock provider
    live_chart.data_provider = MockAlpacaProvider()

    print("✅ Demo setup complete!")
    print("\n📈 Generating sample data and testing strategy...")

    # Test data fetching and processing
    data = live_chart.fetch_and_process_data()

    if data is not None and len(data) > 0:
        print(f"✅ Successfully generated {len(data)} data points")
        print(f"📊 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")

        # Check for signals
        signal_names = strategy.get_signal_names()
        buy_signals = data[data[signal_names['buy']]].shape[0]
        sell_signals = data[data[signal_names['sell']]].shape[0]

        print(f"🎯 Generated {buy_signals} buy signals and {sell_signals} sell signals")

        # Show sample of data
        print(f"\n📋 Sample data (last 5 rows):")
        display_cols = ['timestamp', 'Close', 'bb_upper', 'bb_lower', 'Buy Signal', 'Sell Signal']
        print(data[display_cols].tail().to_string(index=False))

        # Create a static chart for demo
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Price chart with Bollinger Bands
        x_axis = range(len(data))
        ax1.plot(x_axis, data['Close'], label='BTC/USD', color='black', linewidth=2)
        ax1.plot(x_axis, data['bb_upper'], label='BB Upper', color='red', linestyle='--', alpha=0.7)
        ax1.plot(x_axis, data['bb_lower'], label='BB Lower', color='green', linestyle='--', alpha=0.7)
        ax1.plot(x_axis, data['bb_middle'], label='BB Middle', color='blue', alpha=0.7)
        ax1.fill_between(x_axis, data['bb_upper'], data['bb_lower'], alpha=0.1, color='gray')

        # Add buy/sell signals
        buy_signals_data = data[data['Buy Signal'] == True]
        sell_signals_data = data[data['Sell Signal'] == True]

        if not buy_signals_data.empty:
            buy_indices = [data.index.get_loc(idx) for idx in buy_signals_data.index]
            ax1.scatter(buy_indices, buy_signals_data['Close'], color='green', marker='^',
                       s=100, label='Buy Signal', zorder=5)

        if not sell_signals_data.empty:
            sell_indices = [data.index.get_loc(idx) for idx in sell_signals_data.index]
            ax1.scatter(sell_indices, sell_signals_data['Close'], color='red', marker='v',
                       s=100, label='Sell Signal', zorder=5)

        ax1.set_title('BTC/USD Demo Chart with Bollinger Bands Strategy')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Time axis
        time_labels = [ts.strftime('%H:%M') for ts in data['timestamp']]
        ax1.set_xticks(x_axis[::len(x_axis)//10])
        ax1.set_xticklabels([time_labels[i] for i in range(0, len(time_labels), len(time_labels)//10)], rotation=45)

        # Bollinger Band width indicator
        bb_width = data['bb_upper'] - data['bb_lower']
        ax2.plot(x_axis, bb_width, label='BB Width', color='purple')
        ax2.set_title('Bollinger Band Width (Volatility Indicator)')
        ax2.set_ylabel('BB Width')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(x_axis[::len(x_axis)//10])
        ax2.set_xticklabels([time_labels[i] for i in range(0, len(time_labels), len(time_labels)//10)], rotation=45)

        plt.tight_layout()
        print(f"\n📊 Displaying demo chart...")
        plt.show()

        # Show performance summary
        performance = live_chart.get_performance_summary()
        print(f"\n📈 Demo Performance Summary:")
        print(f"Current Balance: ${performance['current_balance']:.2f}")
        print(f"Total Return: ${performance['total_return']:.2f}")
        print(f"Total Trades: {performance['total_trades']}")

    else:
        print("❌ Failed to generate demo data")

    print(f"\n🎯 Demo complete!")
    print(f"To run with real data, use: python3 run_live_trading.py")


if __name__ == "__main__":
    demo_live_trading()