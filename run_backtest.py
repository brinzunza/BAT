#!/usr/bin/env python3
"""
BAT Backtesting System for Stocks and Cryptocurrencies
Comprehensive backtesting with strategy selection and performance analysis
"""

import os
import sys
from typing import Optional
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_providers.alpaca_provider import AlpacaDataProvider
from engines.backtest_engine import BacktestEngine
from strategies.bollinger_bands_strategy import BollingerBandsStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.moving_average import MovingAverageStrategy


def get_alpaca_credentials():
    """Get Alpaca API credentials from environment or user input"""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("⚠️  Alpaca API credentials not found in environment variables")
        print("You can either:")
        print("1. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("2. Enter them now (they will be used for this session only)")
        print("3. Continue without credentials (limited to public crypto data)")
        print()

        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "2":
            api_key = input("Enter Alpaca API Key: ").strip()
            secret_key = input("Enter Alpaca Secret Key: ").strip()
        elif choice == "3":
            print("Continuing with public data access (crypto only)...")
            api_key = None
            secret_key = None
        else:
            print("Please set environment variables and try again.")
            sys.exit(1)

    return api_key, secret_key


def select_symbol():
    """Let user select trading symbol from crypto or stocks"""
    print("\n📊 Symbol Selection:")
    print("===================")
    print("1. Cryptocurrency")
    print("2. Stocks")
    print()

    # Popular crypto options
    crypto_symbols = {
        "1": ("BTC/USD", "Bitcoin"),
        "2": ("ETH/USD", "Ethereum"),
        "3": ("DOGE/USD", "Dogecoin"),
        "4": ("LTC/USD", "Litecoin"),
        "5": ("BCH/USD", "Bitcoin Cash"),
        "6": ("AVAX/USD", "Avalanche"),
        "7": ("LINK/USD", "Chainlink"),
        "8": ("UNI/USD", "Uniswap"),
        "9": ("CUSTOM", "Custom Crypto Symbol")
    }

    # Popular stock options
    stock_symbols = {
        "1": ("AAPL", "Apple Inc."),
        "2": ("MSFT", "Microsoft Corporation"),
        "3": ("GOOGL", "Alphabet Inc."),
        "4": ("AMZN", "Amazon.com Inc."),
        "5": ("TSLA", "Tesla Inc."),
        "6": ("NVDA", "NVIDIA Corporation"),
        "7": ("META", "Meta Platforms Inc."),
        "8": ("NFLX", "Netflix Inc."),
        "9": ("SPY", "SPDR S&P 500 ETF"),
        "10": ("QQQ", "Invesco QQQ Trust"),
        "11": ("CUSTOM", "Custom Stock Symbol")
    }

    while True:
        asset_choice = input("Select asset type (1-2): ").strip()

        if asset_choice == "1":  # Cryptocurrency
            print("\n₿ Available Cryptocurrencies:")
            print("============================")
            for key, (symbol, name) in crypto_symbols.items():
                if symbol == "CUSTOM":
                    print(f"{key}. {name}")
                else:
                    print(f"{key}. {symbol} - {name}")

            while True:
                crypto_choice = input(f"\nSelect cryptocurrency (1-{len(crypto_symbols)}): ").strip()
                if crypto_choice in crypto_symbols:
                    if crypto_symbols[crypto_choice][0] == "CUSTOM":
                        custom_symbol = input("Enter custom crypto symbol (e.g., ADA/USD): ").strip().upper()
                        if "/" not in custom_symbol:
                            custom_symbol += "/USD"
                        return custom_symbol
                    else:
                        return crypto_symbols[crypto_choice][0]
                print(f"❌ Invalid choice. Please enter 1-{len(crypto_symbols)}.")

        elif asset_choice == "2":  # Stocks
            print("\n📈 Available Stocks:")
            print("===================")
            for key, (symbol, name) in stock_symbols.items():
                if symbol == "CUSTOM":
                    print(f"{key}. {name}")
                else:
                    print(f"{key}. {symbol} - {name}")

            while True:
                stock_choice = input(f"\nSelect stock (1-{len(stock_symbols)}): ").strip()
                if stock_choice in stock_symbols:
                    if stock_symbols[stock_choice][0] == "CUSTOM":
                        custom_symbol = input("Enter custom stock symbol (e.g., AMZN): ").strip().upper()
                        return custom_symbol
                    else:
                        return stock_symbols[stock_choice][0]
                print(f"❌ Invalid choice. Please enter 1-{len(stock_symbols)}.")

        print("❌ Invalid choice. Please enter 1 or 2.")


def select_strategy():
    """Let user select a trading strategy"""
    strategies = {
        "1": ("Bollinger Bands", BollingerBandsStrategy()),
        "2": ("RSI Strategy", RSIStrategy()),
        "3": ("MACD Strategy", MACDStrategy()),
        "4": ("Moving Average Crossover", MovingAverageStrategy())
    }

    print("\n📊 Available Trading Strategies:")
    print("================================")
    for key, (name, _) in strategies.items():
        print(f"{key}. {name}")

    while True:
        choice = input("\nSelect strategy (1-4): ").strip()
        if choice in strategies:
            return strategies[choice][1]
        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


def select_trading_mode():
    """Let user select trading mode"""
    print("\n🔄 Trading Mode Selection:")
    print("==========================")
    print("1. Buy & Close Only (Long-only trading)")
    print("   - Buy signals → Buy positions")
    print("   - Sell signals → Close positions")
    print("   - No short selling")
    print()
    print("2. Buy & Short Trading (Long/Short trading)")
    print("   - Buy signals → Buy positions (or close short)")
    print("   - Sell signals → Short positions (or close long)")
    print("   - More complex but potentially higher returns")
    print()

    while True:
        choice = input("Select trading mode (1-2): ").strip()
        if choice == "1":
            return "long_only"
        elif choice == "2":
            return "long_short"
        print("❌ Invalid choice. Please enter 1 or 2.")


def select_time_period():
    """Let user select backtesting time period"""
    print("\n📅 Time Period Selection:")
    print("=========================")
    print("1. Last 7 days")
    print("2. Last 30 days")
    print("3. Last 90 days (3 months)")
    print("4. Last 180 days (6 months)")
    print("5. Last 365 days (1 year)")
    print("6. Custom period")
    print()

    periods = {
        "1": 7,
        "2": 30,
        "3": 90,
        "4": 180,
        "5": 365
    }

    while True:
        choice = input("Select time period (1-6): ").strip()
        if choice in periods:
            days = periods[choice]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        elif choice == "6":
            while True:
                try:
                    start_str = input("Enter start date (YYYY-MM-DD): ").strip()
                    end_str = input("Enter end date (YYYY-MM-DD): ").strip()

                    # Validate dates
                    start_date = datetime.strptime(start_str, '%Y-%m-%d')
                    end_date = datetime.strptime(end_str, '%Y-%m-%d')

                    if start_date >= end_date:
                        print("❌ Start date must be before end date.")
                        continue

                    return start_str, end_str
                except ValueError:
                    print("❌ Invalid date format. Please use YYYY-MM-DD.")
        print("❌ Invalid choice. Please enter 1-6.")


def select_initial_balance():
    """Let user select initial balance for backtesting"""
    print("\n💰 Initial Balance Selection:")
    print("=============================")
    print("1. $1,000")
    print("2. $5,000")
    print("3. $10,000")
    print("4. $25,000")
    print("5. $50,000")
    print("6. Custom amount")
    print()

    balances = {
        "1": 1000,
        "2": 5000,
        "3": 10000,
        "4": 25000,
        "5": 50000
    }

    while True:
        choice = input("Select initial balance (1-6): ").strip()
        if choice in balances:
            return balances[choice]
        elif choice == "6":
            while True:
                try:
                    amount = float(input("Enter custom amount ($): ").strip())
                    if amount <= 0:
                        print("❌ Amount must be positive.")
                        continue
                    return amount
                except ValueError:
                    print("❌ Invalid amount. Please enter a number.")
        print("❌ Invalid choice. Please enter 1-6.")


def main():
    """Main function to run backtesting system"""
    print("🚀 BAT Backtesting System")
    print("=" * 40)

    # Get API credentials
    api_key, secret_key = get_alpaca_credentials()

    # Select symbol
    symbol = select_symbol()

    # Check if stocks require API credentials
    is_crypto = '/' in symbol
    if not is_crypto and not api_key:
        print("❌ Stock data requires Alpaca API credentials.")
        print("Please set your API credentials and try again.")
        sys.exit(1)

    # Select strategy
    strategy = select_strategy()

    # Select trading mode
    trading_mode = select_trading_mode()

    # Select time period
    start_date, end_date = select_time_period()

    # Select initial balance
    initial_balance = select_initial_balance()

    # Determine asset type and unit for display
    if "/" in symbol:
        asset_type = "Cryptocurrency"
    else:
        asset_type = "Stock"

    print(f"\n⚙️  Backtest Configuration:")
    print(f"Asset Type: {asset_type}")
    print(f"Symbol: {symbol}")
    print(f"Strategy: {strategy.name}")
    print(f"Trading Mode: {'Long-only' if trading_mode == 'long_only' else 'Long/Short'}")
    print(f"Time Period: {start_date} to {end_date}")
    print(f"Initial Balance: ${initial_balance:,.2f}")

    if api_key:
        print(f"API Status: ✅ Connected to Alpaca")
    else:
        print(f"API Status: ⚠️  Public data only")

    print("\n" + "=" * 40)

    try:
        # Initialize data provider
        data_provider = AlpacaDataProvider(api_key, secret_key)

        # Fetch historical data
        print(f"\n🔍 Fetching historical data for {symbol}...")
        df = data_provider.get_data(
            ticker=symbol,
            timespan='1Min',
            from_date=start_date,
            to_date=end_date,
            limit=50000
        )

        if df.empty:
            print(f"❌ No data available for {symbol} in the specified period.")
            print("Please try a different symbol or time period.")
            return

        print(f"✅ Retrieved {len(df)} data points")
        print(f"📅 Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"💰 Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

        # Initialize backtest engine
        engine = BacktestEngine(initial_balance=initial_balance, trading_mode=trading_mode, symbol=symbol)

        # Run backtest
        print(f"\n🎯 Running backtest for {strategy.name}...")
        print("⏳ Please wait...")

        results = engine.backtest(df, strategy)

        # Display results
        if len(results) > 0:
            print("\n" + "🎉" + "="*48 + "🎉")
            print("                BACKTEST RESULTS")
            print("🎉" + "="*48 + "🎉")

            engine.print_analysis(results)

            # Show trade details
            print(f"\n📊 Trade Summary:")
            print("=" * 20)
            print(f"Total signals generated: {len(results)}")

            if len(results) > 0:
                completed_trades = results.dropna(subset=['Profit'], errors='ignore')
                print(f"Completed trades: {len(completed_trades)}")

                if len(completed_trades) > 0:
                    profitable_trades = completed_trades[completed_trades['Profit'] > 0]
                    losing_trades = completed_trades[completed_trades['Profit'] <= 0]
                    print(f"Profitable trades: {len(profitable_trades)}")
                    print(f"Losing trades: {len(losing_trades)}")

                print("\n📋 Recent Trades:")
                print("=" * 20)
                if len(results) > 5:
                    print("Last 5 trades:")
                    print(results.tail().to_string(index=False))
                else:
                    print("All trades:")
                    print(results.to_string(index=False))

            print(f"\n📈 Generating charts...")

            # Generate interactive chart
            try:
                engine.plot_interactive_chart(results)
                print("✅ Interactive chart displayed!")
            except Exception as e:
                print(f"⚠️ Could not display interactive chart: {e}")
                print("Showing basic chart instead...")
                engine.plot_results(results)

        else:
            print("\n⚠️  No trades were generated.")
            print("This could mean:")
            print("- The strategy didn't find any signals in the given timeframe")
            print("- The data period was too short")
            print("- The strategy parameters need adjustment")
            print("\nTry:")
            print("- Extending the time period")
            print("- Choosing a different strategy")
            print("- Selecting a more volatile asset")

    except KeyboardInterrupt:
        print("\n\n🛑 Backtest interrupted by user")

    except Exception as e:
        print(f"\n❌ Error during backtesting: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your configuration and try again.")

    finally:
        print("\n✅ Backtesting session completed!")


if __name__ == "__main__":
    main()