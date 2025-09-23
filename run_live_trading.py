#!/usr/bin/env python3
"""
Live Trading System for BTC/USD using Alpaca API
Integrates live charting with trading strategies
"""

import os
import sys
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_trading_chart import LiveTradingChart
from strategies.bollinger_bands_strategy import BollingerBandsStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy


def get_alpaca_credentials():
    """Get Alpaca API credentials from environment or user input"""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("⚠️  Alpaca API credentials not found in environment variables")
        print("You can either:")
        print("1. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("2. Enter them now (they will be used for this session only)")
        print("3. Continue without credentials (simulation mode only)")
        print()

        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "2":
            api_key = input("Enter Alpaca API Key: ").strip()
            secret_key = input("Enter Alpaca Secret Key: ").strip()
        elif choice == "3":
            print("Continuing in simulation mode...")
            api_key = None
            secret_key = None
        else:
            print("Please set environment variables and try again.")
            sys.exit(1)

    return api_key, secret_key


def select_strategy():
    """Let user select a trading strategy"""
    strategies = {
        "1": ("Bollinger Bands", BollingerBandsStrategy()),
        "2": ("RSI Strategy", RSIStrategy()),
        "3": ("MACD Strategy", MACDStrategy())
    }

    print("\n📊 Available Trading Strategies:")
    print("================================")
    for key, (name, _) in strategies.items():
        print(f"{key}. {name}")

    while True:
        choice = input("\nSelect strategy (1-3): ").strip()
        if choice in strategies:
            return strategies[choice][1]
        print("❌ Invalid choice. Please enter 1, 2, or 3.")


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
                        return custom_symbol, 0.1  # Default quantity for custom crypto
                    else:
                        symbol = crypto_symbols[crypto_choice][0]
                        # Set appropriate quantities based on crypto type
                        if "BTC" in symbol:
                            quantity = 0.01
                        elif "ETH" in symbol:
                            quantity = 0.1
                        else:
                            quantity = 1.0
                        return symbol, quantity
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
                        return custom_symbol, 1  # Default quantity for custom stock
                    else:
                        symbol = stock_symbols[stock_choice][0]
                        # Set appropriate quantities based on stock price ranges
                        if symbol in ["SPY", "QQQ"]:
                            quantity = 2  # ETFs - buy a few shares
                        elif symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NFLX"]:
                            quantity = 1  # High-priced stocks - buy 1 share
                        elif symbol == "TSLA":
                            quantity = 1  # Tesla - buy 1 share
                        elif symbol == "NVDA":
                            quantity = 1  # NVIDIA - buy 1 share
                        else:
                            quantity = 1  # Default for other stocks
                        return symbol, quantity
                print(f"❌ Invalid choice. Please enter 1-{len(stock_symbols)}.")

        print("❌ Invalid choice. Please enter 1 or 2.")


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
    print("   - Requires margin account with short selling enabled")
    print()

    while True:
        choice = input("Select trading mode (1-2): ").strip()
        if choice == "1":
            return "long_only"
        elif choice == "2":
            return "long_short"
        print("❌ Invalid choice. Please enter 1 or 2.")


def select_broker_type():
    """Let user select broker type"""
    print("\n🏦 Broker Type Selection:")
    print("========================")
    print("1. Alpaca Paper Trading")
    print("   - Uses Alpaca's paper trading account")
    print("   - Connects to real Alpaca API")
    print("   - Real order execution in paper environment")
    print()
    print("2. Simulated Broker")
    print("   - Uses live Alpaca market data")
    print("   - Local account management")
    print("   - Instant order fills (no slippage)")
    print("   - Perfect for strategy testing")
    print()

    while True:
        choice = input("Select broker type (1-2): ").strip()
        if choice == "1":
            return False  # use_simulated_broker = False
        elif choice == "2":
            return True   # use_simulated_broker = True
        print("❌ Invalid choice. Please enter 1 or 2.")


def main():
    """Main function to run live trading system"""
    print("🚀 BAT Live Trading System")
    print("=" * 40)

    # Get API credentials
    api_key, secret_key = get_alpaca_credentials()

    # Select symbol and quantity
    symbol, quantity = select_symbol()

    # Select strategy
    strategy = select_strategy()

    # Select trading mode
    trading_mode = select_trading_mode()

    # Select broker type
    use_simulated_broker = select_broker_type()

    # Configuration
    paper_trading = True  # Always use paper trading for safety when using Alpaca
    initial_balance = 10000

    # Determine asset type and unit for display
    if "/" in symbol:
        # Crypto
        asset_unit = symbol.split("/")[0]
        asset_type = "Cryptocurrency"
    else:
        # Stock
        asset_unit = "shares"
        asset_type = "Stock"

    print(f"\n⚙️  Configuration:")
    print(f"Asset Type: {asset_type}")
    print(f"Symbol: {symbol}")
    print(f"Strategy: {strategy.name}")
    print(f"Trading Mode: {'Long-only' if trading_mode == 'long_only' else 'Long/Short'}")
    print(f"Broker Type: {'SimulatedBroker' if use_simulated_broker else 'Alpaca Paper Trading'}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    if "/" in symbol:
        print(f"Position Size: {quantity} {asset_unit}")
    else:
        print(f"Position Size: {quantity} {asset_unit}")

    if api_key:
        print(f"API Status: ✅ Connected to Alpaca")
    else:
        print(f"API Status: ⚠️  Simulation mode (no actual trading)")

    print("\n" + "=" * 40)

    try:
        # Create live trading chart
        live_chart = LiveTradingChart(
            strategy=strategy,
            api_key=api_key,
            secret_key=secret_key,
            symbol=symbol,
            paper_trading=paper_trading,
            quantity=quantity,
            trading_mode=trading_mode,
            use_simulated_broker=use_simulated_broker,
            initial_balance=initial_balance
        )

        print("\n🎯 Starting live trading...")
        print("📈 Chart will update every 60 seconds")
        print("🔄 Trading signals will be processed automatically")
        print("⏹️  Press Ctrl+C to stop\n")

        # Start live trading with 60-second updates
        animation = live_chart.start_live_trading(update_interval=60000)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopping live trading...")
        if 'live_chart' in locals():
            live_chart.stop_trading()

            # Show final performance summary
            performance = live_chart.get_performance_summary()
            trade_history = live_chart.get_trade_history()

            print("\n📊 Final Performance Summary:")
            print("=" * 30)
            print(f"Total Trades: {performance['total_trades']}")
            print(f"Profitable Trades: {performance['profitable_trades']}")
            print(f"Losing Trades: {performance['losing_trades']}")
            print(f"Win Rate: {performance['win_rate']:.1f}%")
            print(f"Final Balance: ${performance['current_balance']:.2f}")
            print(f"Total Return: ${performance['total_return']:.2f}")
            print(f"Percent Return: {performance['percent_return']:.2f}%")

            # Show detailed account summary for SimulatedBroker
            if use_simulated_broker and hasattr(live_chart.broker, 'print_account_summary'):
                live_chart.broker.print_account_summary()

            if len(trade_history) > 0:
                print(f"\n📈 Trade History:")
                print(trade_history.to_string(index=False))

        print("\n✅ System stopped successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()