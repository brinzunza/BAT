import os
import sys
import tempfile
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('research')
from research.optimization.find_best import find_best_main

from strategies.mean_reversion import MeanReversionStrategy, MeanReversionExtremeStrategy
from strategies.moving_average import MovingAverageStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_bands_strategy import BollingerBandsStrategy
from strategies.candlestick_strategy import CandlestickPatternsStrategy
from data_providers.polygon_provider import PolygonDataProvider
from data_providers.alpaca_provider import AlpacaDataProvider, AlpacaBroker
from data_providers.oanda_provider import OandaProvider
from engines.backtest_engine import BacktestEngine
from engines.live_trading_engine import LiveTradingEngine
from engines.brokers import SimulatedBroker
from engines.ib_broker import IBBroker
from ui.live_trading_chart import LiveTradingChart


class TradingCLI:
    """Command Line Interface for the trading system"""
    
    def __init__(self):
        self.strategies = {
            '1': ('Mean Reversion (Conservative)', MeanReversionStrategy),
            '2': ('Mean Reversion (Extreme)', MeanReversionExtremeStrategy),
            '3': ('Moving Average', MovingAverageStrategy),
            '4': ('RSI', RSIStrategy),
            '5': ('MACD', MACDStrategy),
            '6': ('Bollinger Bands', BollingerBandsStrategy),
            '7': ('Candlestick Patterns', CandlestickPatternsStrategy)
        }
        
        self.data_provider = None
        self.broker = None
        self.alpaca_data_provider = None
        self.alpaca_broker = None
        self.oanda_provider = None
        self.ib_broker = None
        
    def display_banner(self):
        """Display application banner"""
        print("\n" + "=" * 50)
        print("            BAT - Backtesting & Automated Trading")
        print("=" * 50)
        print()
    
    def setup_data_provider(self):
        """Setup data provider with API key validation"""
        while True:
            print("\nData Provider Setup")
            print("-" * 20)

            api_key = input("Enter your Polygon API key (or press Enter to use default): ").strip()
            if not api_key:
                api_key = "your-api-key-here"  # Default placeholder

            try:
                # Create data provider instance
                self.data_provider = PolygonDataProvider(api_key)

                # Test the connection
                print("Testing API key...")
                success, message = self.data_provider.test_connection()

                if success:
                    print(f"✓ {message}")
                    print("✓ Data provider configured successfully")
                    return True
                else:
                    # API key validation failed
                    print(f"✗ {message}")
                    print("\nWhat would you like to do?")
                    print("1. Retry with a different API key")
                    print("2. Return to main menu")

                    choice = input("\nSelect option (1-2): ").strip()

                    if choice == '2':
                        self.data_provider = None
                        return False
                    # If choice is '1' or anything else, loop continues

            except Exception as e:
                print(f"✗ Error setting up data provider: {e}")
                print("\nWhat would you like to do?")
                print("1. Retry with a different API key")
                print("2. Return to main menu")

                choice = input("\nSelect option (1-2): ").strip()

                if choice == '2':
                    self.data_provider = None
                    return False
                # If choice is '1' or anything else, loop continues
    
    def setup_broker(self):
        """Setup broker interface"""
        print("\nBroker Setup")
        print("-" * 20)
        print("1. Simulated Broker (for testing)")
        print("2. Alpaca Broker (live trading)")
        
        choice = input("Select broker (1-2): ").strip()
        
        if choice == '1':
            initial_balance = float(input("Enter initial balance (default 10000): ") or "10000")
            self.broker = SimulatedBroker(initial_balance)
            print("✓ Simulated broker configured")
        
        elif choice == '2':
            print("Enter Alpaca credentials:")
            api_key = input("API Key: ").strip()
            secret_key = input("Secret Key: ").strip()
            base_url = input("Base URL (default: paper-api.alpaca.markets): ").strip()
            
            if not base_url:
                base_url = "https://paper-api.alpaca.markets/"
            
            try:
                self.broker = AlpacaBroker(api_key, secret_key, base_url)
                print("✓ Alpaca broker configured")
            except Exception as e:
                print(f"✗ Error setting up Alpaca broker: {e}")
                return False
        else:
            print("Invalid choice")
            return False
        
        return True

    def setup_alpaca_credentials(self):
        """Setup Alpaca credentials for live trading"""
        print("\nAlpaca Setup for Live Trading")
        print("-" * 30)
        print("Enter your Alpaca API credentials:")
        print("(You can get these from https://alpaca.markets/)")

        api_key = input("Alpaca API Key: ").strip()
        secret_key = input("Alpaca Secret Key: ").strip()

        if not api_key or not secret_key:
            print(" API credentials are required for live trading")
            return False

        # Ask about paper trading
        paper_trading = input("Use paper trading? (y/n, recommended: y): ").strip().lower()
        paper_trading = paper_trading != 'n'  # Default to paper trading

        try:
            # Test connection
            self.alpaca_data_provider = AlpacaDataProvider(api_key, secret_key)
            self.alpaca_broker = AlpacaBroker(api_key, secret_key, paper_trading)

            # Test account access
            account_info = self.alpaca_broker.get_account()
            if account_info:
                trading_mode = "Paper Trading" if paper_trading else "Live Trading"
                print(f" Connected to Alpaca ({trading_mode})")
                print(f"Account Status: {account_info.get('status', 'Unknown')}")
                if 'buying_power' in account_info:
                    print(f"Buying Power: ${float(account_info['buying_power']):.2f}")
                return True
            else:
                print(" Failed to connect to Alpaca account")
                return False

        except Exception as e:
            print(f" Error connecting to Alpaca: {e}")
            return False

    def setup_forex_credentials(self):
        """Setup OANDA and Interactive Brokers for forex trading"""
        print("\n💱 Forex Trading Setup (OANDA + Interactive Brokers)")
        print("=" * 60)
        print("This setup requires:")
        print("  1. OANDA account for live forex data")
        print("  2. Interactive Brokers TWS/Gateway for trade execution")
        print()

        # OANDA Setup
        print("📊 OANDA Configuration:")
        print("-" * 30)

        oanda_token = input("OANDA Access Token (or press Enter for default): ").strip()
        if not oanda_token:
            oanda_token = "4783ce686cc4960d43f7ac27c3e9c542-7a14009cdf2d7109a793fab7b4d0d462"

        oanda_account = input("OANDA Account ID (or press Enter for default): ").strip()
        if not oanda_account:
            oanda_account = "101-001-27040015-001"

        oanda_env = input("Environment (practice/live, default: practice): ").strip().lower() or "practice"

        try:
            # Test OANDA connection
            self.oanda_provider = OandaProvider(
                access_token=oanda_token,
                account_id=oanda_account,
                environment=oanda_env
            )

            # Test by fetching latest candle
            test_candle = self.oanda_provider.get_latest_candle("EURUSD")
            if test_candle:
                print(f"✓ Connected to OANDA ({oanda_env})")
                print(f"  Latest EUR/USD: {test_candle['close']:.5f}")
            else:
                print("✗ Failed to fetch data from OANDA")
                return False

        except Exception as e:
            print(f"✗ OANDA connection failed: {e}")
            return False

        # Interactive Brokers Setup
        print("\n🏦 Interactive Brokers Configuration:")
        print("-" * 30)
        print("Make sure TWS or IB Gateway is running with API enabled")
        print()

        ib_host = input("IB Host (default: 127.0.0.1): ").strip() or "127.0.0.1"
        ib_port = input("IB Port (7497=paper, 7496=live, default: 7497): ").strip()
        ib_port = int(ib_port) if ib_port else 7497
        ib_client_id = input("IB Client ID (default: 1): ").strip()
        ib_client_id = int(ib_client_id) if ib_client_id else 1

        try:
            # Test IB connection
            self.ib_broker = IBBroker()
            if self.ib_broker.connect_to_tws(ib_host, ib_port, ib_client_id):
                print(f"✓ Connected to IB TWS")

                # Get account info
                account = self.ib_broker.get_account()
                print(f"  Account Equity: ${account['equity']:,.2f}")
                print(f"  Buying Power: ${account['buying_power']:,.2f}")
                return True
            else:
                print("✗ Failed to connect to IB TWS")
                print("  Make sure TWS/Gateway is running and API is enabled")
                return False

        except Exception as e:
            print(f"✗ IB connection failed: {e}")
            return False

    def select_strategy(self):
        """Strategy selection menu"""
        print("\nAvailable Strategies:")
        print("-" * 50)

        for key, (name, _) in self.strategies.items():
            print(f"{key}. {name}")

        choice = input("Select strategy (1-7): ").strip()
        
        if choice not in self.strategies:
            print("Invalid choice")
            return None
        
        strategy_name, strategy_class = self.strategies[choice]

        # Get strategy parameters
        if choice == '1':  # Mean Reversion (Conservative)
            window = int(input("Enter window size (default 20): ") or "20")
            num_std = float(input("Enter standard deviations (default 2.0): ") or "2.0")
            return strategy_class(window=window, num_std=num_std)

        elif choice == '2':  # Mean Reversion (Extreme)
            window = int(input("Enter window size (default 20): ") or "20")
            num_std = float(input("Enter standard deviations (default 2.0): ") or "2.0")
            return strategy_class(window=window, num_std=num_std)

        elif choice == '3':  # Moving Average
            short = int(input("Enter short window (default 1): ") or "1")
            medium = int(input("Enter medium window (default 5): ") or "5")
            long_win = int(input("Enter long window (default 25): ") or "25")
            return strategy_class(short_window=short, medium_window=medium, long_window=long_win)
        
        elif choice == '4':  # RSI
            window = int(input("Enter RSI window (default 14): ") or "14")
            oversold = float(input("Enter oversold threshold (default 30): ") or "30")
            overbought = float(input("Enter overbought threshold (default 70): ") or "70")
            return strategy_class(window=window, oversold_threshold=oversold, overbought_threshold=overbought)

        elif choice == '5':  # MACD
            fast = int(input("Enter fast EMA period (default 12): ") or "12")
            slow = int(input("Enter slow EMA period (default 26): ") or "26")
            signal = int(input("Enter signal line period (default 9): ") or "9")
            return strategy_class(fast=fast, slow=slow, signal=signal)

        elif choice == '6':  # Bollinger Bands
            window = int(input("Enter window size (default 20): ") or "20")
            num_std = float(input("Enter standard deviations (default 2): ") or "2")
            return strategy_class(window=window, num_std=num_std)

        elif choice == '7':  # Candlestick Patterns
            return strategy_class()

        return None

    def select_trading_mode(self):
        """Let user select trading mode"""
        print("\nTrading Mode Selection:")
        print("==========================")
        print("1. Buy & Close Only (Long-only trading)")
        print("   - Buy signals → Buy positions")
        print("   - Sell signals → Close positions")
        print("   - No short selling")
        print()
        print("2. Buy & Short Trading (Long/Short trading)")
        print("   - Buy signals → Buy positions (or close short)")
        print("   - Sell signals → Short positions (or close long)")
        print("   - Allows short selling for advanced strategies")
        print()

        while True:
            choice = input("Select trading mode (1-2): ").strip()
            if choice == "1":
                return "long_only"
            elif choice == "2":
                return "long_short"
            print(" Invalid choice. Please enter 1 or 2.")

    def select_asset_type(self):
        """Let user select between crypto, stocks, and forex for backtesting (Polygon only)"""
        print("\n Asset Type Selection (Backtesting):")
        print("======================================")
        print("1. Cryptocurrency (Polygon API)")
        print("2. Stocks (Polygon API)")
        print("3. Forex (Polygon API)")
        print()

        while True:
            choice = input("Select asset type (1-3): ").strip()
            if choice in ['1', '2', '3']:
                return choice
            print(" Invalid choice. Please enter 1, 2, or 3.")

    def configure_data_parameters(self):
        """Configure data parameters with support for crypto, stocks, and forex"""
        print("\nData Configuration:")
        print("-" * 20)

        # Select asset type
        asset_choice = self.select_asset_type()

        # Configure ticker based on asset type (Polygon only for backtesting)
        if asset_choice == '1':  # Crypto (Polygon)
            print("\n₿ Cryptocurrency Configuration (Polygon):")
            ticker = input("Enter crypto ticker (default X:BTCUSD): ").strip() or "X:BTCUSD"
            print("Common crypto tickers: X:BTCUSD, X:ETHUSD, X:DOGEUSD, X:LTCUSD")
        elif asset_choice == '2':  # Stocks (Polygon)
            print("\n Stock Configuration (Polygon):")
            ticker = input("Enter stock ticker (default AAPL): ").strip() or "AAPL"
            print("Common stock tickers: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, SPY, QQQ")
        else:  # asset_choice == '3' - Forex (Polygon)
            print("\n Forex Configuration (Polygon):")
            ticker = input("Enter forex pair (default C:EURUSD): ").strip() or "C:EURUSD"

            # Show available forex pairs if data provider is set up
            if hasattr(self.data_provider, 'get_available_forex_pairs'):
                try:
                    forex_pairs = self.data_provider.get_available_forex_pairs()
                    print("Available forex pairs:")
                    for i, pair in enumerate(forex_pairs):
                        if i % 4 == 0 and i > 0:
                            print()  # New line every 4 pairs
                        print(f"{pair:<12}", end=" ")
                    print()  # Final newline
                except:
                    print("Common forex pairs: C:EURUSD, C:GBPUSD, C:USDJPY, C:USDCHF, C:AUDUSD, C:USDCAD")
                    print("                    C:NZDUSD, C:EURGBP, C:EURJPY, C:GBPJPY, C:CHFJPY, C:EURCHF")
            else:
                print("Common forex pairs: C:EURUSD, C:GBPUSD, C:USDJPY, C:USDCHF, C:AUDUSD, C:USDCAD")
                print("                    C:NZDUSD, C:EURGBP, C:EURJPY, C:GBPJPY, C:CHFJPY, C:EURCHF")

        # Configure timespan (Polygon API only)
        timespan = input("Enter timespan (minute/hour/day, default minute): ").strip() or "minute"

        # Date configuration
        use_defaults = input("Use default date range? (y/n, default y): ").strip().lower()
        if use_defaults != 'n':
            # Default: last 30 days
            to_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        else:
            from_date = input("Enter from date (YYYY-MM-DD): ").strip()
            to_date = input("Enter to date (YYYY-MM-DD): ").strip()

        limit = int(input("Enter data limit (default 50000): ") or "50000")

        return {
            'ticker': ticker,
            'timespan': timespan,
            'from_date': from_date,
            'to_date': to_date,
            'limit': limit,
            'asset_type': asset_choice
        }
    
    def run_backtest(self):
        """Run backtesting workflow"""
        print("\n" + "=" * 40)
        print("           BACKTESTING MODE")
        print("=" * 40)

        # Select strategy
        strategy = self.select_strategy()
        if not strategy:
            return

        # Select trading mode
        trading_mode = self.select_trading_mode()

        # Configure data
        data_params = self.configure_data_parameters()

        # Get initial balance
        initial_balance = float(input("Enter initial balance (default 10000): ") or "10000")

        # Get position sizing percentage
        position_percentage = float(input("Enter percentage of account to use per trade (1-100, default 100): ") or "100")
        if position_percentage < 1 or position_percentage > 100:
            print("Invalid percentage. Using 100% of account.")
            position_percentage = 100

        # Get spread in pips for forex
        spread_pips = 0.0
        if data_params['asset_type'] == '3':  # Forex
            print("\n💱 Forex Spread Configuration:")
            print("Typical spreads: EUR/USD: 0.5-2 pips, GBP/USD: 1-3 pips, USD/JPY: 0.5-2 pips")
            spread_pips = float(input("Enter spread in pips (default 1.0): ") or "1.0")
            if spread_pips < 0:
                print("Invalid spread. Using 1.0 pip.")
                spread_pips = 1.0

        print(f"\n Running backtest for {strategy.name}...")
        print(f" Ticker: {data_params['ticker']}")
        print(f"Trading Mode: {'Long-only' if trading_mode == 'long_only' else 'Long/Short'}")
        print(f"⏰ Timespan: {data_params['timespan']}")
        print(f"📅 From: {data_params['from_date']} To: {data_params['to_date']}")
        if data_params['asset_type'] == '3':
            print(f"💱 Spread: {spread_pips} pips")

        try:
            # Get data based on asset type
            print("Fetching data...")

            # Remove asset_type from params before passing to data provider
            asset_type = data_params.pop('asset_type')

            # Backtesting only uses Polygon API now
            if not self.data_provider:
                print(" Polygon data provider not configured. Please configure it first.")
                return
            df = self.data_provider.get_data(**data_params)

            print(f"✓ Retrieved {len(df)} data points")

            if df.empty:
                print(f" No data available for {data_params['ticker']} in the specified period.")
                print("Please try a different ticker or time period.")
                return

            # Run backtest
            print("Running backtest...")
            # Pass the ticker symbol to the engine for proper formatting
            engine = BacktestEngine(initial_balance, trading_mode, data_params['ticker'], position_percentage, spread_pips)
            results = engine.backtest(df, strategy)
            
            # Display results
            print("\n" + "=" * 40)
            print("           BACKTEST RESULTS")
            print("=" * 40)
            
            if len(results) > 0:
                engine.print_analysis(results)
                
                # Ask if user wants to see detailed results or plot
                show_details = input("\nShow detailed trade results? (y/n): ").strip().lower() == 'y'
                if show_details:
                    print("\nDetailed Trade Overview:")
                    self._print_detailed_trade_results(results)

                    # Create CSV export
                    self._export_trade_results_to_csv(results, strategy)

                plot_results = input("Show balance plot? (y/n): ").strip().lower() == 'y'
                if plot_results:
                    engine.plot_results(results)

                # Ask for interactive candlestick chart
                interactive_chart = input("Want to see interactive bar chart? (y/n): ").strip().lower() == 'y'
                if interactive_chart:
                    engine.plot_interactive_chart(results)
            else:
                print("No trades generated by this strategy with the given data.")
        
        except Exception as e:
            print(f"✗ Backtest failed: {e}")
    
    def run_forex_live_trading(self):
        """Run forex live trading with OANDA data + IB execution"""
        print("\n" + "=" * 60)
        print("       💱 FOREX LIVE TRADING")
        print("          OANDA Data + Interactive Brokers Execution")
        print("=" * 60)

        # Setup credentials if not already configured
        if not self.oanda_provider or not self.ib_broker:
            print("\n🔑 Forex trading requires OANDA + IB credentials")
            if not self.setup_forex_credentials():
                return

        # Select strategy
        strategy = self.select_strategy()
        if not strategy:
            return

        # Select trading mode
        trading_mode = self.select_trading_mode()

        # Configure trading parameters
        print("\n⚙️  Forex Trading Configuration:")
        print("-" * 30)

        # Forex pair selection
        forex_pair = input("Enter forex pair (default EURUSD): ").strip().upper() or "EURUSD"

        # Historical lookback
        lookback = int(input("Historical candles to fetch (default 200): ") or "200")

        # Position sizing
        print("\nPosition Sizing:")
        print("1. Percentage of account")
        print("2. Fixed quantity (base currency units)")

        sizing_choice = input("Select method (1-2, default 1): ").strip() or "1"

        if sizing_choice == "2":
            quantity = float(input(f"Enter {forex_pair[:3]} quantity (e.g., 20000 = 20K): ") or "20000")
            position_percentage = None
        else:
            position_percentage = float(input("Position size as % of account (1-100, default 100): ") or "100")
            if position_percentage < 1 or position_percentage > 100:
                print("Invalid percentage. Using 100%.")
                position_percentage = 100
            quantity = None

        # Update interval
        update_interval = int(input("Update interval in seconds (default 60): ") or "60")

        print(f"\n📋 Configuration Summary:")
        print(f"   Strategy: {strategy.name}")
        print(f"   Symbol: {forex_pair}")
        print(f"   Trading Mode: {'Long-only' if trading_mode == 'long_only' else 'Long/Short'}")
        print(f"   Historical Lookback: {lookback} candles")
        if position_percentage:
            print(f"   Position Size: {position_percentage}% of account")
        else:
            print(f"   Position Size: {quantity:,.0f} {forex_pair[:3]}")
        print(f"   Update Interval: {update_interval}s")

        confirm = input("\nStart forex live trading? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Forex trading cancelled.")
            return

        try:
            # Get account info
            initial_balance = self.ib_broker.get_account()['equity']

            print(f"\n🚀 Initializing live trading chart...")
            print(f"Chart will open in a new window")
            print(f"Data updates every {update_interval} seconds")
            print(f"All trades will be logged to console")
            print(f"  Press Ctrl+C or close chart window to stop")
            print(f"\n{'='*60}")

            # Create live trading chart with forex support
            forex_chart = LiveTradingChart(
                strategy=strategy,
                symbol=forex_pair,
                trading_mode=trading_mode,
                position_percentage=position_percentage if position_percentage else 100,
                quantity=quantity,
                data_provider=self.oanda_provider,
                broker_interface=self.ib_broker,
                lookback=lookback
            )

            print("✓ Trading engine ready")
            print("✓ Live chart ready")
            print("\n" + "=" * 60)
            print(" 🟢 LIVE TRADING STARTED")
            print("=" * 60)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Symbol: {forex_pair}")
            print("Chart will show: Price candles | Strategy indicators | Buy/Sell signals")
            print("Terminal shows: Price | Position | Unrealized P&L | Realized P&L\n")

            # Start live chart with animation (this will block until chart closed)
            forex_chart.start_live_trading(update_interval * 1000)  # Convert to milliseconds

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print(" 🛑 STOPPING FOREX LIVE TRADING")
            print("=" * 60)

        finally:
            # Print final summary
            if 'forex_chart' in locals():
                print("\n" + "=" * 60)
                print(" 📊 FOREX TRADING SESSION SUMMARY")
                print("=" * 60)

                performance = forex_chart.get_performance_summary()
                trade_history = forex_chart.get_trade_history()

                print(f"\nStrategy: {strategy.name}")
                print(f"Symbol: {forex_pair}")
                print(f"Total Trades: {performance['total_trades']}")
                print(f"Profitable Trades: {performance['profitable_trades']}")
                print(f"Losing Trades: {performance['losing_trades']}")
                print(f"Win Rate: {performance['win_rate']:.1f}%")
                print(f"Final Balance: ${performance['current_balance']:,.2f}")
                print(f"Total Return: ${performance['total_return']:,.2f}")
                print(f"Percent Return: {performance['percent_return']:.2f}%")

                if len(trade_history) > 0:
                    print(f"\n📝 Recent Trades:")
                    self._print_detailed_trade_results(trade_history.tail(10))

                print("\n" + "=" * 60)

    def run_live_trading(self):
        """Run live trading workflow with real-time charting"""
        print("\n" + "=" * 50)
        print("       LIVE TRADING WITH REAL-TIME CHARTS")
        print("=" * 50)

        # Setup Alpaca credentials if not already configured
        if not self.alpaca_data_provider or not self.alpaca_broker:
            print("🔑 Alpaca credentials required for live trading.")
            print(" Live trading uses Alpaca for both data and execution.")
            if not self.setup_alpaca_credentials():
                return

        # Select strategy
        strategy = self.select_strategy()
        if not strategy:
            return

        # Select trading mode
        trading_mode = self.select_trading_mode()

        # Configure trading parameters
        print("\n Live Trading Configuration:")
        print("-" * 30)

        # Select asset type for live trading
        print(" Asset Selection for Live Trading:")
        print("===================================")
        print("1. Cryptocurrency - BTC/USD")
        print("2. Cryptocurrency - ETH/USD")
        print("3. Cryptocurrency - DOGE/USD")
        print("4. Cryptocurrency - Custom Crypto")
        print("5. Stock - AAPL")
        print("6. Stock - MSFT")
        print("7. Stock - GOOGL")
        print("8. Stock - TSLA")
        print("9. Stock - Custom Stock")
        print()

        asset_symbols = {
            "1": ("BTC/USD", 0.01, "crypto"),
            "2": ("ETH/USD", 0.1, "crypto"),
            "3": ("DOGE/USD", 100, "crypto"),
            "4": ("CUSTOM_CRYPTO", 1.0, "crypto"),
            "5": ("AAPL", 1, "stock"),
            "6": ("MSFT", 1, "stock"),
            "7": ("GOOGL", 1, "stock"),
            "8": ("TSLA", 1, "stock"),
            "9": ("CUSTOM_STOCK", 1, "stock")
        }

        while True:
            choice = input("Select asset (1-9): ").strip()
            if choice in asset_symbols:
                symbol, default_quantity, asset_type = asset_symbols[choice]

                if choice == "4":  # Custom crypto
                    symbol = input("Enter crypto pair (e.g., LTC/USD): ").strip().upper()
                    if "/" not in symbol:
                        symbol += "/USD"
                elif choice == "9":  # Custom stock
                    symbol = input("Enter stock ticker (e.g., AMZN): ").strip().upper()
                    asset_type = "stock"

                break
            print(" Invalid choice. Please enter 1-9.")

        print(f"Selected: {symbol}")

        # Configure position sizing method
        print("\nPosition Sizing Options:")
        print("1. Fixed quantity (shares/units)")
        print("2. Percentage of account")

        sizing_choice = input("Select position sizing method (1-2, default 2): ").strip() or "2"

        if sizing_choice == "1":
            # Fixed quantity method
            if asset_type == "crypto":
                unit = symbol.split("/")[0] if "/" in symbol else "units"
                quantity = float(input(f"Enter {unit} position size (default {default_quantity}): ") or str(default_quantity))
            else:  # stock
                quantity = int(input(f"Enter number of shares (default {default_quantity}): ") or str(default_quantity))
            position_percentage = None
        else:
            # Percentage method
            position_percentage = float(input("Enter percentage of account to use per trade (1-100, default 20): ") or "20")
            if position_percentage < 1 or position_percentage > 100:
                print("Invalid percentage. Using 20% of account.")
                position_percentage = 20
            quantity = None  # Will be calculated dynamically

        # Update interval
        update_interval = int(input("Chart update interval in seconds (default 60): ") or "60")

        # Select broker type
        print("\n🏦 Broker Selection:")
        print("====================")
        print("1. Alpaca Paper Trading")
        print("2. Simulated Broker")
        print()

        while True:
            broker_choice = input("Select broker (1-2): ").strip()
            if broker_choice == "1":
                use_simulated_broker = False
                break
            elif broker_choice == "2":
                use_simulated_broker = True
                break
            print(" Invalid choice. Please enter 1 or 2.")

        print(f"\n  Configuration Summary:")
        print(f"   Asset Type: {'Cryptocurrency' if asset_type == 'crypto' else 'Stock'}")
        print(f"   Strategy: {strategy.name}")
        print(f"   Symbol: {symbol}")
        print(f"   Trading Mode: {'Long-only' if trading_mode == 'long_only' else 'Long/Short'}")

        if position_percentage is not None:
            print(f"   Position Size: {position_percentage}% of account per trade")
        else:
            if asset_type == "crypto":
                unit = symbol.split("/")[0] if "/" in symbol else "units"
                print(f"   Position Size: {quantity} {unit}")
            else:
                print(f"   Position Size: {quantity} shares")

        print(f"   Update Interval: {update_interval} seconds")
        print(f"   Broker Type: {'SimulatedBroker' if use_simulated_broker else 'Alpaca Paper Trading'}")

        if use_simulated_broker:
            print(f"   Initial Balance: $10,000 (simulated)")
        else:
            print(f"   Account Balance: Will be retrieved from Alpaca")

        print(f"\n Features:")
        print(f"    Real-time candlestick chart")
        print(f"    Strategy indicators overlay")
        print(f"    Buy/sell signals on chart")
        print(f"    Live P&L tracking")
        print(f"    Automated trade execution")
        print(f"    Console trade logging")

        confirm = input(f"\nStart live trading? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Live trading cancelled.")
            return

        try:
            # Create live trading chart
            live_chart = LiveTradingChart(
                strategy=strategy,
                api_key=self.alpaca_data_provider.api_key,
                secret_key=self.alpaca_data_provider.secret_key,
                symbol=symbol,
                paper_trading=self.alpaca_broker.paper_trading,
                quantity=quantity,
                trading_mode=trading_mode,
                use_simulated_broker=use_simulated_broker,
                initial_balance=10000,
                position_percentage=position_percentage
            )

            print(f"\nStarting live trading with charts...")
            print(f"Chart will open in a new window")
            print(f"Data updates every {update_interval} seconds")
            print(f"All trades will be logged to console")
            print(f"  Press Ctrl+C to stop")
            print(f"\n{'='*50}")

            # Start live trading with charts
            animation = live_chart.start_live_trading(update_interval * 1000)  # Convert to milliseconds

        except KeyboardInterrupt:
            print(f"\n\n  Live trading stopped by user")

            # Show final performance
            if 'live_chart' in locals():
                performance = live_chart.get_performance_summary()
                trade_history = live_chart.get_trade_history()

                print(f"\n" + "=" * 40)
                print(f"         FINAL TRADING SUMMARY")
                print(f"=" * 40)
                print(f"Strategy: {strategy.name}")
                print(f"Total Trades: {performance['total_trades']}")
                print(f"Profitable Trades: {performance['profitable_trades']}")
                print(f"Losing Trades: {performance['losing_trades']}")
                print(f"Win Rate: {performance['win_rate']:.1f}%")
                print(f"Final Balance: ${performance['current_balance']:.2f}")
                print(f"Total Return: ${performance['total_return']:.2f}")
                print(f"Percent Return: {performance['percent_return']:.2f}%")
                print(f"Current Position: {performance['current_position']}")

                if len(trade_history) > 0:
                    print(f"\n Recent Trades:")
                    # Show last 5 trades with detailed format
                    recent_trades = trade_history.tail()
                    self._print_detailed_trade_results(recent_trades)
                else:
                    print(f"\n No trades executed during this session")

        except Exception as e:
            print(f"\n Live trading error: {e}")
            print(f"Please check your Alpaca credentials and internet connection.")

    def _print_detailed_trade_results(self, results):
        """Print detailed trade overview with enhanced formatting"""
        if len(results) == 0:
            print("No trades executed.")
            return

        print("=" * 170)
        print(f"{'#':<3} {'Time':<19} {'Price':<10} {'Position':<8} {'Action':<12} {'Shares':<12} {'Cost/Proceeds':<15} {'Last Trade P&L':<15} {'Cash Balance':<15} {'Total Worth*':<15} {'Total Profit*':<15} {'Result':<8}")
        print("=" * 170)
        print("=" * 170)

        for i, (idx, trade) in enumerate(results.iterrows(), 1):
            # Format values
            time_str = trade['Time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(trade['Time'], 'strftime') else str(trade['Time'])
            price = f"${trade['Price']:.4f}"
            position = "LONG" if trade['Position'] == 1 else ("SHORT" if trade['Position'] == -1 else "FLAT")
            action = trade['Action']
            shares = f"{trade['Shares']:.6f}" if 'Shares' in trade else "N/A"

            # Cost/Proceeds
            if 'Cost' in trade and pd.notna(trade['Cost']):
                cost_proceeds = f"${trade['Cost']:.2f}"
            elif 'Proceeds' in trade and pd.notna(trade['Proceeds']):
                cost_proceeds = f"${trade['Proceeds']:.2f}"
            else:
                cost_proceeds = "N/A"

            # Last trade realized P&L
            last_trade_pnl = f"${trade['Last_Trade_Realized']:.2f}" if 'Last_Trade_Realized' in trade and pd.notna(trade['Last_Trade_Realized']) else "N/A"

            # Cash balance
            cash_balance = f"${trade['Balance']:.2f}" if 'Balance' in trade else "N/A"

            # Total account worth
            total_worth = f"${trade['Total_Account_Worth']:.2f}" if 'Total_Account_Worth' in trade else "N/A"

            # Total profit
            total_profit = f"${trade['Total_Profit']:.2f}" if 'Total_Profit' in trade else "N/A"

            # Trade result
            trade_result = trade.get('Trade_Result', trade.get('Result', 'N/A'))

            print(f"{i:<3} {time_str:<19} {price:<10} {position:<8} {action:<12} {shares:<12} {cost_proceeds:<15} {last_trade_pnl:<15} {cash_balance:<15} {total_worth:<15} {total_profit:<15} {trade_result:<8}")

        print("=" * 170)
        print(f"Total Trades: {len(results)}")
        print("\nNote: This display shows account worth based on realized gains/losses only.")
        print("Open positions do not affect the total worth until they are closed.")

    def _export_trade_results_to_csv(self, results, strategy):
        """Export detailed trade results to CSV file in a temporary folder"""
        try:
            # Create a temporary directory for CSV exports
            temp_dir = tempfile.mkdtemp(prefix="bat_exports_")

            # Generate filename with timestamp and strategy name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            strategy_name = getattr(strategy, 'name', 'UnknownStrategy').replace(' ', '_')
            filename = f"backtest_results_{strategy_name}_{timestamp}.csv"
            filepath = os.path.join(temp_dir, filename)

            # Export to CSV
            results.to_csv(filepath, index=False)

            print(f"\n Trade results exported to CSV:")
            print(f"   File: {filename}")
            print(f"   Location: {temp_dir}")
            print(f"   Full path: {filepath}")
            print(f"   Records: {len(results)} trades")

        except Exception as e:
            print(f" Error exporting to CSV: {e}")

    def run_live_trading_menu(self):
        """Consolidated live trading menu"""
        print("\n" + "=" * 50)
        print("           LIVE TRADING")
        print("=" * 50)
        print("\nSelect market type:")
        print("1. Stocks/Crypto (Alpaca)")
        print("2. Forex (OANDA + Interactive Brokers)")
        print("3. Back to Main Menu")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == '1':
            self.run_live_trading()
        elif choice == '2':
            self.run_forex_live_trading()
        elif choice == '3':
            return
        else:
            print("Invalid choice. Please try again.")

    def download_dataset(self, ticker, start, end, timeframe, limit=50000):

    
        print("Downloading dataset...")
        try:
            df = self.data_provider.get_data(
                ticker=ticker,
                timespan=timeframe,
                from_date=start,
                to_date=end,
                limit=limit
            )

            clean_ticker = ticker.replace(':', '_').replace('/', '_')
            filename = f"{clean_ticker}_{timeframe}_{start}_to_{end}.csv"

            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            datasets_dir = os.path.join(script_dir, 'research', 'datasets')

            os.makedirs(datasets_dir, exist_ok=True)

            filepath = os.path.join(datasets_dir, filename)

            df.to_csv(filepath, index=False)

            print(f"✓ Dataset successfully downloaded")

        except Exception as e:
            print(f"✗ Dataset download failed: {e}")

    def optimize_strategy(self):
        strategy = input("Choose Strategy to optimize (Mean Reversion: 1): ").strip()
        if strategy == '1':
            dataset_path = input("Enter path to dataset CSV (default: /research/datasets/X_BTCUSD_minute_2025-01-01_to_2025-09-01.csv): ").strip()
            if not dataset_path:
                dataset_path = "/esearch/datasets/X_BTCUSD_minute_2025-01-01_to_2025-09-01.csv"
            print(f"\nStarting optimization for {dataset_path}...")
            find_best_main(dataset_path)
        return

    def main_menu(self):
        """Main application menu"""
        while True:
            self.display_banner()

            print("Main Menu:")
            print("-" * 10)
            print("1. Backtest")
            print("2. Live Trading")
            print("3. Research/Optimization")
            print("4. Exit")

            choice = input("\nSelect option (1-4): ").strip()

            if choice == '1':
                if not self.data_provider:
                    print("Data provider not configured. Setting up now...")
                    if not self.setup_data_provider():
                        continue
                self.run_backtest()
                input("\nPress Enter to continue...")

            elif choice == '2':
                self.run_live_trading_menu()
                input("\nPress Enter to continue...")

            elif choice == '3':
                print("\n" + "=" * 50)
                print("                           RESEARCH & OPTIMIZATION")
                print("=" * 50)
                new_dataset_or_not = input("Want to use a new or existing dataset? (New: 1, Existing: 0): ").strip()
                if new_dataset_or_not == '1':
                    self.setup_data_provider()
                    ticker = input("Choose a ticker: ")
                    start = input("Choose a start date: ")
                    end = input("Choose an end date: ")
                    timeframe = input("Choose a timeframe: ")
                    limit = input("Choose a limit: ")
                    self.download_dataset(ticker, start, end, timeframe, limit)
                    self.optimize_strategy()
                elif new_dataset_or_not == '0':
                    self.optimize_strategy()


            elif choice == '4':
                print("Thank you for using BAT!")
                # Disconnect IB if connected
                if self.ib_broker and self.ib_broker.connected:
                    print("Disconnecting from IB TWS...")
                    self.ib_broker.disconnect_from_tws()
                break

            else:
                print("Invalid choice. Please try again.")
                input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    cli = TradingCLI()
    cli.main_menu()


if __name__ == "__main__":
    main()