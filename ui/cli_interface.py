import os
import sys
import tempfile
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.mean_reversion import MeanReversionStrategy
from strategies.moving_average import MovingAverageStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.macd_strategy import MACDStrategy
from strategies.bollinger_bands_strategy import BollingerBandsStrategy
from strategies.candlestick_strategy import CandlestickPatternsStrategy
from data_providers.polygon_provider import PolygonDataProvider
from data_providers.alpaca_provider import AlpacaDataProvider, AlpacaBroker
from engines.backtest_engine import BacktestEngine
from engines.live_trading_engine import LiveTradingEngine
from engines.brokers import SimulatedBroker
from live_trading_chart import LiveTradingChart


class TradingCLI:
    """Command Line Interface for the trading system"""
    
    def __init__(self):
        self.strategies = {
            '1': ('Mean Reversion', MeanReversionStrategy),
            '2': ('Moving Average', MovingAverageStrategy),
            '3': ('RSI', RSIStrategy),
            '4': ('MACD', MACDStrategy),
            '5': ('Bollinger Bands', BollingerBandsStrategy),
            '6': ('Candlestick Patterns', CandlestickPatternsStrategy)
        }
        
        self.data_provider = None
        self.broker = None
        self.alpaca_data_provider = None
        self.alpaca_broker = None
        
    def display_banner(self):
        """Display application banner"""
        print("=" * 60)
        print("         BAT - Backtesting & Automated Trading")
        print("            Stocks & Cryptocurrency Trading ")
        print("=" * 60)
        print()
    
    def setup_data_provider(self):
        """Setup data provider"""
        print("Data Provider Setup")
        print("-" * 20)
        
        api_key = input("Enter your Polygon API key (or press Enter to use default): ").strip()
        if not api_key:
            api_key = "your-api-key-here"  # Default placeholder
        
        try:
            self.data_provider = PolygonDataProvider(api_key)
            print("✓ Data provider configured successfully")
        except Exception as e:
            print(f"✗ Error setting up data provider: {e}")
            return False
        
        return True
    
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

    def select_strategy(self):
        """Strategy selection menu"""
        print("\nAvailable Strategies:")
        print("-" * 20)
        
        for key, (name, _) in self.strategies.items():
            print(f"{key}. {name}")
        
        choice = input("Select strategy (1-6): ").strip()
        
        if choice not in self.strategies:
            print("Invalid choice")
            return None
        
        strategy_name, strategy_class = self.strategies[choice]
        
        # Get strategy parameters
        if choice == '1':  # Mean Reversion
            window = int(input("Enter window size (default 20): ") or "20")
            num_std = float(input("Enter standard deviations (default 2.0): ") or "2.0")
            return strategy_class(window=window, num_std=num_std)
        
        elif choice == '2':  # Moving Average
            short = int(input("Enter short window (default 1): ") or "1")
            medium = int(input("Enter medium window (default 5): ") or "5")
            long_win = int(input("Enter long window (default 25): ") or "25")
            return strategy_class(short_window=short, medium_window=medium, long_window=long_win)
        
        elif choice == '3':  # RSI
            window = int(input("Enter RSI window (default 14): ") or "14")
            oversold = float(input("Enter oversold threshold (default 30): ") or "30")
            overbought = float(input("Enter overbought threshold (default 70): ") or "70")
            return strategy_class(window=window, oversold_threshold=oversold, overbought_threshold=overbought)

        elif choice == '4':  # MACD
            fast = int(input("Enter fast EMA period (default 12): ") or "12")
            slow = int(input("Enter slow EMA period (default 26): ") or "26")
            signal = int(input("Enter signal line period (default 9): ") or "9")
            return strategy_class(fast=fast, slow=slow, signal=signal)

        elif choice == '5':  # Bollinger Bands
            window = int(input("Enter window size (default 20): ") or "20")
            num_std = float(input("Enter standard deviations (default 2): ") or "2")
            return strategy_class(window=window, num_std=num_std)

        elif choice == '6':  # Candlestick Patterns
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
        """Let user select between crypto and stocks for backtesting (Polygon only)"""
        print("\n Asset Type Selection (Backtesting):")
        print("======================================")
        print("1. Cryptocurrency (Polygon API)")
        print("2. Stocks (Polygon API)")
        print()

        while True:
            choice = input("Select asset type (1-2): ").strip()
            if choice in ['1', '2']:
                return choice
            print(" Invalid choice. Please enter 1 or 2.")

    def configure_data_parameters(self):
        """Configure data parameters with support for both crypto and stocks"""
        print("\nData Configuration:")
        print("-" * 20)

        # Select asset type
        asset_choice = self.select_asset_type()

        # Configure ticker based on asset type (Polygon only for backtesting)
        if asset_choice == '1':  # Crypto (Polygon)
            print("\n₿ Cryptocurrency Configuration (Polygon):")
            ticker = input("Enter crypto ticker (default X:BTCUSD): ").strip() or "X:BTCUSD"
            print("Common crypto tickers: X:BTCUSD, X:ETHUSD, X:DOGEUSD, X:LTCUSD")
        else:  # asset_choice == '2' - Stocks (Polygon)
            print("\n Stock Configuration (Polygon):")
            ticker = input("Enter stock ticker (default AAPL): ").strip() or "AAPL"
            print("Common stock tickers: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, SPY, QQQ")

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
        
        print(f"\n Running backtest for {strategy.name}...")
        print(f" Ticker: {data_params['ticker']}")
        print(f"Trading Mode: {'Long-only' if trading_mode == 'long_only' else 'Long/Short'}")
        print(f"⏰ Timespan: {data_params['timespan']}")
        print(f"📅 From: {data_params['from_date']} To: {data_params['to_date']}")

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
            engine = BacktestEngine(initial_balance, trading_mode, data_params['ticker'], position_percentage)
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

    def main_menu(self):
        """Main application menu"""
        while True:
            self.display_banner()
            
            print("Main Menu:")
            print("-" * 10)
            print("1. Run Backtest")
            print("2. Run Live Trading")
            print("3. Setup Data Provider")
            print("4. Setup Broker")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                if not self.data_provider:
                    print("Data provider not configured. Setting up now...")
                    if not self.setup_data_provider():
                        continue
                self.run_backtest()
                input("\nPress Enter to continue...")
            
            elif choice == '2':
                # Live trading uses Alpaca directly, no need for general data provider
                self.run_live_trading()
                input("\nPress Enter to continue...")
            
            elif choice == '3':
                self.setup_data_provider()
                input("\nPress Enter to continue...")
            
            elif choice == '4':
                self.setup_broker()
                input("\nPress Enter to continue...")
            
            elif choice == '5':
                print("Thank you for using BAT!")
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