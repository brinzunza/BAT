import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.mean_reversion import MeanReversionStrategy
from strategies.moving_average import MovingAverageStrategy
from strategies.trend_structure import TrendStructureStrategy
from data_providers.polygon_provider import PolygonDataProvider
from engines.backtest_engine import BacktestEngine
from engines.live_trading_engine import LiveTradingEngine
from engines.brokers import SimulatedBroker, AlpacaBroker


class TradingCLI:
    """Command Line Interface for the trading system"""
    
    def __init__(self):
        self.strategies = {
            '1': ('Mean Reversion', MeanReversionStrategy),
            '2': ('Moving Average', MovingAverageStrategy), 
            '3': ('Trend Structure', TrendStructureStrategy)
        }
        
        self.data_provider = None
        self.broker = None
        
    def display_banner(self):
        """Display application banner"""
        print("=" * 60)
        print("         BAT - Backtesting & Automated Trading")
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
    
    def select_strategy(self):
        """Strategy selection menu"""
        print("\nAvailable Strategies:")
        print("-" * 20)
        
        for key, (name, _) in self.strategies.items():
            print(f"{key}. {name}")
        
        choice = input("Select strategy (1-3): ").strip()
        
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
        
        elif choice == '3':  # Trend Structure
            return strategy_class()
        
        return None
    
    def configure_data_parameters(self):
        """Configure data parameters"""
        print("\nData Configuration:")
        print("-" * 20)
        
        ticker = input("Enter ticker (default C:EURUSD): ").strip() or "C:EURUSD"
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
            'limit': limit
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
        
        # Configure data
        data_params = self.configure_data_parameters()
        
        # Get initial balance
        initial_balance = float(input("Enter initial balance (default 10000): ") or "10000")
        
        print(f"\n📊 Running backtest for {strategy.name}...")
        print(f"📈 Ticker: {data_params['ticker']}")
        print(f"⏰ Timespan: {data_params['timespan']}")
        print(f"📅 From: {data_params['from_date']} To: {data_params['to_date']}")
        
        try:
            # Get data
            print("Fetching data...")
            df = self.data_provider.get_data(**data_params)
            print(f"✓ Retrieved {len(df)} data points")
            
            # Run backtest
            print("Running backtest...")
            engine = BacktestEngine(initial_balance)
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
                    print("\nDetailed Results:")
                    print(results.to_string(index=False))
                
                plot_results = input("Show balance plot? (y/n): ").strip().lower() == 'y'
                if plot_results:
                    engine.plot_results(results)
            else:
                print("No trades generated by this strategy with the given data.")
        
        except Exception as e:
            print(f"✗ Backtest failed: {e}")
    
    def run_live_trading(self):
        """Run live trading workflow"""
        print("\n" + "=" * 40)
        print("          LIVE TRADING MODE")
        print("=" * 40)
        
        if not self.broker:
            print("Broker not configured. Setting up now...")
            if not self.setup_broker():
                return
        
        # Select strategy
        strategy = self.select_strategy()
        if not strategy:
            return
        
        # Configure trading parameters
        print("\nTrading Configuration:")
        print("-" * 20)
        
        symbol = input("Enter trading symbol (default BTC): ").strip() or "BTC"
        quantity = float(input("Enter position size (default 1): ") or "1")
        sleep_interval = int(input("Enter check interval in seconds (default 60): ") or "60")
        max_iterations = input("Enter max iterations (default unlimited): ").strip()
        max_iterations = int(max_iterations) if max_iterations else None
        
        # Setup live trading engine
        initial_balance = float(input("Enter initial balance (default 10000): ") or "10000")
        engine = LiveTradingEngine(self.data_provider, self.broker, initial_balance)
        
        print(f"\n🚀 Starting live trading for {strategy.name}...")
        print(f"📈 Symbol: {symbol}")
        print(f"💰 Quantity: {quantity}")
        print(f"⏱️  Check interval: {sleep_interval} seconds")
        print(f"🔄 Max iterations: {max_iterations or 'Unlimited'}")
        print("\nPress Ctrl+C to stop trading...")
        
        try:
            engine.run_strategy(strategy, symbol, quantity, sleep_interval, max_iterations)
        except KeyboardInterrupt:
            print("\n\n⏹️  Trading stopped by user")
        finally:
            # Show final performance
            performance = engine.get_performance_summary()
            print("\n" + "=" * 40)
            print("         TRADING SUMMARY")
            print("=" * 40)
            print(f"Total Trades: {performance['total_trades']}")
            print(f"Win Rate: {performance['win_rate']:.2f}%")
            print(f"Final Balance: ${performance['current_balance']:.2f}")
            print(f"Total Return: ${performance['total_return']:.2f}")
            print(f"Percent Return: {performance['percent_return']:.2f}%")
            print(f"Current Position: {performance['current_position']}")
    
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
                if not self.data_provider:
                    print("Data provider not configured. Setting up now...")
                    if not self.setup_data_provider():
                        continue
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