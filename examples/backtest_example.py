#!/usr/bin/env python3
"""
Example: Running a backtest with the modular system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.mean_reversion import MeanReversionStrategy
from strategies.moving_average import MovingAverageStrategy
from data_providers.polygon_provider import PolygonDataProvider
from engines.backtest_engine import BacktestEngine


def main():
    """Example backtest execution"""
    
    # Setup data provider
    api_key = "your-polygon-api-key-here"  # Replace with your actual API key
    data_provider = PolygonDataProvider(api_key)
    
    # Setup strategy
    strategy = MeanReversionStrategy(window=20, num_std=2.0)
    
    # Get data
    print("Fetching data...")
    df = data_provider.get_data(
        ticker='C:EURUSD',
        timespan='minute',
        from_date='2023-01-01',
        to_date='2023-02-01',
        limit=50000
    )
    print(f"Retrieved {len(df)} data points")
    
    # Run backtest
    print(f"Running backtest for {strategy.name}...")
    engine = BacktestEngine(initial_balance=10000)
    results = engine.backtest(df, strategy)
    
    # Display results
    if len(results) > 0:
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        
        engine.print_analysis(results)
        
        # Show first few trades
        print("\nFirst 5 trades:")
        print(results.head().to_string(index=False))
        
        # Plot results
        engine.plot_results(results)
    else:
        print("No trades generated.")


if __name__ == "__main__":
    main()