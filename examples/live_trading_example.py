#!/usr/bin/env python3
"""
Example: Running live trading with simulated broker
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.mean_reversion import MeanReversionStrategy
from data_providers.polygon_provider import PolygonDataProvider
from engines.live_trading_engine import LiveTradingEngine
from engines.brokers import SimulatedBroker


def main():
    """Example live trading execution"""
    
    # Setup data provider
    api_key = "your-polygon-api-key-here"  # Replace with your actual API key
    data_provider = PolygonDataProvider(api_key)
    
    # Setup simulated broker
    broker = SimulatedBroker(initial_balance=10000)
    
    # Setup strategy
    strategy = MeanReversionStrategy(window=20, num_std=2.0)
    
    # Setup live trading engine
    engine = LiveTradingEngine(data_provider, broker, initial_balance=10000)
    
    print(f"Starting live trading simulation for {strategy.name}...")
    print("This will run for 5 iterations (5 minutes) as a demo")
    print("Press Ctrl+C to stop early")
    
    try:
        # Run for limited iterations as demo
        engine.run_strategy(
            strategy=strategy,
            symbol='BTC',
            quantity=0.1,
            sleep_interval=60,  # Check every minute
            max_iterations=5    # Run for 5 iterations only
        )
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
    
    # Show results
    performance = engine.get_performance_summary()
    trade_history = engine.get_trade_history()
    
    print("\n" + "="*50)
    print("LIVE TRADING RESULTS")
    print("="*50)
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Win Rate: {performance['win_rate']:.2f}%")
    print(f"Final Balance: ${performance['current_balance']:.2f}")
    print(f"Total Return: ${performance['total_return']:.2f}")
    print(f"Percent Return: {performance['percent_return']:.2f}%")
    
    if len(trade_history) > 0:
        print("\nTrade History:")
        print(trade_history.to_string(index=False))
    
    # Show broker account info
    broker_info = broker.get_account_info()
    print(f"\nBroker Account Info:")
    print(f"Balance: ${broker_info['equity']:.2f}")
    print(f"Positions: {broker_info['positions']}")
    print(f"Total Trades Executed: {broker_info['trade_count']}")


if __name__ == "__main__":
    main()