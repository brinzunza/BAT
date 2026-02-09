#!/usr/bin/env python3
"""
Quick test for Synth live trading integration

This script tests that the Synth provider works with LiveTradingChart
without actually starting the GUI.

Usage:
    python test_synth_live_trading.py API_KEY
"""

import sys
from data_providers.synth_provider import SynthDataProvider
from engines.brokers import SimulatedBroker
from engines.live_trading_engine import LiveTradingEngine
from strategies.mean_reversion import MeanReversionStrategy


def test_synth_live_trading(api_key: str):
    """Test Synth integration with live trading engine"""

    print("=" * 60)
    print("SYNTH LIVE TRADING INTEGRATION TEST")
    print("=" * 60)

    # Create Synth provider
    print("\n1. Creating Synth provider...")
    try:
        synth = SynthDataProvider(api_key=api_key)
        print("   ✓ Synth provider created")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test connection
    print("\n2. Testing Synth API connection...")
    try:
        success, message = synth.test_connection()
        if success:
            print(f"   ✓ {message}")
        else:
            print(f"   ✗ {message}")
            return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Create simulated broker
    print("\n3. Creating SimulatedBroker...")
    try:
        broker = SimulatedBroker(initial_balance=10000)
        print("   ✓ SimulatedBroker created with $10,000 balance")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Create strategy
    print("\n4. Creating Mean Reversion strategy...")
    try:
        strategy = MeanReversionStrategy(
            lookback_period=20,
            std_dev_multiplier=2.0,
            holding_period=5
        )
        print("   ✓ Strategy created")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Create trading engine
    print("\n5. Creating LiveTradingEngine...")
    try:
        engine = LiveTradingEngine(
            data_provider=synth,
            broker_interface=broker,
            initial_balance=10000,
            trading_mode="long_only",
            position_percentage=20.0
        )
        print("   ✓ LiveTradingEngine created")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Fetch data and process signals (single iteration)
    print("\n6. Testing data fetch and signal generation...")
    try:
        df = synth.get_live_data('SYNTH')
        print(f"   ✓ Fetched data: {len(df)} rows")
        print(f"   ✓ Latest price: ${df.iloc[-1]['Close']:.2f}")

        # Generate signals
        df_with_signals = strategy.generate_signals(df)
        signal_names = strategy.get_signal_names()
        buy_signal = df_with_signals.iloc[-1][signal_names['buy']]
        sell_signal = df_with_signals.iloc[-1][signal_names['sell']]

        print(f"   ✓ Buy signal: {buy_signal}")
        print(f"   ✓ Sell signal: {sell_signal}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # Test LiveTradingChart initialization (without GUI)
    print("\n7. Testing LiveTradingChart initialization...")
    try:
        from ui.live_trading_chart import LiveTradingChart

        chart = LiveTradingChart(
            strategy=strategy,
            symbol='SYNTH',
            trading_mode='long_only',
            position_percentage=20,
            use_simulated_broker=True,
            initial_balance=10000,
            data_provider=synth,
            broker_interface=broker
        )
        print("   ✓ LiveTradingChart initialized successfully")
        print(f"   ✓ Is forex: {chart.is_forex}")
        print(f"   ✓ Symbol: {chart.symbol}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSynth live trading integration is working correctly.")
    print("You can now use it via the CLI:")
    print("  python ui/cli_interface.py")
    print("  → Live Trading")
    print("  → Synthetic Data (Synth)")
    print()

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_synth_live_trading.py API_KEY")
        print("\nExample:")
        print("  python test_synth_live_trading.py bruno")
        sys.exit(1)

    api_key = sys.argv[1]
    success = test_synth_live_trading(api_key)

    if not success:
        print("\n❌ Tests failed!")
        sys.exit(1)
