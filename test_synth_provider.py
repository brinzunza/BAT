#!/usr/bin/env python3
"""
Test script for Synth data provider

This script tests the Synth provider's ability to fetch real-time data
at high frequency (1 data point per second).

Usage:
    python test_synth_provider.py API_KEY [BASE_URL] [NUM_FETCHES]

    API_KEY is required
    BASE_URL defaults to 'http://35.209.219.174:8000'
    NUM_FETCHES defaults to 10

Examples:
    python test_synth_provider.py bruno
    python test_synth_provider.py your_key http://35.209.219.174:8000 20
"""

import sys
import time
from datetime import datetime
from data_providers.synth_provider import SynthDataProvider


def test_synth_provider(api_key: str, base_url: str = "http://35.209.219.174:8000", num_fetches: int = 10):
    """
    Test the Synth provider by fetching data multiple times

    Args:
        api_key: Synth API key
        base_url: Synth API base URL
        num_fetches: Number of times to fetch data (default: 10)
    """
    print("=" * 60)
    print("SYNTH DATA PROVIDER TEST")
    print("=" * 60)
    print(f"Base URL: {base_url}")
    print(f"Number of fetches: {num_fetches}")
    print(f"Interval: ~1 second between fetches")
    print("=" * 60)

    # Create provider
    try:
        synth = SynthDataProvider(base_url, api_key)
        print("\n✓ SynthDataProvider created successfully")
    except Exception as e:
        print(f"\n✗ Error creating provider: {e}")
        return False

    # Test connection
    print("\n1. Testing API connection...")
    success, message = synth.test_connection()
    if not success:
        print(f"✗ Connection test failed: {message}")
        return False
    print(f"✓ {message}")

    # Fetch data multiple times to test real-time updates
    print(f"\n2. Fetching live data {num_fetches} times (1-second intervals)...")
    print("-" * 60)

    previous_price = None
    previous_timestamp = None

    for i in range(num_fetches):
        try:
            # Record fetch time
            fetch_start = time.time()

            # Fetch data
            df = synth.get_live_data('SYNTH')

            # Calculate fetch duration
            fetch_duration = time.time() - fetch_start

            if df.empty:
                print(f"Fetch {i+1}: ✗ No data returned")
                continue

            # Extract data from DataFrame
            row = df.iloc[0]
            timestamp = row['timestamp']
            price = row['Close']
            open_price = row['Open']
            high = row['High']
            low = row['Low']
            volume = row['Volume']

            # Calculate change from previous fetch
            price_change = ""
            time_change = ""

            if previous_price is not None:
                price_diff = price - previous_price
                price_change = f" (Δ ${price_diff:+.2f})"

            if previous_timestamp is not None:
                time_diff = (timestamp - previous_timestamp).total_seconds()
                time_change = f" ({time_diff:.1f}s since last)"

            # Display results
            print(f"\nFetch {i+1}/{num_fetches} [{fetch_duration*1000:.0f}ms]:")
            print(f"  Timestamp: {timestamp}{time_change}")
            print(f"  Price:     ${price:.2f}{price_change}")
            print(f"  OHLC:      O=${open_price:.2f} H=${high:.2f} L=${low:.2f} C=${price:.2f}")
            print(f"  Volume:    {volume:,.0f}")

            # Store for next iteration
            previous_price = price
            previous_timestamp = timestamp

            # Wait 1 second before next fetch (except on last iteration)
            if i < num_fetches - 1:
                time.sleep(1)

        except Exception as e:
            print(f"Fetch {i+1}: ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

    return True


def test_raw_tick_data(api_key: str, base_url: str = "http://35.209.219.174:8000"):
    """Test raw tick data retrieval"""
    print("\n" + "=" * 60)
    print("TESTING RAW TICK DATA")
    print("=" * 60)

    synth = SynthDataProvider(base_url, api_key)

    try:
        tick = synth.get_latest_tick('SYNTH')
        print("\nRaw API Response:")
        print("-" * 60)
        for key, value in tick.items():
            print(f"  {key}: {value}")
        print("-" * 60)
        return True
    except Exception as e:
        print(f"✗ Error fetching raw tick: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_synth_provider.py API_KEY [BASE_URL] [NUM_FETCHES]")
        print("\nExamples:")
        print("  python test_synth_provider.py bruno")
        print("  python test_synth_provider.py your_key http://35.209.219.174:8000 20")
        print("\nAPI_KEY is required")
        sys.exit(1)

    api_key = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://35.209.219.174:8000"
    num_fetches = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Run tests
    success = test_synth_provider(api_key, base_url, num_fetches)

    if success:
        print("\n")
        test_raw_tick_data(api_key, base_url)
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
