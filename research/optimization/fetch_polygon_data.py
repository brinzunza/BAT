#!/usr/bin/env python3
"""
Fetch historical data from Polygon API and save to CSV format compatible with backtest.c

Usage:
    python fetch_polygon_data.py <ticker> <timespan> <days> <output_file> [api_key]

Examples:
    # Fetch 7 days of minute data for Bitcoin
    python fetch_polygon_data.py X:BTCUSD minute 7 btc_data.csv YOUR_API_KEY

    # Fetch 30 days of hourly data for SPY
    python fetch_polygon_data.py SPY hour 30 spy_data.csv YOUR_API_KEY

    # Fetch 365 days of daily data for AAPL
    python fetch_polygon_data.py AAPL day 365 aapl_data.csv YOUR_API_KEY

Arguments:
    ticker       - Symbol to fetch (e.g., AAPL, SPY, X:BTCUSD)
    timespan     - Bar timespan: minute, hour, day, week, month
    days         - Number of days to fetch (lookback from today)
    output_file  - Output CSV filename
    api_key      - Polygon API key (optional if set in environment)
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import requests


def fetch_polygon_data(ticker, timespan, days, api_key):
    """
    Fetch historical aggregate bars from Polygon API

    Args:
        ticker: Stock/crypto symbol
        timespan: Bar interval (minute, hour, day, week, month)
        days: Number of days to look back
        api_key: Polygon API key

    Returns:
        DataFrame with OHLCV data
    """
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)

    to_date_str = to_date.strftime('%Y-%m-%d')
    from_date_str = from_date.strftime('%Y-%m-%d')

    # Polygon API endpoint
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/{timespan}/"
        f"{from_date_str}/{to_date_str}?adjusted=true&sort=asc&limit=100000&apiKey={api_key}"
    )

    print(f"Fetching data for {ticker}...")
    print(f"  Date range: {from_date_str} to {to_date_str}")
    print(f"  Timespan: {timespan}")

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)

    data = response.json()

    if 'results' not in data:
        print(f"Error: No results returned from API")
        print(f"Response: {data}")
        sys.exit(1)

    # Convert to DataFrame
    df = pd.DataFrame(data['results'])

    # Convert timestamp from milliseconds to readable format
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')

    # Rename columns to match expected format: timestamp,open,high,low,close,volume
    df = df.rename(columns={
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    })

    # Keep only required columns in correct order
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    print(f"  Fetched {len(df)} bars")

    return df


def save_to_csv(df, output_file):
    """
    Save DataFrame to CSV in format compatible with backtest.c

    Format: timestamp,open,high,low,close,volume
    """
    # Convert timestamp to ISO format string
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Save to CSV with header
    df.to_csv(output_file, index=False)

    print(f"\nData saved to: {output_file}")
    print(f"  Format: timestamp,open,high,low,close,volume")
    print(f"  Rows: {len(df)}")

    # Show first few rows
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    ticker = sys.argv[1]
    timespan = sys.argv[2]
    days = int(sys.argv[3])
    output_file = sys.argv[4]

    # Get API key from argument or environment
    if len(sys.argv) > 5:
        api_key = sys.argv[5]
    else:
        api_key = os.environ.get('POLYGON_API_KEY')
        if not api_key:
            print("Error: API key not provided and POLYGON_API_KEY environment variable not set")
            print("\nUsage:")
            print(__doc__)
            sys.exit(1)

    # Validate timespan
    valid_timespans = ['minute', 'hour', 'day', 'week', 'month']
    if timespan not in valid_timespans:
        print(f"Error: Invalid timespan '{timespan}'")
        print(f"Valid options: {', '.join(valid_timespans)}")
        sys.exit(1)

    # Fetch data
    df = fetch_polygon_data(ticker, timespan, days, api_key)

    # Save to CSV
    save_to_csv(df, output_file)

    print(f"\n✓ Ready to use with backtest.c:")
    print(f"  gcc -o backtest backtest.c -lm")
    print(f"  ./backtest {output_file}")


if __name__ == '__main__':
    main()
